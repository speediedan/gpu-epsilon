# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Phase evidence against the two implementations it replaces, kept verbatim in ``tests/oracles``.

The new reports must equal each oracle's on a synthetic corpus of per-rank records, except where an intended
difference (:data:`INTENDED`) explains it; each intended difference must actually occur; and the harness must fail when
the code under test is broken (a planted ``classify`` defect, a planted rank-merge defect). The corpus must reach every
classification and every row of each project's gate.
"""
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from oracles import load

from gpu_epsilon import evidence
from gpu_epsilon import phase_evidence as pe
from gpu_epsilon import release_eligibility as re_
from gpu_epsilon.evidence import CLASSIFICATIONS
from gpu_epsilon.records import node_outcome, read_sessions

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="phase evidence reads pyproject.toml with tomllib")

CONFIGS = Path(__file__).parent / "configs"
FTS, IT = "finetuning-scheduler", "interpretune"
ORACLE = {FTS: "fts_phase_manifest", IT: "it_phase_report"}
SCOPES = {FTS: ("full", "smoke"), IT: ("full", "selected")}
JOB_CLASSES = ("ci-any-vendor", "ci-vendor-marked", "ci-release")
NODES = [f"tests/test_x.py::test_{c}" for c in "abcd"]
STRAY = "tests/test_x.py::test_stray"
STAGES = (
    ("setup", "errors"),
    ("call", "passed"),
    ("call", "failed"),
    ("setup", "skipped"),
    ("call", "xfailed"),
    ("call", "xpassed"),
    ("teardown", "errors"),
)
# Where the new code is meant to differ from an oracle, by project. Every difference the corpus finds must be one of
# these, and every one of these must be found.
INTENDED = {
    FTS: {
        # both forks crash on a line a dying process left half written; records.read_sessions skips it
        "half-written-line",
        # a rank seen in several sessions: a session that never finished, a failure or an error in any of them now
        # wins, where finetuning-scheduler kept the last verdict it read (a failure followed by a pass read as a pass)
        "rank-seen-in-several-sessions",
    },
    IT: {
        "half-written-line",
        # rank 0's verdict decides, and other ranks can only fail a node; interpretune kept the last verdict of any
        # rank, so a node seen only on rank 1 counted as executed
        "nonzero-rank",
    },
}
CORPUS_SIZE = 400


def config(project: str) -> pe.EvidenceConfig:
    return pe.EvidenceConfig.load(CONFIGS / f"{project}.phase-evidence.toml")


# the corpus


@dataclass
class Session:
    """One pytest session's records, rendered as the plugin writes them."""

    rank: int = 0
    collected: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)  # (nodeid, when, outcome)
    started: bool = True
    collected_event: bool = True
    finished: bool = True
    world_size: int | None = None
    half: bool = False

    def text(self) -> str:
        ident = {"rank": self.rank, "world_size": self.world_size, "pid": 1}
        records = [{"event": "start", **ident}] if self.started else []
        if self.collected_event:
            records.append({"event": "collected", "nodeids": list(self.collected), **ident})
        records += [{"event": "outcome", "nodeid": n, "when": w, "outcome": o, **ident} for n, w, o in self.outcomes]
        if self.finished:
            records.append({"event": "finish", "exitstatus": 0, **ident})
        lines = [json.dumps(r) for r in records]
        if self.half:
            lines.append('{"event": "finish", "exitst')
        return "\n".join(lines) + ("\n" if lines and not self.half else "")

    def entries(self) -> dict[str, list[str]]:
        """What this session contributes to each node's verdicts on its rank, in the oracles' own terms."""
        verdicts: dict[str, list[str]] = {}
        for n, _, o in self.outcomes:
            verdicts.setdefault(n, []).append(o)
        out = {n: next((o for o in ("failed", "errors") if o in v), v[0]) for n, v in verdicts.items()}
        if self.started and not self.finished:
            out.update({n: "incomplete" for n in (self.collected if self.collected_event else []) if n not in out})
        return out


@dataclass
class Case:
    seed: int
    manifest: dict
    sessions: list
    reviewed: dict
    vendor: str
    evidence_class: str
    job_class: str

    def write(self, root: Path) -> Path:
        root.mkdir(parents=True)
        for i, s in enumerate(self.sessions):
            (root / f"records-rank{s.rank}-pid{i:04d}.jsonl").write_text(s.text(), encoding="utf-8")
        return root

    @property
    def half_written(self) -> bool:
        return any(s.half for s in self.sessions)

    @property
    def nonzero_rank(self) -> bool:
        return any(s.rank != 0 and s.text() for s in self.sessions)

    @property
    def rank_merge_changes_a_verdict(self) -> bool:
        """Whether some rank saw a node in several sessions and the last verdict differs from the merged one."""
        seen: dict[tuple, list[str]] = {}
        for s in self.sessions:
            if s.text():
                for n, verdict in s.entries().items():
                    seen.setdefault((n, s.rank), []).append(verdict)
        merged = lambda v: next((o for o in ("incomplete", "failed", "errors") if o in v), v[-1])  # noqa: E731
        return any(len(v) > 1 and v[-1] != merged(v) for v in seen.values())

    def explains(self, project: str) -> set[str]:
        reasons = set()
        if self.half_written:
            reasons.add("half-written-line")
        if project == FTS and self.rank_merge_changes_a_verdict:
            reasons.add("rank-seen-in-several-sessions")
        if project == IT and self.nonzero_rank:
            reasons.add("nonzero-rank")
        return reasons


def _random_outcomes(rng, nodes):
    return [(n, *rng.choice(STAGES)) for n in nodes if rng.random() < 0.8]


def _another_stage(rng, case):
    rng.choice(case.sessions).outcomes.append((rng.choice(NODES + [STRAY]), *rng.choice(STAGES)))


def _no_verdict(rng, case):
    node = rng.choice(NODES)
    for s in case.sessions:
        s.outcomes[:] = [o for o in s.outcomes if o[0] != node]


def _never_finished(rng, case):
    rng.choice(case.sessions).finished = False


def _another_session(rank):
    def mutate(rng, case):
        nodes = rng.sample(NODES, rng.randint(1, 4))
        finished = rng.random() < 0.75
        case.sessions.append(Session(rank, nodes, _random_outcomes(rng, nodes), finished=finished,
                                     world_size=2 if rank else None))
    return mutate


def _unmanifested(rng, case):
    rng.choice(case.sessions).outcomes.append((STRAY, "call", "passed"))


def _half_written(rng, case):
    session = rng.choice(case.sessions)
    session.half, session.finished = True, False


def _empty_file(rng, case):
    case.sessions.append(Session(started=False, collected_event=False, finished=False))


def _died_before_collecting(rng, case):
    case.sessions.append(Session(collected_event=False, finished=False))


def _world_size(rng, case):
    rng.choice(case.sessions).world_size = rng.choice([1, 2, 4])


def _all_skipped(rng, case):
    for s in case.sessions:
        s.outcomes[:] = [(n, "setup", "skipped") for n, _, _ in s.outcomes]


def _earlier_failure(rng, case):
    """An earlier session on the same rank failed the node a later one passes (a two-stage run, a retried rank)."""
    node, rank = rng.choice(NODES), rng.choice([0, 0, 1])
    case.sessions.insert(0, Session(rank, [node], [(node, "call", rng.choice(["failed", "errors"]))]))
    if rank:
        case.sessions.append(Session(rank, [node], [(node, "call", "passed")], world_size=2))


def _nothing_selected(rng, case):
    case.manifest.update(selected_node_ids=[])
    case.sessions[:] = [Session(0, [], [])]


MUTATIONS = {
    "another stage": _another_stage,
    "no verdict": _no_verdict,
    "never finished": _never_finished,
    "another rank": _another_session(1),
    "another rank-0 session": _another_session(0),
    "unmanifested node": _unmanifested,
    "half-written line": _half_written,
    "empty file": _empty_file,
    "died before collecting": _died_before_collecting,
    "world size": _world_size,
    "unreviewed": lambda rng, case: case.reviewed.clear(),
    "an earlier session failed it": _earlier_failure,
    "all skipped": _all_skipped,
    "nothing selected": _nothing_selected,
    "empty selection, nodes still ran": lambda rng, case: case.manifest.update(selected_node_ids=[]),
    "narrower selection": lambda rng, case: case.manifest.update(selected_node_ids=rng.sample(NODES, 2)),
    "collection error": lambda rng, case: case.manifest.update(collection_errors=["tests/test_broken.py"]),
}


def make_case(project: str, seed: int, phases: list[str]) -> Case:
    rng = random.Random(f"{project}-{seed}")
    outcomes = []
    for n in NODES:
        r = rng.random()
        kind = "passed" if r < 0.8 else "skipped" if r < 0.9 else "xfailed" if r < 0.96 else "xpassed"
        outcomes.append((n, "setup" if kind == "skipped" else "call", kind))
    manifest = {
        "repository": f"speediedan/{project}",
        "revision": "0" * 40,
        "phase": rng.choice(phases),
        "scope": rng.choice(SCOPES[project]),
        "selector": "s",
        "tree_dirty_paths": [],
        "raw_node_ids": list(NODES),
        "selected_node_ids": list(NODES),
        "collection_errors": [],
        "collect_exit_codes": {"raw": 0, "selected": 0},
    }
    case = Case(
        seed,
        manifest,
        [Session(0, list(NODES), outcomes)],
        {n: "reviewed" for n, _, o in outcomes if o in ("skipped", "xfailed") and rng.random() < 0.7},
        rng.choice(["nvidia", "amd"]),
        rng.choice(pe.EVIDENCE_CLASSES),
        rng.choice(JOB_CLASSES),
    )
    for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
        MUTATIONS[rng.choice(sorted(MUTATIONS))](rng, case)
    return case


# the harness


def oracle_gate(project: str, oracle, classification: str, scope: str, job_class: str) -> bool:
    if project == IT:
        return oracle.gate(classification, scope, job_class)
    return classification == "clean"  # finetuning-scheduler's report exits 0 only on a clean phase


def comparable(project: str, report: dict, new: bool) -> dict:
    """A report in the terms both sides share. interpretune's schema changes (its ``missing_completions`` and per-node
    ``sessions`` became finetuning-scheduler's rank fields) are an intended difference checked on its own."""
    if project == FTS:
        return report
    view = dict(report)
    nodes = view.pop("nodes")
    if new:
        view.pop("rank_failures")
        view["missing"] = view.pop("missing_rank_completions")
    else:
        view["missing"] = view.pop("missing_completions")
    view["node_outcomes"] = {n: v["outcome"] for n, v in nodes.items()}
    return view


@dataclass
class Result:
    unexplained: list = field(default_factory=list)
    explained: dict = field(default_factory=dict)  # reason -> cases that differ and are explained by it
    classifications: set = field(default_factory=set)
    gate_rows: set = field(default_factory=set)  # (classification, full scope, release job, passes)


def compare(project: str, cases: list[Case], tmp_path: Path) -> Result:
    cfg, oracle, result = config(project), load(ORACLE[project]), Result()
    versions = {"python": "3.13.0"}
    for case in cases:
        record_dir = case.write(tmp_path / f"{project}-{case.seed}")
        kwargs = dict(vendor=case.vendor, evidence_class=case.evidence_class, lease_mode="tool", image="i",
                      versions=versions, reviewed_exclusions=case.reviewed)
        new = pe.build_report(cfg, case.manifest, read_sessions(record_dir), **kwargs)
        scope = case.manifest["scope"]
        passes = pe.gate(cfg, new["classification"], scope, case.job_class)
        result.classifications.add(new["classification"])
        result.gate_rows.add((new["classification"], scope == "full", case.job_class == "ci-release", passes))
        try:
            old = oracle.build_report(case.manifest, oracle.read_records(record_dir), **kwargs)
        except ValueError:  # json.JSONDecodeError
            differs = "the oracle raised"
        else:
            old_passes = oracle_gate(project, oracle, old["classification"], scope, case.job_class)
            same = comparable(project, new, True) == comparable(project, old, False) and passes == old_passes
            differs = None if same else f"{old['classification']} -> {new['classification']}"
        if differs is None:
            continue
        # the oracle raising is explained only by a half-written line, and a different report only by the others
        reasons = case.explains(project) & INTENDED[project]
        if differs == "the oracle raised":
            reasons &= {"half-written-line"}
        else:
            reasons -= {"half-written-line"}
        for reason in reasons:
            result.explained.setdefault(reason, []).append(case.seed)
        if not reasons:
            result.unexplained.append(f"seed {case.seed}: {differs}")
    return result


def corpus(project: str) -> list[Case]:
    phases = [p.name for p in config(project).phases]
    return [make_case(project, seed, phases) for seed in range(CORPUS_SIZE)]


# the gate each project's configuration must reach, row by row: (classification, full scope, release job, passes)
GATE_TABLE = {
    FTS: {
        ("clean", True, False, True),
        ("empty", True, False, False),
        ("unreviewed-exclusions", True, False, False),
        ("failing", True, False, False),
    },
    IT: {
        ("clean", True, True, True),
        ("failing", True, False, False),
        ("empty", True, False, False),
        ("empty", False, False, True),
        ("unreviewed-exclusions", True, False, True),
        ("unreviewed-exclusions", True, True, False),
    },
}


def problems(project: str, result: Result) -> list[str]:
    """What makes a comparison fail: a difference no intended difference explains, or an intended difference the
    corpus never finds (the new code agreeing with the oracle where it is meant not to)."""
    missing = sorted(INTENDED[project] - set(result.explained))
    return result.unexplained + [f"intended difference not found: {reason}" for reason in missing]


@pytest.mark.parametrize("project", [FTS, IT])
def test_reports_match_the_oracle_except_where_intended(project, tmp_path):
    result = compare(project, corpus(project), tmp_path)
    assert problems(project, result) == []
    assert result.classifications == set(CLASSIFICATIONS)
    assert GATE_TABLE[project] <= result.gate_rows


def _no_xpass_classify(report, failure_counts=()):
    return evidence.classify({**report, "outcomes": {**report["outcomes"], "xpassed": 0}}, failure_counts)


@pytest.mark.parametrize("project", [FTS, IT])
def test_a_planted_classify_defect_fails_the_harness(project, tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "classify", _no_xpass_classify)
    assert compare(project, corpus(project), tmp_path).unexplained


@pytest.mark.parametrize(
    "project, defect, shows_as",
    [
        # interpretune: a rank-0 node seen in two sessions is not an intended difference, so keeping the first
        # verdict instead of the last must show
        (IT, lambda verdicts: verdicts[0], "seed"),
        # finetuning-scheduler: the allowance is exact (only where the merge changes the last verdict), so keeping
        # the first verdict shows too
        (FTS, lambda verdicts: verdicts[0], "seed"),
        # and reverting to finetuning-scheduler's last-verdict rule makes the intended difference vanish
        (FTS, lambda verdicts: verdicts[-1], "intended difference not found: rank-seen-in-several-sessions"),
    ],
    ids=["it-first", "fts-first", "fts-last"],
)
def test_a_planted_rank_merge_defect_fails_the_harness(project, defect, shows_as, tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "_rank_verdict", defect)
    found = problems(project, compare(project, corpus(project), tmp_path))
    assert any(p.startswith(shows_as) for p in found), found


# each intended difference, as one named case that must differ


def _manifest(project, selected, phase="standalone", scope="full"):
    return {"repository": f"speediedan/{project}", "revision": "r", "phase": phase, "scope": scope, "selector": "s",
            "raw_node_ids": list(selected), "selected_node_ids": list(selected), "collection_errors": []}


def _both(project, manifest, sessions, tmp_path):
    record_dir = Case(0, manifest, sessions, {}, "amd", "pipeline", "ci-any-vendor").write(tmp_path / "r")
    oracle = load(ORACLE[project])
    kwargs = dict(vendor="amd", evidence_class="pipeline")
    new = pe.build_report(config(project), manifest, read_sessions(record_dir), **kwargs)
    try:
        old = oracle.build_report(manifest, oracle.read_records(record_dir), **kwargs)
    except ValueError as exc:
        old = exc
    return old, new


def test_drift_a_failure_then_a_pass_on_one_rank_fails_the_node(tmp_path):
    a = NODES[0]
    sessions = [Session(0, [a], [(a, "call", "failed")]), Session(0, [a], [(a, "call", "passed")])]
    old, new = _both(FTS, _manifest(FTS, [a]), sessions, tmp_path)
    assert old["classification"] == "clean" and old["nodes"][a]["ranks"] == {"0": "passed"}
    assert new["classification"] == "failing" and new["nodes"][a]["ranks"] == {"0": "failed"}


@pytest.mark.parametrize(
    "sessions, old_outcome, new_outcome",
    [
        # seen only on rank 1: interpretune counted it executed
        ([Session(1, [NODES[0]], [(NODES[0], "call", "passed")])], "passed", "unexecuted"),
        # rank 0 skipped, rank 1 passed: interpretune kept the last verdict read
        (
            [Session(0, [NODES[0]], [(NODES[0], "setup", "skipped")]),
             Session(1, [NODES[0]], [(NODES[0], "call", "passed")])],
            "passed",
            "skipped",
        ),
    ],
)
def test_drift_rank_zero_decides(tmp_path, sessions, old_outcome, new_outcome):
    old, new = _both(IT, _manifest(IT, [NODES[0]]), sessions, tmp_path)
    assert old["nodes"][NODES[0]]["outcome"] == old_outcome
    assert new["nodes"][NODES[0]]["outcome"] == new_outcome


@pytest.mark.parametrize("project", [FTS, IT])
def test_drift_a_half_written_line_is_read_not_fatal(project, tmp_path):
    a = NODES[0]
    session = Session(0, [a], [(a, "call", "passed")], finished=False, half=True)
    old, new = _both(project, _manifest(project, [a]), [session], tmp_path)
    assert isinstance(old, ValueError)
    # the line was the session's finish, so the session never finished
    assert new["missing_rank_completions"] == 1 and new["classification"] == "failing"
    assert new["nodes"][a]["outcome"] == "passed"


def test_drift_interpretunes_report_takes_the_rank_schema(tmp_path):
    a = NODES[0]
    old, new = _both(IT, _manifest(IT, [a]), [Session(0, [a], [(a, "call", "passed")])], tmp_path)
    assert set(new) - set(old) == {"rank_failures", "missing_rank_completions"}
    assert set(old) - set(new) == {"missing_completions"}
    assert old["nodes"][a] == {"outcome": "passed", "sessions": 1}
    assert new["nodes"][a] == {"outcome": "passed", "ranks": {"0": "passed"}, "world_size": None}
    fts_old, fts_new = _both(FTS, _manifest(FTS, [a]), [Session(0, [a], [(a, "call", "passed")])], tmp_path / "f")
    assert fts_old == fts_new  # finetuning-scheduler's schema is the one kept


def test_drift_a_rerun_attempt_records_no_outcome():
    rerun = SimpleNamespace(outcome="rerun", failed=False, skipped=False, passed=False, when="call")
    assert node_outcome(rerun) is None
    assert load("it_phase_plugin").node_outcome(rerun) is None
    assert load("fts_phase_plugin").node_outcome(rerun) == "passed"  # an attempt that failed, recorded as passed


@pytest.mark.parametrize(
    "when, outcome, wasxfail",
    [(w, o, x) for w in ("setup", "call", "teardown") for o in ("passed", "failed", "skipped") for x in (False, True)],
)
def test_node_outcome_matches_both_plugins_outside_reruns(when, outcome, wasxfail):
    report = SimpleNamespace(when=when, outcome=outcome, failed=outcome == "failed", skipped=outcome == "skipped")
    if wasxfail:
        report.wasxfail = "reason"
    expected = node_outcome(report)
    assert load("it_phase_plugin").node_outcome(report) == expected
    assert load("fts_phase_plugin").node_outcome(report) == expected


# the gate, exhaustively


@pytest.mark.parametrize("project", [FTS, IT])
def test_the_gate_matches_the_oracle_everywhere(project):
    cfg, oracle = config(project), load(ORACLE[project])
    for classification in CLASSIFICATIONS:
        for scope in SCOPES[project]:
            for job_class in JOB_CLASSES:
                expected = oracle_gate(project, oracle, classification, scope, job_class)
                assert pe.gate(cfg, classification, scope, job_class) is expected, (classification, scope, job_class)


# configuration against each oracle's own tables


def test_finetuning_schedulers_config_is_its_phase_table():
    cfg, pm = config(FTS), load(ORACLE[FTS])
    assert cfg.repository == pm.REPOSITORY and [p.name for p in cfg.phases] == list(pm.PHASES)
    for phase in cfg.phases:
        old = pm.PHASES[phase.name]
        assert (phase.selection_env, phase.args, phase.command, phase.release, phase.measured) == (
            old.selection_env, old.args, old.command, old.release, old.measured), phase.name
        assert phase.paths == old.paths or phase.paths_from == {"glob": "tests/test_*.py", "contains": pm.POOL_FIXTURE}
    assert cfg.selection_vars == pm.SELECTION_VARS and cfg.collect_env == pm.RUN_ENV
    assert cfg.device_vars == pm.DEVICE_VARS and cfg.vendors == pm.VENDORS
    assert cfg.scopes["smoke"].select_file == pm.SMOKE_FILE and cfg.scopes["smoke"].phases == ("standalone",)
    assert cfg.run_env_exports_selection


def test_interpretunes_config_is_its_phase_table():
    cfg, pr = config(IT), load(ORACLE[IT])
    assert cfg.repository == pr.REPOSITORY and [p.name for p in cfg.phases] == list(pr.PHASES)
    for phase in cfg.phases:
        old = pr.PHASES[phase.name]
        assert (phase.paths, phase.selection_env, phase.args, phase.command, phase.release, phase.measured) == (
            old.paths, old.selection_env, (), old.command, old.release, False), phase.name
    assert cfg.selection_vars == pr.SELECTION_VARS and cfg.collect_env == {}
    assert cfg.device_vars == pr.DEVICE_VARS and cfg.vendors == pr.VENDORS
    assert cfg.env_scope is not None and cfg.env_scope.select_file_env == "IT_GPU_SELECTION_FILE"
    assert not cfg.run_env_exports_selection


# run identity


FAKE_TORCH = SimpleNamespace(
    __version__="2.15.0+cu132",
    version=SimpleNamespace(git_version="c" * 40, cuda="13.2", hip=None),
    cuda=SimpleNamespace(device_count=lambda: 0, get_device_name=lambda i: "gpu"),
)


def test_drift_a_missing_package_is_recorded_as_none(monkeypatch):
    for name in ("torch", "lightning", "finetuning_scheduler"):
        monkeypatch.setitem(sys.modules, name, None)  # import fails, as when the package is absent
    old, new = load(ORACLE[FTS]).installed_versions(), pe.installed_versions(config(FTS))
    for key in ("torch", "lightning", "finetuning_scheduler"):
        assert old[key].startswith("unavailable: ") and new[key] is None
        # the release checker's presence test reads the old placeholder as an installed version
        assert (old.get(key) or "").split("+")[0] and not (new.get(key) or "").split("+")[0]
    old_it = load(ORACLE[IT]).installed_versions()
    new_it = pe.installed_versions(config(IT))
    assert old_it["torch"].startswith("unavailable: ") and new_it["torch"] is None
    assert {k: v for k, v in old_it.items() if k != "torch"}.items() <= new_it.items()


@pytest.mark.parametrize("project", [FTS, IT])
def test_installed_versions_match_the_oracle_when_installed(project, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FAKE_TORCH)
    monkeypatch.setitem(sys.modules, "lightning", SimpleNamespace(__version__="2.6.1"))
    monkeypatch.setitem(sys.modules, "finetuning_scheduler", SimpleNamespace(__version__="2.13.0"))
    assert pe.installed_versions(config(project)) == load(ORACLE[project]).installed_versions()
    assert pe.device_identity(config(project)) == load(ORACLE[project]).device_identity()


def test_version_sources_read_the_installed_distribution():
    from importlib import metadata

    assert pe.version_of("dist:pytest") == metadata.version("pytest") == pe.version_of("module:pytest")
    assert len(pe.version_of("record:pytest") or "") == 64
    assert pe.version_of("vcs:gpu-epsilon") is None  # an editable or index install names no commit
    for kind in pe.VERSION_SOURCES:
        assert pe.version_of(f"{kind}:not-an-installed-distribution") is None


@pytest.mark.parametrize("project", [FTS, IT])
def test_device_identity_matches_the_oracle_without_torch(project, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-a,GPU-b")
    assert pe.device_identity(config(project)) == load(ORACLE[project]).device_identity()


# run-env


@pytest.mark.parametrize("project", [FTS, IT])
def test_run_env_matches_the_oracle_under_the_shared_names(project, tmp_path, monkeypatch):
    cfg, oracle = config(project), load(ORACLE[project])
    monkeypatch.setattr(oracle, "ROOT", cfg.root)
    monkeypatch.setenv("PYTEST_ADDOPTS", "-x")
    old_dir_env = "FTS_PHASE_RECORD_DIR" if project == FTS else "IT_PHASE_RECORD_DIR"
    for phase in cfg.phases:
        old = oracle.run_env(phase.name, tmp_path / "old")
        new = pe.run_env(cfg, phase.name, tmp_path / "new")
        renamed = {k: v.replace(str(tmp_path / "old"), str(tmp_path / "new")) for k, v in old.items()}
        renamed[pe.PHASE_RECORD_DIR_ENV] = renamed.pop(old_dir_env)
        renamed["PYTEST_ADDOPTS"] = renamed["PYTEST_ADDOPTS"].replace(oracle.PLUGIN, pe.PLUGIN)
        if project == IT:
            renamed[pe.PHASE_RECORD_ROOT_ENV] = renamed.pop("IT_PHASE_RECORD_ROOT")
        else:  # drift: finetuning-scheduler's phases are now recorded only from repository-rooted sessions
            assert pe.PHASE_RECORD_ROOT_ENV not in renamed
            renamed[pe.PHASE_RECORD_ROOT_ENV] = str(cfg.root)
        assert new == renamed, phase.name
        assert new["PYTEST_ADDOPTS"] == f"-x -p {pe.PLUGIN}"


# coverage


@pytest.fixture
def measured_runs(tmp_path):
    pytest.importorskip("coverage")
    import os
    import subprocess

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text((CONFIGS / f"{FTS}.phase-evidence.toml").read_text())
    runs = {}
    for name, x in (("ref", 1), ("cand", 0)):
        (project / "mod.py").write_text(f"x = {x}\nif x:\n    y = 2\nelse:\n    y = 3\n")
        out = tmp_path / name
        data = pe.coverage_file(out, "standalone")
        data.parent.mkdir(parents=True)
        env = {**os.environ, "COVERAGE_FILE": str(data)}
        subprocess.run([sys.executable, "-m", "coverage", "run", "mod.py"], cwd=project, env=env, check=True)
        runs[name] = out
    return project, runs


def test_coverage_summary_and_diff_match_the_oracle(measured_runs):
    project, runs = measured_runs
    cfg, pm = pe.EvidenceConfig.load(project / "pyproject.toml"), load(ORACLE[FTS])
    for phase in ("standalone", "examples", "gpu-nonstandalone"):
        new, old = pe.coverage_summary(cfg, phase, runs["ref"]), pm.coverage_summary(phase, runs["ref"], root=project)
        # the export carries coverage's own timestamp, so its digest changes from one export to the next
        assert {k: v for k, v in new.items() if k != "json_sha256"} == {
            k: v for k, v in old.items() if k != "json_sha256"
        }, phase
    pe.coverage_summary(cfg, "standalone", runs["cand"])
    diff = pe.coverage_diff(cfg, runs["ref"], runs["cand"], ["standalone"])
    assert diff == pm.coverage_diff(runs["ref"], runs["cand"], ["standalone"])
    assert diff["standalone"]["lost"] == {"mod.py": [3]} and diff["standalone"]["gained"] == {"mod.py": [5]}
    argv = ["--config", str(project / "pyproject.toml"), "coverage-diff"]
    assert pe.main([*argv, str(runs["ref"]), str(runs["ref"]), "--phase", "standalone"]) == 0
    assert pe.main([*argv, str(runs["ref"]), str(runs["cand"]), "--phase", "standalone"]) == 1
    # a measured phase without coverage on one side is an evidence gap, never a pass
    assert pe.main([*argv, str(runs["ref"]), str(runs["cand"]), "--phase", "gpu-nonstandalone"]) == 1


# the release checker reads these reports


@pytest.mark.parametrize("project, phase", [(FTS, "examples"), (IT, "cuda")])
def test_a_clean_report_passes_the_release_checker(project, phase, tmp_path):
    a, b = NODES[:2]
    manifest = {**_manifest(project, [a, b], phase=phase), "revision": "0" * 40, "tree_dirty_paths": [],
                "collect_exit_codes": {"raw": 0, "selected": 0}}
    record_dir = Case(0, manifest, [Session(0, [a, b], [(a, "call", "passed"), (b, "call", "passed")])], {}, "nvidia",
                      "pipeline", "ci-release").write(tmp_path / "r")
    report = pe.build_report(config(project), manifest, read_sessions(record_dir), vendor="nvidia",
                             evidence_class="pipeline", lease_mode="tool", image="i@sha256:" + "a" * 64)
    release = re_.ReleaseConfig.load(CONFIGS / f"{project}.json")
    if project == IT:
        # interpretune's release configuration still names its old count: its consumer change must rename it
        with pytest.raises(re_.Ineligible, match="counts must be nonnegative integers"):
            re_.check_phase(release, "nvidia", phase, report, manifest, "0" * 40)
        release = re_.ReleaseConfig.from_mapping(
            {**json.loads((CONFIGS / f"{project}.json").read_text()), "failure_counts": list(pe.FAILURE_COUNTS)}
        )
    re_.check_phase(release, "nvidia", phase, report, manifest, "0" * 40)
