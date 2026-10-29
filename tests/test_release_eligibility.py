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
"""The release eligibility checker, over evidence shaped as each consuming project writes it.

Ported from the checkers' own tests in finetuning-scheduler and interpretune. Their reports were built there by each
project's report writer; here :func:`_report` builds the same shape by hand, so these tests pin the checker's rules
and each project keeps a contract test running the shared checker over its own writer's output.
"""
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from gpu_epsilon import release_eligibility as re_
from gpu_epsilon.evidence import OUTCOMES, classify, manifest_digest

CONFIGS = Path(__file__).parent / "configs"
SRC = Path(re_.__file__).resolve().parents[1]
REVISION = "0" * 40
NODES = [f"tests/test_x.py::test_{i}" for i in range(4)]
COVERAGE_EXPORT = b'{"totals": {"covered_lines": 10}}'
RESULTS = {"nvidia": "Succeeded", "amd": "Succeeded"}


@dataclass
class Profile:
    """One consuming project: its release configuration and the versions its legs install."""

    name: str
    config: re_.ReleaseConfig
    images: dict
    torch: dict
    versions: dict

    @property
    def phases(self) -> tuple[str, ...]:
        return self.config.phases


PROFILES = {
    "finetuning-scheduler": dict(
        images={"nvidia": "fts:cuda@sha256:" + "a" * 64, "amd": "fts:rocm@sha256:" + "b" * 64},
        torch={"nvidia": "2.15.0+cu132", "amd": "2.15.0+rocm10.0"},
        versions={
            "python": "3.13.16",
            "torch_git_version": "c" * 40,
            "lightning": "2.6.1",
            "lightning_commit": None,
            "finetuning_scheduler": "2.15.0",
        },
    ),
    "interpretune": dict(
        images={"nvidia": "it:cuda@sha256:" + "a" * 64, "amd": "it:rocm@sha256:" + "b" * 64},
        torch={"nvidia": "2.15.0+cu130", "amd": "2.15.0+rocm10.0"},
        versions={
            "python": "3.13.16",
            "interpretune": "0.1.0.dev0",
            "torch_git_version": "c" * 40,
            "lightning": "2.6.1",
            "finetuning-scheduler": "2.15.0",
            "transformer-lens": "4.0.0",
            "nnsight": "0.5.0",
            "sae-lens": "6.0.0",
        },
    ),
}


@pytest.fixture(params=sorted(PROFILES))
def profile(request):
    return Profile(request.param, re_.ReleaseConfig.load(CONFIGS / f"{request.param}.json"), **PROFILES[request.param])


def _manifest(profile, phase):
    return {
        "repository": profile.config.repository,
        "revision": REVISION,
        "phase": phase,
        "scope": "full",
        "selector": f"selector for {phase}",
        "tree_dirty_paths": [],
        "raw_node_ids": NODES + ["tests/test_x.py::test_deselected"],
        "selected_node_ids": list(NODES),
        "collection_errors": [],
        "collect_exit_codes": {"raw": 0, "selected": 0},
    }


def _coverage(profile, phase):
    if phase not in profile.config.measured_phases:
        return {"measured": False}
    sha = hashlib.sha256(COVERAGE_EXPORT).hexdigest()
    return {
        "measured": True,
        "missing": False,
        "json": f"coverage/{phase}.json",
        "json_sha256": sha,
        "covered_lines": 10,
    }


def _report(profile, vendor, manifest, outcomes, reviewed=None):
    """A phase report of the shape both projects' report writers produce."""
    reviewed = reviewed or {}
    selected = manifest["selected_node_ids"]
    nodes = {n: {"outcome": outcomes.get(n, "unexecuted")} for n in selected}
    counts = dict.fromkeys(OUTCOMES, 0)
    for node in nodes.values():
        if node["outcome"] != "unexecuted":
            counts[node["outcome"]] += 1
    unexecuted = sum(node["outcome"] == "unexecuted" for node in nodes.values())
    exclusions = [
        {"nodeid": n, "outcome": v["outcome"], "reason": reviewed.get(n)}
        for n, v in nodes.items()
        if v["outcome"] in ("skipped", "xfailed")
    ]
    report = {
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "phase": manifest["phase"],
        "scope": manifest["scope"],
        "selector": manifest["selector"],
        "evidence_class": "pipeline",
        "vendor": vendor,
        "backend": profile.config.vendors[vendor],
        "lease_mode": "tool",
        "image": profile.images[vendor],
        "versions": {**profile.versions, "torch": profile.torch[vendor]},
        "count_unit": "unique-node-id",
        "collection": {
            "raw": len(manifest["raw_node_ids"]),
            "deselected": len(manifest["raw_node_ids"]) - len(selected),
            "selected": len(selected),
            "errors": len(manifest.get("collection_errors", [])),
        },
        "outcomes": {"attempted": len(selected) - unexecuted, **counts, "unexecuted": unexecuted},
        **dict.fromkeys(profile.config.failure_counts, 0),
        "exclusions_reviewed": all(e["reason"] for e in exclusions),
        "exclusions": exclusions,
        "unmanifested_nodes": [],
        "nodes": nodes,
        "manifest_sha256": manifest_digest(manifest),
        "coverage": _coverage(profile, manifest["phase"]),
    }
    report["classification"] = classify(report, profile.config.failure_counts)
    return report


def _leg(profile, vendor, outcomes=None, reviewed=None):
    leg = {}
    for phase in profile.phases:
        manifest = _manifest(profile, phase)
        report = _report(profile, vendor, manifest, outcomes or dict.fromkeys(NODES, "passed"), reviewed)
        leg[phase] = {"report": report, "manifest": manifest, "coverage_sha256": report["coverage"].get("json_sha256")}
    return leg


def _legs(profile):
    return {vendor: _leg(profile, vendor) for vendor in profile.config.vendors}


def _check(profile, legs, results=None):
    return re_.check_release(profile.config, legs, results or RESULTS, REVISION)


def _write(profile, root, vendor, leg, attempt=1):
    phases = root / f"{profile.config.artifact_prefix}-{vendor}-{attempt}" / profile.config.phases_dir
    for kind in ("reports", "manifests"):
        (phases / kind).mkdir(parents=True, exist_ok=True)
    for phase, docs in leg.items():
        (phases / "reports" / f"{phase}.json").write_text(json.dumps(docs["report"]))
        (phases / "manifests" / f"{phase}.json").write_text(json.dumps(docs["manifest"]))
        if docs["report"]["coverage"].get("json"):
            (phases / "coverage").mkdir(exist_ok=True)
            (phases / docs["report"]["coverage"]["json"]).write_bytes(COVERAGE_EXPORT)


def _argv(artifacts, output, config):
    return ["--config", str(config), "--artifacts", str(artifacts), "--revision", REVISION, "--output", str(output)]


def _results_argv(results=None):
    return [arg for vendor, result in (results or RESULTS).items() for arg in (f"--{vendor}-result", result)]


def test_project_configs_name_their_release_phases(profile):
    expected = {
        "finetuning-scheduler": ("gpu-nonstandalone", "standalone", "examples", "examples-multigpu"),
        "interpretune": ("cuda", "standalone", "profile_ci"),
    }
    assert profile.phases == expected[profile.name]


def test_two_clean_full_legs_are_eligible(profile):
    verdict = _check(profile, _legs(profile))
    assert verdict["eligible"] and verdict["revision"] == REVISION
    assert verdict["repository"] == profile.config.repository
    assert {v: e["image"] for v, e in verdict["evidence"].items()} == profile.images
    assert set(verdict["evidence"]["amd"]["phases"]) == set(profile.phases)
    assert verdict["checker"]["config"]["artifact_prefix"] == profile.config.artifact_prefix


# Mutations of two clean legs, each taking the profile and the legs. ``first``, ``mid`` and ``last`` are positions in
# the profile's phase list, so each case reads the same against every project.


def _phase(profile, at):
    return {"first": profile.phases[0], "mid": profile.phases[1], "last": profile.phases[-1]}[at]


def _set(vendor, at, key, value, doc="report"):
    def mutate(profile, legs):
        docs = legs[vendor][_phase(profile, at)]
        docs[doc][key] = value
        if doc == "manifest":
            # a run that really collected this way: its report carries the changed manifest's digest
            docs["report"]["manifest_sha256"] = manifest_digest(docs["manifest"])

    return mutate


def _set_image(vendor, at, other_vendor):
    def mutate(profile, legs):
        legs[vendor][_phase(profile, at)]["report"]["image"] = profile.images[other_vendor]

    return mutate


def _drop_phase(profile, legs):
    del legs["amd"][profile.phases[-1]]


def _drop_vendor(profile, legs):
    del legs["nvidia"]


def _tamper_manifest(profile, legs):
    legs["nvidia"][profile.phases[1]]["manifest"]["selected_node_ids"].append("tests/test_x.py::test_deselected")


def _inflate_passes(profile, legs):
    report = legs["amd"][profile.phases[1]]["report"]
    report["outcomes"]["passed"] += 1
    report["outcomes"]["attempted"] += 1


def _other_selection(profile, legs):
    # a vendor that selected a different node set, consistently with its own manifest and report
    docs = legs["amd"][profile.phases[1]]
    manifest, report = docs["manifest"], docs["report"]
    manifest["raw_node_ids"].append("tests/test_x.py::test_amd_only")
    report["collection"].update(raw=len(manifest["raw_node_ids"]), deselected=report["collection"]["deselected"] + 1)
    report["manifest_sha256"] = manifest_digest(manifest)


def _flip_node(profile, legs):
    # a node's recorded outcome is a failure while the summary counts still read clean
    legs["amd"][profile.phases[1]]["report"]["nodes"][NODES[0]]["outcome"] = "failed"


def _drop_node(profile, legs):
    del legs["amd"][profile.phases[1]]["report"]["nodes"][NODES[0]]


def _unmanifested(profile, legs):
    legs["nvidia"][profile.phases[-1]]["report"]["unmanifested_nodes"] = ["tests/test_x.py::test_extra"]


def _with_reviewed_skip(profile, legs):
    for vendor in legs:
        legs[vendor] = _leg(
            profile, vendor, {**dict.fromkeys(NODES, "passed"), NODES[0]: "skipped"}, {NODES[0]: "reviewed"}
        )


def _skip_without_reason(profile, legs):
    # a skipped node whose exclusion has no reason, in a report that nonetheless claims its exclusions were reviewed
    _with_reviewed_skip(profile, legs)
    legs["amd"][profile.phases[-1]]["report"]["exclusions"][0]["reason"] = None


def _drop_exclusion(profile, legs):
    _with_reviewed_skip(profile, legs)
    legs["nvidia"][profile.phases[-1]]["report"]["exclusions"] = []


def _failure_count(value):
    # a project's own failure count: a rank or process that failed or never finished
    def mutate(profile, legs):
        report = legs["amd"][profile.phases[0]]["report"]
        if value is None:
            del report[profile.config.failure_counts[0]]
        else:
            report[profile.config.failure_counts[0]] = value

    return mutate


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (_drop_vendor, "nvidia: no release evidence"),
        (_drop_phase, r"amd: no evidence for required phase\(s\) "),
        (_set("nvidia", "mid", "revision", "1" * 40), "revision"),
        (_set("amd", "last", "scope", "smoke"), "scope"),
        (_set("amd", "mid", "evidence_class", "parity-pre-agent"), "evidence_class"),
        (_set("nvidia", "mid", "lease_mode", "plain"), "lease_mode"),
        (_set("nvidia", "mid", "backend", "hip"), "backend"),
        (_set("nvidia", "mid", "repository", "someone/else"), "repository"),
        (_set("amd", "mid", "count_unit", "per-rank"), "count_unit"),
        (_set("amd", "mid", "image", "project:rocm-latest"), "not digest-pinned"),
        (_set_image("amd", "last", "nvidia"), "image differs"),
        (_set("nvidia", "mid", "tree_dirty_paths", ["src/x.py"], doc="manifest"), "dirty tree"),
        (_set("nvidia", "mid", "collect_exit_codes", {"raw": 0, "selected": 2}, doc="manifest"), "exited"),
        (_tamper_manifest, "manifest digest mismatch"),
        (_inflate_passes, "unaccounted nodes"),
        (_set("amd", "mid", "classification", "failing"), "report claims"),
        (_other_selection, "the vendors collected or selected different nodes"),
        (_flip_node, "outcome counts disagree with the node records"),
        (_drop_node, "node records are not the selection"),
        (_unmanifested, "nodes ran that the manifest did not select"),
        (_skip_without_reason, "exclusions_reviewed disagrees with reasons"),
        (_drop_exclusion, "exclusions do not list exactly the skipped and xfailed nodes"),
        (_failure_count(1), "phase is failing"),
        (_failure_count(None), "counts must be nonnegative integers"),
        (_failure_count(-1), "counts must be nonnegative integers"),
    ],
)
def test_ineligible_evidence(profile, mutate, reason):
    legs = _legs(profile)
    mutate(profile, legs)
    with pytest.raises(re_.Ineligible, match=reason):
        _check(profile, legs)


def _coverage_missing(profile, legs):
    legs["amd"][profile.config.measured_phases[0]]["report"]["coverage"] = {"measured": True, "missing": True}


def _coverage_tampered(profile, legs):
    legs["nvidia"][profile.config.measured_phases[0]]["coverage_sha256"] = "d" * 64


def _coverage_empty(profile, legs):
    legs["nvidia"][profile.config.measured_phases[0]]["report"]["coverage"]["covered_lines"] = 0


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (_coverage_missing, "coverage evidence is missing"),
        (_coverage_tampered, "coverage export missing or not the one the report digested"),
        (_coverage_empty, "coverage records no executed lines"),
    ],
)
def test_measured_phase_needs_its_coverage(profile, mutate, reason):
    if not profile.config.measured_phases:
        pytest.skip(f"{profile.name} measures no release phase")
    legs = _legs(profile)
    mutate(profile, legs)
    with pytest.raises(re_.Ineligible, match=reason):
        _check(profile, legs)


def test_an_unmeasured_phase_needs_no_coverage(profile):
    legs = _legs(profile)
    for vendor in legs:
        for phase in set(profile.phases) - set(profile.config.measured_phases):
            legs[vendor][phase]["report"]["coverage"] = {"measured": False}
    assert _check(profile, legs)["eligible"]


# Versions: (key, value written on the amd leg, expected reason or None for eligible), per project.
VERSION_CASES = {
    "finetuning-scheduler": [
        ("lightning_commit", "abc123", "installed versions differ across vendors"),
        ("torch_git_version", "e" * 40, "installed versions differ across vendors"),
        ("lightning", "2.7.0", "installed versions differ across vendors"),
        ("finetuning_scheduler", None, "installed versions missing"),
        ("torch", "2.15.0+rocm10.1", None),
    ],
    "interpretune": [
        ("torch_git_version", "e" * 40, "installed versions differ across vendors"),
        ("transformer-lens", "4.0.1", "installed versions differ across vendors"),
        # a package one vendor's image lacks is a difference, not a pass
        ("nnsight", None, "installed versions differ across vendors"),
        ("interpretune", None, "installed versions missing"),
        ("torch", "2.15.0+rocm10.1", None),
    ],
}


@pytest.mark.parametrize("case", range(5))
def test_versions_must_agree_across_vendors(profile, case):
    key, value, reason = VERSION_CASES[profile.name][case]
    legs = _legs(profile)
    for docs in legs["amd"].values():
        docs["report"]["versions"] = {**docs["report"]["versions"], key: value}
    if reason is None:
        assert _check(profile, legs)["evidence"]["amd"]["versions"][key] == value
    else:
        with pytest.raises(re_.Ineligible, match=reason):
            _check(profile, legs)


def test_versions_must_agree_within_a_leg(profile):
    legs = _legs(profile)
    report = legs["nvidia"][profile.phases[-1]]["report"]
    report["versions"] = {**report["versions"], "python": "3.12.0"}
    with pytest.raises(re_.Ineligible, match="versions differs from the leg's other phases"):
        _check(profile, legs)


@pytest.mark.parametrize(
    "outcome, classification",
    [("failed", "failing"), ("errors", "failing"), ("xpassed", "failing"), ("skipped", "unreviewed-exclusions")],
)
def test_unclean_phase_is_ineligible(profile, outcome, classification):
    legs = _legs(profile)
    legs["amd"] = _leg(profile, "amd", {**dict.fromkeys(NODES, "passed"), NODES[0]: outcome})
    with pytest.raises(re_.Ineligible, match=f"amd/{profile.phases[0]}: phase is {classification}"):
        _check(profile, legs)


def test_reviewed_skip_is_eligible(profile):
    legs = _legs(profile)
    _with_reviewed_skip(profile, legs)
    assert _check(profile, legs)["eligible"]


def test_vendor_torch_tags_differ_but_releases_agree(profile):
    verdict = _check(profile, _legs(profile))
    assert {v: e["versions"]["torch"] for v, e in verdict["evidence"].items()} == profile.torch


@pytest.mark.parametrize("results, reason", [
    ({"nvidia": "Succeeded", "amd": "SucceededWithIssues"}, "amd leg result is 'SucceededWithIssues'"),
    ({"nvidia": "Succeeded"}, "amd leg result is None"),
])
def test_leg_result_must_be_succeeded(profile, results, reason):
    with pytest.raises(re_.Ineligible, match=reason):
        _check(profile, _legs(profile), results)


def test_revision_is_required(profile):
    with pytest.raises(re_.Ineligible, match="candidate revision is missing"):
        re_.check_release(profile.config, _legs(profile), RESULTS, "")


# The command line.


def test_cli_writes_verdict_from_latest_attempt(profile, tmp_path, capsys):
    artifacts, output = tmp_path / "artifacts", tmp_path / "out" / "eligibility.json"
    legs = _legs(profile)
    _write(profile, artifacts, "nvidia", legs["nvidia"])
    _write(profile, artifacts, "amd", _leg(profile, "amd", {**dict.fromkeys(NODES, "passed"), NODES[0]: "failed"}))
    _write(profile, artifacts, "amd", legs["amd"], attempt=2)
    config = CONFIGS / f"{profile.name}.json"
    assert re_.main(_argv(artifacts, output, config) + _results_argv()) == 0
    assert json.loads(output.read_text())["eligible"] is True
    assert "ELIGIBLE" in capsys.readouterr().out


def test_cli_ignores_another_projects_artifacts(profile, tmp_path, capsys):
    artifacts, output = tmp_path / "artifacts", tmp_path / "eligibility.json"
    other = next(p for p in PROFILES if p != profile.name)
    other_profile = Profile(other, re_.ReleaseConfig.load(CONFIGS / f"{other}.json"), **PROFILES[other])
    for vendor, leg in _legs(other_profile).items():
        _write(other_profile, artifacts, vendor, leg)
    assert re_.main(_argv(artifacts, output, CONFIGS / f"{profile.name}.json") + _results_argv()) == 1
    assert "INELIGIBLE: nvidia: no release evidence" in capsys.readouterr().err


def test_cli_removes_a_stale_verdict_when_ineligible(profile, tmp_path, capsys):
    artifacts, output = tmp_path / "artifacts", tmp_path / "eligibility.json"
    _write(profile, artifacts, "nvidia", _legs(profile)["nvidia"])
    output.write_text('{"eligible": true}')
    argv = _argv(artifacts, output, CONFIGS / f"{profile.name}.json")
    assert re_.main(argv + _results_argv({"nvidia": "Succeeded", "amd": "Failed"})) == 1
    assert not output.exists()
    assert "INELIGIBLE: amd leg result is 'Failed'" in capsys.readouterr().err


@pytest.mark.parametrize("content", ["[]", "{"], ids=["wrong-shape", "unreadable"])
def test_cli_reports_bad_evidence(profile, tmp_path, capsys, content):
    artifacts = tmp_path / "artifacts"
    for vendor, leg in _legs(profile).items():
        _write(profile, artifacts, vendor, leg)
    phases = artifacts / f"{profile.config.artifact_prefix}-amd-1" / profile.config.phases_dir
    (phases / "reports" / f"{profile.phases[-1]}.json").write_text(content)
    argv = _argv(artifacts, tmp_path / "e.json", CONFIGS / f"{profile.name}.json")
    assert re_.main(argv + _results_argv()) == 1
    assert "INELIGIBLE" in capsys.readouterr().err


def test_cli_rejects_a_bad_config_and_still_removes_a_stale_verdict(tmp_path, capsys):
    config, output = tmp_path / "release.json", tmp_path / "eligibility.json"
    config.write_text(json.dumps({"repository": "o/r", "artifact_prefix": "x", "phases_dir": "p", "phase": ["a"]}))
    output.write_text('{"eligible": true}')
    assert re_.main(_argv(tmp_path, output, config) + _results_argv()) == 2
    assert not output.exists()
    assert "unknown release configuration key(s): phase" in capsys.readouterr().err


def test_cli_result_options_are_the_configs_vendors(tmp_path, capsys):
    doc = json.loads((CONFIGS / "interpretune.json").read_text())
    config = tmp_path / "release.json"
    config.write_text(json.dumps({**doc, "vendors": {"nvidia": "cuda", "amd": "hip", "intel": "xpu"}}))
    argv = _argv(tmp_path / "artifacts", tmp_path / "e.json", config)
    with pytest.raises(SystemExit):
        re_.main(argv + _results_argv())
    assert "--intel-result" in capsys.readouterr().err
    assert re_.main(argv + _results_argv({**RESULTS, "intel": "Succeeded"})) == 1
    assert "INELIGIBLE: nvidia: no release evidence" in capsys.readouterr().err


def test_runs_on_the_standard_library_alone(tmp_path):
    """The release job runs the checker on the agent's own Python: ``-I -S`` gives it no site-packages and no
    environment, and the checker must import nothing outside the standard library and this package."""
    profile = Profile("interpretune", re_.ReleaseConfig.load(CONFIGS / "interpretune.json"), **PROFILES["interpretune"])
    artifacts, output = tmp_path / "artifacts", tmp_path / "e.json"
    for vendor, leg in _legs(profile).items():
        _write(profile, artifacts, vendor, leg)
    argv = _argv(artifacts, output, CONFIGS / "interpretune.json") + _results_argv()
    code = (
        f"import sys; sys.path.insert(0, {str(SRC)!r}); from gpu_epsilon.release_eligibility import main\n"
        f"rc = main({argv!r})\n"
        "top = {n.split('.')[0] for n in sys.modules}\n"
        "print(sorted(top - set(sys.stdlib_module_names) - {'gpu_epsilon', '__main__'})); sys.exit(rc)"
    )
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "[]"
    assert json.loads(output.read_text())["eligible"] is True


def test_runs_as_a_module(profile, tmp_path):
    artifacts, output = tmp_path / "artifacts", tmp_path / "e.json"
    for vendor, leg in _legs(profile).items():
        _write(profile, artifacts, vendor, leg)
    argv = _argv(artifacts, output, CONFIGS / f"{profile.name}.json") + _results_argv()
    proc = subprocess.run(
        [sys.executable, "-m", "gpu_epsilon.release_eligibility", *argv], cwd=tmp_path, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(output.read_text())["eligible"] is True
