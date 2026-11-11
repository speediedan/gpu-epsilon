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
"""Phase evidence: the manifest and the report each GPU CI phase publishes, written one way for every project.

A GPU CI run is reported phase by phase (:mod:`gpu_epsilon.evidence` describes the two documents). This module writes
them:

- ``collect`` writes ``<out-dir>/manifests/<phase>.json``: the node ids the phase's paths collect (``raw_node_ids``)
  and the ones its selection keeps (``selected_node_ids``), with the repository, revision, scope, selector string, the
  tree's differences from the revision, any collection errors and both collections' exit codes. Collection only: it
  runs with ``CUDA_VISIBLE_DEVICES`` emptied and executes nothing, so it is evidence about selection, never a pass. The
  project's selection variables and any inherited ``PYTEST_ADDOPTS`` are removed first, so neither the caller's shell
  nor a stray ``-k`` changes what a phase is.
- ``run-env`` prints the exports a phase's pipeline command runs under, so every pytest process it starts (each node
  the special-test runner runs, each rank a launcher starts) loads :mod:`gpu_epsilon.records` and writes its records
  to the phase's own directory. Only sessions rooted at the repository are recorded: a test that runs pytest on a tree
  of its own inherits the exports, and its session is not the phase's.
- ``report`` writes ``<out-dir>/reports/<phase>.json``: each selected node's outcome, read from those records and
  counted against the manifest, with the run's identity (vendor, backend, evidence class, image, lease mode, installed
  versions, devices), the manifest's digest and, for a measured phase, its coverage export. It exits by the project's
  gate (below).
- ``coverage-diff`` lists, per measured phase and file, the lines one run executed and another did not.

A node's outcome is rank 0's (the launching process) unless any rank failed or errored on it or started and never
finished, which fails it. A rank seen in several sessions (a two-stage run, a rerun) keeps a session that never
finished over a failure, a failure over an error, and otherwise its last verdict. A selected node with no rank-0
verdict is ``unexecuted``. A session that started and never finished counts in ``missing_rank_completions`` and fails
every node it collected and gave no verdict for. A node that ran but is not in the manifest's selection is a collection
error: the run and the manifest disagree about what the phase is. Classification is
:func:`gpu_epsilon.evidence.classify` with ``rank_failures`` and ``missing_rank_completions`` as failure counts.

What differs between projects is configuration, read from ``[tool.gpu-epsilon.phase-evidence]`` in the project's
``pyproject.toml`` (:class:`EvidenceConfig` lists the keys): the phases and their paths and selection variables, the
variables stripped before collecting and the ones always set, the device variables a report records, how a scope
narrows a phase, the versions a report records, and the gate. The repository root is the configuration file's
directory.

Usage, from the repository root::

    python -m gpu_epsilon.phase_evidence list
    python -m gpu_epsilon.phase_evidence collect --phase standalone --out-dir /tmp/phases
    eval "$(python -m gpu_epsilon.phase_evidence run-env --phase standalone --out-dir /tmp/phases)"
    python -m gpu_epsilon.phase_evidence report --phase standalone --out-dir /tmp/phases \\
        --vendor amd --evidence-class parity-pre-agent --lease-mode tool
    python -m gpu_epsilon.phase_evidence coverage-diff /tmp/phases-nvidia /tmp/phases-amd

Keep ``--out-dir`` outside the repository: the manifest records the tree's untracked files, so an out-dir inside the
checkout marks every later collection dirty. Importing this module needs only the standard library; reading the
configuration needs Python 3.11 (``tomllib``).
"""
import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from gpu_epsilon.evidence import OUTCOMES, classify, manifest_digest
from gpu_epsilon.records import PHASE_RECORD_DIR_ENV, PHASE_RECORD_ROOT_ENV, RECORD_DIR_ENV, read_sessions

PLUGIN = "gpu_epsilon.records"
FULL = "full"
EVIDENCE_CLASSES = ("pipeline", "parity-pre-agent")
DEFAULT_VENDORS = {"nvidia": "cuda", "amd": "hip"}
DEFAULT_DEVICE_VARS = ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
# the report counts that fail a phase when nonzero, as the release checker's ``failure_counts`` names them
FAILURE_COUNTS = ("rank_failures", "missing_rank_completions")
# versions every report records, whatever the project configures
BUILTIN_VERSIONS = ("python", "torch", "torch_git_version", "torch_cuda", "torch_hip")
VERSION_SOURCES = ("dist", "module", "vcs", "record")
GATE_RULES = {"empty": ("never", "outside-full"), "unreviewed_exclusions": ("never", "except-release")}


class ConfigError(ValueError):
    """The project's phase-evidence configuration is malformed."""


def _table(doc, where: str, allowed: set[str]) -> dict:
    """``doc`` with kebab-case keys in snake case; a key outside ``allowed`` is refused by its own name."""
    if not isinstance(doc, dict):
        raise ConfigError(f"{where} must be a table")
    unknown = sorted(k for k in doc if not isinstance(k, str) or k.replace("-", "_") not in allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(map(str, unknown))}")
    return {k.replace("-", "_"): v for k, v in doc.items()}


def _strings(value, where: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ConfigError(f"{where} must be a list of non-empty strings")
    if not all(isinstance(v, str) and v for v in value):
        raise ConfigError(f"{where} must be a list of non-empty strings")
    return tuple(value)


def _env(value, where: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and k and isinstance(v, str) for k, v in value.items()
    ):
        raise ConfigError(f"{where} must map variable names to strings")
    return dict(value)


def _bool(value, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be true or false")
    return value


@dataclass(frozen=True)
class EvidencePhase:
    """One phase (an entry of ``[[tool.gpu-epsilon.phase-evidence.phase]]``, keys in kebab case).

    Attributes:
        name: the phase's name, as the pipeline and the release configuration use it.
        paths: the pytest paths the phase collects. Exactly one of ``paths`` and ``paths_from``.
        paths_from: ``{"glob": ..., "contains": ...}``: the files under the root matching ``glob`` whose text contains
            ``contains``, sorted (a phase that runs the modules using a fixture, say).
        selection_env: the variables that select the phase's tests from its paths.
        args: pytest arguments that narrow the selection (``-k``), applied to the selected collection only.
        command: the pipeline command that runs the phase, for ``list``.
        release: whether a release leg runs the phase.
        measured: whether the pipeline runs the phase under ``coverage run``; ``run-env`` then gives it its own data
            file and ``report`` exports and digests it.
    """

    name: str
    paths: tuple[str, ...] = ()
    paths_from: dict[str, str] = field(default_factory=dict)
    selection_env: dict[str, str] = field(default_factory=dict)
    args: tuple[str, ...] = ()
    command: str = ""
    release: bool = True
    measured: bool = False

    @classmethod
    def from_mapping(cls, doc) -> "EvidencePhase":
        name = doc.get("name") if isinstance(doc, dict) else None
        fields = _table(doc, f"phase {name or '?'}", set(cls.__dataclass_fields__))
        if not isinstance(name, str) or not name:
            raise ConfigError("every phase needs a non-empty name")
        where = f"phase {name}"
        for key in ("paths", "args"):
            if key in fields:
                fields[key] = _strings(fields[key], f"{where}: {key}")
        if "selection_env" in fields:
            fields["selection_env"] = _env(fields["selection_env"], f"{where}: selection-env")
        if "paths_from" in fields:
            rule = fields["paths_from"]
            if not isinstance(rule, dict) or set(rule) != {"glob", "contains"} or not all(
                isinstance(v, str) and v for v in rule.values()
            ):
                raise ConfigError(f"{where}: paths-from must be exactly {{glob, contains}}, both non-empty strings")
            fields["paths_from"] = dict(rule)
        for key in ("release", "measured"):
            if key in fields:
                fields[key] = _bool(fields[key], f"{where}: {key}")
        if not isinstance(fields.get("command", ""), str):
            raise ConfigError(f"{where}: command must be a string")
        phase = cls(**fields)
        if bool(phase.paths) == bool(phase.paths_from):
            raise ConfigError(f"{where}: needs exactly one of 'paths' and 'paths-from'")
        return phase

    def resolved_paths(self, root: Path) -> tuple[str, ...]:
        """The pytest paths, relative to ``root``: ``paths``, or the files ``paths_from`` names today."""
        if not self.paths_from:
            return self.paths
        glob, contains = self.paths_from["glob"], self.paths_from["contains"]
        return tuple(
            p.relative_to(root).as_posix()
            for p in sorted(root.glob(glob))
            if p.is_file() and contains in p.read_text(encoding="utf-8")
        )


@dataclass(frozen=True)
class Scope:
    """A narrower run of some phases (``[tool.gpu-epsilon.phase-evidence.scopes.<name>]``).

    Attributes:
        name: the scope's name, recorded in the manifest and the report (never ``full``).
        select_file: a file of node ids, one per line (``#`` comments allowed): the manifest keeps the selected nodes
            it lists, in its order, and a listed node not collected is a collection error. Chosen with ``--scope``.
        select_file_env: a variable naming a selection file the project's own ``conftest.py`` narrows by. The scope
            applies whenever the variable is set; the selection is the project's, so the manifest is not filtered
            further. Exactly one of ``select_file`` and ``select_file_env``.
        phases: the phases the scope is defined for (every phase when empty; a ``select_file`` scope only).
    """

    name: str
    select_file: str | None = None
    select_file_env: str | None = None
    phases: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, name: str, doc) -> "Scope":
        where = f"scope {name}"
        fields = _table(doc, where, set(cls.__dataclass_fields__) - {"name"})
        if name == FULL:
            raise ConfigError(f"{where}: 'full' is every phase's unnarrowed scope and cannot be configured")
        if "phases" in fields:
            fields["phases"] = _strings(fields["phases"], f"{where}: phases")
        scope = cls(name=name, **fields)
        sources = [v for v in (scope.select_file, scope.select_file_env) if v is not None]
        if len(sources) != 1 or not all(isinstance(v, str) and v for v in sources):
            raise ConfigError(f"{where}: needs exactly one of 'select-file' and 'select-file-env', a non-empty string")
        if scope.select_file_env and scope.phases:
            raise ConfigError(f"{where}: a select-file-env scope applies to every phase, so it takes no 'phases'")
        return scope


@dataclass(frozen=True)
class Gate:
    """Which classifications pass a job, besides ``clean`` (always) and ``failing`` (never).

    Attributes:
        empty: ``never``, or ``outside-full``: an empty phase passes a narrowed scope, which may legitimately select
            nothing, and fails a full run, where it means the collector broke.
        unreviewed_exclusions: ``never``, or ``except-release``: a skip without a reviewed reason passes every job but
            a release leg, whose evidence must name a reason for each.
        release_job_classes: the ``--job-class`` values that are release legs.
    """

    empty: str = "never"
    unreviewed_exclusions: str = "never"
    release_job_classes: tuple[str, ...] = ("ci-release",)

    @classmethod
    def from_mapping(cls, doc) -> "Gate":
        fields = _table(doc, "gate", set(cls.__dataclass_fields__))
        for key, allowed in GATE_RULES.items():
            if key in fields and fields[key] not in allowed:
                raise ConfigError(f"gate: {key.replace('_', '-')} must be one of {', '.join(allowed)}")
        if "release_job_classes" in fields:
            fields["release_job_classes"] = _strings(fields["release_job_classes"], "gate: release-job-classes")
        return cls(**fields)

    def passes(self, classification: str, scope: str, job_class: str) -> bool:
        if classification == "clean":
            return True
        if classification == "empty":
            return self.empty == "outside-full" and scope != FULL
        if classification == "unreviewed-exclusions":
            return self.unreviewed_exclusions == "except-release" and job_class not in self.release_job_classes
        return False


@dataclass(frozen=True)
class EvidenceConfig:
    """A project's phase-evidence configuration (``[tool.gpu-epsilon.phase-evidence]``, keys in kebab case).

    Attributes:
        repository: the ``owner/name`` every manifest names.
        phases: the phases (``[[...phase]]`` entries, :class:`EvidencePhase`), in order.
        root: the repository root: the configuration file's directory (not a key).
        selection_vars: variables the project's ``conftest.py`` selects or orders on, removed before every collection
            so an inherited one cannot change what a phase collects.
        collect_env: variables set for every collection (a GPU opt-in that gates skips, never selection); part of
            every selector.
        device_vars: the visible-device variables a report records.
        run_env_exports_selection: whether ``run-env`` also exports ``collect_env`` and the phase's ``selection_env``
            (when the pipeline command does not set them itself).
        scopes: the narrower scopes (:class:`Scope`), by name.
        versions: ``{report key: source}`` recorded beside the built-in interpreter and torch versions. A source is
            ``dist:<distribution>`` (its metadata version), ``module:<module>`` (its ``__version__``),
            ``vcs:<distribution>`` (the commit its ``direct_url.json`` names) or ``record:<distribution>`` (the
            SHA-256 of its installed ``RECORD``). Anything not installed is recorded as ``None``.
        gate: which classifications pass a job (:class:`Gate`).
        vendors: each vendor and the backend its reports name.
    """

    repository: str
    phases: tuple[EvidencePhase, ...]
    root: Path = Path(".")
    selection_vars: tuple[str, ...] = ()
    collect_env: dict[str, str] = field(default_factory=dict)
    device_vars: tuple[str, ...] = DEFAULT_DEVICE_VARS
    run_env_exports_selection: bool = False
    scopes: dict[str, Scope] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    gate: Gate = field(default_factory=Gate)
    vendors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_VENDORS))

    @classmethod
    def from_mapping(cls, doc, root: Path = Path(".")) -> "EvidenceConfig":
        allowed = (set(cls.__dataclass_fields__) - {"phases", "root"}) | {"phase", "description"}
        fields = _table(doc, "[tool.gpu-epsilon.phase-evidence]", allowed)
        fields.pop("description", None)
        if not isinstance(fields.get("repository"), str) or "/" not in fields["repository"]:
            raise ConfigError("repository must be the 'owner/name' every manifest names")
        phases = fields.pop("phase", None)
        if not isinstance(phases, list) or not phases:
            raise ConfigError("at least one [[tool.gpu-epsilon.phase-evidence.phase]] is needed")
        fields["phases"] = tuple(EvidencePhase.from_mapping(p) for p in phases)
        names = [p.name for p in fields["phases"]]
        if len(set(names)) != len(names):
            raise ConfigError("each phase name must appear once")
        for key in ("selection_vars", "device_vars"):
            if key in fields:
                fields[key] = _strings(fields[key], key.replace("_", "-"))
        if "collect_env" in fields:
            fields["collect_env"] = _env(fields["collect_env"], "collect-env")
        if "run_env_exports_selection" in fields:
            fields["run_env_exports_selection"] = _bool(
                fields["run_env_exports_selection"], "run-env-exports-selection"
            )
        scopes = fields.get("scopes", {})
        if not isinstance(scopes, dict):
            raise ConfigError("scopes must be a table of scopes")
        fields["scopes"] = {name: Scope.from_mapping(name, spec) for name, spec in scopes.items()}
        for scope in fields["scopes"].values():
            if not set(scope.phases) <= set(names):
                raise ConfigError(f"scope {scope.name}: names a phase that does not exist")
        if sum(bool(s.select_file_env) for s in fields["scopes"].values()) > 1:
            raise ConfigError("at most one scope may be chosen by a select-file-env variable")
        versions = fields.get("versions", {})
        if not isinstance(versions, dict):
            raise ConfigError("versions must map report keys to sources")
        for key, source in versions.items():
            kind, _, name = source.partition(":") if isinstance(source, str) else ("", "", "")
            if key in BUILTIN_VERSIONS:
                raise ConfigError(f"versions: {key} is recorded by every report and cannot be configured")
            if kind not in VERSION_SOURCES or not name:
                raise ConfigError(f"versions: {key} must be '<{'|'.join(VERSION_SOURCES)}>:<name>', not {source!r}")
        fields["versions"] = dict(versions)
        if "gate" in fields:
            fields["gate"] = Gate.from_mapping(fields["gate"])
        vendors = fields.get("vendors", DEFAULT_VENDORS)
        if not isinstance(vendors, dict) or not vendors or not all(
            isinstance(k, str) and k.isalpha() and k.islower() and isinstance(v, str) and v for k, v in vendors.items()
        ):
            raise ConfigError("vendors must map lowercase vendor names to backends")
        fields["vendors"] = dict(vendors)
        return cls(root=Path(root), **fields)

    @classmethod
    def load(cls, path: Path) -> "EvidenceConfig":
        try:
            import tomllib  # pyright: ignore[reportMissingImports]  (3.11+; the except handles 3.10)
        except ModuleNotFoundError as exc:  # Python 3.10
            raise ConfigError("reading pyproject.toml needs Python 3.11 or newer (tomllib)") from exc
        path = Path(path)
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
        table = doc.get("tool", {}).get("gpu-epsilon", {}).get("phase-evidence")
        if table is None:
            raise ConfigError(f"{path} has no [tool.gpu-epsilon.phase-evidence] table")
        return cls.from_mapping(table, root=path.resolve().parent)

    def phase(self, name: str) -> EvidencePhase:
        """The phase named ``name``; ``KeyError`` for a phase the project does not define."""
        for phase in self.phases:
            if phase.name == name:
                return phase
        raise KeyError(name)

    @property
    def env_scope(self) -> Scope | None:
        """The scope a ``select-file-env`` variable chooses, if the project has one."""
        return next((s for s in self.scopes.values() if s.select_file_env), None)

    def resolve_scope(self, phase_name: str, requested: str | None = None) -> tuple[str, Scope | None]:
        """The scope a phase is collected in: ``requested`` (``--scope``), or the ``select-file-env`` scope when its
        variable is set, else ``full``; ``ValueError`` names a scope that does not apply."""
        env_scope = self.env_scope
        if env_scope and env_scope.select_file_env and os.environ.get(env_scope.select_file_env):
            if requested is not None:
                raise ValueError(
                    f"{env_scope.select_file_env} is set, which chooses scope {env_scope.name!r}; drop --scope"
                )
            return env_scope.name, env_scope
        if requested in (None, FULL):
            return FULL, None
        scope = self.scopes.get(requested)
        if scope is None or scope.select_file_env or (scope.phases and phase_name not in scope.phases):
            raise ValueError(f"scope {requested!r} is not defined for phase {phase_name!r}")
        return scope.name, scope

    def selector(self, phase: EvidencePhase, scope: Scope | None = None) -> str:
        """The phase's selection, as one comparable string: selection variables, then the pytest paths and arguments,
        then the scope's selection file."""
        env = " ".join(f"{k}={v}" for k, v in sorted({**self.collect_env, **phase.selection_env}.items()))
        parts = [env, "; pytest", *phase.resolved_paths(self.root), *phase.args]
        if scope is not None:
            source = scope.select_file or os.environ[scope.select_file_env or ""]
            parts.append(f"; select_file={source}")
        return " ".join(parts).replace(" ; ", "; ")


# collection


def _collect(config: EvidenceConfig, paths: tuple[str, ...], args: tuple[str, ...], env_update: dict):
    """Collect once through the plugin: the selected node ids, the failed collectors and pytest's exit code."""
    env = {k: v for k, v in os.environ.items() if k not in config.selection_vars}
    env.update(config.collect_env)
    env.update(env_update)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join([str(config.root), *filter(None, [env.get("PYTHONPATH")])])
    # an inherited PYTEST_ADDOPTS (a -k, a -m, another plugin) would change what the phase collects
    for name in ("PYTEST_ADDOPTS", RECORD_DIR_ENV, PHASE_RECORD_ROOT_ENV):
        env.pop(name, None)
    record_dir = Path(tempfile.mkdtemp(prefix="gpu-epsilon-phase-collect-"))
    env[PHASE_RECORD_DIR_ENV] = str(record_dir)
    cmd = [sys.executable, "-m", "pytest", *paths, *args, "--collect-only", "-q"]
    cmd += ["-p", "no:cacheprovider", "-p", PLUGIN]
    try:
        proc = subprocess.run(cmd, cwd=config.root, env=env, capture_output=True, text=True)
        records = [r for session in read_sessions(record_dir) for r in session]
    finally:
        shutil.rmtree(record_dir, ignore_errors=True)
    collected = [r["nodeids"] for r in records if r["event"] == "collected"]
    errors = sorted(r["nodeid"] for r in records if r["event"] == "collect_error")
    if len(collected) != 1:
        errors.append(f"collection recorded {len(collected)} times (pytest exit {proc.returncode})")
    return (collected[0] if collected else []), errors, proc.returncode


def git_state(root: Path) -> tuple[str, list[str] | None]:
    """``HEAD`` and the paths that differ from it (untracked included); ``None`` when git cannot say."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, capture_output=True, text=True
    )
    dirty = None if status.returncode else sorted(line[3:] for line in status.stdout.splitlines() if line.strip())
    return (head.stdout.strip() if head.returncode == 0 else "unknown"), dirty


def build_manifest(config: EvidenceConfig, phase_name: str, scope: str | None = None) -> dict:
    """Collect ``phase_name`` twice on the CPU (all of its paths, then its selection) and describe both."""
    phase = config.phase(phase_name)
    scope_name, scope_def = config.resolve_scope(phase_name, scope)
    paths = phase.resolved_paths(config.root)
    if not paths:
        raise ValueError(f"phase {phase_name!r} resolved to no paths")
    raw, raw_errors, raw_exit = _collect(config, paths, (), {})
    selected, sel_errors, sel_exit = _collect(config, paths, phase.args, phase.selection_env)
    if scope_def is not None and scope_def.select_file:
        lines = (config.root / scope_def.select_file).read_text(encoding="utf-8").splitlines()
        wanted = [w for w in (line.strip() for line in lines) if w and not w.startswith("#")]
        sel_errors += [f"{scope_name} node not collected: {w}" for w in wanted if w not in selected]
        selected = [w for w in wanted if w in selected]
    revision, dirty = git_state(config.root)
    return {
        "repository": config.repository,
        "revision": revision,
        "phase": phase_name,
        "scope": scope_name,
        "selector": config.selector(phase, scope_def),
        # A tree that differs from the revision makes the revision a wrong name for what was collected.
        "tree_dirty_paths": dirty,
        "raw_node_ids": raw,
        "selected_node_ids": selected,
        "collection_errors": sorted(set(raw_errors + sel_errors)),
        # 5 is pytest's "no tests collected": a narrowed scope may legitimately select nothing in a phase
        "collect_exit_codes": {"raw": raw_exit, "selected": sel_exit},
    }


# reporting


def _session_node_outcomes(records: list[dict]) -> dict[str, str]:
    """A session's verdict per node: a failure or error in any stage wins, else the deciding stage's outcome."""
    per_node: dict[str, list[str]] = {}
    for r in records:
        if r["event"] == "outcome":
            per_node.setdefault(r["nodeid"], []).append(r["outcome"])
    return {
        nodeid: next((o for o in ("failed", "errors") if o in outcomes), outcomes[0])
        for nodeid, outcomes in per_node.items()
    }


def _rank_verdict(verdicts: list[str]) -> str:
    """One rank's verdict on a node across the sessions it ran in: a session that never finished, then a failure,
    then an error, else the last verdict read."""
    return next((o for o in ("incomplete", "failed", "errors") if o in verdicts), verdicts[-1])


def build_report(
    config: EvidenceConfig,
    manifest: dict,
    sessions: list[list[dict]],
    *,
    vendor: str,
    evidence_class: str,
    lease_mode: str | None = None,
    image: str | None = None,
    versions: dict | None = None,
    reviewed_exclusions: dict | None = None,
    coverage: dict | None = None,
) -> dict:
    """Count one phase's per-node, per-rank outcomes against its manifest and describe the run that produced them (the
    module docstring states the rules)."""
    if vendor not in config.vendors:
        raise ValueError(f"vendor must be one of {sorted(config.vendors)}")
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(f"evidence class must be one of {EVIDENCE_CLASSES}")
    reviewed = reviewed_exclusions or {}
    phase = next((p for p in config.phases if p.name == manifest["phase"]), None)
    selected = manifest["selected_node_ids"]
    seen: dict[str, dict[str, list[str]]] = {}
    world_sizes: dict[str, set] = {}
    missing_completions = 0
    for records in sessions:
        if not records:
            continue
        rank = str(records[0]["rank"])
        verdicts = _session_node_outcomes(records)
        world_size = next((r["world_size"] for r in records if r.get("world_size")), None)
        for nodeid, verdict in verdicts.items():
            seen.setdefault(nodeid, {}).setdefault(rank, []).append(verdict)
            if world_size:
                world_sizes.setdefault(nodeid, set()).add(world_size)
        if not any(r["event"] == "finish" for r in records) and any(r["event"] == "start" for r in records):
            missing_completions += 1
            collected = next((r["nodeids"] for r in records if r["event"] == "collected"), [])
            for nodeid in collected:
                if nodeid not in verdicts:
                    seen.setdefault(nodeid, {}).setdefault(rank, []).append("incomplete")
    ranks = {nodeid: {rank: _rank_verdict(v) for rank, v in per_rank.items()} for nodeid, per_rank in seen.items()}
    nodes: dict[str, dict] = {}
    counts: dict[str, int] = dict.fromkeys(OUTCOMES, 0)
    unexecuted, rank_failures = [], 0
    for nodeid in selected:
        per_rank = ranks.get(nodeid, {})
        rank_failures += sum(1 for r, o in per_rank.items() if r != "0" and o in ("failed", "errors", "incomplete"))
        if any(o == "incomplete" for o in per_rank.values()) or any(
            o in ("failed", "errors") for r, o in per_rank.items() if r != "0"
        ):
            outcome = "failed"
        elif "0" in per_rank:
            outcome = per_rank["0"]
        else:
            unexecuted.append(nodeid)
            nodes[nodeid] = {"outcome": "unexecuted", "ranks": per_rank}
            continue
        counts[outcome] += 1
        # the launcher's world size, when one was set: two hosts can agree on a verdict yet run a node on different
        # numbers of ranks
        world_size = max(world_sizes.get(nodeid, ()), default=None)
        nodes[nodeid] = {"outcome": outcome, "ranks": per_rank, "world_size": world_size}
    unmanifested = sorted(set(ranks) - set(selected))
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
        "evidence_class": evidence_class,
        "vendor": vendor,
        "backend": config.vendors[vendor],
        "lease_mode": lease_mode,
        "image": image,
        "versions": versions or {},
        "count_unit": "unique-node-id",
        "collection": {
            "raw": len(manifest["raw_node_ids"]),
            "deselected": len(manifest["raw_node_ids"]) - len(selected),
            "selected": len(selected),
            "errors": len(manifest.get("collection_errors", [])) + len(unmanifested),
        },
        "outcomes": {"attempted": len(selected) - len(unexecuted), **counts, "unexecuted": len(unexecuted)},
        "rank_failures": rank_failures,
        "missing_rank_completions": missing_completions,
        "exclusions_reviewed": all(e["reason"] for e in exclusions),
        "exclusions": exclusions,
        "unmanifested_nodes": unmanifested,
        "nodes": nodes,
        "manifest_sha256": manifest_digest(manifest),
        "coverage": coverage or {"measured": phase.measured if phase else None},
    }
    report["classification"] = classify(report, FAILURE_COUNTS)
    return report


def gate(config: EvidenceConfig, classification: str, scope: str, job_class: str) -> bool:
    """Whether a phase with ``classification`` passes a ``job_class`` job, by the project's :class:`Gate`."""
    return config.gate.passes(classification, scope, job_class)


# run identity


def _import(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _distribution(name: str):
    from importlib import metadata

    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


def _vcs_commit(name: str) -> str | None:
    """The commit a distribution was installed from (its ``direct_url.json``), or ``None`` for an index install."""
    dist = _distribution(name)
    direct_url = dist.read_text("direct_url.json") if dist else None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id") if direct_url else None


def _record_sha256(name: str) -> str | None:
    """SHA-256 of a distribution's installed ``RECORD``, which hashes every file it installed."""
    dist = _distribution(name)
    record = dist.read_text("RECORD") if dist else None
    return hashlib.sha256(record.encode()).hexdigest() if record else None


def version_of(source: str) -> str | None:
    """The value one ``versions`` source names, or ``None`` when it is not installed (never a truthy placeholder: the
    release checker reads any truthy value as present)."""
    kind, _, name = source.partition(":")
    if kind == "dist":
        dist = _distribution(name)
        return dist.version if dist else None
    if kind == "module":
        module = _import(name)
        return getattr(module, "__version__", None) if module else None
    if kind == "vcs":
        return _vcs_commit(name)
    if kind == "record":
        return _record_sha256(name)
    raise ValueError(f"unknown version source {source!r}")


def installed_versions(config: EvidenceConfig) -> dict:
    """The versions a report records: the interpreter, torch (with its commit and backend), then the project's."""
    import platform

    versions: dict = {"python": platform.python_version()}
    torch = _import("torch")
    versions.update(
        torch=torch.__version__ if torch else None,
        torch_git_version=torch.version.git_version if torch else None,
        torch_cuda=torch.version.cuda if torch else None,
        torch_hip=getattr(torch.version, "hip", None) if torch else None,
    )
    for key, source in config.versions.items():
        versions[key] = version_of(source)
    return versions


def device_identity(config: EvidenceConfig) -> dict:
    """The devices the run saw: the visible-device variables, the device count and each device's name."""
    ident: dict = {var: os.environ.get(var) for var in config.device_vars}
    try:
        torch = importlib.import_module("torch")
        count = torch.cuda.device_count()
        ident.update(device_count=count, devices=[torch.cuda.get_device_name(i) for i in range(count)])
    except (ImportError, RuntimeError, AssertionError) as exc:
        ident.update(device_count=None, devices=f"unavailable: {exc}")
    return ident


# coverage


def coverage_file(out_dir: Path, phase_name: str) -> Path:
    """The phase's coverage data file (the special-test runner appends every test's run to it)."""
    return Path(out_dir).resolve() / "coverage" / phase_name / ".coverage"


def coverage_summary(config: EvidenceConfig, phase_name: str, out_dir: Path) -> dict:
    """Export a measured phase's coverage data as JSON (file names relative to the root) and summarize it.

    A measured phase without a data file is reported as missing, which a comparison must treat as an evidence gap, not
    as zero lines.
    """
    if not config.phase(phase_name).measured:
        return {"measured": False}
    data = coverage_file(out_dir, phase_name)
    if not data.exists():
        return {"measured": True, "missing": True, "data_file": str(data)}
    export = Path(out_dir).resolve() / "coverage" / f"{phase_name}.json"
    cmd = [sys.executable, "-m", "coverage", "json", f"--data-file={data}", "-o", str(export), "--ignore-errors"]
    proc = subprocess.run(cmd, cwd=config.root, capture_output=True, text=True)
    if proc.returncode or not export.exists():
        return {"measured": True, "missing": True, "data_file": str(data), "error": proc.stderr.strip()[-500:]}
    doc = json.loads(export.read_text(encoding="utf-8"))
    return {
        "measured": True,
        "missing": False,
        "data_file_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "json": export.relative_to(Path(out_dir).resolve()).as_posix(),
        "json_sha256": hashlib.sha256(export.read_bytes()).hexdigest(),
        "files": len(doc["files"]),
        "num_statements": doc["totals"]["num_statements"],
        "covered_lines": doc["totals"]["covered_lines"],
        "percent_covered": round(doc["totals"]["percent_covered"], 2),
    }


def executed_lines(out_dir: Path, phase_name: str) -> dict[str, set[int]] | None:
    """The lines a phase's coverage export records as executed, per relative file name; ``None`` if absent."""
    export = Path(out_dir) / "coverage" / f"{phase_name}.json"
    if not export.exists():
        return None
    doc = json.loads(export.read_text(encoding="utf-8"))
    return {name: set(f["executed_lines"]) for name, f in doc["files"].items()}


def coverage_diff(config: EvidenceConfig, reference: Path, candidate: Path, phase_names=None) -> dict:
    """Per measured phase, and for their union, the lines the reference run executed that the candidate did not
    (``lost``) and the reverse (``gained``). Coverage parity is an empty ``lost`` everywhere."""
    names = [n for n in (phase_names or [p.name for p in config.phases]) if config.phase(n).measured]
    result: dict = {}
    union: dict[str, dict[str, set[int]]] = {"reference": {}, "candidate": {}}
    for name in names:
        ref, cand = executed_lines(reference, name), executed_lines(candidate, name)
        if ref is None or cand is None:
            result[name] = {"missing": [side for side, v in (("reference", ref), ("candidate", cand)) if v is None]}
            continue
        for side, lines in (("reference", ref), ("candidate", cand)):
            for f, ls in lines.items():
                union[side].setdefault(f, set()).update(ls)
        result[name] = _line_delta(ref, cand)
    result["union"] = _line_delta(union["reference"], union["candidate"])
    return result


def _line_delta(ref: dict[str, set[int]], cand: dict[str, set[int]]) -> dict:
    files = sorted(set(ref) | set(cand))
    lost = {f: sorted(ref.get(f, set()) - cand.get(f, set())) for f in files}
    gained = {f: sorted(cand.get(f, set()) - ref.get(f, set())) for f in files}
    n_ref = sum(map(len, ref.values()))
    kept = n_ref - sum(map(len, lost.values()))
    return {
        "reference_lines": n_ref,
        "candidate_lines": sum(map(len, cand.values())),
        "lost": {f: v for f, v in lost.items() if v},
        "gained": {f: v for f, v in gained.items() if v},
        "kept_fraction": round(kept / n_ref, 4) if n_ref else None,
    }


# the pipeline's side


def run_env(config: EvidenceConfig, phase_name: str, out_dir: Path) -> dict:
    """The environment a phase's pipeline command runs under to be recorded (and selected, and measured, as the
    project configures); every pytest process it starts inherits it."""
    phase = config.phase(phase_name)
    addopts = " ".join(filter(None, [os.environ.get("PYTEST_ADDOPTS", ""), "-p", PLUGIN]))
    env: dict[str, str] = {}
    if config.run_env_exports_selection:
        env.update(config.collect_env)
        env.update(phase.selection_env)
    env[PHASE_RECORD_DIR_ENV] = str(Path(out_dir).resolve() / "records" / phase_name)
    # only the phase's own sessions: a test that runs pytest on a tree of its own inherits these too
    env[PHASE_RECORD_ROOT_ENV] = str(config.root)
    env["PYTEST_ADDOPTS"] = addopts.strip()
    if phase.measured:
        data = coverage_file(out_dir, phase_name)
        data.parent.mkdir(parents=True, exist_ok=True)
        env["COVERAGE_FILE"] = str(data)
    return env


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _summary(doc: dict) -> str:
    o, c = doc["outcomes"], doc["collection"]
    return (
        f"phase={doc['phase']} {doc['evidence_class']} vendor={doc['vendor']} scope={doc['scope']}"
        f" collected={c['raw']} selected={c['selected']} passed={o['passed']} failed={o['failed']}"
        f" errors={o['errors']} skipped={o['skipped']} xfailed={o['xfailed']} xpassed={o['xpassed']}"
        f" unexecuted={o['unexecuted']} rank_failures={doc['rank_failures']}"
        f" missing_rank_completions={doc['missing_rank_completions']} -> {doc['classification']}"
    )


def _parser(config: EvidenceConfig | None) -> argparse.ArgumentParser:
    phases = [p.name for p in config.phases] if config else []
    measured = [p.name for p in config.phases if p.measured] if config else []
    scopes = [FULL, *(n for n, s in config.scopes.items() if s.select_file)] if config else [FULL]
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=Path("pyproject.toml"), help="the project's pyproject.toml")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[common]
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", parents=[common], help="the phases, their selectors and the commands that run them")
    collect = sub.add_parser(
        "collect", parents=[common], help="write <out-dir>/manifests/<phase>.json (CPU only, executes nothing)"
    )
    collect.add_argument("--phase", action="append", choices=phases, required=True)
    collect.add_argument("--scope", choices=scopes, default=None, help="a select-file scope (default: full)")
    collect.add_argument("--out-dir", type=Path, required=True)
    env = sub.add_parser("run-env", parents=[common], help="print the exports that record a phase")
    env.add_argument("--phase", choices=phases, required=True)
    env.add_argument("--out-dir", type=Path, required=True)
    report = sub.add_parser(
        "report", parents=[common], help="write <out-dir>/reports/<phase>.json; exit by the project's gate"
    )
    report.add_argument("--phase", choices=phases, required=True)
    report.add_argument("--out-dir", type=Path, required=True)
    report.add_argument("--vendor", choices=sorted(config.vendors) if config else None, required=True)
    report.add_argument("--evidence-class", choices=EVIDENCE_CLASSES, required=True)
    report.add_argument("--job-class", default="ci-any-vendor", help="the GPU job's class, read by the gate")
    report.add_argument("--lease-mode", default=None)
    report.add_argument("--image", default=None)
    report.add_argument("--reviewed-exclusions", type=Path, default=None, help="JSON object: node id -> reason")
    diff = sub.add_parser(
        "coverage-diff", parents=[common], help="lines one run's measured phases executed and the other's did not"
    )
    diff.add_argument("reference", type=Path, help="out-dir of the reference run (for example the NVIDIA host)")
    diff.add_argument("candidate", type=Path, help="out-dir of the run compared with it")
    diff.add_argument("--phase", action="append", choices=measured)
    diff.add_argument("--out", type=Path, default=None, help="write the full result as JSON")
    return parser


def _collect_main(config: EvidenceConfig, args: argparse.Namespace) -> int:
    status = 0
    for name in args.phase:
        try:
            manifest = build_manifest(config, name, args.scope)
        except (ValueError, OSError) as exc:
            print(f"error: phase {name}: {exc}", file=sys.stderr)
            return 2
        _write_json(args.out_dir / "manifests" / f"{name}.json", manifest)
        # collected beside phase, always: a count without its phase reads as a suite result
        print(
            f"phase={name} scope={manifest['scope']} raw={len(manifest['raw_node_ids'])}"
            f" selected={len(manifest['selected_node_ids'])} errors={len(manifest['collection_errors'])}"
            f" (collection only) sha256={manifest_digest(manifest)}"
        )
        status |= bool(manifest["collection_errors"])
    return status


def _coverage_diff_main(config: EvidenceConfig, args: argparse.Namespace) -> int:
    result = coverage_diff(config, args.reference, args.candidate, args.phase)
    if args.out:
        _write_json(args.out, result)
    lost_any = False
    for name, r in result.items():
        if "missing" in r:
            print(f"{name}: coverage export missing on {', '.join(r['missing'])}")
            lost_any = True
            continue
        lost = sum(map(len, r["lost"].values()))
        lost_any |= bool(lost)
        print(
            f"{name}: reference={r['reference_lines']} candidate={r['candidate_lines']} lost={lost}"
            f" gained={sum(map(len, r['gained'].values()))} kept={r['kept_fraction']}"
        )
        for f, lines in r["lost"].items():
            print(f"  lost {f}: {len(lines)} lines")
    return 1 if lost_any else 0


def _report_main(config: EvidenceConfig, args: argparse.Namespace) -> int:
    manifest = json.loads((args.out_dir / "manifests" / f"{args.phase}.json").read_text(encoding="utf-8"))
    reviewed = json.loads(args.reviewed_exclusions.read_text(encoding="utf-8")) if args.reviewed_exclusions else None
    versions = installed_versions(config)
    for key in ("torch", *config.versions):
        if versions.get(key) is None:
            print(f"note: {key} is not installed; recorded as null", file=sys.stderr)
    doc = build_report(
        config,
        manifest,
        read_sessions(args.out_dir / "records" / args.phase),
        vendor=args.vendor,
        evidence_class=args.evidence_class,
        lease_mode=args.lease_mode,
        image=args.image,
        versions={**versions, **device_identity(config)},
        reviewed_exclusions=reviewed,
        coverage=coverage_summary(config, args.phase, args.out_dir),
    )
    _write_json(args.out_dir / "reports" / f"{args.phase}.json", doc)
    print(_summary(doc))
    for e in doc["exclusions"]:
        if not e["reason"]:
            print(f"  unreviewed {e['outcome']}: {e['nodeid']}")
    cov = doc["coverage"]
    if cov.get("measured"):
        status = "MISSING" if cov.get("missing") else f"{cov['covered_lines']}/{cov['num_statements']} lines"
        print(f"phase={args.phase} coverage: {status}")
    passes = gate(config, doc["classification"], doc["scope"], args.job_class)
    print(f"phase={args.phase} {'passes' if passes else 'fails'} a {args.job_class} job")
    return 0 if passes else 1


def main(argv: list[str] | None = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=Path("pyproject.toml"))
    known, _ = pre.parse_known_args(argv)
    try:
        config = EvidenceConfig.load(known.config)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"error: phase-evidence configuration {known.config}: {exc}", file=sys.stderr)
        return 2
    args = _parser(config).parse_args(argv)
    if args.cmd == "list":
        for phase in config.phases:
            print(
                f"{phase.name}\trelease={phase.release}\tmeasured={phase.measured}"
                f"\tselector={config.selector(phase)}\t{phase.command}"
            )
        return 0
    if args.cmd == "collect":
        return _collect_main(config, args)
    if args.cmd == "run-env":
        for key, value in run_env(config, args.phase, args.out_dir).items():
            print(f"export {key}={json.dumps(value)}")
        return 0
    if args.cmd == "coverage-diff":
        return _coverage_diff_main(config, args)
    return _report_main(config, args)


if __name__ == "__main__":
    sys.exit(main())
