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
"""Release eligibility: one full, clean, like-for-like GPU run on each vendor, at the release candidate's revision.

A release needs a demanded green on each vendor. A project's release pipeline runs one full leg per vendor, each
publishing its phase evidence as an artifact named ``<artifact_prefix>-<vendor>-<attempt>``, then runs this checker on
a CPU job over all of them. Its verdict is the only result a publication step may consume: the legs' own results are
necessary, not sufficient.

The checker re-derives what it can rather than trusting a report's own summary:

- each required phase has a manifest and a report, the report's ``manifest_sha256`` is the manifest's digest (the rule
  :func:`gpu_epsilon.evidence.manifest_digest` defines), and its counts agree with the manifest's node lists;
- the phase is ``clean`` by :func:`gpu_epsilon.evidence.classify` applied to the counts, and the report says so too;
- the run is the candidate's: repository and revision match, the scope is full, the tree it collected from was clean,
  the evidence class is ``pipeline`` (a parity run made outside the pipeline is never release evidence), the lease was
  taken through the host's lease tool, and the image is digest-pinned;
- within a leg, every phase names the same image and installed versions; across the legs, each phase collected and
  selected the same node ids with the same selector, and the installed versions and builds the project names agree
  (a required version without its local build tag, the one part of it that names the vendor).

Each phase's summary counts are recounted from its per-node records and its exclusions from the nodes it skipped or
expected to fail, and a measured phase's coverage export is checked against the digest its report recorded.

What differs between projects is data, read from a JSON file (:class:`ReleaseConfig` lists the keys): the artifact
names, the required and measured phases, the failure counts its reports carry and the versions that must agree. JSON
rather than a ``pyproject.toml`` table because the release job runs on the agent's own Python, which may predate
``tomllib``.

Usage, on the downloaded artifacts (standard library only)::

    python3 -m gpu_epsilon.release_eligibility --config tests/gpu_ci/release.json \\
        --artifacts "$(Pipeline.Workspace)" --revision "$(Build.SourceVersion)" \\
        --nvidia-result Succeeded --amd-result Succeeded --output eligibility.json

The ``--<vendor>-result`` options are the config's vendors.
"""
import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gpu_epsilon import __version__
from gpu_epsilon.evidence import OUTCOMES, classify, manifest_digest

DIGEST_PINNED = re.compile(r".+@sha256:[0-9a-f]{64}")
DEFAULT_VENDORS = {"nvidia": "cuda", "amd": "hip"}


class Ineligible(ValueError):
    """The evidence does not make the candidate releasable."""


class ConfigError(ValueError):
    """The project's release configuration is malformed."""


@dataclass(frozen=True)
class ReleaseConfig:
    """What a project's release evidence looks like and which parts of it must agree across vendors.

    Attributes:
        repository: the ``owner/name`` every report and manifest must name.
        artifact_prefix: a leg's artifacts are ``<artifact_prefix>-<vendor>-<attempt>``; the latest attempt counts.
        phases_dir: the directory inside each artifact holding ``reports/<phase>.json`` and
            ``manifests/<phase>.json``.
        phases: the phases a release leg runs, in order.
        measured_phases: the phases whose reports carry coverage evidence (run under ``coverage run``).
        failure_counts: report counts that fail a phase when nonzero (:func:`gpu_epsilon.evidence.classify`).
        required_versions: installed versions that must be present and agree across vendors, compared without their
            local tag (``2.15.0+cu132`` and ``2.15.0+rocm10.0`` agree).
        shared_versions: installed versions compared exactly as recorded, absence included: a package one vendor's
            image lacks is a difference, not a pass.
        shared_builds: what names the build behind a version string that does not (a test-channel torch release
            candidate carries the final version), compared as recorded.
        vendors: each vendor and the backend its reports name.
    """

    repository: str
    artifact_prefix: str
    phases_dir: str
    phases: tuple[str, ...]
    measured_phases: tuple[str, ...] = ()
    failure_counts: tuple[str, ...] = ()
    required_versions: tuple[str, ...] = ()
    shared_versions: tuple[str, ...] = ()
    shared_builds: tuple[str, ...] = ()
    vendors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_VENDORS))

    def __post_init__(self) -> None:
        for name in ("repository", "artifact_prefix", "phases_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ConfigError(f"{name} must be a non-empty string")
        lists = ("phases", "measured_phases", "failure_counts", "required_versions", "shared_versions", "shared_builds")
        for name in lists:
            value = getattr(self, name)
            if isinstance(value, str) or not all(isinstance(v, str) and v for v in value):
                raise ConfigError(f"{name} must be a list of non-empty strings")
            object.__setattr__(self, name, tuple(value))
        if not self.phases or len(set(self.phases)) != len(self.phases):
            raise ConfigError("phases must name at least one phase, each once")
        if not set(self.measured_phases) <= set(self.phases):
            raise ConfigError("measured_phases must be a subset of phases")
        if not self.required_versions:
            raise ConfigError("required_versions must name at least the project's own package")
        if not isinstance(self.vendors, dict) or not self.vendors or not all(
            isinstance(k, str) and re.fullmatch(r"[a-z]+", k) and isinstance(v, str) and v
            for k, v in self.vendors.items()
        ):
            raise ConfigError("vendors must map lowercase vendor names to backends")
        object.__setattr__(self, "vendors", dict(self.vendors))

    @classmethod
    def from_mapping(cls, doc: dict) -> "ReleaseConfig":
        if not isinstance(doc, dict):
            raise ConfigError("the release configuration must be a JSON object")
        unknown = sorted(set(doc) - set(cls.__dataclass_fields__) - {"description"})
        if unknown:
            raise ConfigError(f"unknown release configuration key(s): {', '.join(unknown)}")
        try:
            return cls(**{k: v for k, v in doc.items() if k != "description"})
        except TypeError as exc:  # a required key is missing
            raise ConfigError(str(exc)) from exc

    @classmethod
    def load(cls, path: Path) -> "ReleaseConfig":
        return cls.from_mapping(json.loads(Path(path).read_text()))

    @property
    def artifact(self) -> re.Pattern:
        return re.compile(re.escape(self.artifact_prefix) + r"-(?P<vendor>[a-z]+)-(?P<attempt>\d+)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Ineligible(message)


def leg_dirs(config: ReleaseConfig, artifacts: Path) -> dict[str, Path]:
    """Each vendor's evidence: its latest attempt's phases directory (a re-run leg publishes a new attempt)."""
    latest: dict[str, tuple[int, Path]] = {}
    for path in sorted(artifacts.iterdir()) if artifacts.is_dir() else ():
        match = config.artifact.fullmatch(path.name)
        if match and path.is_dir():
            vendor, attempt = match["vendor"], int(match["attempt"])
            if vendor not in latest or attempt > latest[vendor][0]:
                latest[vendor] = (attempt, path / config.phases_dir)
    return {vendor: root for vendor, (_, root) in latest.items()}


def load_leg(config: ReleaseConfig, root: Path) -> dict[str, dict]:
    """``{phase: {"report": ..., "manifest": ..., "coverage_sha256": ...}}`` for every required phase under
    ``root``.

    ``coverage_sha256`` is the digest of the coverage export the report names, when that file is present.
    """
    phases = {}
    for phase in config.phases:
        report, manifest = root / "reports" / f"{phase}.json", root / "manifests" / f"{phase}.json"
        if report.is_file() and manifest.is_file():
            doc = json.loads(report.read_text())
            export = root / str((doc.get("coverage") or {}).get("json") or "")
            phases[phase] = {
                "report": doc,
                "manifest": json.loads(manifest.read_text()),
                "coverage_sha256": hashlib.sha256(export.read_bytes()).hexdigest() if export.is_file() else None,
            }
    return phases


def check_phase(
    config: ReleaseConfig,
    vendor: str,
    phase: str,
    report: dict,
    manifest: dict,
    revision: str,
    coverage_sha256: str | None = None,
) -> None:
    """One phase of one leg: identity, digest, counts against the manifest and the node records, exclusions,
    coverage evidence, and a clean classification."""
    where = f"{vendor}/{phase}"
    for key, expected in (
        ("repository", config.repository),
        ("revision", revision),
        ("phase", phase),
        ("scope", "full"),
        ("evidence_class", "pipeline"),
        ("vendor", vendor),
        ("backend", config.vendors[vendor]),
        ("lease_mode", "tool"),
        ("count_unit", "unique-node-id"),
    ):
        require(report.get(key) == expected, f"{where}: {key} is {report.get(key)!r}, not {expected!r}")
    require(DIGEST_PINNED.fullmatch(report.get("image") or "") is not None, f"{where}: image is not digest-pinned")
    for key in ("repository", "revision", "phase", "scope", "selector"):
        require(report[key] == manifest[key], f"{where}: report and manifest disagree on {key}")
    require(report["manifest_sha256"] == manifest_digest(manifest), f"{where}: manifest digest mismatch")
    require(not manifest.get("tree_dirty_paths"), f"{where}: collected from a dirty tree")
    require(not any(manifest.get("collect_exit_codes", {}).values()), f"{where}: a collection exited non-zero")
    raw, selected = manifest["raw_node_ids"], manifest["selected_node_ids"]
    require(len(raw) == len(set(raw)) and len(selected) == len(set(selected)), f"{where}: duplicate manifest node")
    require(set(selected) <= set(raw), f"{where}: a selected node is absent from the raw collection")
    collection, outcomes = report["collection"], report["outcomes"]
    failure_counts = [report.get(key) for key in config.failure_counts]
    require(
        all(type(v) is int and v >= 0 for v in (*collection.values(), *outcomes.values(), *failure_counts)),
        f"{where}: counts must be nonnegative integers",
    )
    require(
        collection["raw"] == len(raw) and collection["selected"] == len(selected),
        f"{where}: counts disagree with the manifest",
    )
    require(
        collection["raw"] == collection["selected"] + collection["deselected"],
        f"{where}: collection counts do not add up",
    )
    require(collection["selected"] == outcomes["attempted"] + outcomes["unexecuted"], f"{where}: unaccounted nodes")
    require(sum(outcomes[k] for k in OUTCOMES) == outcomes["attempted"], f"{where}: outcomes do not add up")
    # The summary counts decide cleanliness, so they are recounted from the per-node records, which must cover exactly
    # the manifest's selection.
    nodes = report.get("nodes")
    if not isinstance(nodes, dict) or set(nodes) != set(selected):
        raise Ineligible(f"{where}: node records are not the selection")
    recount = dict.fromkeys((*OUTCOMES, "unexecuted"), 0)
    for nodeid, node in nodes.items():
        require(node.get("outcome") in recount, f"{where}: {nodeid} has outcome {node.get('outcome')!r}")
        recount[node["outcome"]] += 1
    require(
        all(outcomes[k] == n for k, n in recount.items()), f"{where}: outcome counts disagree with the node records"
    )
    require(not report.get("unmanifested_nodes"), f"{where}: nodes ran that the manifest did not select")
    # A skip or expected failure is clean only with a reviewed reason: the exclusions must list exactly those nodes,
    # and "reviewed" must mean every one of them has its reason.
    excluded = {n for n, node in nodes.items() if node["outcome"] in ("skipped", "xfailed")}
    reasons = {e.get("nodeid"): e.get("reason") for e in report.get("exclusions", [])}
    require(set(reasons) == excluded, f"{where}: exclusions do not list exactly the skipped and xfailed nodes")
    require(
        report["exclusions_reviewed"] == all(reasons.values()), f"{where}: exclusions_reviewed disagrees with reasons"
    )
    # Coverage is part of a measured phase's evidence: the export must be present and be the one the report digested.
    # It is not compared across vendors here: a backend may run lines of its own.
    if phase in config.measured_phases:
        coverage = report.get("coverage") or {}
        measured = bool(coverage.get("measured")) and not coverage.get("missing")
        require(measured, f"{where}: coverage evidence is missing")
        require(
            coverage_sha256 is not None and coverage_sha256 == coverage.get("json_sha256"),
            f"{where}: coverage export missing or not the one the report digested",
        )
        require((coverage.get("covered_lines") or 0) > 0, f"{where}: coverage records no executed lines")
    classification = classify(report, config.failure_counts)
    require(classification == "clean", f"{where}: phase is {classification}")
    require(report.get("classification") == classification, f"{where}: report claims {report.get('classification')!r}")


def check_release(
    config: ReleaseConfig, legs: dict[str, dict[str, dict]], results: dict[str, str], revision: str
) -> dict:
    """The verdict over every vendor's leg, or :class:`Ineligible` naming the first reason the candidate is not
    releasable."""
    require(bool(revision), "candidate revision is missing")
    for vendor in config.vendors:
        require(results.get(vendor) == "Succeeded", f"{vendor} leg result is {results.get(vendor)!r}, not Succeeded")
        require(vendor in legs, f"{vendor}: no release evidence ({config.artifact_prefix}-{vendor}-<attempt>)")
    selections: dict[str, tuple] = {}
    installed_by_vendor: dict[str, dict] = {}
    evidence = {}
    recorded = ("python", *config.required_versions, *config.shared_versions, *config.shared_builds)
    for vendor in config.vendors:
        phases = legs[vendor]
        missing = sorted(set(config.phases) - set(phases))
        require(not missing, f"{vendor}: no evidence for required phase(s) {', '.join(missing)}")
        first = phases[config.phases[0]]["report"]
        for phase in config.phases:
            report, manifest = phases[phase]["report"], phases[phase]["manifest"]
            check_phase(config, vendor, phase, report, manifest, revision, phases[phase].get("coverage_sha256"))
            for key in ("image", "versions"):
                require(report[key] == first[key], f"{vendor}/{phase}: {key} differs from the leg's other phases")
            selection = (manifest["selector"], sorted(manifest["raw_node_ids"]), sorted(manifest["selected_node_ids"]))
            require(
                selections.setdefault(phase, selection) == selection,
                f"{phase}: the vendors collected or selected different nodes; classify the difference before release",
            )
        versions = first["versions"]
        installed = {k: (versions.get(k) or "").split("+")[0] for k in config.required_versions}
        require(
            all(installed.values()), f"{vendor}: installed versions missing ({', '.join(config.required_versions)})"
        )
        installed_by_vendor[vendor] = {
            **installed,
            **{k: versions.get(k) for k in (*config.shared_versions, *config.shared_builds)},
        }
        evidence[vendor] = {
            "image": first["image"],
            "versions": {k: versions.get(k) for k in recorded},
            "phases": {p: phases[p]["report"]["manifest_sha256"] for p in config.phases},
        }
    require(
        len({json.dumps(v, sort_keys=True) for v in installed_by_vendor.values()}) == 1,
        f"installed versions differ across vendors: {installed_by_vendor}",
    )
    return {
        "eligible": True,
        "repository": config.repository,
        "revision": revision,
        "results": dict(results),
        "required_phases": list(config.phases),
        "evidence": evidence,
        "checker": {"name": "gpu-epsilon", "version": __version__, "config": asdict(config)},
    }


def _parser(vendors) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the project's release configuration (JSON)")
    parser.add_argument("--artifacts", type=Path, required=True, help="the directory the legs' artifacts downloaded to")
    parser.add_argument("--revision", required=True, help="the release candidate's commit")
    for vendor in vendors:
        parser.add_argument(f"--{vendor}-result", required=True, help=f"the {vendor} leg's job result (Succeeded, ...)")
    parser.add_argument("--output", type=Path, required=True, help="where to write the verdict when eligible")
    return parser


def main(argv: list[str] | None = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    pre.add_argument("--output", type=Path)
    known, _ = pre.parse_known_args(argv)
    # A verdict left by an earlier invocation must never survive a failed check as current evidence.
    if known.output:
        known.output.unlink(missing_ok=True)
    config = None
    if known.config:
        try:
            config = ReleaseConfig.load(known.config)
        except (ConfigError, OSError, json.JSONDecodeError) as exc:
            print(f"error: release configuration {known.config}: {exc}", file=sys.stderr)
            return 2
    args = _parser(config.vendors if config else DEFAULT_VENDORS).parse_args(argv)
    assert config is not None  # --config is required, so parse_args exits without one
    results = {vendor: getattr(args, f"{vendor}_result") for vendor in config.vendors}
    try:
        legs = {vendor: load_leg(config, root) for vendor, root in leg_dirs(config, args.artifacts).items()}
        verdict = check_release(config, legs, results, args.revision)
    except (Ineligible, KeyError, TypeError, AttributeError, OSError, json.JSONDecodeError) as exc:
        print(f"INELIGIBLE: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"ELIGIBLE: {', '.join(config.vendors)}, {len(config.phases)} full phases each, revision {args.revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
