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
"""Scheduled-run disposition: whether a scheduled GPU build of a branch should run, by whether its tree is covered.

A scheduled build exists to give a branch a full GPU run, so it has to meet the same two tests a manual run does: there
is a reason to run, and the GPU agents are otherwise idle. The verdict:

- ``REJECT covered``: a green **full** run already ran this exact tree (same git tree object). "Full" is the run's own
  decision, read from the log of the step the project names (``run-step``) with the project's pattern
  (``run-pattern``), so a narrowed run (a selected gate, a smoke set) never counts: catching what a narrowed run misses
  is half of what a scheduled run is for. A schedule the project lists in ``always-run`` (one whose extra work runs in
  no other build) is never rejected this way.
- ``RELEASE``: the tree is uncovered and the host is quiet (nothing running, nothing approved and waiting, no
  releasable pull request gate, no lease held or waited on).
- ``HOLD``: uncovered but the host is busy, named; ask again later.
- ``REJECT stale``: still not releasable once the build has waited ``approval-cap-minutes``, naming what held it, so a
  ``batch: true`` schedule is never blocked behind an old build. The next scheduled run is a fresh attempt.

It decides and prints; releasing or rejecting the approval stays with whoever operates the pool. Anything it cannot
establish is refused by name rather than guessed: a build that is not a queued scheduled build, a missing credential,
no lease state (pass ``--no-lease`` to say there is none), a tree that cannot be read.

**In the job** (``--in-job``, an early step of a scheduled build): the coverage test runs inside the job instead, after
any approval. A scheduled run whose tree a green full run since the last green scheduled run already covered cancels
itself, through the build REST API with the job's access token (``SYSTEM_ACCESSTOKEN`` mapped into the step). An
``always-run`` schedule and every other build reason always run. This direction never fails closed: anything the step
cannot establish (the configuration, the API, the tree, the cancel) lets the run go ahead in full, because an extra
full run costs a slot and a skipped one could hide a regression.

What differs between projects is configuration, read from ``[tool.gpu-epsilon.scheduled-runs]`` in the project's
``pyproject.toml`` (:class:`ScheduledRunsConfig` lists the keys).

Usage, from the repository root::

    python -m gpu_epsilon.scheduled_runs --build 1704 \\
        --org https://dev.azure.com/<org> --project <project> --pool-project <another project on the pool> \\
        --lease-status-cmd "$GPU_LEASE_CMD --status"
    python -m gpu_epsilon.scheduled_runs --in-job   # in a pipeline step, from its predefined variables

Exit status: 0 release, 1 reject, 3 hold, 2 refused (nothing decided). ``--in-job`` always exits 0. Importing this
module needs only the standard library; reading the configuration needs Python 3.11 (``tomllib``).
"""
import argparse
import base64
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

#: A held lease, or a ticket waiting for one, in a lease tool's ``--status`` output.
LEASE_HELD = re.compile(r"^(\S+)\s+HELD\s+(.*)$")
LEASE_TICKET = re.compile(r"^\s*ticket\s+\S+\s+(\S+)")
#: Check runs that belong to the GPU pipeline itself, not to the hosted CPU matrix a gate's release waits for.
GPU_CHECK_APPS = frozenset({"azure-pipelines"})
GREEN_CONCLUSIONS = frozenset({"success", "skipped", "neutral"})
RELEASE, REJECT, HOLD = "RELEASE", "REJECT", "HOLD"
EXIT = {RELEASE: 0, REJECT: 1, HOLD: 3}
CONFIG_TABLE = "[tool.gpu-epsilon.scheduled-runs]"


class ConfigError(ValueError):
    """A project's scheduled-run configuration is malformed."""


class Refused(RuntimeError):
    """The inputs do not let the disposition be decided."""


@dataclass(frozen=True)
class ScheduledRunsConfig:
    """A project's scheduled-run configuration (``[tool.gpu-epsilon.scheduled-runs]``, keys in kebab case).

    Attributes:
        repository: the GitHub ``owner/name`` the GPU pipeline builds, whose commits' trees are compared.
        run_step: the display name of the pipeline step whose log says whether a run was full.
        run_pattern: a regular expression with one group, the run's mode, matched against that step's log; the last
            match wins. It should match only a resolved value, never the step's source echoed without running.
        full_mode: the mode that makes a green run cover its tree.
        always_run: schedule display names never rejected as covered (their extra work runs in no other build).
        approval_cap_minutes: how long a queued scheduled build may wait to be released before it is rejected as
            stale.
    """

    repository: str
    run_step: str
    run_pattern: re.Pattern
    full_mode: str = "full"
    always_run: tuple[str, ...] = ()
    approval_cap_minutes: int = 120

    @property
    def approval_cap(self) -> timedelta:
        return timedelta(minutes=self.approval_cap_minutes)

    @classmethod
    def from_mapping(cls, doc) -> "ScheduledRunsConfig":
        if not isinstance(doc, dict):
            raise ConfigError(f"{CONFIG_TABLE} must be a table")
        fields = {key.replace("-", "_"): value for key, value in doc.items() if "_" not in key}
        fields.pop("description", None)
        unknown = {k.replace("-", "_") for k in doc if "_" in k} | (set(fields) - set(cls.__dataclass_fields__))
        if unknown:
            raise ConfigError(f"unknown {CONFIG_TABLE} key(s): {', '.join(sorted(unknown))}")
        if not isinstance(fields.get("repository"), str) or fields["repository"].count("/") != 1:
            raise ConfigError("repository must be the 'owner/name' the GPU pipeline builds")
        for key in ("run_step", "full_mode"):
            if key in fields and not (isinstance(fields[key], str) and fields[key]):
                raise ConfigError(f"{key.replace('_', '-')} must be a non-empty string")
        if "run_step" not in fields:
            raise ConfigError("run-step must name the step whose log says whether a run was full")
        try:
            pattern = re.compile(fields.get("run_pattern", ""))
        except (re.error, TypeError) as exc:
            raise ConfigError(f"run-pattern is not a regular expression: {exc}") from exc
        if not pattern.pattern or pattern.groups != 1:
            raise ConfigError("run-pattern must be a regular expression with exactly one group, the run's mode")
        fields["run_pattern"] = pattern
        always = fields.get("always_run", [])
        if not isinstance(always, list) or not all(isinstance(s, str) and s for s in always):
            raise ConfigError("always-run must be a list of schedule display names")
        fields["always_run"] = tuple(always)
        cap = fields.get("approval_cap_minutes", 120)
        if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
            raise ConfigError("approval-cap-minutes must be a positive integer")
        return cls(**fields)

    @classmethod
    def load(cls, path: Path) -> "ScheduledRunsConfig":
        try:
            import tomllib  # pyright: ignore[reportMissingImports]  (3.11+; the except handles 3.10)
        except ModuleNotFoundError as exc:  # Python 3.10
            raise ConfigError("reading pyproject.toml needs Python 3.11 or newer (tomllib)") from exc
        path = Path(path)
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
        table = doc.get("tool", {}).get("gpu-epsilon", {}).get("scheduled-runs")
        if table is None:
            raise ConfigError(f"{path} has no {CONFIG_TABLE} table")
        return cls.from_mapping(table)


@dataclass(frozen=True)
class FullRun:
    """A green run of the GPU pipeline that ran in full, and the tree it ran (``None`` if unreadable)."""

    build_id: int
    sha: str
    tree: str | None


@dataclass
class HostState:
    """What would make releasing now cost someone else: each list names its builds or holders."""

    running: list[str] = field(default_factory=list)
    approved_waiting: list[str] = field(default_factory=list)
    releasable_gates: list[str] = field(default_factory=list)
    lease: list[str] = field(default_factory=list)

    def blockers(self) -> list[str]:
        named = (
            ("running", self.running),
            ("approved and waiting", self.approved_waiting),
            ("releasable PR gate", self.releasable_gates),
            ("lease", self.lease),
        )
        return [f"{what}: {', '.join(items)}" for what, items in named if items]


@dataclass(frozen=True)
class Verdict:
    action: str
    reason: str

    def line(self, build_id: int) -> str:
        return f"{self.action} scheduled build {build_id}: {self.reason}"


def dispose(
    *,
    tree: str,
    always_run: str | None,
    full_runs: Iterable[FullRun],
    host: HostState,
    waited: timedelta,
    cap: timedelta = timedelta(minutes=120),
) -> Verdict:
    """The verdict for a queued scheduled build whose commit has ``tree``, after ``waited`` in the queue.

    ``always_run`` names the build's schedule when the project lists it in ``always-run``, else ``None``.
    """
    if always_run is None and (run := covering_run(tree, full_runs)) is not None:
        return Verdict(REJECT, f"covered by build {run.build_id} ({run.sha}), a green full run of tree {tree}")
    blockers = host.blockers()
    why = f"schedule {always_run!r} always runs" if always_run else f"tree {tree} has no green full run"
    if not blockers:
        return Verdict(RELEASE, f"{why}; host quiet")
    if waited >= cap:
        return Verdict(REJECT, f"stale: not releasable after {_minutes(waited)} min; held by {'; '.join(blockers)}")
    return Verdict(HOLD, f"{why}; {'; '.join(blockers)} (waited {_minutes(waited)} min of {_minutes(cap)} min)")


def run_mode(log_text: str, pattern: re.Pattern) -> str | None:
    """The mode a run's step log reports through ``pattern``'s group, or ``None`` when the log carries none."""
    found = pattern.findall(log_text)
    return found[-1] if found else None


def lease_blockers(status_text: str) -> list[str]:
    """Held leases and waiting tickets in a lease tool's ``--status`` output (free leases name nothing)."""
    out = []
    for line in status_text.splitlines():
        held = LEASE_HELD.match(line.strip())
        if held:
            out.append(f"{held[1]} held ({held[2].strip()})")
            continue
        ticket = LEASE_TICKET.match(line)
        if ticket:
            out.append(f"{ticket[1]} waiting")
    return out


def hosted_checks_green(check_runs: list[dict]) -> bool:
    """Whether a pull request head's hosted checks are all finished and green; no hosted check is not green."""
    hosted = [c for c in check_runs if (c.get("app") or {}).get("slug") not in GPU_CHECK_APPS]
    return bool(hosted) and all(
        c.get("status") == "completed" and c.get("conclusion") in GREEN_CONCLUSIONS for c in hosted
    )


def _minutes(delta: timedelta) -> int:
    return int(delta.total_seconds() // 60)


def _parse_time(value: str) -> datetime:
    # Azure returns up to 7 fractional digits; datetime accepts 6
    base, _, rest = value.rstrip("Z").partition(".")
    frac = re.match(r"\d*", rest)
    return datetime.fromisoformat(f"{base}.{(frac[0] if frac else '')[:6] or '0'}").replace(tzinfo=timezone.utc)


class Azure:
    """The few Azure DevOps REST reads the disposition needs, with a PAT or the job's bearer token."""

    def __init__(self, org: str, pat: str | None = None, *, bearer: str | None = None):
        if bearer:
            self.auth = f"Bearer {bearer}"
        elif pat:
            self.auth = "Basic " + base64.b64encode(f":{pat}".encode()).decode()
        else:
            raise Refused("AZURE_DEVOPS_EXT_PAT is not set: the build and approval state cannot be read")
        self.org = org.rstrip("/")

    def get(self, project: str, path: str, **query) -> dict:
        query.setdefault("api-version", "7.1")
        url = f"{self.org}/{urllib.parse.quote(project)}/_apis/{path}?{urllib.parse.urlencode(query)}"
        req = urllib.request.Request(url, headers={"Authorization": self.auth, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
        try:
            return json.loads(body)
        except ValueError as exc:
            # a rejected credential can come back as a sign-in page, not an error: never read it as an empty list
            raise Refused(f"Azure answered {path} with something that is not JSON (is the PAT valid?)") from exc

    def patch(self, project: str, path: str, body: dict) -> None:
        url = f"{self.org}/{urllib.parse.quote(project)}/_apis/{path}?api-version=7.1"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            method="PATCH",
            headers={"Authorization": self.auth, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30):
            pass

    def text(self, project: str, path: str) -> str:
        url = f"{self.org}/{urllib.parse.quote(project)}/_apis/{path}?api-version=7.1"
        req = urllib.request.Request(url, headers={"Authorization": self.auth, "Accept": "text/plain"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode(errors="replace")


def github(path: str) -> dict | list:
    """A public GitHub API read (``GITHUB_TOKEN`` used when set, for the rate limit)."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "gpu-epsilon"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    req = urllib.request.Request(f"https://api.github.com/{path}", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def commit_tree(repo: str, sha: str) -> str | None:
    """The tree of ``sha``, or ``None`` when GitHub no longer serves the commit (an old merge ref, say)."""
    try:
        commit = github(f"repos/{repo}/git/commits/{sha}")
    except urllib.error.HTTPError as e:
        if e.code in (404, 422):
            return None
        raise
    return commit["tree"]["sha"] if isinstance(commit, dict) else None


def full_runs_since(
    az: Azure, project: str, definition: int, config: ScheduledRunsConfig, since: str | None
) -> list[FullRun]:
    """Green runs of ``definition`` since ``since`` (or ever) whose ``run-step`` log reports ``full-mode``."""
    query = {"definitions": definition, "resultFilter": "succeeded", "queryOrder": "finishTimeDescending", "$top": 50}
    if since:
        query["minTime"] = since
    runs = []
    for build in az.get(project, "build/builds", **query)["value"]:
        timeline = az.get(project, f"build/builds/{build['id']}/timeline")
        step_logs = [
            r["log"]["id"]
            for r in timeline.get("records") or []
            if r.get("type") == "Task" and r.get("name") == config.run_step and r.get("log")
        ]
        modes = [run_mode(az.text(project, f"build/builds/{build['id']}/logs/{log}"), config.run_pattern)
                 for log in step_logs]
        if config.full_mode in modes:
            runs.append(FullRun(build["id"], build["sourceVersion"], commit_tree(config.repository,
                                                                                 build["sourceVersion"])))
    return runs


def last_green_scheduled(az: Azure, project: str, definition: int) -> str | None:
    found = az.get(
        project,
        "build/builds",
        definitions=definition,
        reasonFilter="schedule",
        resultFilter="succeeded",
        queryOrder="finishTimeDescending",
        **{"$top": 1},
    )["value"]
    return found[0]["finishTime"] if found else None


def host_state(
    az: Azure, projects: list[str], candidate: int, lease_cmds: list[str], run: Callable[[str], str]
) -> HostState:
    """The pool's state across every project that shares it, plus each host's lease status."""
    state = HostState()
    for project in projects:
        state.running += [
            f"{project} {b['id']}" for b in az.get(project, "build/builds", statusFilter="inProgress")["value"]
        ]
        pending = {
            int(a["pipeline"]["owner"]["id"])
            for a in az.get(project, "pipelines/approvals", state="pending", **{"api-version": "7.1-preview.1"})[
                "value"
            ]
        }
        for b in az.get(project, "build/builds", statusFilter="notStarted")["value"]:
            if b["id"] == candidate:
                continue
            if b["id"] not in pending:
                state.approved_waiting.append(f"{project} {b['id']}")
            elif b.get("reason") == "pullRequest":
                head = (b.get("triggerInfo") or {}).get("pr.sourceSha")
                repo = b.get("repository", {}).get("id", "")
                checks = github(f"repos/{repo}/commits/{head}/check-runs") if head and repo else {}
                if isinstance(checks, dict) and hosted_checks_green(checks.get("check_runs", [])):
                    state.releasable_gates.append(f"{project} {b['id']}")
    for cmd in lease_cmds:
        state.lease += lease_blockers(run(cmd))
    return state


def _run(cmd: str) -> str:
    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=60)
    if proc.returncode:
        raise Refused(f"lease status command failed (exit {proc.returncode}): {cmd}: {proc.stderr.strip()[-300:]}")
    return proc.stdout


def covering_run(tree: str, runs: Iterable[FullRun]) -> FullRun | None:
    """The first green full run of exactly ``tree``, if any; a run whose tree could not be read covers nothing."""
    return next((run for run in runs if run.tree is not None and run.tree == tree), None)


def in_job(env: dict, config_path: Path, sleep: Callable[[float], None] = time.sleep) -> int:
    """The in-job coverage test for a scheduled build: cancel a scheduled run of a covered tree, run anything else."""

    def proceed(why: str) -> int:
        print(f"scheduled-run coverage: {why}; running")
        return 0

    if env.get("BUILD_REASON") != "Schedule":
        return proceed(f"build reason {env.get('BUILD_REASON')!r}, not a scheduled build")
    try:
        config = ScheduledRunsConfig.load(config_path)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"##vso[task.logissue type=warning]Scheduled-run configuration could not be read ({exc}); running.")
        return 0
    schedule = env.get("BUILD_CRONSCHEDULE_DISPLAYNAME")
    if schedule in config.always_run:
        return proceed(f"schedule {schedule!r} always runs")
    try:
        az = Azure(env["SYSTEM_COLLECTIONURI"], bearer=env.get("SYSTEM_ACCESSTOKEN"))
        project, build_id, sha = env["SYSTEM_TEAMPROJECT"], int(env["BUILD_BUILDID"]), env["BUILD_SOURCEVERSION"]
        definition = az.get(project, f"build/builds/{build_id}")["definition"]["id"]
        tree = commit_tree(config.repository, sha)
        if tree is None:
            return proceed(f"the tree of {sha} cannot be read")
        covering = covering_run(
            tree, full_runs_since(az, project, definition, config, last_green_scheduled(az, project, definition))
        )
    except (Refused, urllib.error.URLError, KeyError, ValueError) as exc:
        print(f"##vso[task.logissue type=warning]Scheduled-run coverage could not be established ({exc}); running.")
        return 0
    if covering is None:
        return proceed(f"tree {tree} has no green full run since the last green scheduled run")
    print(
        f"##vso[task.logissue type=warning]This scheduled run's tree {tree} was already run green in full by build "
        f"{covering.build_id} ({covering.sha}); cancelling this run rather than repeating it."
    )
    try:
        az.patch(project, f"build/builds/{build_id}", {"status": "cancelling"})
    except (urllib.error.URLError, ValueError) as exc:
        print(f"##vso[task.logissue type=warning]The cancel request failed ({exc}); running.")
        return 0
    sleep(300)  # the cancel ends this step; reaching the next line means it did not take effect
    print("##vso[task.logissue type=warning]The cancel did not take effect within five minutes; running.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("pyproject.toml"), help="the project's pyproject.toml")
    ap.add_argument("--in-job", action="store_true", help="the coverage test inside a scheduled build (always exits 0)")
    ap.add_argument("--build", type=int, help="the queued scheduled build")
    ap.add_argument("--org", help="https://dev.azure.com/<org>")
    ap.add_argument("--project", help="the build's Azure project")
    ap.add_argument(
        "--pool-project", action="append", default=[], help="another project whose builds share the pool (repeatable)"
    )
    leases = ap.add_mutually_exclusive_group()
    leases.add_argument(
        "--lease-status-cmd", action="append", default=[], help="a host's lease status command (repeatable)"
    )
    leases.add_argument("--no-lease", action="store_true", help="the hosts have no lease tool; skip the lease check")
    args = ap.parse_args(argv)
    if args.in_job:
        return in_job(dict(os.environ), args.config)
    if args.build is None or not args.org or not args.project:
        ap.error("--build, --org and --project are required (or --in-job)")
    if not args.lease_status_cmd and not args.no_lease:
        ap.error("one of --lease-status-cmd or --no-lease is required")
    try:
        config = ScheduledRunsConfig.load(args.config)
        az = Azure(args.org, os.environ.get("AZURE_DEVOPS_EXT_PAT"))
        build = az.get(args.project, f"build/builds/{args.build}")
        if build.get("reason") != "schedule" or build.get("status") != "notStarted":
            raise Refused(
                f"build {args.build} is not a queued scheduled build "
                f"(reason {build.get('reason')!r}, status {build.get('status')!r})"
            )
        tree = commit_tree(config.repository, build["sourceVersion"])
        if tree is None:
            raise Refused(f"the tree of {build['sourceVersion']} cannot be read from {config.repository}")
        definition = build["definition"]["id"]
        schedule = (build.get("triggerInfo") or {}).get("scheduleName")
        always = schedule if schedule in config.always_run else None
        since = last_green_scheduled(az, args.project, definition)
        runs = [] if always else full_runs_since(az, args.project, definition, config, since)
        host = host_state(az, [args.project, *args.pool_project], args.build, args.lease_status_cmd, _run)
        waited = datetime.now(timezone.utc) - _parse_time(build["queueTime"])
        verdict = dispose(tree=tree, always_run=always, full_runs=runs, host=host, waited=waited,
                          cap=config.approval_cap)
    except (ConfigError, OSError, Refused, urllib.error.URLError, KeyError) as exc:
        print(f"REFUSED scheduled build {args.build}: {exc}", file=sys.stderr)
        return 2
    print(verdict.line(args.build))
    return EXIT[verdict.action]


if __name__ == "__main__":
    sys.exit(main())
