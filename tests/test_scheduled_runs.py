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
"""The scheduled-run disposition: each verdict, the parsers it reads, the configuration, and that it refuses rather
than guesses (and, in a job, runs rather than guesses)."""
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from gpu_epsilon import scheduled_runs as srd
from gpu_epsilon.scheduled_runs import ConfigError, FullRun, HostState, ScheduledRunsConfig, dispose

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="the disposition reads pyproject.toml with tomllib")

CONFIGS = Path(__file__).parent / "configs"
SRC = Path(__file__).resolve().parents[1] / "src"
IT_CONFIG = CONFIGS / "interpretune.scheduled-runs.toml"
FTS_CONFIG = CONFIGS / "finetuning-scheduler.scheduled-runs.toml"
TREE = "0567c2989ecf3387734a54afd5b42771cb452ecf"
OTHER = "d33e1997" + "0" * 32
FULL = [FullRun(1724, "ea2b8ce9" + "0" * 32, TREE), FullRun(1719, "d6438182" + "0" * 32, OTHER)]
QUIET = HostState()
CAP = timedelta(minutes=120)
SOON, LATE = timedelta(minutes=5), CAP + timedelta(minutes=1)
MINIMAL = {"repository": "o/r", "run-step": "Plan", "run-pattern": r"mode=(full|none)\b"}


def _dispose(tree=TREE, always_run=None, full_runs=FULL, host=QUIET, waited=SOON):
    return dispose(tree=tree, always_run=always_run, full_runs=full_runs, host=host, waited=waited, cap=CAP)


# the verdict


def test_covered_tree_is_rejected_naming_the_full_run():
    v = _dispose()
    assert v.action == srd.REJECT and v.reason.startswith("covered by build 1724")


def test_an_always_run_schedule_is_never_rejected_as_covered():
    v = _dispose(always_run="weekly extended GPU run")
    assert v.action == srd.RELEASE and "'weekly extended GPU run' always runs" in v.reason


def test_uncovered_tree_on_a_quiet_host_is_released():
    assert _dispose(tree="f" * 40).action == srd.RELEASE


def test_an_unreadable_full_run_tree_covers_nothing():
    assert _dispose(full_runs=[FullRun(1600, "a" * 40, None)]).action == srd.RELEASE


@pytest.mark.parametrize(
    "host, named",
    [
        (HostState(running=["interpretune 1730"]), "running: interpretune 1730"),
        (HostState(approved_waiting=["finetuning-scheduler 1731"]), "approved and waiting: finetuning-scheduler 1731"),
        (HostState(releasable_gates=["interpretune 1729"]), "releasable PR gate: interpretune 1729"),
        (HostState(lease=["gpu held (pid=1 project=x)"]), "lease: gpu held"),
    ],
)
def test_a_busy_host_holds_and_names_what_holds_it(host, named):
    v = _dispose(tree="f" * 40, host=host)
    assert v.action == srd.HOLD and named in v.reason and "of 120 min" in v.reason


def test_a_build_still_held_at_the_cap_is_rejected_as_stale():
    v = _dispose(tree="f" * 40, host=HostState(lease=["gpu held (x)"]), waited=LATE)
    assert v.action == srd.REJECT and v.reason.startswith("stale") and "gpu held" in v.reason


def test_the_cap_is_the_projects():
    v = dispose(tree="f" * 40, always_run=None, full_runs=[], host=HostState(running=["p 1"]),
                waited=timedelta(minutes=31), cap=timedelta(minutes=30))
    assert v.action == srd.REJECT and v.reason.startswith("stale")


def test_a_quiet_host_releases_even_past_the_cap():
    assert _dispose(tree="f" * 40, full_runs=[], waited=LATE).action == srd.RELEASE


# the parsers


@pytest.mark.parametrize(
    "log, mode",
    [
        (
            "2026-10-25T20:40:01Z GPU run plan: mode=full extended=False lease_wait=2400s on_lease_timeout=fail: x",
            "full",
        ),
        ("GPU run plan: mode=selected extended=False lease_wait=2400s on_lease_timeout=fail: 3 areas", "selected"),
        ("GPU run plan: mode=none extended=False lease_wait=2400s", "none"),
        # the step's source echoed without running prints no resolved mode, so it must not read as a plan
        ('python -m tests.gpu_ci.plan_gpu_run\nprint(f"GPU run plan: mode={plan.mode}")', None),
        ("", None),
    ],
)
def test_interpretunes_run_mode_reads_only_a_resolved_plan_line(log, mode):
    assert srd.run_mode(log, ScheduledRunsConfig.load(IT_CONFIG).run_pattern) == mode


@pytest.mark.parametrize(
    "log, mode",
    [
        ("2026-11-15T16:30:02Z standalone scope: full\n2026-11-15T16:30:03Z ...", "full"),
        ("standalone scope: smoke", "smoke"),
        # the container script echoed with its variable unexpanded is not a scope
        ('echo "standalone scope: ${FTS_STANDALONE_SCOPE}"', None),
        ("standalone scope full", None),  # the host's own line, in another step, has no colon
    ],
)
def test_finetuning_schedulers_run_mode_reads_the_standalone_scope(log, mode):
    assert srd.run_mode(log, ScheduledRunsConfig.load(FTS_CONFIG).run_pattern) == mode


def test_lease_blockers_name_holders_and_waiters_and_nothing_when_free():
    status = (
        "gpu        HELD  pid=294555 project=v7-research since=2026-10-25T11:13:59-07:00\n"
        "  NVIDIA GeForce RTX 4090 (24564 MiB) GPU-af39 free\n"
        "cpu-heavy  free\n"
        "queue (oldest first):\n"
        "  ticket 0000000057  fts-core-104-nvidia  wants all\n"
    )
    assert srd.lease_blockers(status) == [
        "gpu held (pid=294555 project=v7-research since=2026-10-25T11:13:59-07:00)",
        "fts-core-104-nvidia waiting",
    ]
    assert srd.lease_blockers("gpu        free\ncpu-heavy  free\n") == []


def _check(name, status="completed", conclusion="success", app="github-actions"):
    return {"name": name, "status": status, "conclusion": conclusion, "app": {"slug": app}}


@pytest.mark.parametrize(
    "runs, green",
    [
        ([_check("cpu (ubuntu)"), _check("Lint"), _check("gpu", "queued", None, "azure-pipelines")], True),
        ([_check("cpu (ubuntu)"), _check("cpu (macos)", "in_progress", None)], False),
        ([_check("cpu (ubuntu)"), _check("Lint", conclusion="failure")], False),
        # no hosted check at all is not a green matrix
        ([_check("gpu", "queued", None, "azure-pipelines")], False),
        ([], False),
    ],
)
def test_hosted_checks_green(runs, green):
    assert srd.hosted_checks_green(runs) is green


def test_parse_time_accepts_azure_precision():
    assert srd._parse_time("2026-10-25T10:00:08.2352154Z").microsecond == 235215
    assert srd._parse_time("2026-10-25T10:00:08Z").second == 8


# the configuration


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.scheduled-runs.toml")), ids=lambda p: p.name.split(".")[0])
def test_project_configs_load(path):
    config = ScheduledRunsConfig.load(path)
    assert config.full_mode == "full" and config.approval_cap == CAP
    assert config.repository.startswith("speediedan/")


def test_interpretune_always_runs_its_weekly_schedule_and_fts_has_none():
    assert ScheduledRunsConfig.load(IT_CONFIG).always_run == ("weekly extended GPU run",)
    assert ScheduledRunsConfig.load(FTS_CONFIG).always_run == ()


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"run_step": "x"}, "unknown"),
        ({"repository": "r"}, "repository must be the 'owner/name'"),
        ({"run-step": ""}, "run-step must be a non-empty string"),
        ({"run-pattern": "mode=full"}, "exactly one group"),
        ({"run-pattern": "(a)(b)"}, "exactly one group"),
        ({"run-pattern": "("}, "not a regular expression"),
        ({"always-run": "weekly"}, "always-run must be a list"),
        ({"approval-cap-minutes": 0}, "positive integer"),
        ({"approval-cap-minutes": True}, "positive integer"),
    ],
)
def test_malformed_configs_are_refused(change, reason):
    with pytest.raises(ConfigError, match=reason):
        ScheduledRunsConfig.from_mapping({**MINIMAL, **change})


def test_a_config_without_a_run_step_is_refused():
    with pytest.raises(ConfigError, match="run-step must name the step"):
        ScheduledRunsConfig.from_mapping({"repository": "o/r", "run-pattern": "(x)"})


def test_a_pyproject_without_the_table_is_refused(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.gpu-epsilon.phases]\n")
    with pytest.raises(ConfigError, match=r"no \[tool.gpu-epsilon.scheduled-runs\] table"):
        ScheduledRunsConfig.load(tmp_path / "pyproject.toml")


def test_imports_on_the_standard_library_alone(tmp_path):
    """The in-job step may run on whatever Python the job has: the module imports nothing outside the standard library
    and this package (``tomllib`` only once it reads a configuration)."""
    code = (
        f"import sys; sys.path.insert(0, {str(SRC)!r})\n"
        "import gpu_epsilon.scheduled_runs\n"
        "top = {n.split('.')[0] for n in sys.modules}\n"
        "print(sorted(top - set(sys.stdlib_module_names) - {'gpu_epsilon', '__main__'}), 'tomllib' in sys.modules)"
    )
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[] False"


# the pool


class FakeAzure:
    """Builds and approvals per project, shaped like the REST answers the disposition reads."""

    def __init__(self, builds: dict, approvals: dict):
        self.builds, self.approvals = builds, approvals

    def get(self, project, path, **query):
        if path == "pipelines/approvals":
            return {"value": [{"pipeline": {"owner": {"id": i}}} for i in self.approvals.get(project, [])]}
        status = query["statusFilter"]
        return {"value": [b for b in self.builds.get(project, []) if b["status"] == status]}


def test_host_state_reads_every_project_on_the_pool(monkeypatch):
    builds = {
        "interpretune": [
            {"id": 1700, "status": "notStarted", "reason": "schedule"},  # the candidate itself
            {
                "id": 1701,
                "status": "notStarted",
                "reason": "pullRequest",
                "triggerInfo": {"pr.sourceSha": "a" * 40},
                "repository": {"id": "speediedan/interpretune"},
            },
        ],
        "finetuning-scheduler": [
            {"id": 1702, "status": "inProgress"},
            {"id": 1703, "status": "notStarted", "reason": "manual"},  # approved, waiting for an agent
        ],
    }
    az = FakeAzure(builds, {"interpretune": [1700, 1701]})
    monkeypatch.setattr(srd, "github", lambda path: {"check_runs": [_check("cpu (ubuntu)")]})
    state = srd.host_state(az, ["interpretune", "finetuning-scheduler"], 1700, ["lease-status"], lambda cmd: "gpu free")
    assert state.running == ["finetuning-scheduler 1702"]
    assert state.approved_waiting == ["finetuning-scheduler 1703"]
    assert state.releasable_gates == ["interpretune 1701"]
    assert state.lease == []


class FakeRuns:
    """Green builds with one log per step, for :func:`full_runs_since`."""

    def __init__(self, logs: dict):
        self.logs = logs  # {build id: {step name: log text}}

    def get(self, project, path, **query):
        if path == "build/builds":
            return {"value": [{"id": i, "sourceVersion": f"{i:040d}"} for i in self.logs]}
        build = int(path.split("/")[2])
        return {"records": [{"type": "Task", "name": name, "log": {"id": f"{build}:{name}"}}
                            for name in self.logs[build]]}

    def text(self, project, path):
        build, name = path.rsplit("/", 1)[1].split(":", 1)
        return self.logs[int(build)][name]


def test_full_runs_since_counts_only_runs_whose_step_reports_the_full_mode(monkeypatch):
    config = ScheduledRunsConfig.load(FTS_CONFIG)
    runs = FakeRuns({
        2049: {"Testing: standalone multi-gpu": "standalone scope: full"},
        2050: {"Testing: standalone multi-gpu": "standalone scope: smoke"},  # a pull request's merge-ref run
        2051: {"Testing: Examples": "standalone scope: full"},  # the right line in another step does not count
    })
    monkeypatch.setattr(srd, "commit_tree", lambda repo, sha: f"tree-of-{repo}")
    found = srd.full_runs_since(runs, "ci-fts", 7, config, None)  # pyright: ignore[reportArgumentType]
    assert found == [FullRun(2049, f"{2049:040d}", "tree-of-speediedan/finetuning-scheduler")]


# the command line


def test_cli_refuses_without_a_credential(monkeypatch, capsys):
    monkeypatch.delenv("AZURE_DEVOPS_EXT_PAT", raising=False)
    argv = ["--config", str(IT_CONFIG), "--build", "1", "--org", "https://dev.azure.com/x", "--project", "p",
            "--no-lease"]
    assert srd.main(argv) == 2
    assert "AZURE_DEVOPS_EXT_PAT is not set" in capsys.readouterr().err


def test_cli_refuses_without_a_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AZURE_DEVOPS_EXT_PAT", "x")
    argv = ["--config", str(tmp_path / "pyproject.toml"), "--build", "1", "--org", "https://dev.azure.com/x",
            "--project", "p", "--no-lease"]
    assert srd.main(argv) == 2
    assert "REFUSED scheduled build 1" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--build", "1", "--org", "https://dev.azure.com/x", "--project", "p"],  # no lease state
        ["--org", "https://dev.azure.com/x", "--project", "p", "--no-lease"],  # no build
    ],
)
def test_cli_requires_a_build_and_the_lease_state_or_an_explicit_no_lease(argv):
    with pytest.raises(SystemExit):
        srd.main(argv)


def test_cli_refuses_a_build_that_is_not_a_queued_scheduled_build(monkeypatch, capsys):
    monkeypatch.setenv("AZURE_DEVOPS_EXT_PAT", "x")
    monkeypatch.setattr(srd.Azure, "get", lambda self, project, path, **q: {"reason": "manual", "status": "completed"})
    argv = ["--config", str(IT_CONFIG), "--build", "1721", "--org", "https://dev.azure.com/x", "--project", "p",
            "--no-lease"]
    assert srd.main(argv) == 2
    assert "not a queued scheduled build (reason 'manual', status 'completed')" in capsys.readouterr().err


def test_azure_refuses_a_non_json_answer(monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"<html>sign in</html>"

    monkeypatch.setattr(srd.urllib.request, "urlopen", lambda req, timeout: Resp())
    with pytest.raises(srd.Refused, match="not JSON"):
        srd.Azure("https://dev.azure.com/x", "pat").get("p", "build/builds")


# --in-job: the coverage test inside a scheduled build

JOB_ENV = {
    "BUILD_REASON": "Schedule",
    "BUILD_CRONSCHEDULE_DISPLAYNAME": "nightly full GPU run",
    "SYSTEM_COLLECTIONURI": "https://dev.azure.com/x/",
    "SYSTEM_TEAMPROJECT": "interpretune",
    "SYSTEM_ACCESSTOKEN": "token",
    "BUILD_BUILDID": "1800",
    "BUILD_REPOSITORY_NAME": "speediedan/interpretune",
    "BUILD_SOURCEVERSION": "a855ed3f" + "0" * 32,
}


@pytest.fixture
def job(monkeypatch):
    """Azure and GitHub replaced: the build's definition, its tree, the full runs, and a record of cancels."""
    calls = {"patch": [], "sleep": []}
    monkeypatch.setattr(srd.Azure, "get", lambda self, project, path, **q: {"definition": {"id": 2}})
    monkeypatch.setattr(srd.Azure, "patch", lambda self, project, path, body: calls["patch"].append((path, body)))
    monkeypatch.setattr(srd, "commit_tree", lambda repo, sha: TREE)
    monkeypatch.setattr(srd, "last_green_scheduled", lambda az, project, definition: "2026-10-12T16:49:20Z")
    monkeypatch.setattr(srd, "full_runs_since", lambda az, project, definition, config, since: FULL)
    return calls


def _in_job(calls, config=IT_CONFIG, **env):
    return srd.in_job({**JOB_ENV, **env}, config, sleep=calls["sleep"].append)


def test_in_job_cancels_a_scheduled_run_of_a_covered_tree(job, capsys):
    assert _in_job(job) == 0
    assert job["patch"] == [("build/builds/1800", {"status": "cancelling"})]
    assert job["sleep"] == [300]
    out = capsys.readouterr().out
    assert "already run green in full by build 1724" in out and "did not take effect" in out


def test_in_job_runs_an_uncovered_tree(job, monkeypatch, capsys):
    monkeypatch.setattr(srd, "commit_tree", lambda repo, sha: "f" * 40)
    assert _in_job(job) == 0 and job["patch"] == []
    assert "has no green full run" in capsys.readouterr().out


@pytest.mark.parametrize(
    "env, why",
    [
        ({"BUILD_CRONSCHEDULE_DISPLAYNAME": "weekly extended GPU run"}, "'weekly extended GPU run' always runs"),
        ({"BUILD_REASON": "PullRequest"}, "not a scheduled build"),
        ({"BUILD_REASON": "Manual"}, "not a scheduled build"),
    ],
)
def test_in_job_never_skips_an_always_run_schedule_or_other_builds(job, capsys, env, why):
    assert _in_job(job, **env) == 0 and job["patch"] == []
    assert why in capsys.readouterr().out


def test_in_job_reads_trees_from_the_configured_repository(job, monkeypatch):
    seen = []
    monkeypatch.setattr(srd, "commit_tree", lambda repo, sha: seen.append(repo) or "f" * 40)
    _in_job(job, config=FTS_CONFIG)
    assert seen == ["speediedan/finetuning-scheduler"]


def test_in_job_runs_when_the_config_cannot_be_read(job, tmp_path, capsys):
    assert _in_job(job, config=tmp_path / "pyproject.toml") == 0 and job["patch"] == []
    assert "configuration could not be read" in capsys.readouterr().out


def test_in_job_runs_when_coverage_cannot_be_established(job, monkeypatch, capsys):
    def down(*a, **k):
        raise srd.urllib.error.URLError("unreachable")

    monkeypatch.setattr(srd, "full_runs_since", down)
    assert _in_job(job) == 0 and job["patch"] == []
    assert "could not be established" in capsys.readouterr().out


def test_in_job_runs_when_the_tree_cannot_be_read(job, monkeypatch, capsys):
    monkeypatch.setattr(srd, "commit_tree", lambda repo, sha: None)
    assert _in_job(job) == 0 and job["patch"] == []
    assert "cannot be read" in capsys.readouterr().out


def test_in_job_runs_when_the_cancel_request_fails(job, monkeypatch, capsys):
    def refused(self, project, path, body):
        raise srd.urllib.error.HTTPError("u", 403, "forbidden", None, None)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(srd.Azure, "patch", refused)
    assert _in_job(job) == 0 and job["sleep"] == []
    assert "cancel request failed" in capsys.readouterr().out


def test_in_job_without_a_token_runs(job, capsys):
    assert _in_job(job, SYSTEM_ACCESSTOKEN="") == 0 and job["patch"] == []
    assert "could not be established" in capsys.readouterr().out


def test_main_dispatches_in_job(monkeypatch):
    seen = {}
    monkeypatch.setattr(srd, "in_job", lambda env, config: seen.update(env=env, config=config) or 0)
    monkeypatch.setenv("BUILD_REASON", "Schedule")
    assert srd.main(["--in-job", "--config", str(FTS_CONFIG)]) == 0
    assert seen["env"]["BUILD_REASON"] == "Schedule" and seen["config"] == FTS_CONFIG
