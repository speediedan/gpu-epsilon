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
"""The phase driver, end to end, on a small project: each phase a real subprocess, each control a project here."""
import json
import os
import sys
import textwrap

import pytest

from gpu_epsilon import phases as ph
from gpu_epsilon.special_tests import ConfigError

needs_tomllib = pytest.mark.skipif(sys.version_info < (3, 11), reason="the driver reads pyproject.toml with tomllib")

RUNNER = """
[tool.gpu-epsilon.special-tests]
name = "Demo"
exec = ["-m", "pytest", "-p", "no:cacheprovider", "-q"]
collect = ["-m", "pytest", "-p", "no:cacheprovider", "-q", "--collect-only"]

[tool.gpu-epsilon.special-tests.marks.standalone]
env = { DEMO_STANDALONE = "1" }
"""

CONFTEST = """
import os


def pytest_configure(config):
    config.addinivalue_line("markers", "standalone: run alone")


def pytest_collection_modifyitems(config, items):
    standalone = bool(os.environ.get("DEMO_STANDALONE"))
    keep = [item for item in items if bool(item.get_closest_marker("standalone")) == standalone]
    config.hook.pytest_deselected(items=[item for item in items if item not in keep])
    items[:] = keep
"""

TESTS = """
import os

import pytest


def test_plain():
    assert os.environ["DEMO_ALL"] == "1"


@pytest.mark.standalone
def test_alone():
    pass
"""

PYTEST = '"-m", "pytest", "-p", "no:cacheprovider", "-q"'
MARK = '"-c", "open(\\"{0}\\", \\"a\\").write(\\"x\\")"'


def phases_toml(body: str) -> str:
    return textwrap.dedent(f"""
        [tool.gpu-epsilon.phases]
        name = "Demo"
        env = {{ DEMO_ALL = "1" }}
        setup = [[{MARK.format("setup.marker")}]]
        report = [[{MARK.format("report.marker")}]]
        """) + textwrap.dedent(body)


MAIN = f"""
[[tool.gpu-epsilon.phases.phase]]
name = "main"
run = [{PYTEST}, "tests"]
"""


def project(tmp_path, phases: str, tests: str = TESTS, conftest: str = CONFTEST):
    """A project at ``tmp_path`` with the given phases; returns a function that runs the driver there."""
    (tmp_path / "pyproject.toml").write_text(RUNNER + phases_toml(phases))
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text(conftest)
    (tmp_path / "tests" / "test_demo.py").write_text(textwrap.dedent(tests))

    def run(*argv):
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            status = ph.main(["--log-dir", str(tmp_path / "logs"), *argv])
        finally:
            os.chdir(old_cwd)
        records = sorted((tmp_path / "logs").glob("phases_*.json"))
        records = [r for r in records if r.stem.count("_") == 1]  # the session record, not a phase's
        doc = json.loads(records[-1].read_text()) if records else None
        return status, doc

    return run


def statuses(doc):
    return {p["name"]: p["status"] for p in doc["phases"]}


@needs_tomllib
def test_a_clean_run_exits_zero_and_runs_setup_and_report(tmp_path):
    status, doc = project(tmp_path, MAIN)()
    assert status == 0
    assert statuses(doc) == {"main": "passed"}
    assert doc["failed"] == [] and doc["not_run"] == []
    assert (tmp_path / "setup.marker").is_file() and (tmp_path / "report.marker").is_file()


FAILING = """
def test_fails():
    assert False
"""

TWO_PHASES = MAIN + f"""
[[tool.gpu-epsilon.phases.phase]]
name = "second"
run = [{PYTEST}, "tests"]
"""


@needs_tomllib
def test_a_failed_phase_stops_the_run_but_still_reports(tmp_path):
    status, doc = project(tmp_path, TWO_PHASES, tests=FAILING)()
    assert status == 1
    assert statuses(doc) == {"main": "failed", "second": "not-run"}
    assert doc["failed"] == ["main"] and doc["not_run"] == ["second"]
    assert (tmp_path / "report.marker").is_file()


@needs_tomllib
def test_allow_failures_runs_every_phase_and_still_exits_one(tmp_path):
    status, doc = project(tmp_path, TWO_PHASES, tests=FAILING)("--allow-failures")
    assert status == 1
    assert statuses(doc) == {"main": "failed", "second": "failed"}
    assert doc["failed"] == ["main", "second"]


EMPTY = """
[[tool.gpu-epsilon.phases.phase]]
name = "empty"
run = ["-m", "pytest", "-p", "no:cacheprovider", "-q", "tests", "-k", "no_such_test"]
"""


@needs_tomllib
def test_a_phase_that_collects_nothing_fails(tmp_path):
    status, doc = project(tmp_path, EMPTY)()
    assert status == 1
    assert doc["phases"][0]["status"] == "failed" and doc["phases"][0]["detail"] == "no tests collected"


@needs_tomllib
def test_allow_empty_accepts_an_empty_phase(tmp_path):
    status, doc = project(tmp_path, EMPTY + "allow-empty = true\n")()
    assert status == 0 and statuses(doc) == {"empty": "passed"}


@needs_tomllib
def test_a_collection_error_fails_the_phase(tmp_path):
    status, doc = project(tmp_path, MAIN, tests="import no_such_module_anywhere\n")()
    assert status == 1 and statuses(doc) == {"main": "failed"}


SKIPPED = MAIN + """skip-unless = ["-c", "raise SystemExit(3)"]\n"""


@needs_tomllib
def test_a_failing_probe_skips_the_phase(tmp_path):
    status, doc = project(tmp_path, SKIPPED)()
    assert status == 0
    assert statuses(doc) == {"main": "skipped"} and doc["phases"][0]["detail"] == "probe exited 3"


AFTER = MAIN + """after = [["-c", "raise SystemExit(1)"]]\n"""


@needs_tomllib
def test_a_failing_after_command_fails_the_phase(tmp_path):
    status, doc = project(tmp_path, AFTER)()
    assert status == 1 and doc["phases"][0]["detail"] == "an 'after' command failed"


@needs_tomllib
def test_a_failing_setup_runs_no_phase(tmp_path):
    run = project(tmp_path, MAIN)
    text = (tmp_path / "pyproject.toml").read_text().replace("setup.marker", "no_dir/setup.marker")
    (tmp_path / "pyproject.toml").write_text(text)
    status, doc = run()
    assert status == 1
    assert statuses(doc) == {"main": "not-run"} and doc["notes"] == ["setup failed (exit 1)"]
    assert not (tmp_path / "report.marker").exists()


SPECIAL = MAIN + """
[[tool.gpu-epsilon.phases.phase]]
name = "standalone"
special-tests = ["--mark_type=standalone"]
"""


@needs_tomllib
def test_a_special_test_phase_records_the_runners_counts(tmp_path):
    status, doc = project(tmp_path, SPECIAL)()
    assert status == 0
    assert statuses(doc) == {"main": "passed", "standalone": "passed"}
    standalone = doc["phases"][1]
    assert standalone["counts"] == {"collected": 1, "succeeded": 1, "skipped": 0, "failed": 0, "other": 0}
    assert standalone["log"].endswith("_standalone.log")


SPECIAL_FAILING = """
import pytest


@pytest.mark.standalone
def test_alone_fails():
    assert False


@pytest.mark.standalone
def test_alone_passes():
    pass
"""


@needs_tomllib
def test_allow_failures_reaches_the_special_test_runner(tmp_path):
    phases = SPECIAL.replace(MAIN, "")
    status, doc = project(tmp_path, phases, tests=SPECIAL_FAILING)("--allow-failures")
    assert status == 1
    assert doc["phases"][0]["counts"]["failed"] == 1 and doc["phases"][0]["counts"]["succeeded"] == 1


OPTIONAL = TWO_PHASES + "optional = true\n"


@needs_tomllib
def test_selection_by_name(tmp_path):
    run = project(tmp_path, OPTIONAL)
    assert statuses(run()[1]) == {"main": "passed"}
    assert statuses(run("--with", "second")[1]) == {"main": "passed", "second": "passed"}
    assert statuses(run("--phases", "second")[1]) == {"second": "passed"}
    assert statuses(run("--with", "second", "--skip", "main")[1]) == {"second": "passed"}


@needs_tomllib
def test_an_unknown_or_empty_selection_is_an_error(tmp_path, capsys):
    run = project(tmp_path, MAIN)
    assert run("--with", "nope")[0] == 2
    assert "no phase named nope" in capsys.readouterr().err
    assert run("--skip", "main")[0] == 2
    assert run("--phases", "main", "--with", "main")[0] == 2


@needs_tomllib
def test_dry_run_prints_the_plan_and_runs_nothing(tmp_path, capsys):
    status, doc = project(tmp_path, SPECIAL)("--dry-run")
    assert status == 0 and doc is None
    out = capsys.readouterr().out
    assert "main: " in out and "gpu_epsilon.special_tests" in out and "report: " in out
    assert not (tmp_path / "setup.marker").exists()


def test_phase_config_errors():
    with pytest.raises(ConfigError, match="exactly one of"):
        ph.Phase.from_mapping({"name": "x"})
    with pytest.raises(ConfigError, match="exactly one of"):
        ph.Phase.from_mapping({"name": "x", "run": ["-c", "1"], "special-tests": ["--mark_type=a"]})
    with pytest.raises(ConfigError, match="unknown key"):
        ph.Phase.from_mapping({"name": "x", "run": ["-c", "1"], "colour": "red"})
    with pytest.raises(ConfigError, match="list of strings"):
        ph.Phase.from_mapping({"name": "x", "run": "-c 1"})
    with pytest.raises(ConfigError, match="allow-empty"):
        ph.Phase.from_mapping({"name": "x", "special-tests": ["--mark_type=a"], "allow-empty": True})
    with pytest.raises(ConfigError, match="without commas"):
        ph.Phase.from_mapping({"name": "a,b", "run": ["-c", "1"]})


def test_driver_config_errors():
    phase = {"name": "x", "run": ["-c", "1"]}
    with pytest.raises(ConfigError, match="no phases"):
        ph.DriverConfig.from_mapping({})
    with pytest.raises(ConfigError, match="repeat"):
        ph.DriverConfig.from_mapping({"phase": [phase, phase]})
    with pytest.raises(ConfigError, match="unknown phases key"):
        ph.DriverConfig.from_mapping({"phase": [phase], "colour": "red"})
    config = ph.DriverConfig.from_mapping({"phase": [phase], "setup": [["-c", "1"]], "log-dir": "/tmp"})
    assert config.setup == (("-c", "1"),) and config.log_dir == "/tmp"
