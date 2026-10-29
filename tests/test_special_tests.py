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
"""The special-test runner, end to end, on a small project whose ``conftest.py`` selects tests by mark variable.

Every case runs real pytest processes: the positive controls the shell runners lacked (an empty selection, a
collection error, a crashing test, a failing non-launching rank, a node id full of regex metacharacters) are each a
project here.
"""
import json
import os
import sys
import textwrap
from types import SimpleNamespace

import pytest

from gpu_epsilon import special_tests as st
from gpu_epsilon.records import node_outcome, read_sessions

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="the runner reads pyproject.toml with tomllib")

CONFIG = """
[tool.gpu-epsilon.special-tests]
name = "Demo"
env = { DEMO_GPU = "1" }
unset = ["DEMO_OTHER"]
exec = ["-m", "pytest", "-p", "no:cacheprovider", "-q"]
collect = ["-m", "pytest", "-p", "no:cacheprovider", "-q", "--collect-only"]
experiments-list = "tests/.experiments"

[tool.gpu-epsilon.special-tests.marks.standalone]
env = { DEMO_STANDALONE = "1" }
description = "Collecting and running standalone tests."

[tool.gpu-epsilon.special-tests.marks.profile]
env = { DEMO_PROFILE = "1" }

[tool.gpu-epsilon.special-tests.marks.exp_patch]
refused = "this project defines no experimental-patch tests"

[tool.gpu-epsilon.special-tests.options.resource-debug]
help = "set the demo's resource-debug variable"
env = { DEMO_RESOURCE_DEBUG = "1" }

[tool.gpu-epsilon.special-tests.options.maxfail]
takes-value = true
default = "5"
args = ["--maxfail", "{value}"]

[tool.gpu-epsilon.special-tests.options.no-maxfail]
clears = ["maxfail"]
"""

CONFTEST = """
import os


def pytest_configure(config):
    config.addinivalue_line("markers", "standalone: run alone")
    config.addinivalue_line("markers", "profile: profiling")


def pytest_collection_modifyitems(config, items):
    mark = "standalone" if os.environ.get("DEMO_STANDALONE") else "profile" if os.environ.get("DEMO_PROFILE") else None
    keep = [item for item in items if mark and item.get_closest_marker(mark)]
    config.hook.pytest_deselected(items=[item for item in items if item not in keep])
    items[:] = keep
"""


def project(tmp_path, tests: str, extra_config: str = "", experiments: str | None = None):
    """A project at ``tmp_path`` whose ``tests/test_demo.py`` is ``tests``; returns a runner for it."""
    (tmp_path / "pyproject.toml").write_text(CONFIG + extra_config)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text(CONFTEST)
    (tmp_path / "tests" / "test_demo.py").write_text(textwrap.dedent(tests))
    if experiments is not None:
        (tmp_path / "tests" / ".experiments").write_text(experiments)

    def run(*argv, env=None):
        old_cwd, old_env = os.getcwd(), dict(os.environ)
        os.chdir(tmp_path)
        os.environ.update(env or {})
        try:
            status = st.main(["--log-dir", str(tmp_path / "logs"), *argv])
        finally:
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)
        logs = sorted((tmp_path / "logs").glob("special_tests_*.json"))
        doc = json.loads(logs[-1].read_text()) if logs else None
        return status, doc

    return run


def verdicts(doc):
    return {r["nodeid"].split("::")[-1]: r["verdict"] for r in doc["results"]}


PASSING = """
import pytest

@pytest.mark.standalone
def test_pass():
    pass

@pytest.mark.standalone
def test_skip():
    pytest.skip("not here")

@pytest.mark.standalone
@pytest.mark.xfail(reason="known")
def test_xfail():
    assert False

def test_unmarked():
    raise AssertionError("never selected")
"""


def test_a_clean_run_exits_zero(tmp_path):
    status, doc = project(tmp_path, PASSING)()
    assert status == 0
    assert verdicts(doc) == {"test_pass": "PASSED", "test_skip": "SKIPPED", "test_xfail": "XFAIL"}
    assert doc["counts"] == {"collected": 3, "succeeded": 2, "skipped": 1, "failed": 0, "other": 0}


FAILING = """
import pytest

@pytest.mark.standalone
def test_a_fails():
    assert False

@pytest.mark.standalone
def test_b_passes():
    pass
"""


def test_a_failure_stops_the_run_and_the_rest_count_as_other(tmp_path):
    status, doc = project(tmp_path, FAILING)()
    assert status == 1
    assert verdicts(doc) == {"test_a_fails": "FAILED"}
    assert doc["counts"]["other"] == 1


def test_allow_failures_runs_everything_and_still_fails(tmp_path):
    status, doc = project(tmp_path, FAILING)("--allow-failures")
    assert status == 1
    assert verdicts(doc) == {"test_a_fails": "FAILED", "test_b_passes": "PASSED"}


def test_a_crashing_test_fails(tmp_path):
    status, doc = project(tmp_path, """
        import os
        import pytest

        @pytest.mark.standalone
        def test_crash():
            os._exit(3)
    """)()
    assert status == 1
    assert verdicts(doc) == {"test_crash": "FAILED"}
    assert doc["results"][0]["detail"] == "exit code 3"


RANKED = """
import os
import subprocess
import sys
import pytest

@pytest.mark.standalone
def test_ranked(request):
    # a launcher's other rank, re-running this test in its own pytest session
    if os.environ.get("LOCAL_RANK") == "1":
        if os.environ.get("DEMO_RANK_MODE") == "crash":
            os._exit(0)
        assert os.environ.get("DEMO_RANK_MODE") != "fail"
        return
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-p", "gpu_epsilon.records"]
    cmd.append(request.node.nodeid)
    subprocess.run(cmd, env={**os.environ, "LOCAL_RANK": "1"}, cwd=request.config.rootpath)
"""


@pytest.mark.parametrize(
    "mode, verdict, detail",
    [("pass", "PASSED", ""), ("fail", "FAILED", ""), ("crash", "FAILED", "2 pytest sessions started, 1 finished")],
)
def test_another_ranks_failure_or_crash_fails_the_node(tmp_path, mode, verdict, detail):
    # the launching rank's own exit is 0 each time: only the other rank's records can fail the node
    status, doc = project(tmp_path, RANKED)(env={"DEMO_RANK_MODE": mode})
    assert (status, doc["results"][0]["verdict"], doc["results"][0]["detail"]) == (int(mode != "pass"), verdict, detail)


def test_a_collection_error_fails_the_run(tmp_path):
    run = project(tmp_path, PASSING)
    (tmp_path / "tests" / "test_broken.py").write_text("import a_module_that_does_not_exist\n")
    status, doc = run()
    assert status == 1 and doc["note"] == "collection failed" and doc["results"] == []


NOTHING = """
def test_unmarked():
    pass
"""


def test_an_empty_selection_fails_unless_allowed(tmp_path):
    run = project(tmp_path, NOTHING)
    status, doc = run()
    assert status == 1 and doc["note"] == "empty selection"
    status, doc = run("--allow-empty")
    assert status == 0 and doc["note"] == "empty selection (allowed)"


def test_a_project_may_allow_an_empty_selection(tmp_path):
    run = project(tmp_path, NOTHING, extra_config="")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace('name = "Demo"', 'name = "Demo"\nallow-empty = true'))
    assert run()[0] == 0


def test_a_node_id_full_of_regex_metacharacters(tmp_path):
    status, doc = project(tmp_path, r"""
        import pytest

        @pytest.mark.standalone
        @pytest.mark.parametrize("x", [1], ids=["a[b]*(c)+.?{2}|$^\\d"])
        def test_meta(x):
            pass
    """)()
    assert status == 0
    # pytest escapes the backslash in the id it reports; the runner passes the id back verbatim
    assert doc["results"][0]["nodeid"].endswith(r"test_meta[a[b]*(c)+.?{2}|$^\\d]")
    assert doc["results"][0]["verdict"] == "PASSED"


def test_select_file_keeps_its_order_and_refuses_a_missing_test(tmp_path):
    run = project(tmp_path, PASSING)
    listing = tmp_path / "smoke"
    listing.write_text("# smoke set\ntests/test_demo.py::test_xfail\n\ntests/test_demo.py::test_pass\n")
    status, doc = run("--select_file", str(listing))
    assert status == 0 and list(verdicts(doc)) == ["test_xfail", "test_pass"]
    listing.write_text("tests/test_demo.py::test_pass\ntests/test_demo.py::test_gone\n")
    status, doc = run("--select_file", str(listing))
    assert status == 1 and doc["note"] == "listed tests not collected"
    assert run("--select_file", str(tmp_path / "absent"))[0] == 1


ENVIRONMENT = """
import os
import pytest

@pytest.mark.standalone
def test_environment():
    seen = {k: os.environ.get(k) for k in (
        "DEMO_GPU", "DEMO_STANDALONE", "DEMO_PROFILE", "DEMO_OTHER", "DEMO_EXP_A", "DEMO_EXP_B", "DEMO_RESOURCE_DEBUG"
    )}
    print("SEEN", seen)
    assert seen == eval(os.environ["DEMO_EXPECT"])

@pytest.mark.profile
def test_profile_only():
    pass
"""


def test_an_inherited_selection_variable_does_not_change_the_selection(tmp_path):
    expect = {"DEMO_GPU": "1", "DEMO_STANDALONE": "1", "DEMO_PROFILE": None, "DEMO_OTHER": None,
              "DEMO_EXP_A": None, "DEMO_EXP_B": None, "DEMO_RESOURCE_DEBUG": None}
    status, doc = project(tmp_path, ENVIRONMENT)(
        env={"DEMO_PROFILE": "1", "DEMO_OTHER": "1", "DEMO_EXPECT": repr(expect)}
    )
    assert status == 0 and list(verdicts(doc)) == ["test_environment"]


def test_experiment_mask_and_options_set_their_variables(tmp_path):
    expect = {"DEMO_GPU": "1", "DEMO_STANDALONE": "1", "DEMO_PROFILE": None, "DEMO_OTHER": None,
              "DEMO_EXP_A": None, "DEMO_EXP_B": "1", "DEMO_RESOURCE_DEBUG": "1"}
    run = project(tmp_path, ENVIRONMENT, experiments="DEMO_EXP_A DEMO_EXP_B\n")
    status, _ = run("--experiment_patch_mask", "0 1", "--resource-debug",
                    env={"DEMO_EXP_A": "1", "DEMO_EXPECT": repr(expect)})
    assert status == 0


def test_a_bad_experiment_mask_is_refused_before_anything_runs(tmp_path):
    run = project(tmp_path, PASSING, experiments="DEMO_EXP_A DEMO_EXP_B\n")
    assert run("--check-args", "--experiment_patch_mask", "1") == (1, None)
    assert run("--check-args", "--experiment_patch_mask", "1 0") == (0, None)


def test_marks_that_are_refused_or_unknown_exit_one(tmp_path):
    run = project(tmp_path, PASSING)
    assert run("--mark_type", "exp_patch") == (1, None)
    assert run("--mark_type", "nonesuch") == (1, None)
    assert run("--check-args", "--mark_type", "profile") == (0, None)


def test_options_and_exec_overrides_build_the_command(tmp_path):
    run = project(tmp_path, PASSING, extra_config="""
[[tool.gpu-epsilon.special-tests.exec-overrides]]
env = "DEMO_COVERAGE_RC"
exec = ["-m", "pytest", "--rcfile={value}"]
""")
    del run
    os.chdir(tmp_path)
    config = st.RunnerConfig.load(tmp_path / "pyproject.toml")
    parser = st._parser(config)

    def exec_args(*argv, env=None):
        old = dict(os.environ)
        os.environ.update(env or {})
        try:
            return st.Run(config, parser.parse_args(list(argv))).exec_args()
        finally:
            os.environ.clear()
            os.environ.update(old)

    assert exec_args() == ["-m", "pytest", "-p", "no:cacheprovider", "-q", "--maxfail", "5"]
    assert exec_args("--maxfail", "2", "--filter_pattern", "a and b")[-4:] == ["--maxfail", "2", "-k", "a and b"]
    assert "--maxfail" not in exec_args("--no-maxfail")
    assert exec_args(env={"DEMO_COVERAGE_RC": "cov.rc"})[:3] == ["-m", "pytest", "--rcfile=cov.rc"]


def test_the_event_command_reports_into_the_log(tmp_path):
    run = project(tmp_path, PASSING)
    status, _ = run(env={st.EVENT_CMD_ENV: "echo snapshot"})
    text = next((tmp_path / "logs").glob("special_tests_standalone_*.log")).read_text()
    assert status == 0
    assert "snapshot special-tests:collect:start" in text
    assert "snapshot special-tests:test:end:tests/test_demo.py::test_pass" in text


def test_publish_and_copy_the_raw_log(tmp_path, capsys):
    run = project(tmp_path, PASSING, extra_config="")
    pyproject = tmp_path / "pyproject.toml"
    text = pyproject.read_text().replace(
        'name = "Demo"', f'name = "Demo"\npublish = "azure"\nraw-log-copy-dir = "{(tmp_path / "copied").as_posix()}"'
    )
    pyproject.write_text(text)
    assert run()[0] == 0
    copied = list((tmp_path / "copied").glob("special_tests_raw_*.log"))
    assert len(copied) == 1 and "1 passed" in copied[0].read_text()
    assert f"##vso[task.uploadfile]{copied[0].resolve()}" in capsys.readouterr().out


def test_check_args_writes_nothing(tmp_path):
    run = project(tmp_path, PASSING)
    assert run("--check-args") == (0, None)
    assert not (tmp_path / "logs").exists()


# Units: the verdict rule, the records plugin and the configuration.


def _session(rank, *outcomes, finished=True, collect_error=None):
    records = [{"event": "start", "rank": rank}]
    if collect_error:
        records.append({"event": "collect_error", "nodeid": collect_error, "rank": rank})
    records += [{"event": "outcome", "nodeid": "t", "when": "call", "outcome": o, "rank": rank} for o in outcomes]
    return records + ([{"event": "finish", "exitstatus": 0, "rank": rank}] if finished else [])


@pytest.mark.parametrize(
    "sessions, rc, verdict",
    [
        ([_session(0, "passed")], 0, "PASSED"),
        ([_session(0, "passed"), _session(1, "failed")], 0, "FAILED"),
        ([_session(0, "passed"), _session(1, "passed", finished=False)], 0, "FAILED"),
        ([_session(0, "passed")], 1, "FAILED"),
        ([_session(0, "skipped"), _session(1, "passed")], 0, "SKIPPED"),
        ([_session(0, "xfailed")], 0, "XFAIL"),
        ([_session(0, "xpassed")], 0, "XPASS"),
        ([_session(0)], 0, "UNKNOWN"),
        ([], 0, "UNKNOWN"),
        ([], -9, "FAILED"),
        ([_session(0, "passed", collect_error="tests/x.py")], 0, "FAILED"),
    ],
)
def test_node_verdict(sessions, rc, verdict):
    assert st.node_verdict(sessions, "t", rc).verdict == verdict


@pytest.mark.parametrize(
    "report, outcome",
    [
        (dict(outcome="rerun", failed=False, skipped=False, when="call"), None),
        (dict(outcome="failed", failed=True, skipped=False, when="call"), "failed"),
        (dict(outcome="failed", failed=True, skipped=False, when="setup"), "errors"),
        (dict(outcome="skipped", failed=False, skipped=True, when="setup"), "skipped"),
        (dict(outcome="skipped", failed=False, skipped=True, when="call", wasxfail="r"), "xfailed"),
        (dict(outcome="passed", failed=False, skipped=False, when="call", wasxfail="r"), "xpassed"),
        (dict(outcome="passed", failed=False, skipped=False, when="teardown"), None),
    ],
)
def test_node_outcome(report, outcome):
    assert node_outcome(SimpleNamespace(**report)) == outcome


def test_read_sessions_skips_a_line_a_crash_truncated(tmp_path):
    (tmp_path / "records-rank0-pid1.jsonl").write_text('{"event": "start", "rank": 0}\n{"event": "outc')
    assert read_sessions(tmp_path) == [[{"event": "start", "rank": 0}]]


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"nonesuch": 1}, "unknown special-tests key"),
        ({"default-mark": "profile"}, "default-mark 'profile' is not a mark"),
        ({"publish": "s3"}, "publish must be"),
        ({"marks": {"standalone": {"envs": {}}}}, "unknown key"),
        ({"exec": []}, "exec and collect must be non-empty"),
        ({"options": {"a": {"clears": ["b"]}}}, "clears an option that does not exist"),
        ({"exec-overrides": [{"env": "X"}]}, "exactly 'env' and 'exec'"),
    ],
)
def test_malformed_configs_are_refused(change, reason):
    base = {"marks": {"standalone": {}}, "exec": ["-m", "pytest"]}
    with pytest.raises(st.ConfigError, match=reason):
        st.RunnerConfig.from_mapping({**base, **change})


def test_a_missing_table_is_a_config_error(tmp_path, capsys):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    assert st.main(["--config", str(tmp_path / "pyproject.toml")]) == 2
    assert "has no [tool.gpu-epsilon.special-tests] table" in capsys.readouterr().err
