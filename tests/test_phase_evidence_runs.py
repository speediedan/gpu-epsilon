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
"""Phase evidence end to end, in real pytest processes on small git repositories shaped like each project.

Each project's repository holds its oracle plugin at the path its oracle loads it from, so the oracle collects
unmodified: the manifests the new code writes must be byte-identical to the oracle's for every phase and scope, except
where an inherited ``PYTEST_ADDOPTS`` changed finetuning-scheduler's collection (an intended difference). Then
``run-env`` must leave a record for every node, under plain pytest and under the special-test runner (which replaces
its own record directory per node), and must not record a nested session rooted elsewhere.
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from oracles import load, path

from gpu_epsilon import phase_evidence as pe
from gpu_epsilon.records import PHASE_RECORD_ROOT_ENV, read_sessions

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="phase evidence reads pyproject.toml with tomllib")

CONFIGS = Path(__file__).parent / "configs"
ORACLES = Path(path("fts_phase_plugin")).parent

FTS_FILES = {
    "conftest.py": """
        import os
        import pytest

        def pytest_configure(config):
            config.addinivalue_line("markers", "standalone: run alone")

        @pytest.fixture
        def dist_pool_fixture():
            return None

        def pytest_collection_modifyitems(config, items):
            if os.getenv("PL_RUN_CUDA_TESTS") != "1":  # always set for a GPU phase: a collection without it differs
                items[:] = [i for i in items if "gated" not in i.name]
            if os.getenv("PL_RUN_STANDALONE_TESTS") == "1":
                items[:] = [i for i in items if i.get_closest_marker("standalone")]
            if os.getenv("FTS_CUDA_TESTS_ONLY") == "1":
                items[:] = [i for i in items if "cuda" in i.name]
            if os.getenv("FTS_EXPERIMENTAL_PATCH_TESTS") == "1":
                items[:] = []
            if os.getenv("FTS_REVERSE_ORDER") == "1":
                items.reverse()
        """,
    "src/finetuning_scheduler/test_core.py": """
        def test_core():
            pass

        def test_core_cuda():
            pass
        """,
    "tests/__init__.py": "",
    "tests/gpu_ci/__init__.py": "",
    "tests/test_a.py": """
        import pytest

        def test_plain():
            pass

        def test_cuda_plain():
            pass

        def test_cuda_gated():
            pass

        @pytest.mark.standalone
        def test_alone():
            pass

        @pytest.mark.standalone
        def test_alone_cuda():
            pass
        """,
    "tests/test_pool.py": """
        def test_pool(dist_pool_fixture):
            pass

        def test_pool_cuda(dist_pool_fixture):
            pass
        """,
    "tests/test_pool_b.py": """
        def test_pool_b(dist_pool_fixture):
            pass
        """,
    "tests/.standalone_smoke": """
        # the smoke selection
        tests/test_a.py::test_alone_cuda
        tests/test_a.py::test_alone
        """,
    "src/fts_examples/test_examples.py": """
        import pytest

        @pytest.mark.standalone
        def test_model_parallel_examples():
            pass

        @pytest.mark.standalone
        def test_other_standalone_example():
            pass

        def test_example():
            pass
        """,
}

IT_FILES = {
    "conftest.py": """
        import os
        from pathlib import Path

        MARKS = {"IT_RUN_CUDA_TESTS": "cuda", "IT_RUN_STANDALONE_TESTS": "standalone",
                 "IT_RUN_PROFILING_TESTS": "profile_ci", "IT_RUN_OPTIONAL_TESTS": "optional"}

        def pytest_configure(config):
            for mark in MARKS.values():
                config.addinivalue_line("markers", f"{mark}: a phase")

        def pytest_collection_modifyitems(config, items):
            for var, mark in MARKS.items():
                if os.getenv(var) == "1":
                    items[:] = [i for i in items if i.get_closest_marker(mark)]
            if os.getenv("IT_GPU_MEM_CLASS"):
                items[:] = [i for i in items if os.environ["IT_GPU_MEM_CLASS"] in i.name]
            if os.getenv("IT_GPU_SELECTION_FILE"):
                wanted = set(Path(os.environ["IT_GPU_SELECTION_FILE"]).read_text().split())
                items[:] = [i for i in items if i.nodeid in wanted]
        """,
    "tests/__init__.py": "",
    "tests/gpu_ci/__init__.py": "",
    "tests/test_core.py": """
        import pytest

        def test_plain():
            pass

        @pytest.mark.cuda
        def test_cuda_small():
            pass

        @pytest.mark.cuda
        def test_cuda_large():
            pass

        @pytest.mark.standalone
        def test_alone_small():
            pass

        @pytest.mark.profile_ci
        def test_profile():
            pass
        """,
    "src/it_examples/tests/test_examples.py": """
        import pytest

        @pytest.mark.cuda
        def test_example_cuda_small():
            pass

        def test_example():
            pass
        """,
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True,
    )


def make_repo(root: Path, project: str) -> Path:
    """A committed repository shaped like ``project``, with its oracle plugin where its oracle loads it from."""
    files = FTS_FILES if project == "finetuning-scheduler" else IT_FILES
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(textwrap.dedent(text).lstrip("\n"))
    plugin = "fts_phase_plugin" if project == "finetuning-scheduler" else "it_phase_plugin"
    shutil.copy(path(plugin), root / "tests/gpu_ci/phase_plugin.py")
    shutil.copy(CONFIGS / f"{project}.phase-evidence.toml", root / "pyproject.toml")
    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--no-verify", "-m", "synthetic")
    return root


@pytest.fixture
def repo(tmp_path, request):
    return make_repo((tmp_path / "repo").resolve(), request.param)


def oracle_bytes(project: str, repo: Path, out: Path, phase: str, scope: str | None, monkeypatch) -> bytes:
    if project == "finetuning-scheduler":
        oracle = load("fts_phase_manifest")
        monkeypatch.setattr(oracle, "ROOT", repo)
        manifest = oracle.build_manifest(phase, scope or "full", root=repo)
    else:
        oracle = load("it_phase_report")
        monkeypatch.setattr(oracle, "ROOT", repo)
        manifest = oracle.build_manifest(phase, root=repo)
    oracle._write_json(out / f"{phase}.json", manifest)
    return (out / f"{phase}.json").read_bytes()


def new_bytes(repo: Path, out: Path, phase: str, scope: str | None) -> bytes:
    argv = ["--config", str(repo / "pyproject.toml"), "collect", "--phase", phase, "--out-dir", str(out)]
    status = pe.main(argv + (["--scope", scope] if scope else []))
    manifest = out / "manifests" / f"{phase}.json"
    assert status == bool(json.loads(manifest.read_text())["collection_errors"])
    return manifest.read_bytes()


def _clean_env(monkeypatch, project):
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    for var in pe.EvidenceConfig.load(CONFIGS / f"{project}.phase-evidence.toml").selection_vars:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("IT_GPU_SELECTION_FILE", raising=False)


FTS_VARIANTS = [
    *((p, None, {}) for p in ("gpu-nonstandalone", "standalone", "examples", "examples-multigpu", "pool-reverse")),
    ("standalone", "smoke", {}),
    # an inherited selection variable is stripped before collecting
    ("gpu-nonstandalone", None, {"PL_RUN_STANDALONE_TESTS": "1", "FTS_EXPERIMENTAL_PATCH_TESTS": "1"}),
]
IT_VARIANTS = [
    *((phase, None, {}) for phase in ("cuda", "standalone", "profile_ci")),
    # the memory class is stripped, so a two-stage phase's manifest covers both stages
    ("cuda", None, {"IT_GPU_MEM_CLASS": "small"}),
    # interpretune already stripped an inherited PYTEST_ADDOPTS
    ("cuda", None, {"PYTEST_ADDOPTS": "-k large"}),
]


@pytest.mark.parametrize("repo", ["finetuning-scheduler"], indirect=True)
@pytest.mark.parametrize("phase, scope, env", FTS_VARIANTS)
def test_fts_manifests_are_byte_identical(repo, tmp_path, monkeypatch, phase, scope, env):
    _clean_env(monkeypatch, "finetuning-scheduler")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    old = oracle_bytes("finetuning-scheduler", repo, tmp_path / "old", phase, scope, monkeypatch)
    assert old == new_bytes(repo, tmp_path / "new", phase, scope)
    assert json.loads(old)["selected_node_ids"], "the variant must select something to compare"


@pytest.mark.parametrize("repo", ["interpretune"], indirect=True)
@pytest.mark.parametrize("phase, scope, env", IT_VARIANTS)
def test_it_manifests_are_byte_identical(repo, tmp_path, monkeypatch, phase, scope, env):
    _clean_env(monkeypatch, "interpretune")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    old = oracle_bytes("interpretune", repo, tmp_path / "old", phase, scope, monkeypatch)
    assert old == new_bytes(repo, tmp_path / "new", phase, scope)
    assert json.loads(old)["selected_node_ids"], "the variant must select something to compare"


@pytest.mark.parametrize("repo", ["interpretune"], indirect=True)
def test_it_selection_file_scope_is_byte_identical(repo, tmp_path, monkeypatch):
    _clean_env(monkeypatch, "interpretune")
    selection = tmp_path / "selection.txt"
    selection.write_text("tests/test_core.py::test_cuda_small\n")
    monkeypatch.setenv("IT_GPU_SELECTION_FILE", str(selection))
    old = oracle_bytes("interpretune", repo, tmp_path / "old", "cuda", None, monkeypatch)
    assert old == new_bytes(repo, tmp_path / "new", "cuda", None)
    doc = json.loads(old)
    assert doc["scope"] == "selected" and doc["selected_node_ids"] == ["tests/test_core.py::test_cuda_small"]
    assert doc["selector"].endswith(f"; select_file={selection}")
    # the variable chooses the scope, so --scope beside it is refused by name
    argv = ["--config", str(repo / "pyproject.toml"), "collect", "--phase", "cuda", "--out-dir", str(tmp_path)]
    assert pe.main([*argv, "--scope", "full"]) == 2


@pytest.mark.parametrize("project", ["finetuning-scheduler", "interpretune"])
def test_a_dirty_tree_and_a_collection_error_are_byte_identical(project, tmp_path, monkeypatch):
    _clean_env(monkeypatch, project)
    repo = make_repo((tmp_path / "repo").resolve(), project)
    (repo / "tests/test_broken.py").write_text("import not_a_module_anywhere\n")
    if project == "finetuning-scheduler":
        # a smoke node that is not collected is a collection error too
        (repo / "tests/.standalone_smoke").write_text("tests/test_a.py::test_alone\ntests/test_a.py::test_gone\n")
    phase, scope = ("standalone", "smoke") if project == "finetuning-scheduler" else ("cuda", None)
    old = oracle_bytes(project, repo, tmp_path / "old", phase, scope, monkeypatch)
    assert old == new_bytes(repo, tmp_path / "new", phase, scope)
    doc = json.loads(old)
    assert "tests/test_broken.py" in doc["tree_dirty_paths"] and "tests/test_broken.py" in doc["collection_errors"]
    assert doc["collect_exit_codes"]["raw"] != 0


@pytest.mark.parametrize("repo", ["finetuning-scheduler"], indirect=True)
def test_drift_an_inherited_pytest_addopts_no_longer_changes_the_manifest(repo, tmp_path, monkeypatch):
    _clean_env(monkeypatch, "finetuning-scheduler")
    clean = new_bytes(repo, tmp_path / "clean", "gpu-nonstandalone", None)
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k plain")
    old = oracle_bytes("finetuning-scheduler", repo, tmp_path / "old", "gpu-nonstandalone", None, monkeypatch)
    new = new_bytes(repo, tmp_path / "new", "gpu-nonstandalone", None)
    assert new == clean and old != new
    assert json.loads(old)["selected_node_ids"] == ["tests/test_a.py::test_cuda_plain"]


@pytest.mark.parametrize("repo", ["finetuning-scheduler"], indirect=True)
def test_a_scope_outside_its_phases_is_refused(repo, tmp_path, monkeypatch, capsys):
    _clean_env(monkeypatch, "finetuning-scheduler")
    with pytest.raises(ValueError, match="scope 'smoke' is not defined for phase 'examples'"):
        load("fts_phase_manifest").build_manifest("examples", "smoke", root=repo)
    argv = ["--config", str(repo / "pyproject.toml"), "collect", "--phase", "examples", "--scope", "smoke"]
    assert pe.main([*argv, "--out-dir", str(tmp_path)]) == 2
    assert "scope 'smoke' is not defined for phase 'examples'" in capsys.readouterr().err


# recording a phase


PROJECT_CONFIG = """
[tool.gpu-epsilon.phase-evidence]
repository = "o/demo"
selection-vars = ["DEMO_STANDALONE"]
run-env-exports-selection = true

[[tool.gpu-epsilon.phase-evidence.phase]]
name = "standalone"
paths = ["tests"]
selection-env = { DEMO_STANDALONE = "1" }

[tool.gpu-epsilon.special-tests]
exec = ["-m", "pytest", "-p", "no:cacheprovider", "-q"]
collect = ["-m", "pytest", "-p", "no:cacheprovider", "-q", "--collect-only"]

[tool.gpu-epsilon.special-tests.marks.standalone]
env = { DEMO_STANDALONE = "1" }
"""
PROJECT_CONFTEST = """
import os


def pytest_configure(config):
    config.addinivalue_line("markers", "standalone: run alone")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DEMO_STANDALONE") == "1":
        items[:] = [i for i in items if i.get_closest_marker("standalone")]
"""
PROJECT_TESTS = """
import os
import subprocess
import sys

import pytest


@pytest.mark.standalone
def test_pass():
    pass


@pytest.mark.standalone
def test_skip():
    pytest.skip("not here")


@pytest.mark.standalone
@pytest.mark.xfail(reason="known", strict=True)
def test_xfail():
    assert False


@pytest.mark.standalone
@pytest.mark.parametrize("x", ["a[b]", "c.d*"])
def test_meta(x):
    pass


@pytest.mark.standalone
def test_nested():
    # runs pytest on a tree of its own, inheriting the phase's environment, as a test of a test runner does
    inner = os.environ["DEMO_INNER"]
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", inner], cwd=inner)
    assert proc.returncode == 0


def test_not_standalone():
    pass
"""
EXPECTED = {
    "tests/test_demo.py::test_pass": "passed",
    "tests/test_demo.py::test_skip": "skipped",
    "tests/test_demo.py::test_xfail": "xfailed",
    "tests/test_demo.py::test_meta[a[b]]": "passed",
    "tests/test_demo.py::test_meta[c.d*]": "passed",
    "tests/test_demo.py::test_nested": "passed",
}


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project with phase evidence and special tests configured, collected; returns ``(root, out_dir, inner)``."""
    root, out, inner = tmp_path / "project", tmp_path / "out", tmp_path / "inner"
    (root / "tests").mkdir(parents=True)
    (root / "pyproject.toml").write_text(PROJECT_CONFIG)
    (root / "tests" / "conftest.py").write_text(PROJECT_CONFTEST)
    (root / "tests" / "test_demo.py").write_text(PROJECT_TESTS)
    inner.mkdir()
    (inner / "test_inner.py").write_text("def test_inner():\n    pass\n")
    (root / ".gitignore").write_text("__pycache__/\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--no-verify", "-m", "demo")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("DEMO_STANDALONE", raising=False)
    argv = ["--config", str(root / "pyproject.toml"), "collect", "--phase", "standalone", "--out-dir", str(out)]
    assert pe.main(argv) == 0
    return root, out, inner


def exports(root: Path, out: Path, capsys) -> dict:
    argv = ["--config", str(root / "pyproject.toml"), "run-env", "--phase", "standalone", "--out-dir", str(out)]
    assert pe.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    return {k: json.loads(v) for k, _, v in (line.removeprefix("export ").partition("=") for line in lines)}


def report(root: Path, out: Path) -> dict:
    argv = ["--config", str(root / "pyproject.toml"), "report", "--phase", "standalone", "--out-dir", str(out)]
    pe.main([*argv, "--vendor", "nvidia", "--evidence-class", "parity-pre-agent"])
    return json.loads((out / "reports" / "standalone.json").read_text())


@pytest.mark.parametrize("runner", ["pytest", "special-tests"])
def test_run_env_records_every_node(project, capsys, runner):
    root, out, inner = project
    env = {**os.environ, **exports(root, out, capsys), "DEMO_INNER": str(inner)}
    if runner == "pytest":
        cmd = [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"]
    else:
        # the runner points GPU_EPSILON_RECORD_DIR at a directory of its own for every node; the phase's own records
        # must survive that
        cmd = [sys.executable, "-m", "gpu_epsilon.special_tests", "--mark_type=standalone"]
        cmd += ["--log-dir", str(out / "logs")]
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = report(root, out)
    assert {n: v["outcome"] for n, v in doc["nodes"].items()} == EXPECTED
    assert doc["unmanifested_nodes"] == [] and doc["classification"] == "unreviewed-exclusions"
    sessions = read_sessions(out / "records" / "standalone")
    outcomes = {r["nodeid"] for s in sessions for r in s if r["event"] == "outcome"}
    assert outcomes == set(EXPECTED)
    # one session for plain pytest; the runner's collection plus one per node
    assert len(sessions) == (1 if runner == "pytest" else 1 + len(EXPECTED))


PYTEST = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]


def test_a_nested_session_rooted_elsewhere_is_not_recorded(project, capsys):
    root, out, inner = project
    env = {**os.environ, **exports(root, out, capsys), "DEMO_INNER": str(inner)}
    subprocess.run(PYTEST + ["tests"], cwd=root, env=env, check=True)
    assert report(root, out)["unmanifested_nodes"] == []
    # positive control: without the root, the same nested session is recorded and breaks the report
    shutil.rmtree(out / "records")
    env.pop(PHASE_RECORD_ROOT_ENV)
    subprocess.run(PYTEST + ["tests"], cwd=root, env=env, check=True)
    doc = report(root, out)
    assert doc["unmanifested_nodes"] == ["test_inner.py::test_inner"] and doc["classification"] == "failing"


FLAKY = """
from pathlib import Path


def test_flaky():
    marker = Path(__file__).with_name("attempted")
    if not marker.exists():
        marker.write_text("1")
        assert False, "first attempt"


def test_broken():
    assert False
"""


def test_drift_a_rerun_attempt_is_not_recorded(tmp_path):
    """Under ``--reruns``, finetuning-scheduler's plugin records the attempt that failed as ``passed``."""
    pytest.importorskip("pytest_rerunfailures")
    runs = {}
    for name, plugin, record_env, pythonpath in (
        ("new", "gpu_epsilon.records", pe.PHASE_RECORD_DIR_ENV, os.environ.get("PYTHONPATH", "")),
        ("fts", "fts_phase_plugin", "FTS_PHASE_RECORD_DIR", str(ORACLES)),
    ):
        tree = tmp_path / name
        tree.mkdir()
        (tree / "test_flaky.py").write_text(FLAKY)
        env = {**os.environ, record_env: str(tree / "records"), "PYTHONPATH": pythonpath}
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", plugin, "--reruns", "1"]
        subprocess.run([*cmd, "test_flaky.py"], cwd=tree, env=env, capture_output=True)
        runs[name] = [(r["nodeid"].split("::")[1], r["outcome"]) for s in read_sessions(tree / "records") for r in s
                      if r["event"] == "outcome"]
    assert runs["new"] == [("test_flaky", "passed"), ("test_broken", "failed")]
    assert runs["fts"] == [("test_flaky", "passed"), ("test_flaky", "passed"), ("test_broken", "passed"),
                           ("test_broken", "failed")]


@pytest.mark.parametrize("repo", ["finetuning-scheduler"], indirect=True)
def test_a_config_that_drifts_from_the_project_changes_the_bytes(repo, tmp_path, monkeypatch):
    """Positive control for the byte comparison: drop finetuning-scheduler's always-set ``PL_RUN_CUDA_TESTS`` and the
    manifest must differ (its selector and, through the synthetic conftest, its node ids)."""
    _clean_env(monkeypatch, "finetuning-scheduler")
    old = oracle_bytes("finetuning-scheduler", repo, tmp_path / "old", "gpu-nonstandalone", None, monkeypatch)
    config = repo / "pyproject.toml"
    config.write_text(config.read_text().replace('collect-env = { PL_RUN_CUDA_TESTS = "1" }\n', ""))
    _git(repo, "commit", "-q", "--no-verify", "-am", "drift")
    new = json.loads(new_bytes(repo, tmp_path / "new", "gpu-nonstandalone", None))
    old_doc = json.loads(old)
    assert new["selector"] != old_doc["selector"]
    assert "tests/test_a.py::test_cuda_gated" in old_doc["raw_node_ids"]
    assert "tests/test_a.py::test_cuda_gated" not in new["raw_node_ids"]
