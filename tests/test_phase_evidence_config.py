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
"""A project's phase-evidence configuration: what it may say, what it is refused for, and the command line around it."""
import subprocess
import sys
from pathlib import Path

import pytest
from oracles import ORACLES, body, header

from gpu_epsilon import phase_evidence as pe

CONFIGS = Path(__file__).parent / "configs"
SRC = Path(pe.__file__).resolve().parents[1]
needs_tomllib = pytest.mark.skipif(sys.version_info < (3, 11), reason="reading pyproject.toml needs tomllib")
MINIMAL = {"repository": "o/r", "phase": [{"name": "a", "paths": ["tests"]}]}


@needs_tomllib
@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.phase-evidence.toml")), ids=lambda p: p.name.split(".")[0])
def test_project_configs_load(path):
    config = pe.EvidenceConfig.load(path)
    assert config.root == CONFIGS.resolve()
    assert config.repository.startswith("speediedan/") and config.phases
    assert config.vendors == {"nvidia": "cuda", "amd": "hip"}


def test_defaults():
    config = pe.EvidenceConfig.from_mapping(MINIMAL)
    assert config.phases[0].paths == ("tests",) and config.phases[0].release and not config.phases[0].measured
    assert config.gate == pe.Gate() and config.scopes == {} and config.device_vars == pe.DEFAULT_DEVICE_VARS
    assert config.resolve_scope("a") == ("full", None)


def _with(**change):
    return {**MINIMAL, **change}


def _phase(**change):
    return _with(phase=[{"name": "a", "paths": ["tests"], **change}])


@pytest.mark.parametrize(
    "doc, reason",
    [
        # unknown keys are refused by name, at every level
        (_with(**{"selection-var": ["X"]}), r"unknown key\(s\) selection-var"),
        (_with(gate={"emtpy": "never"}), r"gate: unknown key\(s\) emtpy"),
        (_with(scopes={"smoke": {"select-files": "f"}}), r"scope smoke: unknown key\(s\) select-files"),
        (_phase(**{"selection_env": {}, "argz": []}), r"phase a: unknown key\(s\) argz"),
        # malformed values
        (_with(repository="r"), "repository must be the 'owner/name'"),
        (_with(phase=[]), "at least one"),
        (_with(phase=[{"name": "a", "paths": ["t"]}, {"name": "a", "paths": ["u"]}]), "each phase name must appear"),
        (_with(phase=[{"paths": ["t"]}]), "needs a non-empty name"),
        (_phase(paths=[]), "needs exactly one of 'paths' and 'paths-from'"),
        (_phase(**{"paths-from": {"glob": "t*.py", "contains": "x"}}), "needs exactly one of 'paths' and 'paths-from'"),
        (_with(phase=[{"name": "a", "paths-from": {"glob": "t*.py"}}]), "paths-from must be exactly"),
        (_phase(paths="tests"), "paths must be a list of non-empty strings"),
        (_phase(**{"selection-env": {"X": 1}}), "selection-env must map variable names to strings"),
        (_phase(measured="yes"), "measured must be true or false"),
        (_with(**{"collect-env": ["X"]}), "collect-env must map variable names to strings"),
        (_with(gate={"empty": "sometimes"}), "gate: empty must be one of never, outside-full"),
        (_with(gate={"unreviewed-exclusions": "always"}), "unreviewed-exclusions must be one of"),
        (_with(scopes={"full": {"select-file": "f"}}), "'full' is every phase's unnarrowed scope"),
        (_with(scopes={"s": {}}), "needs exactly one of 'select-file' and 'select-file-env'"),
        (_with(scopes={"s": {"select-file": "f", "select-file-env": "V"}}), "needs exactly one of"),
        (_with(scopes={"s": {"select-file": "f", "phases": ["b"]}}), "names a phase that does not exist"),
        (_with(scopes={"s": {"select-file-env": "V", "phases": ["a"]}}), "takes no 'phases'"),
        (_with(scopes={"s": {"select-file-env": "V"}, "t": {"select-file-env": "W"}}), "at most one scope"),
        (_with(versions={"torch": "dist:torch"}), "torch is recorded by every report"),
        (_with(versions={"x": "pip:x"}), "x must be '<dist|module|vcs|record>:<name>'"),
        (_with(versions={"x": "dist:"}), "x must be"),
        (_with(vendors={"NVIDIA": "cuda"}), "vendors must map lowercase vendor names"),
    ],
)
def test_malformed_configs_are_refused(doc, reason):
    with pytest.raises(pe.ConfigError, match=reason):
        pe.EvidenceConfig.from_mapping(doc)


def test_scopes_resolve(monkeypatch):
    config = pe.EvidenceConfig.from_mapping(
        _with(phase=[{"name": "a", "paths": ["t"]}, {"name": "b", "paths": ["t"]}],
              scopes={"smoke": {"select-file": "s.txt", "phases": ["a"]}, "selected": {"select-file-env": "SEL"}})
    )
    monkeypatch.delenv("SEL", raising=False)
    assert config.resolve_scope("a", "smoke")[0] == "smoke"
    with pytest.raises(ValueError, match="scope 'smoke' is not defined for phase 'b'"):
        config.resolve_scope("b", "smoke")
    with pytest.raises(ValueError, match="scope 'selected' is not defined"):
        config.resolve_scope("a", "selected")  # chosen by its variable, never by --scope
    monkeypatch.setenv("SEL", "/tmp/sel.txt")
    assert config.resolve_scope("b")[0] == "selected"
    assert config.selector(config.phase("b"), config.scopes["selected"]).endswith("; select_file=/tmp/sel.txt")
    with pytest.raises(ValueError, match="SEL is set, which chooses scope 'selected'"):
        config.resolve_scope("a", "smoke")


@needs_tomllib
def test_cli_refuses_a_bad_config_by_name(tmp_path, capsys):
    path = tmp_path / "pyproject.toml"
    path.write_text('[tool.gpu-epsilon.phase-evidence]\nrepository = "o/r"\nphases = []\n')
    assert pe.main(["--config", str(path), "list"]) == 2
    assert "unknown key(s) phases" in capsys.readouterr().err
    path.write_text("[project]\nname = 'x'\n")
    assert pe.main(["--config", str(path), "list"]) == 2
    assert "has no [tool.gpu-epsilon.phase-evidence] table" in capsys.readouterr().err


@needs_tomllib
def test_cli_list_and_run_env(capsys, tmp_path):
    config = CONFIGS / "finetuning-scheduler.phase-evidence.toml"
    assert pe.main(["list", "--config", str(config)]) == 0  # --config after the subcommand works too
    lines = capsys.readouterr().out.splitlines()
    assert [line.split("\t")[0] for line in lines] == [
        "gpu-nonstandalone", "standalone", "examples", "examples-multigpu", "pool-reverse"]
    assert "release=False\tmeasured=False" in lines[-1]
    assert pe.main(["--config", str(config), "run-env", "--phase", "standalone", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert 'export PL_RUN_STANDALONE_TESTS="1"' in out and f"export {pe.PHASE_RECORD_ROOT_ENV}=" in out
    with pytest.raises(SystemExit):
        pe.main(["--config", str(config), "run-env", "--phase", "extended", "--out-dir", str(tmp_path)])


@needs_tomllib
@pytest.mark.parametrize(
    "project, job_class, status",
    [
        ("interpretune", "ci-any-vendor", 0),  # an unreviewed skip passes a gate job
        ("interpretune", "ci-release", 1),  # and fails a release leg
        ("finetuning-scheduler", "ci-any-vendor", 1),  # finetuning-scheduler passes only a clean phase
    ],
)
def test_cli_report_exits_by_the_projects_gate(project, job_class, status, tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(pe, "device_identity", lambda config: {"device_count": 0, "devices": []})
    config = tmp_path / "pyproject.toml"
    config.write_text((CONFIGS / f"{project}.phase-evidence.toml").read_text())
    phase = "examples" if project == "finetuning-scheduler" else "cuda"
    a, b = "t.py::test_a", "t.py::test_b"
    manifest = {"repository": f"speediedan/{project}", "revision": "r", "phase": phase, "scope": "full",
                "selector": "s", "raw_node_ids": [a, b], "selected_node_ids": [a, b], "collection_errors": []}
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / f"{phase}.json").write_text(json.dumps(manifest))
    records = tmp_path / "records" / phase
    records.mkdir(parents=True)
    lines = [{"event": "start", "rank": 0}, {"event": "outcome", "nodeid": a, "when": "call", "outcome": "passed",
             "rank": 0}, {"event": "outcome", "nodeid": b, "when": "setup", "outcome": "skipped", "rank": 0},
             {"event": "finish", "exitstatus": 0, "rank": 0}]
    (records / "records-rank0-pid1.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    argv = ["--config", str(config), "report", "--phase", phase, "--out-dir", str(tmp_path), "--vendor", "amd",
            "--evidence-class", "pipeline", "--job-class", job_class]
    assert pe.main(argv) == status
    doc = json.loads((tmp_path / "reports" / f"{phase}.json").read_text())
    assert doc["classification"] == "unreviewed-exclusions" and doc["backend"] == "hip"


@pytest.mark.parametrize("name", ORACLES)
def test_each_oracle_is_the_file_its_header_names(name):
    import hashlib

    provenance = header(name)
    assert set(provenance) == {"repository", "path", "commit", "sha256"}
    assert len(provenance["commit"]) == 40 and provenance["path"].startswith("tests/gpu_ci/")
    assert hashlib.sha256(body(name)).hexdigest() == provenance["sha256"]


def test_imports_on_the_standard_library_alone(tmp_path):
    """The release job imports gpu_epsilon on the agent's own Python: ``-I -S`` gives it no site-packages, and these
    modules must import nothing outside the standard library and this package (``tomllib`` only once they read a
    configuration)."""
    code = (
        f"import sys; sys.path.insert(0, {str(SRC)!r})\n"
        "import gpu_epsilon.phase_evidence, gpu_epsilon.records\n"
        "top = {n.split('.')[0] for n in sys.modules}\n"
        "print(sorted(top - set(sys.stdlib_module_names) - {'gpu_epsilon', '__main__'}), 'tomllib' in sys.modules)"
    )
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[] False"
