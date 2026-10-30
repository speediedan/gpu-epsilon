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
"""The lease seam: each branch of :func:`gpu_epsilon.lease.lease_command`, and a real re-exec under a stand-in tool."""
import os
import stat
import subprocess
import sys
import textwrap

import pytest

from gpu_epsilon import lease

posix_only = pytest.mark.skipif(os.name == "nt", reason="the lease tool and its re-exec are POSIX only")


def tool(tmp_path):
    """A stand-in lease tool: records its arguments, exports GPU_LEASE_HELD as the real one does, runs the rest."""
    path = tmp_path / "lease_tool"
    path.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import os, sys
        args = sys.argv[1:]
        open({str(tmp_path / "tool_args")!r}, "w").write(" ".join(args))
        rest = args[args.index("--") + 1:]
        os.environ["GPU_LEASE_HELD"] = "gpu:" + str(os.getpid())
        os.execvp(rest[0], rest)
        """))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_no_lease_command_runs_directly():
    assert lease.lease_command(["x"], {}) is None
    assert lease.lease_command(["x"], {"GPU_LEASE_CMD": "/no/such/tool"}) is None


@posix_only
def test_a_configured_tool_wraps_the_command(tmp_path):
    env = {"GPU_LEASE_CMD": tool(tmp_path), "GPU_LEASE_ARGS": "--cpu-heavy --wait 5", "GPU_LEASE_PROJECT": "demo"}
    cmd = lease.lease_command(["python", "-m", "x"], env)
    assert cmd == [env["GPU_LEASE_CMD"], "--cpu-heavy", "--wait", "5", "--project", "demo", "--", "python", "-m", "x"]


@posix_only
def test_a_live_holder_means_run_directly(tmp_path):
    env = {"GPU_LEASE_CMD": tool(tmp_path), "GPU_LEASE_HELD": f"gpu:{os.getpid()}"}
    assert lease.lease_command(["x"], env) is None
    assert lease.lease_command(["x"], {**env, "GPU_LEASE_HELD": "gpu"}) is None  # untagged: trusted


@posix_only
def test_a_stale_holder_is_ignored(tmp_path, capsys):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    env = {"GPU_LEASE_CMD": tool(tmp_path), "GPU_LEASE_HELD": f"gpu:{dead.pid}", "GPU_LEASE_PROJECT": "demo"}
    assert lease.lease_command(["x"], env) is not None
    assert "GPU_LEASE_HELD" not in env
    assert "stale" in capsys.readouterr().err


@posix_only
def test_a_reexecuted_run_never_loops(tmp_path):
    env = {"GPU_LEASE_CMD": tool(tmp_path), lease.REEXEC_ENV: "1"}
    assert lease.lease_command(["x"], env) is None


@posix_only
@pytest.mark.skipif(sys.version_info < (3, 11), reason="the runner reads pyproject.toml with tomllib")
def test_the_runner_reexecutes_under_the_lease(tmp_path):
    (tmp_path / "pyproject.toml").write_text(textwrap.dedent("""
        [tool.gpu-epsilon.special-tests]
        exec = ["-m", "pytest", "-p", "no:cacheprovider", "-q"]
        collect = ["-m", "pytest", "-p", "no:cacheprovider", "-q", "--collect-only"]
        collect-dirs = ["tests"]
        allow-empty = true

        [tool.gpu-epsilon.special-tests.marks.standalone]
        env = { DEMO_STANDALONE = "1" }
        """))
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_none.py").write_text("")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GPU_LEASE")}
    env.update({"GPU_LEASE_CMD": tool(tmp_path), "GPU_LEASE_PROJECT": "demo"})
    cmd = [sys.executable, "-m", "gpu_epsilon.special_tests", "--lease", "--log-dir", str(tmp_path / "logs")]
    proc = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "acquiring the GPU lease" in proc.stderr
    args = (tmp_path / "tool_args").read_text()
    assert args.startswith("--project demo -- ") and "gpu_epsilon.special_tests --lease" in args
