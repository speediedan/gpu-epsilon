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
"""Take the host's GPU lease for a whole run by re-executing it under the lease command, once, before any GPU work.

The Python form of the ``gpu_lease_reexec`` seam each project's ``scripts/gpu_lease_wrap.sh`` carries, so a runner
needs no shell entry script to take the lease. It is a no-op unless ``GPU_LEASE_CMD`` names an executable, so a host
without a lease tool (a contributor's machine, hosted CI) runs exactly as before:

- ``GPU_LEASE_HELD`` naming a live process means an ancestor already holds the lease: run directly. A value whose
  process is dead is reported and ignored, as the shell seam does.
- Else, with ``GPU_LEASE_CMD`` set, replace this process with
  ``$GPU_LEASE_CMD [$GPU_LEASE_ARGS] --project <project> -- <this command line>``. The lease command exports
  ``GPU_LEASE_HELD`` to the command it runs, so the re-executed run goes straight on.
- Else run directly.

The project defaults to the name of the git top level (else the working directory); ``GPU_LEASE_PROJECT`` overrides it.
The lease tool itself, and the CI primitives that hold a lease across the steps of an Azure job, stay outside this
package.
"""
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import MutableMapping
from pathlib import Path

LEASE_CMD_ENV = "GPU_LEASE_CMD"
LEASE_ARGS_ENV = "GPU_LEASE_ARGS"
LEASE_PROJECT_ENV = "GPU_LEASE_PROJECT"
LEASE_HELD_ENV = "GPU_LEASE_HELD"
# set on the re-executed command, so a lease command that does not export GPU_LEASE_HELD cannot loop
REEXEC_ENV = "GPU_EPSILON_LEASE_REEXEC"


def _alive(pid: int) -> bool:
    if os.name == "nt":  # no lease tool runs there, and signal 0 is not a probe on Windows
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # alive, owned by another user
        return True
    return True


def held_by_ancestor(env: MutableMapping[str, str]) -> bool:
    """Whether ``env`` shows a live holder of the lease; a stale ``GPU_LEASE_HELD`` is reported and removed."""
    value = env.get(LEASE_HELD_ENV)
    if not value:
        return False
    pid = value.rsplit(":", 1)[-1]
    if not pid.isdigit():  # untagged (older format): trust it
        return True
    if _alive(int(pid)):
        return True
    print(f"gpu-epsilon: ignoring stale {LEASE_HELD_ENV}={value!r} (holder pid {pid} is dead)", file=sys.stderr)
    env.pop(LEASE_HELD_ENV, None)
    return False


def project_name(cwd: Path | None = None) -> str:
    cwd = cwd or Path.cwd()
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True, text=True)
        if top.returncode == 0 and top.stdout.strip():
            return Path(top.stdout.strip()).name
    except OSError:
        pass
    return cwd.name


def lease_command(argv: list[str], env: MutableMapping[str, str]) -> list[str] | None:
    """The command that runs ``argv`` under the lease, or ``None`` when ``argv`` should run directly."""
    if env.get(REEXEC_ENV) or held_by_ancestor(env):
        return None
    tool = env.get(LEASE_CMD_ENV)
    if not tool or not (shutil.which(tool) or (Path(tool).is_file() and os.access(tool, os.X_OK))):
        return None
    project = env.get(LEASE_PROJECT_ENV) or project_name()
    return [tool, *shlex.split(env.get(LEASE_ARGS_ENV, "")), "--project", project, "--", *argv]


def reexec_under_lease(argv: list[str]) -> None:
    """Replace this process with ``argv`` run under the lease; return (and run directly) when no lease applies."""
    cmd = lease_command(argv, os.environ)
    if cmd is None:
        return
    print(f"gpu-epsilon: acquiring the GPU lease via {cmd[0]} ...", file=sys.stderr, flush=True)
    os.environ[REEXEC_ENV] = "1"
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvp(cmd[0], cmd)
