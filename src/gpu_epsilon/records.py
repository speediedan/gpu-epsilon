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
"""A pytest plugin recording what a session collected and how each node ended, per rank.

Load it with ``-p gpu_epsilon.records``. It has two sinks, each doing nothing unless its variable names a directory:

- ``GPU_EPSILON_RECORD_DIR``, the special-test runner's: it points each node's process at a directory of its own;
- ``GPU_EPSILON_PHASE_RECORD_DIR``, :mod:`gpu_epsilon.phase_evidence`'s: one directory for a whole GPU CI phase, set
  through ``run-env`` and so inherited by every pytest process the phase starts, the runner's included. It needs its
  own variable because the runner replaces the first one for every node. When ``GPU_EPSILON_PHASE_RECORD_ROOT`` is
  set, only a session rooted there is recorded: a test that runs pytest on a tree of its own inherits the phase's
  variables, and its session is not the phase's.

Each pytest session (each rank's, under a launcher such as Lightning's ``ddp`` that re-runs the test in one pytest
session per rank) appends JSON lines to its own file in each active sink:

- ``{"event": "start"}`` when the session starts;
- ``{"event": "collect_error", "nodeid": ...}`` for each collector that failed (an import error, say);
- ``{"event": "collected", "nodeids": [...]}`` after collection and every deselection, so the node ids the session
  runs, in the order it runs them;
- ``{"event": "outcome", "nodeid": ..., "when": ..., "outcome": ...}`` for each test stage that decides a node's
  outcome;
- ``{"event": "finish", "exitstatus": ...}`` when the session finishes. A rank that died or hung in a collective never
  writes it.

Every line carries the session's ``rank`` (``LOCAL_RANK``, else ``RANK``, else 0 for the launching process),
``world_size`` when the launcher set it, and ``pid``. This is the record format both consuming projects' phase plugins
write, so either reader can read the other's.
"""
import json
import os
from pathlib import Path

RECORD_DIR_ENV = "GPU_EPSILON_RECORD_DIR"
PHASE_RECORD_DIR_ENV = "GPU_EPSILON_PHASE_RECORD_DIR"
PHASE_RECORD_ROOT_ENV = "GPU_EPSILON_PHASE_RECORD_ROOT"


def identity() -> dict:
    """This process's rank, world size and pid, as every record carries them."""
    world_size = os.environ.get("WORLD_SIZE", "")
    return {
        "rank": int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))),
        "world_size": int(world_size) if world_size.isdigit() else None,
        "pid": os.getpid(),
    }


def node_outcome(report) -> str | None:
    """The outcome one ``pytest_runtest_logreport`` report decides, or ``None`` when it decides nothing.

    Mirrors pytest's own terminal categories: a failed setup or teardown is an error, a failed call a failure, an
    expected failure ``xfailed`` (a strict XPASS already arrives as a failed call), a skip at any stage ``skipped``. A
    report pytest-rerunfailures marks ``rerun`` decides nothing: the attempt that follows it does.
    """
    if getattr(report, "outcome", None) == "rerun":
        return None
    if report.failed:
        return "failed" if report.when == "call" else "errors"
    if report.skipped:
        return "xfailed" if hasattr(report, "wasxfail") else "skipped"
    if report.when == "call":
        return "xpassed" if hasattr(report, "wasxfail") else "passed"
    return None


class Recorder:
    """Appends one session's records to its own JSON-lines file, so ranks never interleave writes."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _write(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**record, **identity()}) + "\n")

    def pytest_sessionstart(self, session):
        self._write({"event": "start"})

    def pytest_collectreport(self, report):
        if report.failed:
            self._write({"event": "collect_error", "nodeid": report.nodeid})

    def pytest_collection_finish(self, session):
        self._write({"event": "collected", "nodeids": [item.nodeid for item in session.items]})

    def pytest_runtest_logreport(self, report):
        outcome = node_outcome(report)
        if outcome is not None:
            self._write({"event": "outcome", "nodeid": report.nodeid, "when": report.when, "outcome": outcome})

    def pytest_sessionfinish(self, session, exitstatus):
        self._write({"event": "finish", "exitstatus": int(exitstatus)})


def read_sessions(record_dir: Path) -> list[list[dict]]:
    """Every session's records under ``record_dir``, one list per session file, in file-name order."""
    sessions = []
    for path in sorted(Path(record_dir).glob("records-*.jsonl")):
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:  # a line a dying process left half written
                continue
        sessions.append(records)
    return sessions


def _register(config, record_dir: str, name: str) -> None:
    Path(record_dir).mkdir(parents=True, exist_ok=True)
    ident = identity()
    path = Path(record_dir) / f"records-rank{ident['rank']}-pid{ident['pid']}.jsonl"
    config.pluginmanager.register(Recorder(path), name)


def pytest_configure(config):
    record_dir = os.environ.get(RECORD_DIR_ENV)
    if record_dir:
        _register(config, record_dir, "gpu_epsilon_recorder")
    phase_dir, phase_root = os.environ.get(PHASE_RECORD_DIR_ENV), os.environ.get(PHASE_RECORD_ROOT_ENV)
    if phase_dir and (not phase_root or Path(config.rootpath).resolve() == Path(phase_root).resolve()):
        _register(config, phase_dir, "gpu_epsilon_phase_recorder")
