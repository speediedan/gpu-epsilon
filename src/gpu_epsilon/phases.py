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
"""The phase driver: run a project's test phases in order, record each one's outcome, and report one final status.

A coverage run is a list of phases: the main suite, each special-test mark, the examples, an optional experimental
partition. Each project used to drive them from shell, inferring each phase's outcome from exit codes threaded through
subshells and re-implementing ``--allow-failures`` by hand. Here the phases are data, read from
``[tool.gpu-epsilon.phases]`` in the project's ``pyproject.toml`` (:class:`DriverConfig` and :class:`Phase` list the
keys), and the driver computes the status:

- A phase is either a command (``run``: the arguments after the interpreter, coverage included) or a special-test run
  (``special-tests``: arguments for :mod:`gpu_epsilon.special_tests`, which reads the same ``pyproject.toml``).
- A phase passes only on exit status 0; pytest's "no tests collected" status passes only where the phase sets
  ``allow-empty``, so a phase that ran nothing never reads as green. A phase whose ``skip-unless`` probe fails (no CUDA,
  say) is recorded as skipped with that reason.
- Without ``--allow-failures`` the driver stops at the first failed phase and records the rest as not run. With it,
  every phase runs, ``--allow-failures`` is passed on to each special-test run, and the driver still exits 1, naming
  each failed phase.
- ``setup`` commands run before the first phase (``coverage erase``, say) and ``report`` commands after the last (the
  coverage report); a failing one fails the run.

One JSON record of the run (each phase's status, exit code, duration, and the special-test runner's counts) is written
beside the session log. ``--dry-run`` prints the resolved plan and runs nothing. ``--lease`` takes the host's GPU lease
for the whole run (:mod:`gpu_epsilon.lease`). ``GPU_EPSILON_EVENT_CMD`` runs at each phase boundary, as it does at
each test boundary in the special-test runner.

Usage, from the project root::

    python -m gpu_epsilon.phases                                  # the default phases
    python -m gpu_epsilon.phases --with examples,exp_patch        # plus optional phases
    python -m gpu_epsilon.phases --skip standalone --allow-failures
    python -m gpu_epsilon.phases --phases main --dry-run          # exactly these phases, planned but not run
"""
import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from gpu_epsilon.lease import reexec_under_lease
from gpu_epsilon.special_tests import EVENT_CMD_ENV, NO_TESTS_COLLECTED, RULE, ConfigError, RunnerConfig

PHASE_KEYS = {"name", "description", "run", "special_tests", "env", "unset", "optional", "allow_empty", "skip_unless",
              "after"}


@dataclass(frozen=True)
class Phase:
    """One phase (an entry of ``[[tool.gpu-epsilon.phases.phase]]``, keys in kebab case).

    Attributes:
        name: the phase's name, used by ``--phases``, ``--with`` and ``--skip`` and in the record.
        run: the arguments after the interpreter that run the phase. Exactly one of ``run`` and ``special_tests``.
        special_tests: arguments for :mod:`gpu_epsilon.special_tests` (``--mark_type=standalone``, say).
        description: a line for the log.
        env: variables set for the phase.
        unset: variables removed for the phase (``CUDA_VISIBLE_DEVICES``, to see every device).
        optional: run only when ``--with`` or ``--phases`` names it.
        allow_empty: whether pytest's "no tests collected" status passes (a ``run`` phase only).
        skip_unless: the arguments after the interpreter of a probe; when it exits non-zero the phase is skipped.
        after: commands (each the arguments after the interpreter) run after the phase, such as ``coverage combine``;
            a failing one fails the phase.
    """

    name: str
    run: tuple[str, ...] = ()
    special_tests: tuple[str, ...] = ()
    description: str = ""
    env: dict = field(default_factory=dict)
    unset: tuple[str, ...] = ()
    optional: bool = False
    allow_empty: bool = False
    skip_unless: tuple[str, ...] = ()
    after: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def from_mapping(cls, doc: dict) -> "Phase":
        if not isinstance(doc, dict):
            raise ConfigError("each phase must be a table")
        fields = {k.replace("-", "_"): v for k, v in doc.items()}
        unknown = sorted(set(fields) - PHASE_KEYS)
        if unknown:
            raise ConfigError(f"phase {fields.get('name', '?')}: unknown key(s) {', '.join(unknown)}")
        for key in ("run", "special_tests", "unset", "skip_unless"):
            if key in fields:
                fields[key] = _strings(fields[key], f"phase {fields.get('name', '?')}: {key}")
        if "after" in fields:
            fields["after"] = tuple(_strings(c, f"phase {fields.get('name', '?')}: after") for c in fields["after"])
        try:
            phase = cls(**fields)
        except TypeError as exc:
            raise ConfigError(str(exc)) from exc
        if not phase.name or not isinstance(phase.name, str) or "," in phase.name:
            raise ConfigError(f"phase name {phase.name!r} must be a non-empty string without commas")
        if bool(phase.run) == bool(phase.special_tests):
            raise ConfigError(f"phase {phase.name}: needs exactly one of 'run' and 'special-tests'")
        if phase.special_tests and phase.allow_empty:
            raise ConfigError(f"phase {phase.name}: allow-empty applies to a 'run' phase (pass --allow-empty instead)")
        return phase


@dataclass(frozen=True)
class DriverConfig:
    """A project's phase configuration (``[tool.gpu-epsilon.phases]``, keys in kebab case).

    Attributes:
        phases: the phases, in the order they run.
        name: the project's name in the log banner.
        env: variables set for every phase.
        setup: commands (each the arguments after the interpreter) run before the first phase.
        report: commands run after the last phase that ran, even after a failure, so a failed run still reports.
        log_dir: where the session log and its record go unless ``--log-dir`` says otherwise.
    """

    phases: tuple[Phase, ...]
    name: str = "test"
    env: dict = field(default_factory=dict)
    setup: tuple[tuple[str, ...], ...] = ()
    report: tuple[tuple[str, ...], ...] = ()
    log_dir: str = "."

    @classmethod
    def from_mapping(cls, doc: dict) -> "DriverConfig":
        if not isinstance(doc, dict):
            raise ConfigError("[tool.gpu-epsilon.phases] must be a table")
        fields = {k.replace("-", "_"): v for k, v in doc.items()}
        fields["phases"] = tuple(Phase.from_mapping(p) for p in fields.pop("phase", []))
        unknown = sorted(set(fields) - set(cls.__dataclass_fields__))
        if unknown:
            raise ConfigError(f"unknown phases key(s): {', '.join(unknown)}")
        for key in ("setup", "report"):
            if key in fields:
                fields[key] = tuple(_strings(c, key) for c in fields[key])
        try:
            config = cls(**fields)
        except TypeError as exc:
            raise ConfigError(str(exc)) from exc
        if not config.phases:
            raise ConfigError("no phases: add [[tool.gpu-epsilon.phases.phase]] entries")
        names = [p.name for p in config.phases]
        if len(set(names)) != len(names):
            raise ConfigError(f"phase names repeat: {', '.join(sorted({n for n in names if names.count(n) > 1}))}")
        return config

    @classmethod
    def load(cls, path: Path) -> "DriverConfig":
        doc = _read_toml(path)
        table = doc.get("tool", {}).get("gpu-epsilon", {}).get("phases")
        if table is None:
            raise ConfigError(f"{path} has no [tool.gpu-epsilon.phases] table")
        config = cls.from_mapping(table)
        if any(p.special_tests for p in config.phases):
            RunnerConfig.load(path)  # a special-test phase needs a valid runner table beside this one
        return config

    def select(self, only: list[str], with_: list[str], skip: list[str]) -> list[Phase]:
        """The phases to run, in configuration order: ``only`` if given, else the non-optional ones plus ``with_``."""
        known = {p.name for p in self.phases}
        unknown = sorted(set(only + with_ + skip) - known)
        if unknown:
            raise ConfigError(f"no phase named {', '.join(unknown)} (phases: {', '.join(p.name for p in self.phases)})")
        if only and with_:
            raise ConfigError("--phases names the whole selection; use it without --with")
        wanted = set(only) if only else {p.name for p in self.phases if not p.optional} | set(with_)
        return [p for p in self.phases if p.name in wanted and p.name not in skip]


@dataclass
class PhaseResult:
    name: str
    status: str  # passed, failed, skipped, not-run
    exit_code: int | None = None
    seconds: float = 0.0
    detail: str = ""
    counts: dict | None = None
    log: str | None = None


def _strings(value, what: str) -> tuple[str, ...]:
    if isinstance(value, str) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{what} must be a list of strings")
    return tuple(value)


def _read_toml(path: Path) -> dict:
    try:
        import tomllib  # pyright: ignore[reportMissingImports]  (3.11+; the except handles 3.10)
    except ModuleNotFoundError as exc:  # Python 3.10
        raise ConfigError("reading pyproject.toml needs Python 3.11 or newer (tomllib)") from exc
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _list(values: list[str] | None) -> list[str]:
    return [name.strip() for value in values or [] for name in value.split(",") if name.strip()]


class Driver:
    """One invocation: configuration, the session log, and each phase's result."""

    def __init__(self, config: DriverConfig, args: argparse.Namespace, phases: list[Phase]) -> None:
        self.config, self.args, self.phases = config, args, phases
        self.stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.log_dir = Path(args.log_dir or config.log_dir)
        self.session_log = Path(args.log_file or self.log_dir / f"phases_{self.stamp}.log")
        self.results: list[PhaseResult] = []
        self.notes: list[str] = []

    def log(self, text: str = "") -> None:
        print(text, flush=True)
        with open(self.session_log, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def heading(self, text: str) -> None:
        self.log(RULE)
        self.log(text)
        self.log(RULE)

    def event(self, label: str) -> None:
        cmd = os.environ.get(EVENT_CMD_ENV)
        if not cmd:
            return
        try:
            proc = subprocess.run([*shlex.split(cmd), label], capture_output=True, text=True, timeout=60)
            output = (proc.stdout + proc.stderr).rstrip()
        except (OSError, subprocess.SubprocessError) as exc:
            output = f"{EVENT_CMD_ENV} failed at {label}: {exc}"
        if output:
            self.log(output)

    def environment(self, phase: Phase) -> dict:
        env = dict(os.environ)
        env.update({k: str(v) for k, v in self.config.env.items()})
        for name in phase.unset:
            env.pop(name, None)
        env.update({k: str(v) for k, v in phase.env.items()})
        return env

    def phase_log(self, phase: Phase) -> Path:
        return self.log_dir / f"phases_{self.stamp}_{phase.name}.log"

    def command(self, phase: Phase) -> list[str]:
        if phase.run:
            return [self.args.python, *phase.run]
        cmd = [self.args.python, "-m", "gpu_epsilon.special_tests", "--config", str(self.args.config)]
        cmd += [*phase.special_tests, *shlex.split(self.args.special_tests_args or "")]
        cmd += ["--log_file", str(self.phase_log(phase)), "--log-dir", str(self.log_dir)]
        return cmd + (["--allow-failures"] if self.args.allow_failures else [])

    def plan(self) -> None:
        self.heading(f"{self.config.name} phases (dry run: nothing is executed):")
        for cmd in self.config.setup:
            self.log(f"setup: {shlex.join([self.args.python, *cmd])}")
        for phase in self.phases:
            self.log(f"{phase.name}: {shlex.join(self.command(phase))}")
            changes = [f"{k}={v}" for k, v in {**self.config.env, **phase.env}.items()] + [f"-{k}" for k in phase.unset]
            if changes:
                self.log(f"    environment: {' '.join(changes)}")
            if phase.skip_unless:
                self.log(f"    skipped unless: {shlex.join([self.args.python, *phase.skip_unless])} exits 0")
            for cmd in phase.after:
                self.log(f"    after: {shlex.join([self.args.python, *cmd])}")
        for cmd in self.config.report:
            self.log(f"report: {shlex.join([self.args.python, *cmd])}")

    def run_command(self, argv: list[str], env: dict, output: Path | None) -> int:
        """Run ``argv``, appending its output to ``output``, or passing it through when ``output`` is ``None``."""
        if output is None:
            sys.stdout.flush()
            return subprocess.run(argv, env=env).returncode
        with open(output, "a", encoding="utf-8") as f:
            return subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT).returncode

    def commands(self, label: str, commands: tuple[tuple[str, ...], ...], env: dict) -> bool:
        for cmd in commands:
            argv = [self.args.python, *cmd]
            self.log(f"{label}: {shlex.join(argv)}")
            code = self.run_command(argv, env, self.session_log)
            if code != 0:
                self.log(f"{label} failed with exit code {code}: {shlex.join(argv)}")
                self.notes.append(f"{label} failed (exit {code})")
                return False
        return True

    def run_phase(self, phase: Phase) -> PhaseResult:
        env = self.environment(phase)
        if phase.skip_unless:
            probe = subprocess.run([self.args.python, *phase.skip_unless], env=env, capture_output=True)
            if probe.returncode != 0:
                self.log(f"{phase.name} skipped: {shlex.join(phase.skip_unless)} exited {probe.returncode}")
                return PhaseResult(phase.name, "skipped", detail=f"probe exited {probe.returncode}")
        self.heading(f"Phase {phase.name}: {phase.description or shlex.join(self.command(phase))}")
        self.event(f"phases:start:{phase.name}")
        started = time.monotonic()
        # the special-test runner writes its own log (and an Azure upload line, which must reach the console)
        code = self.run_command(self.command(phase), env, None if phase.special_tests else self.session_log)
        result = PhaseResult(phase.name, "passed", exit_code=code)
        if phase.special_tests:
            result.log = str(self.phase_log(phase))
            record = self.phase_log(phase).with_suffix(".json")
            if record.is_file():
                doc = json.loads(record.read_text(encoding="utf-8"))
                result.counts, result.detail = doc.get("counts"), doc.get("note", "")
        if code == NO_TESTS_COLLECTED and phase.run and not phase.allow_empty:
            result.status, result.detail = "failed", "no tests collected"
        elif code != 0 and not (code == NO_TESTS_COLLECTED and phase.allow_empty):
            result.status = "failed"
        if result.status == "passed" and not self.commands(f"{phase.name} after", phase.after, env):
            result.status, result.detail = "failed", "an 'after' command failed"
        result.seconds = round(time.monotonic() - started, 1)
        self.event(f"phases:end:{phase.name}")
        self.log(f"{phase.name}: {result.status} (exit code {code}{', ' + result.detail if result.detail else ''})")
        return result

    def execute(self) -> int:
        self.heading(f"{self.config.name} phases beginning at {self.stamp}: {', '.join(p.name for p in self.phases)}")
        base_env = {**os.environ, **{k: str(v) for k, v in self.config.env.items()}}
        setup_ok = self.commands("setup", self.config.setup, base_env)
        stopped = not setup_ok
        for phase in self.phases:
            if stopped:
                self.results.append(PhaseResult(phase.name, "not-run"))
                continue
            result = self.run_phase(phase)
            self.results.append(result)
            if result.status == "failed" and not self.args.allow_failures:
                self.log(f"Stopping after the failed phase {phase.name} (use --allow-failures to run every phase)")
                stopped = True
        ran = any(r.status in ("passed", "failed") for r in self.results)
        report_ok = self.commands("report", self.config.report, base_env) if ran else True
        return self.summarize(setup_ok and report_ok)

    def summarize(self, commands_ok: bool) -> int:
        failed = [r.name for r in self.results if r.status == "failed"]
        not_run = [r.name for r in self.results if r.status == "not-run"]
        status = 0 if commands_ok and not failed and not not_run else 1
        self.heading("Phase summary:")
        for r in self.results:
            counts = f" {r.counts}" if r.counts else ""
            self.log(f"{r.name}: {r.status} (exit {r.exit_code}, {r.seconds}s){counts}")
        if failed:
            self.log(f"Failed phases: {', '.join(failed)}")
        if not_run:
            self.log(f"Not run: {', '.join(not_run)}")
        for note in self.notes:
            self.log(note)
        self.log(f"Exiting with status {status}.")
        doc = {
            "status": status,
            "failed": failed,
            "not_run": not_run,
            "notes": self.notes,
            "phases": [r.__dict__ for r in self.results],
            "session_log": str(self.session_log),
        }
        self.session_log.with_suffix(".json").write_text(json.dumps(doc, indent=2) + "\n")
        return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("pyproject.toml"), help="the project's pyproject.toml")
    parser.add_argument("--phases", action="append", help="run exactly these phases (comma separated), in config order")
    parser.add_argument("--with", dest="with_", action="append", help="add these optional phases (comma separated)")
    parser.add_argument("--skip", action="append", help="leave out these phases (comma separated)")
    parser.add_argument("--allow-failures", dest="allow_failures", action="store_true",
                        help="run every phase after a failure (the run still exits 1, naming each failed phase)")
    parser.add_argument("--special-tests-args", dest="special_tests_args",
                        help="extra arguments for every special-test phase, as one shell-quoted string")
    parser.add_argument("--log_file", help="the session log (default: phases_<time>.log in the log directory)")
    parser.add_argument("--log-dir", dest="log_dir", help="the directory for the session and phase logs")
    parser.add_argument(
        "--python", default=sys.executable, help="the interpreter each phase runs under (default: this one)"
    )
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="print the plan and run nothing")
    parser.add_argument(
        "--lease",
        action="store_true",
        help="run under the host's GPU lease (GPU_LEASE_CMD), taken once after the arguments are validated",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = DriverConfig.load(args.config)
        phases = config.select(_list(args.phases), _list(args.with_), _list(args.skip))
    except (ConfigError, OSError, ValueError) as exc:
        print(f"error: phases configuration {args.config}: {exc}", file=sys.stderr)
        return 2
    if not phases:
        print("error: the selection leaves no phase to run", file=sys.stderr)
        return 2
    driver = Driver(config, args, phases)
    driver.log_dir.mkdir(parents=True, exist_ok=True)
    driver.session_log.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        driver.plan()
        return 0
    if args.lease:
        rest = sys.argv[1:] if argv is None else argv
        reexec_under_lease([sys.executable, "-m", "gpu_epsilon.phases", *rest])
    return driver.execute()


if __name__ == "__main__":
    sys.exit(main())
