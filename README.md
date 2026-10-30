# gpu-epsilon

Shared GPU CI tooling for [finetuning-scheduler](https://github.com/speediedan/finetuning-scheduler) and
[interpretune](https://github.com/speediedan/interpretune): the rules their GPU phase evidence is read by, the
release eligibility check both projects' release pipelines run, the special-test runner that runs their
one-process-per-test selections, and the phase driver that runs a coverage run's phases in order.

Both projects run their GPU CI on the same self-hosted agents, one per GPU vendor, and report each run phase by phase
in the same evidence format. Each used to carry its own copy of the code that reads that evidence. This repository
holds the one copy.

## Design constraints

- **Standard library only, Python 3.10 and newer.** The release check runs on a CI agent's own Python, without
  installing either project or any test dependency. The tests hold this as a contract (`-I -S`, no site-packages).
  The special-test runner and the phase driver run in the project's own environment and read their configuration from
  `pyproject.toml`, so they need Python 3.11 (`tomllib`); pytest is the project's, not a dependency of this package.
- **A CI dependency, never a runtime one.** A consuming project pins this repository by commit in its CI requirements
  (`gpu-epsilon @ git+https://github.com/speediedan/gpu-epsilon@<sha>`) and never in its package metadata, so nothing
  of it reaches the project's published `Requires-Dist`.
- **What differs between projects is data.** Each project describes its own evidence in a small JSON file; the rules
  are shared.

## Release eligibility

A release needs one full, clean, like-for-like GPU run on each vendor at the release candidate's revision. A project's
release pipeline runs one leg per vendor, each publishing its phase evidence as an artifact named
`<artifact_prefix>-<vendor>-<attempt>`, then runs the checker on a CPU job over the downloaded artifacts:

```bash
python3 -m gpu_epsilon.release_eligibility --config tests/gpu_ci/release.json \
    --artifacts "$(Pipeline.Workspace)" --revision "$(Build.SourceVersion)" \
    --nvidia-result Succeeded --amd-result Succeeded --output eligibility.json
```

It exits 0 and writes the verdict when the candidate is eligible, exits 1 naming the first reason it is not (removing
any verdict an earlier invocation left), and exits 2 for a malformed configuration. The module docstring lists every
rule it checks. The `--<vendor>-result` options are the configuration's vendors.

### Configuration

| key | meaning |
| --- | --- |
| `repository` | the `owner/name` every report and manifest must name |
| `artifact_prefix` | a leg's artifacts are `<artifact_prefix>-<vendor>-<attempt>`; the latest attempt counts |
| `phases_dir` | the directory in each artifact holding `reports/<phase>.json` and `manifests/<phase>.json` |
| `phases` | the phases a release leg runs, in order |
| `measured_phases` | phases whose reports carry coverage evidence, checked against the export's digest |
| `failure_counts` | report counts that fail a phase when nonzero (a rank or process that failed or never finished) |
| `required_versions` | installed versions that must be present and agree across vendors, without their local tag |
| `shared_versions` | installed versions compared exactly as recorded, absence included |
| `shared_builds` | build identifiers behind a version string that does not name them (a torch commit) |
| `vendors` | each vendor and the backend its reports name; defaults to `{"nvidia": "cuda", "amd": "hip"}` |
| `description` | free text, ignored |

Any other key is refused. [`tests/configs/`](tests/configs) holds the two projects' configurations as the tests use
them.

## Special-test runner

Some tests cannot share a process: they start a process group, take a whole device, or launch one pytest session per
rank. A project marks them, and its `conftest.py` selects them from environment variables. The runner sets the
variables for one mark, collects the selection, runs each node id in a fresh process and exits with one status:

```bash
python -m gpu_epsilon.special_tests --mark_type=standalone --filter_pattern='test_f'
python -m gpu_epsilon.special_tests --mark_type=standalone --select_file=tests/.standalone_smoke --allow-failures
```

Outcomes come from records that `gpu_epsilon.records` (a pytest plugin the runner loads into every session) writes per
rank, never from the log. A node fails when any rank failed or errored on it, when its process exited non-zero, or
when a session started and never finished; a node with no verdict at all is `UNKNOWN` and fails the run. An empty
selection fails unless allowed, a collection error fails, and a node id is passed to pytest verbatim, regex
metacharacters included. Each run writes its session log, the raw test output, and a JSON results file beside the
session log.

`--lease` runs the whole selection under the host's GPU lease (see "GPU lease" below), taken after the arguments are
validated, so a bad flag never waits for a lease. `GPU_EPSILON_EVENT_CMD` names a command run at each collection and
test boundary, whose output goes to the log.

The project describes itself in `pyproject.toml`:

```toml
[tool.gpu-epsilon.special-tests]
name = "FTS"
env = { PL_RUN_CUDA_TESTS = "1" }
exec = ["-m", "coverage", "run", "--source", "src/finetuning_scheduler", "--append", "-m", "pytest",
        "--capture=no", "--no-header", "-v", "-s", "-rA"]
experiments-list = "tests/.experiments"
allow-empty = true
raw-log-copy-dir = "lightning_logs"
publish = "azure"

[tool.gpu-epsilon.special-tests.marks.standalone]
env = { PL_RUN_STANDALONE_TESTS = "1" }

[tool.gpu-epsilon.special-tests.marks.exp_patch]
env = { FTS_EXPERIMENTAL_PATCH_TESTS = "1" }
```

| key | meaning |
| --- | --- |
| `marks` | each mark's `env` (the variables that select it), `description`, and `refused` (a message for a mark with no tests) |
| `default-mark` | the mark run without `--mark_type` (default `standalone`) |
| `collect-dirs` | collection roots, unless `--collect_dir` overrides them (default `["tests"]`) |
| `env` / `unset` | variables set for every mark / removed before any run, beside every mark's own variables |
| `exec` | the arguments after the interpreter that run one node, coverage included |
| `exec-overrides` | `[{env, exec}]`: the first whose variable is set replaces `exec`, with `{value}` substituted |
| `collect` | the arguments after the interpreter that collect (default `-m pytest -q --collect-only --pythonwarnings ignore`) |
| `experiments-list` | the file naming experimental-patch variables, in `--experiment_patch_mask` order |
| `allow-empty` | whether an empty selection exits 0 without `--allow-empty` |
| `log-dir`, `raw-log-copy-dir`, `publish` | where logs go, a second home for the raw log, and `azure` to upload it |
| `options` | project flags: `env` and `args` (with `{value}` for `takes-value` options), `default`, `clears` |

## Phase driver

A coverage run is a list of phases: the main suite, each special-test mark, the examples, an optional experimental
partition. The driver runs them in order from `[tool.gpu-epsilon.phases]`, records each one, and exits with one status:

```bash
python -m gpu_epsilon.phases                                # the default phases
python -m gpu_epsilon.phases --with examples --allow-failures
python -m gpu_epsilon.phases --phases main --dry-run        # print the plan, run nothing
```

A phase passes only on exit status 0. pytest's "no tests collected" status passes only where the phase allows it, and
a phase whose `skip-unless` probe fails is recorded as skipped with the reason. Without `--allow-failures` the driver
stops at the first failed phase and records the rest as not run; with it, every phase runs (and each special-test phase
gets `--allow-failures` too), and the driver still exits 1, naming each failed phase. The `report` commands run even
after a failure. A JSON record of the run (each phase's status, exit code, duration, and the special-test runner's
counts) is written beside the session log. `--lease` takes the host's GPU lease once for the whole run.

```toml
[tool.gpu-epsilon.phases]
name = "FTS"
env = { PL_RUN_CUDA_TESTS = "1" }
setup = [["-m", "coverage", "erase"]]
report = [["-m", "coverage", "report", "-m"]]

[[tool.gpu-epsilon.phases.phase]]
name = "main"
run = ["-m", "coverage", "run", "--append", "--source", "src/finetuning_scheduler", "-m", "pytest",
       "src/finetuning_scheduler", "tests", "-v"]

[[tool.gpu-epsilon.phases.phase]]
name = "standalone"
special-tests = ["--mark_type=standalone"]

[[tool.gpu-epsilon.phases.phase]]
name = "exp_patch"
optional = true
special-tests = ["--mark_type=exp_patch", "--filter_pattern=test_f", "--experiment_patch_mask=1 0 0 1"]
```

| key | meaning |
| --- | --- |
| `name`, `log-dir` | the banner name, and where the session log and its record go |
| `env` | variables set for every phase |
| `setup` / `report` | commands (each the arguments after the interpreter) run before the first phase / after the last |
| `phase` | the phases, in order (below) |

| phase key | meaning |
| --- | --- |
| `name` | used by `--phases`, `--with`, `--skip` and in the record |
| `run` / `special-tests` | exactly one: the arguments after the interpreter, or arguments for the special-test runner |
| `description` | a line for the log |
| `env` / `unset` | variables set / removed for this phase |
| `optional` | run only when `--with` or `--phases` names it |
| `allow-empty` | a `run` phase passes on pytest's "no tests collected" status |
| `skip-unless` | a probe (the arguments after the interpreter); the phase is skipped when it exits non-zero |
| `after` | commands run after the phase (`coverage combine`, say); a failing one fails the phase |

`--special-tests-args` passes extra arguments to every special-test phase (a project's own runner options, say).

## GPU lease

On a host shared by several GPU consumers, `--lease` (on the runner and the driver) re-executes the run once under
the host's lease command, so one lease covers the whole run. It is a no-op unless `GPU_LEASE_CMD` names an executable:

- `GPU_LEASE_HELD` naming a live process means an ancestor already holds the lease, and the run goes straight on;
- else the run is replaced by `$GPU_LEASE_CMD [$GPU_LEASE_ARGS] --project <project> -- <the same command line>`, where
  the project is `GPU_LEASE_PROJECT` or the git top level's name;
- else the run goes straight on, as on a host with no lease tool.

This is the Python form of the `gpu_lease_reexec` seam in each project's `scripts/gpu_lease_wrap.sh`, so a project
needs no shell entry script in front of the runner. The lease tool itself, and the CI steps that hold a lease across a
job, stay with the projects.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
pyright
```

## License

Apache-2.0. See [LICENSE](LICENSE).
