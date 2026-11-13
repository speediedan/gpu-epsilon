# gpu-epsilon

Shared GPU CI tooling for [finetuning-scheduler](https://github.com/speediedan/finetuning-scheduler) and
[interpretune](https://github.com/speediedan/interpretune): the rules their GPU phase evidence is read by, the tool
that writes that evidence, the release eligibility check both projects' release pipelines run, the special-test runner
that runs their one-process-per-test selections, the phase driver that runs a coverage run's phases in order, and the
Azure Pipelines GPU job both projects' GPU pipelines run.

Both projects run their GPU CI on the same self-hosted agents, one per GPU vendor, and report each run phase by phase
in the same evidence format. Each used to carry its own copy of the code that reads that evidence. This repository
holds the one copy.

## Design constraints

- **Standard library only, Python 3.10 and newer.** The release check runs on a CI agent's own Python, without
  installing either project or any test dependency. The tests hold this as a contract (`-I -S`, no site-packages).
  The special-test runner, the phase driver and the phase-evidence writer run in the project's own environment and read
  their configuration from `pyproject.toml`, so they need Python 3.11 (`tomllib`); pytest is the project's, not a
  dependency of this package.
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

## Phase evidence

The release checker reads each phase's manifest and report; `gpu_epsilon.phase_evidence` writes them. A GPU job
collects every phase's manifest on the CPU first, runs each phase under `run-env`'s exports, then reports each one:

```bash
python -m gpu_epsilon.phase_evidence collect --phase standalone --out-dir /evidence/phases
eval "$(python -m gpu_epsilon.phase_evidence run-env --phase standalone --out-dir /evidence/phases)"
python -m gpu_epsilon.special_tests --mark_type=standalone
python -m gpu_epsilon.phase_evidence report --phase standalone --out-dir /evidence/phases \
    --vendor amd --evidence-class pipeline --job-class ci-release --image "$IMAGE" --lease-mode tool
```

`collect` strips the project's selection variables and any inherited `PYTEST_ADDOPTS` before collecting, and exits 1
when a collection failed. `run-env` makes every pytest process the phase starts load `gpu_epsilon.records` and write to
the phase's own record directory (`GPU_EPSILON_PHASE_RECORD_DIR`, separate from the runner's per-node one), recording
only sessions rooted at the repository. `report` counts each node by rank 0's verdict, failing it when any rank failed,
errored or never finished, and exits by the project's gate. `coverage-diff` compares two runs' measured phases line by
line. The module docstring states every rule.

```toml
[tool.gpu-epsilon.phase-evidence]
repository = "speediedan/finetuning-scheduler"
selection-vars = ["PL_RUN_STANDALONE_TESTS", "FTS_CUDA_TESTS_ONLY"]
collect-env = { PL_RUN_CUDA_TESTS = "1" }
run-env-exports-selection = true
gate = { empty = "never", unreviewed-exclusions = "never" }
versions = { lightning = "module:lightning", lightning_commit = "vcs:lightning" }
scopes.smoke = { select-file = "tests/.standalone_smoke", phases = ["standalone"] }

[[tool.gpu-epsilon.phase-evidence.phase]]
name = "standalone"
paths = ["tests"]
selection-env = { PL_RUN_STANDALONE_TESTS = "1" }
command = "python -m gpu_epsilon.special_tests --mark_type=standalone"
measured = true
```

| key | meaning |
| --- | --- |
| `repository` | the `owner/name` every manifest names |
| `phase` | the phases, in order (below) |
| `selection-vars` | variables the project's `conftest.py` selects on, removed before every collection |
| `collect-env` | variables set for every collection and named in every selector (a GPU opt-in that gates skips) |
| `device-vars` | the visible-device variables a report records (default: the CUDA, ROCR and HIP ones) |
| `run-env-exports-selection` | whether `run-env` also exports `collect-env` and the phase's `selection-env` |
| `scopes` | narrower scopes: `select-file` (chosen by `--scope`, filtered here, limited to `phases`) or `select-file-env` (chosen when the variable is set, narrowed by the project's `conftest.py`) |
| `versions` | report key to `dist:<distribution>`, `module:<module>`, `vcs:<distribution>` or `record:<distribution>`, beside the built-in interpreter and torch versions; anything not installed is `null` |
| `gate` | `empty` (`never` or `outside-full`), `unreviewed-exclusions` (`never` or `except-release`) and `release-job-classes` (default `["ci-release"]`); `clean` always passes and `failing` never does |
| `vendors` | each vendor and the backend its reports name; defaults to `{nvidia = "cuda", amd = "hip"}` |
| `description` | free text, ignored |

| phase key | meaning |
| --- | --- |
| `name` | the phase's name, as the pipeline and the release configuration use it |
| `paths` / `paths-from` | exactly one: the pytest paths, or `{glob, contains}`: the files matching `glob` whose text contains `contains` |
| `selection-env` | the variables that select the phase's tests |
| `args` | pytest arguments that narrow the selected collection (`-k`) |
| `command` | the pipeline command, for `list` |
| `release` | whether a release leg runs the phase (default true) |
| `measured` | whether the pipeline runs it under `coverage run`, so `run-env` sets `COVERAGE_FILE` and `report` digests the export |

Any other key, at any level, is refused by name (exit 2). [`tests/configs/`](tests/configs) holds both projects'
configurations, checked against the implementations they replace, which are kept verbatim in
[`tests/oracles/`](tests/oracles).

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

Building or activating the environment is out of scope: it runs before a usable Python environment exists, so it stays
in each project's shell entry script, which then calls the driver with the venv's interpreter.

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

## GPU job template

`templates/gpu-job.yml` is the Azure Pipelines job both projects' GPU pipelines run: one job that takes either GPU
agent and chooses its image and device options at run time from the vendor the agent declares (`GPU_VENDOR`). An
Azure container resource is fixed before the agent is known, so it is a host job that starts its own container. A
pipeline uses it from this repository as a resource, pinned to a tag, so the template does not move under it:

```yaml
resources:
  repositories:
    - repository: ci
      type: github
      name: speediedan/gpu-epsilon
      endpoint: speediedan  # the project's GitHub service connection
      ref: refs/tags/<tag>
jobs:
  - template: templates/gpu-job.yml@ci
    parameters: ...
```

The template owns what the two jobs shared: admission by demands (the job class and the project are agent
capabilities, and a vendor-demanded job adds `GPU_VENDOR -equals <vendor>`), the vendor table, the digest-pinned
image check, the container's start with the checkout, lease directory, lease tool and evidence directory mounted, a
container sleep 5 minutes longer than the job timeout (so a container leaked by an agent crash exits and the kernel
drops its lease locks), evidence publishing and the `always()` cleanup. What differs is a parameter:

| parameter | what it is |
| --- | --- |
| `jobName`, `displayName`, `dependsOn` | the job's own |
| `vendor`, `jobClass`, `project`, `images` | admission and the digest-pinned image for each vendor |
| `variablePrefix` | names the variables the template sets (`<P>_CTR`, `<P>_EXPECTED_VENDOR`, `<P>_GPU_VENDOR`, `<P>_IMAGE`, `<P>_DEVICE_OPTIONS`, `<P>_GPU_ONE_STAGE`) and the one it reads (`<P>_PHASES_COLLECTED`, which a step sets once the phase manifests are collected, and which publishes the evidence), so a project's step scripts keep their own names |
| `artifactPrefix` | the container's and the evidence artifacts' name prefix: `<a>-release-<vendor>-<attempt>` for a release leg, `<a>-<job>-<vendor>-<attempt>` otherwise |
| `timeoutInMinutes`, `fetchDepth`, `workspaceClean` | the job's timeout, the checkout's depth (left to the pipeline unless set), the workspace's `clean` |
| `hostPrep` | shell run on the host just before `docker run`, in the same script, so a variable it sets can be named in `containerOptions` |
| `containerOptions` | extra `docker run` arguments, as bash words, one or more per line |
| `steps` | the job's work, between the container's start and evidence publishing: host steps that run their commands in the container |
| `releaseCommand` | a host command the cleanup runs before it removes the container, to release the job's host leases |
| `leaseDir`, `leaseTool` | the host's lease directory and lease tool, mounted at `/gpu_leases` and `/gpu_lease_tool` |

Azure is the template's only interpreter, so hosted CI holds it to text checks (`tests/test_gpu_job_template.py`);
what shows it expands is a consumer's pipeline preview (`POST .../pipelines/<id>/preview` with `previewRun`), which
resolves the template from this repository through the consumer's service connection without running anything.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
pyright
```

## License

Apache-2.0. See [LICENSE](LICENSE).
