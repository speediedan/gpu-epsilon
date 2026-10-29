# gpu-epsilon

Shared GPU CI tooling for [finetuning-scheduler](https://github.com/speediedan/finetuning-scheduler) and
[interpretune](https://github.com/speediedan/interpretune): the rules their GPU phase evidence is read by, the
release eligibility check both projects' release pipelines run, and the special-test runner that runs their
one-process-per-test selections.

Both projects run their GPU CI on the same self-hosted agents, one per GPU vendor, and report each run phase by phase
in the same evidence format. Each used to carry its own copy of the code that reads that evidence. This repository
holds the one copy.

## Design constraints

- **Standard library only, Python 3.10 and newer.** The release check runs on a CI agent's own Python, without
  installing either project or any test dependency. The tests hold this as a contract (`-I -S`, no site-packages).
  The special-test runner runs in the project's own environment and reads its configuration from `pyproject.toml`, so
  it needs Python 3.11 (`tomllib`); pytest is the project's, not a dependency of this package.
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

The runner is meant to be `exec`-ed by a short entry script that keeps the project's existing command line and takes
the host's GPU lease, after `--check-args` has validated the arguments (so a bad flag never waits for a lease).
`GPU_EPSILON_EVENT_CMD` names a command run at each collection and test boundary, whose output goes to the log.

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

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
pyright
```

## License

Apache-2.0. See [LICENSE](LICENSE).
