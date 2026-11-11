# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- The phase-evidence writer (`gpu_epsilon.phase_evidence`): `list`, `collect`, `run-env`, `report` and
  `coverage-diff`, configured per project in `[tool.gpu-epsilon.phase-evidence]`, replacing finetuning-scheduler's
  `tests/gpu_ci/phase_manifest.py` and interpretune's `tests/gpu_ci/phase_report.py`. Manifests are byte-identical to
  both. Reports take finetuning-scheduler's rank rules (`rank_failures`, `missing_rank_completions`), so
  interpretune's release configuration names those counts in place of `missing_completions`. Where the two disagreed,
  the more correct side wins: a rerun attempt records nothing, only sessions rooted at the repository are recorded, an
  inherited `PYTEST_ADDOPTS` is stripped before collecting, rank 0 decides and other ranks can only fail a node, a
  failure in any of a rank's sessions wins, a package that is not installed is recorded as `null`, and a half-written
  record line is skipped ([speediedan/interpretune#924](https://github.com/speediedan/interpretune/issues/924)).

- A second sink in `gpu_epsilon.records` for phase evidence, under its own variables (`GPU_EPSILON_PHASE_RECORD_DIR`,
  `GPU_EPSILON_PHASE_RECORD_ROOT`), so a phase's records survive the special-test runner's per-node record directories.

- The phase driver (`gpu_epsilon.phases`): a coverage run's phases as data in `[tool.gpu-epsilon.phases]`, each a
  command or a special-test run, with optional phases selected by name, a `skip-unless` probe, `setup` and `report`
  commands, one JSON record of the run, `--dry-run`, and one final status that `--allow-failures` cannot turn green
  ([#3](https://github.com/speediedan/gpu-epsilon/issues/3)).

- `--lease` on the special-test runner and the phase driver (`gpu_epsilon.lease`): the GPU lease re-exec seam each
  project carried in shell, so a project needs no entry script in front of the runner
  ([#4](https://github.com/speediedan/gpu-epsilon/issues/4)).

- The special-test runner (`gpu_epsilon.special_tests`) and the per-rank record plugin it reads outcomes from
  (`gpu_epsilon.records`): one process per selected node id, outcomes from records rather than the log, one exit
  status, and the positive controls the shell runners lacked (an empty selection, a collection error, a crashing test,
  a failing or dying non-launching rank, regex metacharacters in a node id). Configured per project in
  `[tool.gpu-epsilon.special-tests]`.

- The release eligibility checker shared by finetuning-scheduler and interpretune, merged from each project's own copy
  and configured per project by a JSON file, with the phase-evidence rules it reads by (`gpu_epsilon.evidence`), tests
  ported from both projects and hosted CI.
