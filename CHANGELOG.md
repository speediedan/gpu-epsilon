# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
