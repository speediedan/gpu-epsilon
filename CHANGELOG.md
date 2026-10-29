# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- The release eligibility checker shared by finetuning-scheduler and interpretune, merged from each project's own copy
  and configured per project by a JSON file, with the phase-evidence rules it reads by (`gpu_epsilon.evidence`), tests
  ported from both projects and hosted CI.
