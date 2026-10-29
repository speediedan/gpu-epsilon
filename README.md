# gpu-epsilon

Shared GPU CI tooling for [finetuning-scheduler](https://github.com/speediedan/finetuning-scheduler) and
[interpretune](https://github.com/speediedan/interpretune): the rules their GPU phase evidence is read by, and the
release eligibility check both projects' release pipelines run. Test orchestration shared by the two projects follows
here.

Both projects run their GPU CI on the same self-hosted agents, one per GPU vendor, and report each run phase by phase
in the same evidence format. Each used to carry its own copy of the code that reads that evidence. This repository
holds the one copy.

## Design constraints

- **Standard library only, Python 3.10 and newer.** The release check runs on a CI agent's own Python, without
  installing either project or any test dependency. The tests hold this as a contract (`-I -S`, no site-packages).
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

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
pyright
```

## License

Apache-2.0. See [LICENSE](LICENSE).
