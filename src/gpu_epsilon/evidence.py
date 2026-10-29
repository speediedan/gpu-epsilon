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
"""The rules a GPU CI phase's evidence is read by, shared by every project that writes it.

A project reports a GPU CI run phase by phase, as two documents per phase:

- a **manifest**: the node ids the phase's paths collect (``raw_node_ids``) and the ones its selector keeps
  (``selected_node_ids``), with the repository, revision, phase, scope and selector string;
- a **report**: the phase's per-node outcomes counted against the manifest, the run's identity (vendor, backend,
  evidence class, image, lease mode, installed versions) and the manifest's digest.

Each project writes them with its own tooling, because collecting and running its phases is specific to it. What a
reader may conclude from them is not, so it lives here: the digest rule that ties a report to its manifest, and the
classification of a phase from its counts.
"""
import hashlib
import json
from collections.abc import Iterable

# per-node outcomes a report counts; a selected node with none of them is ``unexecuted``
OUTCOMES = ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
CLASSIFICATIONS = ("clean", "unreviewed-exclusions", "empty", "failing")


def manifest_digest(manifest: dict) -> str:
    """SHA-256 of the manifest as sorted, compact UTF-8 JSON without a trailing newline."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def classify(report: dict, failure_counts: Iterable[str] = ()) -> str:
    """One word for the phase: ``clean``, ``unreviewed-exclusions``, ``empty`` or ``failing``.

    ``failure_counts`` names the report's own counts that fail a phase when nonzero: how a project records a process
    that started and never finished, or a non-launching rank that failed. finetuning-scheduler's reports carry
    ``rank_failures`` and ``missing_rank_completions``, interpretune's ``missing_completions``.
    """
    o, c = report["outcomes"], report["collection"]
    if c["errors"] or o["failed"] or o["errors"] or o["xpassed"] or o["unexecuted"]:
        return "failing"
    if any(report[key] for key in failure_counts):
        return "failing"
    if not c["selected"] or not o["passed"]:
        return "empty"
    return "clean" if report["exclusions_reviewed"] else "unreviewed-exclusions"
