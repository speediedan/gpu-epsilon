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
"""The shared phase-evidence rules: the manifest digest and the phase classification."""
import hashlib

import pytest

from gpu_epsilon.evidence import CLASSIFICATIONS, OUTCOMES, classify, manifest_digest


def test_manifest_digest_is_sorted_compact_json():
    manifest = {"phase": "standalone", "raw_node_ids": ["b", "a"], "revision": "0" * 40}
    expected = '{"phase":"standalone","raw_node_ids":["b","a"],"revision":"' + "0" * 40 + '"}'
    assert manifest_digest(manifest) == hashlib.sha256(expected.encode()).hexdigest()
    assert manifest_digest(dict(reversed(manifest.items()))) == manifest_digest(manifest)


def _report(selected=4, errors=0, reviewed=True, **outcomes):
    counts = {**dict.fromkeys(OUTCOMES, 0), "unexecuted": 0, **outcomes}
    return {
        "collection": {"selected": selected, "errors": errors},
        "outcomes": counts,
        "exclusions_reviewed": reviewed,
        "rank_failures": 0,
        "missing_rank_completions": 0,
        "missing_completions": 0,
    }


@pytest.mark.parametrize(
    "report, expected",
    [
        (_report(passed=4), "clean"),
        (_report(passed=3, skipped=1), "clean"),
        (_report(passed=3, skipped=1, reviewed=False), "unreviewed-exclusions"),
        (_report(passed=3, failed=1), "failing"),
        (_report(passed=3, errors=1), "failing"),
        (_report(passed=3, xpassed=1), "failing"),
        (_report(passed=3, unexecuted=1), "failing"),
        (_report(passed=4, errors=1), "failing"),
        (_report(selected=0), "empty"),
        (_report(skipped=4), "empty"),
    ],
)
def test_classify(report, expected):
    assert expected in CLASSIFICATIONS
    assert classify(report) == expected


@pytest.mark.parametrize("key", ["rank_failures", "missing_rank_completions", "missing_completions"])
def test_a_projects_failure_count_fails_the_phase(key):
    report = {**_report(passed=4), key: 1}
    assert classify(report) == "clean"  # a count the project does not name is not read
    assert classify(report, (key,)) == "failing"


def test_a_named_failure_count_must_be_present():
    report = _report(passed=4)
    del report["missing_completions"]
    with pytest.raises(KeyError):
        classify(report, ("missing_completions",))
