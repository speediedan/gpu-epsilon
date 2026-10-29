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
"""A project's release configuration: what it may say and what it is refused for."""
import json
from pathlib import Path

import pytest

from gpu_epsilon.release_eligibility import ConfigError, ReleaseConfig

CONFIGS = Path(__file__).parent / "configs"
MINIMAL = {"repository": "o/r", "artifact_prefix": "r-release", "phases_dir": "r-phases", "phases": ["a"],
           "required_versions": ["r"]}


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.json")), ids=lambda p: p.stem)
def test_project_configs_load(path):
    config = ReleaseConfig.load(path)
    assert config.vendors == {"nvidia": "cuda", "amd": "hip"}
    assert config.artifact.fullmatch(f"{config.artifact_prefix}-amd-3")["attempt"] == "3"


def test_lists_become_tuples_and_vendors_are_copied():
    vendors = {"nvidia": "cuda"}
    config = ReleaseConfig.from_mapping({**MINIMAL, "vendors": vendors})
    vendors["amd"] = "hip"
    assert config.phases == ("a",) and config.vendors == {"nvidia": "cuda"}


def test_the_artifact_prefix_is_literal():
    config = ReleaseConfig.from_mapping({**MINIMAL, "artifact_prefix": "r.release"})
    assert config.artifact.fullmatch("r.release-amd-1")
    assert not config.artifact.fullmatch("rxrelease-amd-1")


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"phase": ["a"]}, "unknown release configuration key"),
        ({"repository": ""}, "repository must be a non-empty string"),
        ({"phases": []}, "phases must name at least one phase"),
        ({"phases": ["a", "a"]}, "phases must name at least one phase, each once"),
        ({"phases": "a"}, "phases must be a list of non-empty strings"),
        ({"measured_phases": ["b"]}, "measured_phases must be a subset of phases"),
        ({"required_versions": []}, "required_versions must name at least"),
        ({"vendors": {"NVIDIA": "cuda"}}, "vendors must map lowercase vendor names"),
        ({"vendors": {}}, "vendors must map lowercase vendor names"),
    ],
)
def test_malformed_configs_are_refused(change, reason):
    with pytest.raises(ConfigError, match=reason):
        ReleaseConfig.from_mapping({**MINIMAL, **change})


def test_a_missing_key_is_refused():
    doc = dict(MINIMAL)
    del doc["phases_dir"]
    with pytest.raises(ConfigError, match="phases_dir"):
        ReleaseConfig.from_mapping(doc)


def test_a_config_must_be_an_object(tmp_path):
    path = tmp_path / "release.json"
    path.write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(ConfigError, match="JSON object"):
        ReleaseConfig.load(path)
