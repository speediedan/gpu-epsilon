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
"""The shared GPU job template, read as text: Azure is its only interpreter, so these checks are what hosted CI can
hold it to, and its consumers' pipeline previews are what show it expands."""
import re
from pathlib import Path

TEMPLATE = Path(__file__).parents[1] / "templates" / "gpu-job.yml"
TEXT = TEMPLATE.read_text()
# The template's own text, without its comments: a comment may name a consumer.
CODE = "\n".join(line for line in TEXT.splitlines() if not line.lstrip().startswith("#"))
DECLARED = re.findall(r"^  - name: (\w+)$", TEXT, flags=re.MULTILINE)


def test_every_parameter_is_declared_once_and_used():
    assert len(DECLARED) == len(set(DECLARED)), DECLARED
    used = set(re.findall(r"parameters\.(\w+)", CODE))
    assert used == set(DECLARED), {"undeclared": used - set(DECLARED), "unused": set(DECLARED) - used}


def test_every_pipeline_variable_it_sets_or_reads_carries_the_consumer_prefix():
    # A variable named without the prefix would collide with, or silently miss, a consumer's own variable.
    set_names = re.findall(r"task\.setvariable variable=([^\]]+)\]", CODE)
    read_names = re.findall(r"variables\['([^']+)'\]", CODE) + re.findall(r"\$\((\$\{\{[^)]*\}\}\w*)\)", CODE)
    assert set_names and read_names
    for name in set_names + read_names:
        assert name.startswith("${{ parameters.variablePrefix }}_"), name


def test_names_no_consumer_or_consumer_path():
    for word in ("interpretune", "finetuning", "fts", ".azure-pipelines", "IT_", "FTS_"):
        lines = [line for line in CODE.splitlines() if word in line and "values:" not in line]
        assert not lines, (word, lines)


def test_the_container_outlives_the_job_timeout():
    # A container leaked by an agent crash must exit on its own, so the kernel drops its lease locks.
    assert "sleep $(( (${{ parameters.timeoutInMinutes }} + 5) * 60 ))" in CODE
    assert "condition: always()" in CODE and 'docker rm -f "$CTR"' in CODE
