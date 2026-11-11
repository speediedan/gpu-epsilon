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
"""The implementations :mod:`gpu_epsilon.phase_evidence` replaces, kept verbatim as oracles.

Each ``<project>_<module>.py`` here is a project's own file at a recorded commit, below a header naming the repository,
path, commit and the SHA-256 of the verbatim part. :func:`load` checks that digest before importing, so an edit (a
formatter, say) is a failure rather than a quietly different oracle.
"""
import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).parent
MARKER = "# ---8<--- verbatim below\n"
ORACLES = ("fts_phase_manifest", "fts_phase_plugin", "it_phase_report", "it_phase_plugin")


def path(name: str) -> Path:
    return HERE / f"{name}.py"


def header(name: str) -> dict:
    """The provenance a header records: ``repository``, ``path``, ``commit`` and ``sha256``."""
    head = path(name).read_text(encoding="utf-8").split(MARKER, 1)[0]
    return dict(re.findall(r"^#   (repository|path|commit|sha256):\s+(\S+)", head, re.MULTILINE))


def body(name: str) -> bytes:
    """Every byte after the marker line: the source file as its repository had it."""
    return path(name).read_bytes().split(MARKER.encode(), 1)[1]


def load(name: str) -> ModuleType:
    """A fresh import of one oracle, after checking its verbatim part against the digest its header records.

    interpretune's ``phase_report`` imports its plugin as ``tests.gpu_ci.phase_plugin``; that name is bound to the
    plugin oracle for the import only, so the oracle runs unmodified.
    """
    if hashlib.sha256(body(name)).hexdigest() != header(name)["sha256"]:
        raise AssertionError(f"oracle {name} differs from the source its header records")
    shims: dict[str, ModuleType] = {}
    if name == "it_phase_report":
        shims = {"tests": ModuleType("tests"), "tests.gpu_ci": ModuleType("tests.gpu_ci")}
        shims["tests.gpu_ci.phase_plugin"] = load("it_phase_plugin")
    saved = {k: sys.modules.get(k) for k in shims}
    sys.modules.update(shims)
    try:
        spec = importlib.util.spec_from_file_location(f"oracle_{name}", path(name))
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        # dataclasses resolve string annotations through the defining module, so it must be importable by name
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module
