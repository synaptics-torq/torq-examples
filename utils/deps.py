# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import importlib.util
from pathlib import Path

from utils.errors import SetupError


def check_requirements(requirements_txt: str | Path):
    """Check that all packages in a requirements.txt file are importable."""
    requirements_txt = Path(requirements_txt)
    missing = []
    for line in requirements_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkg = line.split(">=")[0].split("<=")[0].split("==")[0].split("!=")[0].split("<")[0].split(">")[0].split("[")[0].strip()
        if not importlib.util.find_spec(pkg):
            missing.append(pkg)
    if missing:
        raise SetupError(
            f"Missing packages: {', '.join(missing)}. "
            f"Run: pip install -r {requirements_txt}"
        )
