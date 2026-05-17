# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import importlib.metadata
import re
from pathlib import Path


class MissingRequirementsError(RuntimeError):
    """Raised when a requirements.txt dependency is not installed."""


def _requirement_name(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("-"):
        return None
    if line.startswith(".") or line.startswith("/") or "://" in line:
        return None
    return re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0].strip() or None


def check_requirements(requirements_txt: str | Path):
    """Check that packages in a requirements.txt file are installed."""
    requirements_txt = Path(requirements_txt)
    missing = []
    for line in requirements_txt.read_text().splitlines():
        pkg = _requirement_name(line)
        if pkg is None:
            continue
        try:
            importlib.metadata.distribution(pkg)
        except importlib.metadata.PackageNotFoundError:
            missing.append(pkg)
    if missing:
        raise MissingRequirementsError(
            f"Missing packages: {', '.join(missing)}. "
            f"Run: pip install -r {requirements_txt}"
        )
