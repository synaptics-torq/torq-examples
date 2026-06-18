# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import logging
import os
from pathlib import Path


def discover_lm_head_path(model_path: str | os.PathLike) -> Path | None:
    """Find a sibling LM head VMFB for *model_path*, when unambiguous."""
    model_path = Path(model_path).resolve()
    candidates = []
    for path in sorted(model_path.parent.glob("*.vmfb*")):
        if path.resolve() == model_path:
            continue
        normalized_stem = path.stem.lower().replace("-", "_")
        if "lm_head" in normalized_stem:
            candidates.append(path)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    candidate_list = ", ".join(str(path) for path in candidates)
    raise ValueError(
        "Found multiple LM head candidates next to model. "
        f"Please pass --lm-head explicitly. Candidates: {candidate_list}"
    )


def resolve_lm_head_path(
    model_path: str | os.PathLike,
    lm_head_path: str | os.PathLike | None = None,
    *,
    disable_lm_head: bool = False,
    logger: logging.Logger | None = None,
) -> Path | None:
    """Resolve explicit, disabled, or auto-discovered LM head selection."""
    if lm_head_path is not None and disable_lm_head:
        raise ValueError("--lm-head and --no-lm-head cannot be used together.")
    if lm_head_path is not None:
        return Path(lm_head_path)
    if disable_lm_head:
        return None

    discovered = discover_lm_head_path(model_path)
    if discovered is not None and logger is not None:
        logger.info("Auto-discovered LM head '%s'", str(discovered))
    return discovered
