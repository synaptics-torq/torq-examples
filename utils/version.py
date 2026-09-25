# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""torq-examples versioning helpers.

The ``VERSION`` file at the repo root records which torq-examples release this
checkout corresponds to. The version tracks the torq-compiler/runtime release
the demos are tested against, and the built-in model repos on Hugging Face are
tagged the same way: a checkout of examples ``2.2.0`` downloads (and maintains)
the models at tag ``v2.2.0``.

The file must be bumped on every release of torq-examples.
"""

import logging
from pathlib import Path
from typing import Final

from utils.download import DownloadError

__all__ = [
    "examples_version",
    "parse_model_specs",
    "resolve_model_version",
]

logger = logging.getLogger(__name__)

_VERSION_FILE: Final[Path] = Path(__file__).resolve().parent.parent / "VERSION"


def examples_version() -> str | None:
    """Return the torq-examples version as an HF-style tag (e.g. ``v2.2.0``).

    Returns ``None`` when the ``VERSION`` file is missing or empty so callers
    can report a clear error (or fall back to an explicit ``--model-version``)
    instead of failing at import time.
    """
    try:
        raw = _VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    return raw if raw.lower().startswith("v") else f"v{raw}"


def parse_model_specs(specs: list[str]) -> list[tuple[str, str | None]]:
    """Split model specs into ``(name, version)`` pairs.

    Each spec is a model name or raw HF repo id, optionally suffixed with
    ``:version`` to pin that single model to a specific version, e.g.
    ``"default:v2.0.0"`` or ``"custom/gemma3"``. The ``:version`` suffix takes
    precedence over a global ``--model-version`` for that model.
    """
    parsed: list[tuple[str, str | None]] = []
    for spec in specs:
        name, sep, version = spec.rpartition(":")
        if sep and name:
            parsed.append((name, version))
        else:
            parsed.append((spec, None))
    return parsed


def resolve_model_version(
    repo_id: str,
    model_version: str | None,
    *,
    builtin_repos: set[str] | frozenset[str],
) -> str | None:
    """Pick the HF version to download for ``repo_id``.

    An explicit ``model_version`` always wins. Otherwise built-in repos track
    the torq-examples release version (repo-root ``VERSION`` file), and custom
    repos fall back to their latest (HEAD) revision, untracked.

    Raises:
        DownloadError: No explicit version was given for a built-in repo and
            the torq-examples version cannot be determined.
    """
    if model_version is not None:
        return model_version
    if repo_id not in builtin_repos:
        logger.info(
            "Custom repo %s has no default version; using its latest (HEAD) revision.",
            repo_id,
        )
        return None
    version = examples_version()
    if version is None:
        raise DownloadError(
            "Cannot determine the torq-examples version (missing or empty VERSION "
            f"file at the repo root) for repo {repo_id}; pass --model-version to "
            "select a model version explicitly."
        )
    return version
