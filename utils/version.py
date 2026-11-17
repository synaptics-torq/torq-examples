# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""torq-examples versioning helpers.

The ``VERSION`` file at the repo root records which torq-examples release this
checkout corresponds to. The version tracks the torq-compiler/runtime release
the demos are tested against, and the built-in model repos on Hugging Face are
tagged the same way: a checkout of examples ``2.2.0`` downloads (and maintains)
the models at tag ``v2.2.0`` (or, when the repo has no such tag, the newest
tag of the same release series, ``v2.2.*``).

Model compatibility is judged on ``major.minor`` only: upstream guarantees
that patch releases do not break compatibility, so torq-examples ``2.3.1``
runs models tagged ``v2.3.*``.

The file must be bumped on every release of torq-examples.
"""

import logging
import re
from pathlib import Path
from typing import Final

from utils.download import DownloadError, list_hf_tags

__all__ = [
    "compatible_version",
    "examples_version",
    "follow_examples_series",
    "parse_model_specs",
    "release_series",
    "resolve_model_version",
    "warn_if_model_incompatible",
]

logger = logging.getLogger(__name__)

_VERSION_FILE: Final[Path] = Path(__file__).resolve().parent.parent / "VERSION"
_VERSION_RE: Final[re.Pattern] = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")


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


def release_series(version: str | None) -> tuple[str, str] | None:
    """Return the ``(major, minor)`` of *version* (``v2.3.4`` -> ``("2", "3")``).

    Compatibility is judged on this pair only. ``None`` for versions that do
    not start with ``X.Y.Z`` (branches like ``q4``, ``None``).
    """
    match = _VERSION_RE.match(version) if version else None
    return match.group(1, 2) if match else None


def compatible_version(repo_id: str, version: str) -> str | None:
    """Return *version* if *repo_id* has that tag, else the newest tag of its series.

    E.g. ``v2.3.4`` resolves to itself when tagged, or to ``v2.3.2`` when that
    is the newest ``v2.3.*`` tag in the repo. ``None`` when *version* has no
    series, the tags cannot be listed (Hub unreachable), or no tag of the
    series exists.
    """
    series = release_series(version)
    tags = list_hf_tags(repo_id) if series else None
    if tags and version in tags:
        return version
    candidates = [
        (int(m.group(3)), tag)
        for tag in tags or []
        if (m := _VERSION_RE.fullmatch(tag)) and m.group(1, 2) == series
    ]
    return max(candidates)[1] if candidates else None


def follow_examples_series(repo_id: str, version: str | None) -> str | None:
    """Move a default-tracked model *version* to the torq-examples release series.

    Returns *version* unchanged when it already is in the ``VERSION`` file's
    ``major.minor`` series (or either has no series). Otherwise returns the
    :func:`compatible_version` of ``VERSION``, or keeps *version* with a
    warning when no tag of the new series exists upstream yet (or the Hub is
    unreachable).
    """
    current = examples_version()
    series = release_series(version)
    if series is None or release_series(current) in (None, series):
        return version
    target = compatible_version(repo_id, current)
    if target is None:
        logger.warning(
            "torq-examples is at %s but %s has no compatible model version yet "
            "(or Hugging Face is unreachable); keeping %s, which may not work "
            "with this release.",
            current, repo_id, version,
        )
        return version
    logger.info(
        "torq-examples moved to %s; updating %s from %s to %s.",
        current, repo_id, version, target,
    )
    return target


def warn_if_model_incompatible(repo_id: str, version: str | None) -> None:
    """Warn when model *version* is not in the ``VERSION`` file's ``major.minor`` series."""
    current = examples_version()
    series = release_series(version)
    if series is not None and series != release_series(current):
        logger.warning(
            "%s is at model version %s, which is not compatible with torq-examples "
            "%s (only matching major.minor versions are); it may fail to load or run.",
            repo_id, version, current,
        )


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
    the torq-examples release version (repo-root ``VERSION`` file), or the
    newest tag of its series when the repo lacks that exact tag (see
    :func:`compatible_version`), and custom
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
    return compatible_version(repo_id, version) or version
