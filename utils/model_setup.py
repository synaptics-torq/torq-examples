# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared orchestration for demo model setup and inference-time refresh.

Each demo's ``setup_demo.py`` keeps only what is demo-specific:

- a ``repo_map`` of model names to HF repo ids,
- a ``files_present(model_dir) -> bool`` integrity check for the assets the
  demo needs,
- a ``download(repo_id, base_dir, *, revision) -> list[str]`` function that
  fetches the missing assets and returns the manifest file list.

Everything else: ``name:version`` spec parsing, default version resolution
against the torq-examples ``VERSION`` file, the per-model refresh/download
loop, the manifest-driven inference-time refresh, and the
``--model-version``/``--no-update`` command line, lives here.
"""

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

from utils.deps import MissingRequirementsError, check_requirements
from utils.download import (
    DownloadError,
    ModelStatus,
    base_dir_for,
    default_models_dir,
    ensure_model,
    get_hf_revision,
    read_manifest,
    resolve_repo_id,
    verify_manifest,
)
from utils.log import add_logging_args, configure_logging
from utils.version import parse_model_specs, resolve_model_version

__all__ = [
    "refresh_model",
    "download_models",
    "ensure_demo_models",
    "setup_demo",
    "demo_main",
]

logger = logging.getLogger(__name__)

# Demo-specific hooks: ``files_present`` checks whether the demo's assets are
# complete in a model dir; ``download`` fetches whatever is missing and returns
# the filenames to record in the manifest. ``tracked_files_present`` (optional)
# overrides how a *tracked* copy is checked for completeness.
FilesPresent = Callable[[Path], bool]
DownloadFn = Callable[..., list[str]]  # (repo_id, base_dir, *, revision)

_MODEL_VERSION_HELP: Final[str] = (
    "Model version tag to download for every model without its own "
    "'name:version' (default: the torq-examples version for built-in "
    "repos, the repo's latest revision for custom repos). A pinned "
    "version is kept in sync with its own tag but never upgraded."
)
_NO_UPDATE_HELP: Final[str] = (
    "Download without tracking: no .manifest.json is written, so the "
    "models are never checked for updates or refreshed (at your own risk)."
)


def refresh_model(
    repo_id: str,
    model_dir: str | Path,
    base_dir: str | Path,
    *,
    version: str | None,
    record: bool = True,
    files_present: FilesPresent,
    download: DownloadFn,
    tracked_files_present: FilesPresent | None = None,
) -> ModelStatus:
    """Check ``model_dir`` and refresh it so it tracks ``version``.

    Tracked copies (``record=True``) are checked for completeness against
    ``tracked_files_present`` (or, by default, ``verify_manifest`` and
    ``files_present``), and ``get_hf_revision`` resolves the commit
    ``version`` currently points to so an upstream tag move is detected.
    Untracked copies (``--no-update``, ``record=False``) are checked against
    ``files_present`` only and never resolve or record a version.
    """
    model_dir = Path(model_dir)
    if record:
        if tracked_files_present is None:
            present = verify_manifest(model_dir) and files_present(model_dir)
        else:
            present = tracked_files_present(model_dir)
        revision = None
        if version is not None:
            revision = get_hf_revision(repo_id, revision=version)
    else:
        present = files_present(model_dir)
        revision = None
    return ensure_model(
        model_dir,
        repo_id,
        files_present=present,
        version=version if record else None,
        revision=revision,
        download=lambda: download(repo_id, base_dir, revision=version),
        record=record,
    )


def download_models(
    repo_map: dict[str, str],
    models: list[str],
    *,
    files_present: FilesPresent,
    download: DownloadFn,
    label: str,
    base_dir: str | Path | None = None,
    model_version: str | None = None,
    no_update: bool = False,
    tracked_files_present: FilesPresent | None = None,
) -> dict[str, Path]:
    """Download/refresh ``models``; return ``{name: model_dir}``.

    ``models`` entries may be built-in names, raw HF repo ids, or
    ``name:version`` to pin a specific version for that model. ``model_version``
    applies to every model without its own ``:version``; built-in repos (the
    values of ``repo_map``) default to the torq-examples version and custom
    repos to their latest (HEAD) revision. ``no_update=True`` downloads without
    writing a manifest, so the models are never tracked or refreshed (at your
    own risk).

    ``label`` only appears in log messages and error text.
    """
    if base_dir is None:
        base_dir = default_models_dir()
    base_dir = Path(base_dir)
    builtin_repos = frozenset(repo_map.values())

    logger.info("Resolving %s models: [%s]", label, ", ".join(models))
    result: dict[str, Path] = {}
    for name, spec_version in parse_model_specs(models):
        repo_id = resolve_repo_id(name, repo_map)
        version = resolve_model_version(
            repo_id, spec_version or model_version, builtin_repos=builtin_repos
        )
        model_dir = base_dir / repo_id
        try:
            refresh_model(
                repo_id,
                model_dir,
                base_dir,
                version=version,
                record=not no_update,
                files_present=files_present,
                download=download,
                tracked_files_present=tracked_files_present,
            )
        except Exception as exc:
            raise DownloadError(f"Unable to download {label} files from {repo_id}") from exc
        result[name] = model_dir
        logger.info(
            "%s model files for version %s ready at '%s'",
            label, version or "latest", model_dir,
        )
    return result


def ensure_demo_models(
    model_dir: str | Path,
    *,
    refresh: bool = True,
    demo_name: str,
    files_present: FilesPresent,
    download: DownloadFn,
    tracked_files_present: FilesPresent | None = None,
) -> None:
    """Verify/refresh the demo's models in ``model_dir`` before inference.

    Re-syncs the local copy to the version recorded in its manifest (never to
    a newer one). Untracked models (``--no-update``) have no manifest and are
    left as-is. When ``refresh`` is ``False`` the check is skipped entirely
    (offline/airgapped runs). Refresh failures are logged, not raised, so
    inference can still proceed on whatever is available locally.
    """
    model_dir = Path(model_dir)
    if not refresh:
        return
    manifest = read_manifest(model_dir)
    if manifest is None:
        logger.info(
            "No manifest in %s; the model is untracked, so skipping the freshness check.",
            model_dir,
        )
        return
    repo_id = manifest.get("repo_id")
    if not repo_id:
        logger.warning(
            "Manifest in %s has no repo_id; cannot verify model freshness. "
            "Run `python setup_demos.py %s` if inference fails.",
            model_dir, demo_name,
        )
        return
    base_dir = base_dir_for(model_dir, repo_id)
    if base_dir is None:
        logger.warning(
            "%s is not laid out as <models dir>/%s; skipping the freshness check "
            "so a refresh cannot fetch a second copy elsewhere. "
            "Run `python setup_demos.py %s` to manage models.",
            model_dir, repo_id, demo_name,
        )
        return
    try:
        refresh_model(
            repo_id,
            model_dir,
            base_dir,
            version=manifest.get("version"),
            files_present=files_present,
            download=download,
            tracked_files_present=tracked_files_present,
        )
    except Exception as e:
        logger.warning(
            "Could not refresh models from %s (%s); using local files.", repo_id, e
        )


def setup_demo(
    repo_map: dict[str, str],
    models: list[str],
    *,
    demo_name: str,
    requirements: str | Path,
    files_present: FilesPresent,
    download: DownloadFn,
    base_dir: str | Path | None = None,
    model_version: str | None = None,
    no_update: bool = False,
    tracked_files_present: FilesPresent | None = None,
) -> dict[str, Path]:
    """Set up a demo: check its requirements, then download/refresh its models.

    Requirements are checked first so a missing dependency fails fast instead
    of after a long download. Returns ``{name: model_dir}`` for the requested
    ``models``; version semantics are those of :func:`download_models`.
    """
    check_requirements(requirements)
    logger.info("Setting up %s demo with models: [%s]", demo_name, ", ".join(models))
    result = download_models(
        repo_map,
        models,
        files_present=files_present,
        download=download,
        label=demo_name,
        base_dir=base_dir,
        model_version=model_version,
        no_update=no_update,
        tracked_files_present=tracked_files_present,
    )
    logger.info("%s setup complete.", demo_name)
    return result


def demo_main(
    setup_fn: Callable[..., dict[str, Path]],
    *,
    description: str,
    default_models: list[str],
    repo_map: dict[str, str],
) -> None:
    """Shared command-line main for the demos' ``setup_demo.py`` scripts.

    Adds the ``models`` positional (built-in names, raw HF repo ids, or
    ``name:version`` specs), ``--model-version``, ``--no-update`` and the
    logging args, then calls
    ``setup_fn(models, model_version=..., no_update=...)``.
    """
    available = ", ".join(f"'{name}' ({repo})" for name, repo in repo_map.items())
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "models", nargs="*", default=list(default_models),
        help=f"Model name or HF repo ID, optionally 'name:version' to pin a "
             f"specific model version for that model. Built-in: [{available}] (default: %(default)s)",
    )
    parser.add_argument("--model-version", default=None, help=_MODEL_VERSION_HELP)
    parser.add_argument("--no-update", action="store_true", help=_NO_UPDATE_HELP)
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_fn(args.models, model_version=args.model_version, no_update=args.no_update)
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
