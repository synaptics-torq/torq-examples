# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.deps import MissingRequirementsError, check_requirements
from utils.download import (
    DownloadError,
    ModelStatus,
    base_dir_for,
    default_models_dir,
    download_from_hf,
    ensure_model,
    get_hf_revision,
    read_manifest,
    verify_manifest,
)
from utils.version import resolve_model_version

logger = logging.getLogger("pose_estimation.setup")

_POSE_HF_REPO: Final[str] = "Synaptics/yolov8-pose-nano-320-int8-torq"
_BUILTIN_REPOS: Final[frozenset[str]] = frozenset({_POSE_HF_REPO})
_MODEL_FILENAME: Final[str] = "yolo_pose.vmfb"
_SAMPLES_PREFIX: Final[str] = "samples/"


def _hf_file_exists(repo_id: str, filename: str, *, revision: str | None = None) -> bool:
    from huggingface_hub import HfApi

    return HfApi().file_exists(repo_id=repo_id, filename=filename, revision=revision)


def _list_sample_files(repo_id: str, *, revision: str | None = None) -> list[str]:
    from huggingface_hub import HfApi

    return [
        path for path in HfApi().list_repo_files(repo_id=repo_id, revision=revision)
        if path.startswith(_SAMPLES_PREFIX) and not path.endswith("/")
    ]


def _has_pose_estimation_files(model_dir: Path) -> bool:
    """Return True when the required pose model file exists."""
    return (model_dir / _MODEL_FILENAME).exists()


def _download_pose_estimation(
    repo_id: str,
    base_dir: Path,
    *,
    revision: str | None = None,
) -> list[str]:
    """Download pose assets; return the manifest file list."""
    manifest_files = []

    if not _hf_file_exists(repo_id, _MODEL_FILENAME, revision=revision):
        raise FileNotFoundError(f"Required file '{_MODEL_FILENAME}' not found in {repo_id}")

    download_from_hf(repo_id, _MODEL_FILENAME, base_dir=base_dir, revision=revision)
    manifest_files.append(_MODEL_FILENAME)

    for sample_file in _list_sample_files(repo_id, revision=revision):
        download_from_hf(repo_id, sample_file, base_dir=base_dir, revision=revision)
        manifest_files.append(sample_file)

    return manifest_files


def _refresh_pose_estimation(
    repo_id: str,
    model_dir: Path,
    base_dir: Path,
    *,
    version: str | None,
    record: bool = True,
) -> ModelStatus:
    files_present = _has_pose_estimation_files(model_dir)
    revision = None
    if record:
        files_present = verify_manifest(model_dir) and files_present
        if version is not None:
            revision = get_hf_revision(repo_id, revision=version)
    return ensure_model(
        model_dir,
        repo_id,
        files_present=files_present,
        version=version if record else None,
        revision=revision,
        download=lambda: _download_pose_estimation(repo_id, base_dir, revision=version),
        record=record,
    )


def ensure_pose_estimation_models(
    model_dir: str | Path,
    *,
    refresh: bool = True,
) -> None:
    """Verify/refresh pose assets before inference.

    Re-syncs the local copy to the version recorded in its manifest (never to a
    newer one). Untracked models (``--no-update``) have no manifest and are
    left as-is. When ``refresh`` is ``False`` the check is skipped entirely
    for offline/airgapped runs. Refresh failures are logged, not raised, so
    inference can still proceed using local files.
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
            "Manifest in %s has no repo_id; cannot verify pose estimation asset freshness. "
            "Run `python setup_demos.py pose_estimation` if inference fails.",
            model_dir,
        )
        return
    base_dir = base_dir_for(model_dir, repo_id)
    if base_dir is None:
        logger.warning(
            "%s is not laid out as <models dir>/%s; skipping the freshness check "
            "so a refresh cannot fetch a second copy elsewhere. "
            "Run `python setup_demos.py pose_estimation` to manage assets.",
            model_dir,
            repo_id,
        )
        return

    try:
        _refresh_pose_estimation(
            repo_id,
            model_dir,
            base_dir,
            version=manifest.get("version"),
        )
    except Exception as e:
        logger.warning(
            "Could not refresh pose estimation assets from %s (%s); using local files.",
            repo_id,
            e,
        )


def setup_pose_estimation(
    model_version: str | None = None,
    no_update: bool = False,
):
    repo_id = _POSE_HF_REPO
    version = resolve_model_version(repo_id, model_version, builtin_repos=_BUILTIN_REPOS)
    base_dir = default_models_dir()
    model_dir = base_dir / repo_id

    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info(
        "Setting up pose estimation demo from %s (version=%s)",
        repo_id, version or "latest",
    )

    try:
        status = _refresh_pose_estimation(
            repo_id, model_dir, base_dir, version=version, record=not no_update
        )
    except Exception as exc:
        raise DownloadError(f"Unable to download pose estimation assets from {repo_id}") from exc

    if status is ModelStatus.UP_TO_DATE:
        logger.info("Using local pose estimation assets from %s", model_dir)
    else:
        logger.info("Downloaded pose estimation assets to %s", model_dir)


if __name__ == "__main__":
    import argparse
    import sys

    from utils.log import add_logging_args, configure_logging

    parser = argparse.ArgumentParser(description="Set up the pose estimation demo.")
    add_logging_args(parser)
    parser.add_argument(
        "--model-version",
        default=None,
        help=(
            "Model version tag to download (default: the torq-examples version for "
            "built-in repos, the repo's latest revision for custom repos). A "
            "pinned version is kept in sync with its own tag but never upgraded."
        ),
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help=(
            "Download without tracking: no .manifest.json is written, so the model "
            "is never checked for updates or refreshed (at your own risk)."
        ),
    )
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_pose_estimation(model_version=args.model_version, no_update=args.no_update)
    except (DownloadError, MissingRequirementsError, ValueError) as exc:
        logger.error("%s", exc)
        if exc.__cause__:
            logger.error("Caused by: %s", exc.__cause__)
        sys.exit(1)
