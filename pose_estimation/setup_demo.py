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

logger = logging.getLogger("pose_estimation.setup")

_POSE_HF_REPO: Final[str] = "Synaptics/yolov8-pose-nano-320-int8-torq"
_MODEL_FILENAME: Final[str] = "yolo_pose.vmfb"
_SAMPLES_PREFIX: Final[str] = "samples/"


def _hf_file_exists(repo_id: str, filename: str) -> bool:
    from huggingface_hub import HfApi

    return HfApi().file_exists(repo_id=repo_id, filename=filename)


def _list_sample_files(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi

    return [
        path for path in HfApi().list_repo_files(repo_id=repo_id)
        if path.startswith(_SAMPLES_PREFIX) and not path.endswith("/")
    ]


def _has_pose_estimation_files(model_dir: Path) -> bool:
    """Return True when the required pose model file exists."""
    return (model_dir / _MODEL_FILENAME).exists()


def _download_pose_estimation(repo_id: str, base_dir: Path) -> list[str]:
    """Download pose assets; return the manifest file list."""
    manifest_files = []

    if not _hf_file_exists(repo_id, _MODEL_FILENAME):
        raise FileNotFoundError(f"Required file '{_MODEL_FILENAME}' not found in {repo_id}")

    download_from_hf(repo_id, _MODEL_FILENAME, base_dir=base_dir)
    manifest_files.append(_MODEL_FILENAME)

    for sample_file in _list_sample_files(repo_id):
        download_from_hf(repo_id, sample_file, base_dir=base_dir)
        manifest_files.append(sample_file)

    return manifest_files


def _refresh_pose_estimation(repo_id: str, model_dir: Path, base_dir: Path) -> ModelStatus:
    files_present = verify_manifest(model_dir) and _has_pose_estimation_files(model_dir)
    revision = get_hf_revision(repo_id)
    return ensure_model(
        model_dir,
        repo_id,
        files_present=files_present,
        revision=revision,
        download=lambda: _download_pose_estimation(repo_id, base_dir),
    )


def ensure_pose_estimation_models(model_dir: str | Path, *, refresh: bool = True) -> None:
    """Verify/refresh pose assets before inference.

    Reads the repo id from the local manifest and applies the same revision
    check as setup. When ``refresh`` is ``False`` the check is skipped entirely
    for offline/airgapped runs. Refresh failures are logged, not raised, so
    inference can still proceed using local files.
    """
    model_dir = Path(model_dir)

    if not refresh:
        return

    manifest = read_manifest(model_dir)
    repo_id = manifest.get("repo_id") if manifest else None
    if not repo_id:
        logger.warning(
            "No manifest in %s; cannot verify pose estimation asset freshness. "
            "Run `python setup_demos.py pose_estimation` if inference fails.",
            model_dir,
        )
        return

    try:
        _refresh_pose_estimation(repo_id, model_dir, base_dir_for(model_dir, repo_id))
    except Exception as exc:
        logger.warning(
            "Could not refresh pose estimation assets from %s (%s); using local files.",
            repo_id,
            exc,
        )


def setup_pose_estimation():
    repo_id = _POSE_HF_REPO
    base_dir = default_models_dir()
    model_dir = base_dir / repo_id

    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("Setting up pose estimation demo from %s", repo_id)

    try:
        status = _refresh_pose_estimation(repo_id, model_dir, base_dir)
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

    parser = argparse.ArgumentParser(description="Verify pose estimation demo dependencies.")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_pose_estimation()
    except (DownloadError, MissingRequirementsError, ValueError) as exc:
        logger.error("%s", exc)
        if exc.__cause__:
            logger.error("Caused by: %s", exc.__cause__)
        sys.exit(1)
