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

logger = logging.getLogger("object_detection.setup")

_HF_REPO_ID: Final[str] = "Synaptics/yolov8-od-nano-320-int8-torq"
_MODEL_FILENAME: Final[str] = "yolo_8n_2.0.0_npu.vmfb"
_LABELS_FILENAME: Final[str] = "labels.json"
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


def _has_object_detection_files(model_dir: Path) -> bool:
    return (model_dir / _MODEL_FILENAME).exists() and (model_dir / _LABELS_FILENAME).exists()


def _download_object_detection(repo_id: str, base_dir: Path) -> list[str]:
    """Download object detection assets; return the manifest file list."""
    manifest_files = []

    for filename in (_MODEL_FILENAME, _LABELS_FILENAME):
        if not _hf_file_exists(repo_id, filename):
            raise FileNotFoundError(f"Required file '{filename}' not found in {repo_id}")
        download_from_hf(repo_id, filename, base_dir=base_dir)
        manifest_files.append(filename)

    for sample_file in _list_sample_files(repo_id):
        download_from_hf(repo_id, sample_file, base_dir=base_dir)
        manifest_files.append(sample_file)

    return manifest_files


def _refresh_object_detection(repo_id: str, model_dir: Path, base_dir: Path) -> ModelStatus:
    files_present = verify_manifest(model_dir) and _has_object_detection_files(model_dir)
    revision = get_hf_revision(repo_id)
    return ensure_model(
        model_dir,
        repo_id,
        files_present=files_present,
        revision=revision,
        download=lambda: _download_object_detection(repo_id, base_dir),
    )


def ensure_object_detection_models(model_dir: str | Path, *, refresh: bool = True) -> None:
    """Verify/refresh object detection assets before inference.

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
            "No manifest in %s; cannot verify object detection asset freshness. "
            "Run `python setup_demos.py object_detection` if inference fails.",
            model_dir,
        )
        return

    try:
        _refresh_object_detection(repo_id, model_dir, base_dir_for(model_dir, repo_id))
    except Exception as e:
        logger.warning(
            "Could not refresh object detection assets from %s (%s); using local files.",
            repo_id,
            e,
        )


def setup_object_detection():
    repo_id = _HF_REPO_ID
    base_dir = default_models_dir()
    model_dir = base_dir / repo_id

    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("Setting up object detection demo from %s", repo_id)

    try:
        status = _refresh_object_detection(repo_id, model_dir, base_dir)
    except Exception as e:
        raise DownloadError(f"Unable to download object detection assets from {repo_id}") from e

    if status is ModelStatus.UP_TO_DATE:
        logger.info("Using local object detection assets from %s", model_dir)
    else:
        logger.info("Downloaded object detection assets to %s", model_dir)


if __name__ == "__main__":
    import argparse
    import sys

    from utils.log import add_logging_args, configure_logging

    parser = argparse.ArgumentParser(description="Verify object detection demo dependencies.")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_object_detection()
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
