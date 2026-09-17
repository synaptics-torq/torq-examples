# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.download import download_from_hf, hf_file_exists, list_hf_files
from utils.model_setup import demo_main, download_models, ensure_demo_models, setup_demo

logger = logging.getLogger("object_detection.setup")

_OD_HF_REPO_MAP: Final[dict[str, str]] = {
    "nano": "Synaptics/yolov8-od-nano-320-int8-torq",
}
_DEFAULT_MODELS: Final[list[str]] = ["nano"]
_MODEL_FILENAME: Final[str] = "yolo_od.vmfb"
_LABELS_FILENAME: Final[str] = "labels.json"
_SAMPLES_PREFIX: Final[str] = "samples/"


def _has_object_detection_files(model_dir: Path) -> bool:
    return (model_dir / _MODEL_FILENAME).exists() and (model_dir / _LABELS_FILENAME).exists()


def _download_object_detection(
    repo_id: str,
    base_dir: Path,
    *,
    revision: str | None = None,
) -> list[str]:
    """Download object detection assets; return the manifest file list."""
    manifest_files = []

    for filename in (_MODEL_FILENAME, _LABELS_FILENAME):
        if not hf_file_exists(repo_id, filename, revision=revision):
            raise FileNotFoundError(f"Required file '{filename}' not found in {repo_id}")
        download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
        manifest_files.append(filename)

    for sample_file in list_hf_files(repo_id, prefix=_SAMPLES_PREFIX, revision=revision):
        download_from_hf(repo_id, sample_file, base_dir=base_dir, revision=revision)
        manifest_files.append(sample_file)

    return manifest_files


def download_object_detection(
    models: list[str] | None = None,
    *,
    base_dir: str | Path | None = None,
    model_version: str | None = None,
    no_update: bool = False,
) -> dict[str, Path]:
    """Download/refresh the given object detection models; return ``{name: model_dir}``.

    Version selection, ``name:version`` specs and ``no_update`` are handled by
    :func:`utils.model_setup.download_models`. Unlike
    :func:`setup_object_detection`, this does not check demo requirements, so
    it can be reused by other projects that manage their own environment and
    models dir.
    """
    if models is None:
        models = _DEFAULT_MODELS
    return download_models(
        _OD_HF_REPO_MAP,
        models,
        files_present=_has_object_detection_files,
        download=_download_object_detection,
        label="object_detection",
        base_dir=base_dir,
        model_version=model_version,
        no_update=no_update,
    )


def ensure_object_detection_models(
    model_dir: str | Path,
    *,
    refresh: bool = True,
) -> None:
    """Verify/refresh object detection assets in ``model_dir`` before inference.

    Delegates to :func:`utils.model_setup.ensure_demo_models`, which re-syncs
    the local copy to the version recorded in its manifest. Refresh failures
    are logged, not raised, so inference can still proceed using local files.
    """
    ensure_demo_models(
        model_dir,
        refresh=refresh,
        demo_name="object_detection",
        files_present=_has_object_detection_files,
        download=_download_object_detection,
    )


def setup_object_detection(
    models: list[str] | None = None,
    model_version: str | None = None,
    no_update: bool = False,
):
    """Set up the object detection demo (requirements, then assets)."""
    if models is None:
        models = _DEFAULT_MODELS
    return setup_demo(
        _OD_HF_REPO_MAP,
        models,
        demo_name="object_detection",
        requirements=Path(__file__).parent / "requirements.txt",
        files_present=_has_object_detection_files,
        download=_download_object_detection,
        model_version=model_version,
        no_update=no_update,
    )


if __name__ == "__main__":
    demo_main(
        setup_object_detection,
        description="Set up the object detection demo.",
        default_models=_DEFAULT_MODELS,
        repo_map=_OD_HF_REPO_MAP,
    )
