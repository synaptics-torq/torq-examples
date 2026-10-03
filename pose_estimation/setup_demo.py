# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.download import default_models_dir, download_from_hf, hf_file_exists, list_hf_files, resolve_repo_id
from utils.model_setup import demo_main, download_models, ensure_demo_models, setup_demo

logger = logging.getLogger("pose_estimation.setup")

_POSE_HF_REPO: Final[str] = "Synaptics/yolov8-pose-nano-320-int8-torq"
_POSE_HF_REPO_MAP: Final[dict[str, str]] = {"nano": _POSE_HF_REPO}
_DEFAULT_MODELS: Final[list[str]] = ["nano"]
_MODEL_FILENAME: Final[str] = "yolo_pose.vmfb"
_SAMPLES_PREFIX: Final[str] = "samples/"


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

    if not hf_file_exists(repo_id, _MODEL_FILENAME, revision=revision):
        raise FileNotFoundError(f"Required file '{_MODEL_FILENAME}' not found in {repo_id}")

    download_from_hf(repo_id, _MODEL_FILENAME, base_dir=base_dir, revision=revision)
    manifest_files.append(_MODEL_FILENAME)

    for sample_file in list_hf_files(repo_id, prefix=_SAMPLES_PREFIX, revision=revision):
        download_from_hf(repo_id, sample_file, base_dir=base_dir, revision=revision)
        manifest_files.append(sample_file)

    return manifest_files


def download_pose_estimation(
    models: list[str] | None = None,
    *,
    base_dir: str | Path | None = None,
    model_version: str | None = None,
    no_update: bool = False,
) -> dict[str, Path]:
    """Download/refresh the given pose estimation models; return ``{name: model_dir}``.

    Version selection, ``name:version`` specs and ``no_update`` are handled by
    :func:`utils.model_setup.download_models`. Unlike
    :func:`setup_pose_estimation`, this does not check demo requirements, so
    it can be reused by other projects that manage their own environment and
    models dir.
    """
    if models is None:
        models = _DEFAULT_MODELS
    return download_models(
        _POSE_HF_REPO_MAP,
        models,
        files_present=_has_pose_estimation_files,
        download=_download_pose_estimation,
        label="pose_estimation",
        base_dir=base_dir,
        model_version=model_version,
        no_update=no_update,
    )


def local_pose_model_path(
    model: str | None = None,
    *,
    base_dir: str | Path | None = None,
) -> Path | None:
    """Return the local pose model VMFB path to default ``--model`` to.

    Looks for the model file in the repo's model directory; with no ``model``,
    tries the built-in repos in repo-map order.
    """
    if base_dir is None:
        base_dir = default_models_dir()
    if model is not None:
        repo_ids = [resolve_repo_id(model, _POSE_HF_REPO_MAP)]
    else:
        repo_ids = list(dict.fromkeys(_POSE_HF_REPO_MAP.values()))
    for repo_id in repo_ids:
        model_dir = Path(base_dir) / repo_id
        if _has_pose_estimation_files(model_dir):
            return model_dir / _MODEL_FILENAME
    return None


def ensure_pose_estimation_models(
    model_dir: str | Path,
    *,
    refresh: bool = True,
) -> None:
    """Verify/refresh pose assets in ``model_dir`` before inference.

    Delegates to :func:`utils.model_setup.ensure_demo_models`, which re-syncs
    the local copy to the version recorded in its manifest. Refresh failures
    are logged, not raised, so inference can still proceed using local files.
    """
    ensure_demo_models(
        model_dir,
        refresh=refresh,
        demo_name="pose_estimation",
        files_present=_has_pose_estimation_files,
        download=_download_pose_estimation,
    )


def setup_pose_estimation(
    models: list[str] | None = None,
    model_version: str | None = None,
    no_update: bool = False,
):
    """Set up the pose estimation demo (requirements, then assets)."""
    if models is None:
        models = _DEFAULT_MODELS
    return setup_demo(
        _POSE_HF_REPO_MAP,
        models,
        demo_name="pose_estimation",
        requirements=Path(__file__).parent / "requirements.txt",
        files_present=_has_pose_estimation_files,
        download=_download_pose_estimation,
        model_version=model_version,
        no_update=no_update,
    )


if __name__ == "__main__":
    demo_main(
        setup_pose_estimation,
        description="Set up the pose estimation demo.",
        default_models=_DEFAULT_MODELS,
        repo_map=_POSE_HF_REPO_MAP,
    )
