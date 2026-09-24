# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.download import default_models_dir, download_from_hf, hf_file_exists, list_hf_files
from utils.model_setup import demo_main, download_models, ensure_demo_models, setup_demo

logger = logging.getLogger("object_detection.setup")

_OD_HF_REPO_MAP: Final[dict[str, str]] = {
    "nano": "Synaptics/yolov8-od-nano-320-int8-torq",
}
# The YOLO26 repo ships both model sizes in one repo and has no version tags
# yet, so it stays out of the repo map and is downloaded at HEAD, untracked,
# like a custom repo. It is part of the default setup.
_YOLO26_REPO_ID: Final[str] = "Synaptics/yolov26n_od"
_DEFAULT_MODELS: Final[list[str]] = ["nano", _YOLO26_REPO_ID]
_MODEL_FILENAME: Final[str] = "yolo_od.vmfb"
_YOLO26_VARIANT_FILES: Final[dict[str, str]] = {
    "yolo26n": "yolo26n_npu.vmfb",
    "yolo26s": "yolo26s_npu.vmfb",
}
# Model VMFBs shipped by each repo (one repo may carry several sizes).
_MODEL_FILENAMES: Final[dict[str, tuple[str, ...]]] = {
    _OD_HF_REPO_MAP["nano"]: (_MODEL_FILENAME,),
    _YOLO26_REPO_ID: tuple(_YOLO26_VARIANT_FILES.values()),
}
_LABELS_FILENAME: Final[str] = "labels.json"
_SAMPLES_PREFIX: Final[str] = "samples/"
_SAMPLE_SUFFIXES: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".mp4")

# (HF repo id, model file) that setup downloads for each inference
# ``--variant``; the default ``--model`` for the demo's infer scripts.
VARIANT_MODEL_FILES: Final[dict[str, tuple[str, str]]] = {
    "yolov8": (_OD_HF_REPO_MAP["nano"], _MODEL_FILENAME),
    **{
        variant: (_YOLO26_REPO_ID, filename)
        for variant, filename in _YOLO26_VARIANT_FILES.items()
    },
}


def default_model_path(
    variant: str = "yolov8",
    base_dir: str | Path | None = None,
) -> Path:
    """The model file setup downloads for ``variant`` (default: ``yolov8``)."""
    repo_id, filename = VARIANT_MODEL_FILES[variant]
    if base_dir is None:
        base_dir = default_models_dir()
    return Path(base_dir) / repo_id / filename


# Model dirs are laid out as <base>/<repo id>, so the dir's last path segment
# identifies which repo it came from.
_REPO_BY_DIR_TAIL: Final[dict[str, str]] = {
    repo_id.split("/")[-1]: repo_id for repo_id in _MODEL_FILENAMES
}


def _model_filenames(repo_id: str | None) -> tuple[str, ...]:
    """Model VMFBs shipped by ``repo_id`` (several sizes for one repo)."""
    return _MODEL_FILENAMES.get(repo_id or "", (_MODEL_FILENAME,))


def _list_sample_files(repo_id: str, revision: str | None) -> list[str]:
    """Sample media under ``samples/``, or at the repo root when there is no ``samples/`` dir."""
    files = list_hf_files(repo_id, revision=revision)
    samples = [path for path in files if path.startswith(_SAMPLES_PREFIX)]
    if samples:
        return samples
    return [path for path in files if "/" not in path and path.lower().endswith(_SAMPLE_SUFFIXES)]


def _has_object_detection_files(model_dir: Path) -> bool:
    repo_id = _REPO_BY_DIR_TAIL.get(model_dir.name)
    files = (*_model_filenames(repo_id), _LABELS_FILENAME)
    return all((model_dir / filename).exists() for filename in files)


def _download_object_detection(
    repo_id: str,
    base_dir: Path,
    *,
    revision: str | None = None,
) -> list[str]:
    """Download object detection assets; return the manifest file list."""
    manifest_files = []

    for filename in (*_model_filenames(repo_id), _LABELS_FILENAME):
        if not hf_file_exists(repo_id, filename, revision=revision):
            raise FileNotFoundError(f"Required file '{filename}' not found in {repo_id}")
        download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
        manifest_files.append(filename)

    for sample_file in _list_sample_files(repo_id, revision):
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
    :func:`utils.model_setup.download_models`. The default covers both the
    tracked YOLOv8n model and the (untracked) YOLO26 nano/small repo. Unlike
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
