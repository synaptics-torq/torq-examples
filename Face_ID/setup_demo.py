"""Download Face ID VMFB models from Hugging Face."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from utils.deps import check_requirements
from utils.download import (
    DownloadError,
    ModelStatus,
    default_models_dir,
    download_from_hf,
    ensure_model,
    get_hf_revision,
    verify_manifest,
)

logger = logging.getLogger("Face_ID.setup")

FACE_ID_HF_REPO: Final[str] = "Synaptics/face-id-torq"
FULL_FACE_ID_MODEL_FILES: Final[tuple[str, ...]] = (
    "face_detection.vmfb",
    "face_keypoint_static.vmfb",
    "face_embeddings_static.vmfb",
)
FACE_DETECTOR_FILES: Final[tuple[str, ...]] = (
    "face_detection.vmfb",
    "face.jpg",
)


def local_face_id_model_path(
    model: str | None = None,
    *,
    base_dir: str | Path | None = None,
) -> Path | None:
    """Return the local detector VMFB path to default ``--model`` to.

    Looks for ``face_detection.vmfb`` in the repo's model directory; with no
    ``model`` that is the built-in repo, an explicit value is a raw HF repo
    id.
    """
    if base_dir is None:
        base_dir = default_models_dir()
    repo_id = model if model is not None else FACE_ID_HF_REPO
    model_dir = Path(base_dir) / repo_id
    if (model_dir / "face_detection.vmfb").exists():
        return model_dir / "face_detection.vmfb"
    return None


def _download_face_id_files(
    repo_id: str,
    base_dir: Path,
    model_files: tuple[str, ...],
) -> list[str]:
    for filename in model_files:
        download_from_hf(repo_id, filename, base_dir=base_dir)
    return list(model_files)


def _download_models(
    model_files: tuple[str, ...],
    *,
    base_dir: str | Path | None = None,
) -> Path:
    """Download or refresh the requested Face ID models."""
    if base_dir is None:
        base_dir = default_models_dir()
    base_dir = Path(base_dir)
    model_dir = base_dir / FACE_ID_HF_REPO
    files_present = verify_manifest(model_dir) and all(
        (model_dir / filename).exists() for filename in model_files
    )
    try:
        status = ensure_model(
            model_dir,
            FACE_ID_HF_REPO,
            files_present=files_present,
            revision=get_hf_revision(FACE_ID_HF_REPO),
            download=lambda: _download_face_id_files(FACE_ID_HF_REPO, base_dir, model_files),
        )
    except Exception as exc:
        raise DownloadError(f"Unable to download Face ID files from {FACE_ID_HF_REPO}") from exc

    logger.info(
        "%s Face ID models at %s",
        "Using existing" if status is ModelStatus.UP_TO_DATE else "Prepared",
        model_dir,
    )
    return model_dir


def download_face_id(*, base_dir: str | Path | None = None) -> Path:
    """Download all models required by the SL2610 Face ID application."""
    return _download_models(FULL_FACE_ID_MODEL_FILES, base_dir=base_dir)


def download_face_detector(*, base_dir: str | Path | None = None) -> Path:
    """Download only the detector model required by this file-based example."""
    return _download_models(FACE_DETECTOR_FILES, base_dir=base_dir)


def setup_face_id(*, base_dir: str | Path | None = None) -> Path:
    """Set up the file-based Torq Face ID example and its models."""
    check_requirements(Path(__file__).parent / "requirements.txt")
    return download_face_detector(base_dir=base_dir)


if __name__ == "__main__":
    setup_face_id()
