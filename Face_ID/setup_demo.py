"""Set up Face ID VMFB models from Hugging Face."""

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
FACE_ID_MODEL_FILES: Final[tuple[str, ...]] = (
    "face_detection.vmfb",
    "face_keypoint_static.vmfb",
    "face_embeddings_static.vmfb",
)


def _download_face_id(repo_id: str, base_dir: Path) -> list[str]:
    """Download all required Face ID model files."""
    for filename in FACE_ID_MODEL_FILES:
        download_from_hf(repo_id, filename, base_dir=base_dir)
    return list(FACE_ID_MODEL_FILES)


def setup_face_id(*, base_dir: str | Path | None = None) -> Path:
    """Download or refresh Face ID VMFBs and return their local directory."""
    check_requirements(Path(__file__).parent / "requirements.txt")
    if base_dir is None:
        base_dir = default_models_dir()
    base_dir = Path(base_dir)
    model_dir = base_dir / FACE_ID_HF_REPO
    files_present = verify_manifest(model_dir) and all(
        (model_dir / filename).exists() for filename in FACE_ID_MODEL_FILES
    )
    revision = get_hf_revision(FACE_ID_HF_REPO)
    try:
        status = ensure_model(
            model_dir,
            FACE_ID_HF_REPO,
            files_present=files_present,
            revision=revision,
            download=lambda: _download_face_id(FACE_ID_HF_REPO, base_dir),
        )
    except Exception as exc:
        raise DownloadError(f"Unable to download Face ID files from {FACE_ID_HF_REPO}") from exc

    if status is ModelStatus.UP_TO_DATE:
        logger.info("Using local Face ID models from %s", model_dir)
    else:
        logger.info("Face ID models ready at %s", model_dir)
    return model_dir


if __name__ == "__main__":
    import argparse

    from utils.log import add_logging_args, configure_logging

    parser = argparse.ArgumentParser(description="Set up Face ID VMFB models.")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)
    setup_face_id()
