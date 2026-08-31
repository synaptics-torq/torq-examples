# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.deps import MissingRequirementsError, check_requirements
from utils.download import (DownloadError, ModelStatus, base_dir_for, default_models_dir, download_from_hf,
                            ensure_model, get_hf_revision, read_manifest, verify_manifest)

logger = logging.getLogger("ppocr.setup")

_PPOCR_HF_REPO: Final[str] = "Synaptics/paddle-paddle-tiny"

# Detection runs at one of two static shapes (800x608 portrait, 640x384 for 16:9
# sources); recognition uses one vmfb per width bucket.
_DET_HWS: Final[tuple[tuple[int, int], ...]] = ((800, 608), (640, 384))
_DET_FILENAMES: Final[tuple[str, ...]] = tuple(f"ppocr_det_{h}x{w}.vmfb" for h, w in _DET_HWS)
_REC_YML_FILENAME: Final[str] = "ppocr_rec.yml"
_REC_BUCKET_WIDTHS: Final[tuple[int, ...]] = (320, 640, 1280, 2432)
_REC_BUCKET_DIR: Final[str] = "rec_buckets"
_SAMPLES_PREFIX: Final[str] = "samples/"


def rec_bucket_filenames() -> list[str]:
    return [f"{_REC_BUCKET_DIR}/rec_w{width}.vmfb" for width in _REC_BUCKET_WIDTHS]


def _required_filenames() -> list[str]:
    return [*_DET_FILENAMES, _REC_YML_FILENAME, *rec_bucket_filenames()]


def _has_ppocr_files(model_dir: Path) -> bool:
    return all((model_dir / name).exists() for name in _required_filenames())


def _download_ppocr(repo_id: str, base_dir: Path) -> list[str]:
    """Download PP-OCR assets; return the manifest file list."""
    from huggingface_hub import HfApi

    api = HfApi()
    manifest_files = []
    for filename in _required_filenames():
        if not api.file_exists(repo_id=repo_id, filename=filename):
            raise FileNotFoundError(f"Required file '{filename}' not found in {repo_id}")
        download_from_hf(repo_id, filename, base_dir=base_dir)
        manifest_files.append(filename)

    for path in api.list_repo_files(repo_id=repo_id):
        if path.startswith(_SAMPLES_PREFIX) and not path.endswith("/"):
            download_from_hf(repo_id, path, base_dir=base_dir)
            manifest_files.append(path)
    return manifest_files


def _refresh_ppocr(repo_id: str, model_dir: Path, base_dir: Path) -> ModelStatus:
    files_present = verify_manifest(model_dir) and _has_ppocr_files(model_dir)
    revision = get_hf_revision(repo_id)
    return ensure_model(model_dir, repo_id, files_present=files_present, revision=revision, download=lambda: _download_ppocr(repo_id, base_dir))


def download_ppocr(*, base_dir: str | Path | None = None) -> Path:
    """Download/refresh the PP-OCR assets; return the model directory.

    Unlike :func:`setup_ppocr` this does not check demo requirements, so other projects managing
    their own environment and models dir can reuse it.
    """
    base_dir = Path(base_dir) if base_dir is not None else default_models_dir()
    logger.info("Resolving PP-OCR model: %s", _PPOCR_HF_REPO)
    model_dir = base_dir / _PPOCR_HF_REPO
    try:
        _refresh_ppocr(_PPOCR_HF_REPO, model_dir, base_dir)
    except Exception as exc:
        raise DownloadError(f"Unable to download PP-OCR files from {_PPOCR_HF_REPO}") from exc
    logger.info("PP-OCR model files ready at '%s'", model_dir)
    return model_dir


def ensure_ppocr_models(model_dir: str | Path, *, refresh: bool = True) -> None:
    """Verify/refresh PP-OCR assets before inference.

    Reads the repo id from the local manifest and applies the same revision check as setup.
    Refresh failures are logged, not raised, so inference can still proceed on local files.
    """
    if not refresh:
        return

    model_dir = Path(model_dir)
    manifest = read_manifest(model_dir)
    repo_id = manifest.get("repo_id") if manifest else None
    if not repo_id:
        logger.warning("No manifest in %s; cannot verify PP-OCR asset freshness. Run `python setup_demos.py ppocr` if inference fails.", model_dir)
        return

    try:
        _refresh_ppocr(repo_id, model_dir, base_dir_for(model_dir, repo_id))
    except Exception as e:
        logger.warning("Could not refresh PP-OCR assets from %s (%s); using local files.", repo_id, e)


def setup_ppocr():
    base_dir = default_models_dir()
    model_dir = base_dir / _PPOCR_HF_REPO

    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("Setting up PP-OCR demo from %s", _PPOCR_HF_REPO)

    try:
        status = _refresh_ppocr(_PPOCR_HF_REPO, model_dir, base_dir)
    except Exception as e:
        raise DownloadError(f"Unable to download PP-OCR assets from {_PPOCR_HF_REPO}") from e

    if status is ModelStatus.UP_TO_DATE:
        logger.info("Using local PP-OCR assets from %s", model_dir)
    else:
        logger.info("Downloaded PP-OCR assets to %s", model_dir)


if __name__ == "__main__":
    import argparse
    import sys

    from utils.log import add_logging_args, configure_logging

    parser = argparse.ArgumentParser(description="Verify PP-OCR demo dependencies.")
    add_logging_args(parser)
    configure_logging(parser.parse_args().logging)

    try:
        setup_ppocr()
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
