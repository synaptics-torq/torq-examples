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

logger = logging.getLogger("ppocr.setup")

_PPOCR_HF_REPO: Final[str] = "Synaptics/ppocrv6-tiny-torq"

# Detection runs at one static shape; recognition uses one vmfb per width bucket
# so each text line is padded to the narrowest width that fits it.
_DET_FILENAME: Final[str] = "ppocr_det_800x608.vmfb"
_REC_YML_FILENAME: Final[str] = "ppocr_rec.yml"
_REC_BUCKET_WIDTHS: Final[tuple[int, ...]] = (320, 640, 1280, 2432)
_REC_BUCKET_DIR: Final[str] = "rec_buckets"
_SAMPLES_PREFIX: Final[str] = "samples/"


def rec_bucket_filenames() -> list[str]:
    return [f"{_REC_BUCKET_DIR}/rec_w{width}.vmfb" for width in _REC_BUCKET_WIDTHS]


def _hf_file_exists(repo_id: str, filename: str) -> bool:
    from huggingface_hub import HfApi

    return HfApi().file_exists(repo_id=repo_id, filename=filename)


def _list_sample_files(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi

    return [
        path for path in HfApi().list_repo_files(repo_id=repo_id)
        if path.startswith(_SAMPLES_PREFIX) and not path.endswith("/")
    ]


def _required_filenames() -> list[str]:
    return [_DET_FILENAME, _REC_YML_FILENAME, *rec_bucket_filenames()]


def _has_ppocr_files(model_dir: Path) -> bool:
    return all((model_dir / name).exists() for name in _required_filenames())


def _download_ppocr(repo_id: str, base_dir: Path) -> list[str]:
    """Download PP-OCR assets; return the manifest file list."""
    manifest_files = []

    for filename in _required_filenames():
        if not _hf_file_exists(repo_id, filename):
            raise FileNotFoundError(f"Required file '{filename}' not found in {repo_id}")
        download_from_hf(repo_id, filename, base_dir=base_dir)
        manifest_files.append(filename)

    for sample_file in _list_sample_files(repo_id):
        download_from_hf(repo_id, sample_file, base_dir=base_dir)
        manifest_files.append(sample_file)

    return manifest_files


def _refresh_ppocr(repo_id: str, model_dir: Path, base_dir: Path) -> ModelStatus:
    files_present = verify_manifest(model_dir) and _has_ppocr_files(model_dir)
    revision = get_hf_revision(repo_id)
    return ensure_model(
        model_dir,
        repo_id,
        files_present=files_present,
        revision=revision,
        download=lambda: _download_ppocr(repo_id, base_dir),
    )


def download_ppocr(*, base_dir: str | Path | None = None) -> Path:
    """Download/refresh the PP-OCR assets; return the model directory.

    Unlike :func:`setup_ppocr`, this does not check demo requirements, so it can
    be reused by other projects that manage their own environment and models dir.
    """
    if base_dir is None:
        base_dir = default_models_dir()
    base_dir = Path(base_dir)

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
            "No manifest in %s; cannot verify PP-OCR asset freshness. "
            "Run `python setup_demos.py ppocr` if inference fails.",
            model_dir,
        )
        return

    try:
        _refresh_ppocr(repo_id, model_dir, base_dir_for(model_dir, repo_id))
    except Exception as e:
        logger.warning(
            "Could not refresh PP-OCR assets from %s (%s); using local files.", repo_id, e
        )


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
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_ppocr()
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
