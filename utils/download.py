# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import json
import logging
import os
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Final

__all__ = [
    "DownloadError",
    "default_models_dir",
    "download_from_url",
    "download_from_hf",
    "get_hf_revision",
    "write_manifest",
    "verify_manifest",
    "read_manifest",
    "ModelStatus",
    "check_model_status",
    "clear_model_dir",
    "base_dir_for",
    "ensure_model",
]

logger = logging.getLogger(__name__)
_MANIFEST_FILENAME: Final[str] = ".manifest.json"


class DownloadError(Exception):
    """Raised when setup cannot download required model files."""


def download_from_url(url: str, filename: str | os.PathLike, chunk_size: int = 8192):
    filename = Path(filename)
    if filename.exists():
        logger.debug("File found locally at: %s", filename)
        return filename

    filename.parent.mkdir(exist_ok=True, parents=True)

    import requests
    from tqdm import tqdm

    logger.debug("Attempting download from %s...", url)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get('content-length', 0))
    progress = tqdm(total=total, unit='B', unit_scale=True, desc=str(filename))

    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))
    progress.close()
    logger.debug("Download completed.")

    return filename


def default_models_dir() -> Path:
    _repo_root = Path(__file__).resolve().parent.parent
    return Path(os.getenv("MODELS", str(_repo_root / "models")))


def download_from_hf(
    repo_id: str,
    filename: str | os.PathLike,
    base_dir: str | os.PathLike | None = None,
) -> Path:
    if base_dir is None:
        base_dir = default_models_dir()
    base_dir = Path(base_dir)
    local_file = base_dir / repo_id / filename
    local_file.parent.mkdir(parents=True, exist_ok=True)

    if local_file.exists():
        logger.debug("File found locally at: %s", local_file)
        return local_file

    from huggingface_hub import hf_hub_download

    logger.debug("Attempting to download %s from %s...", filename, repo_id)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(base_dir / repo_id),
    )
    logger.debug("Download from HuggingFace completed.")
    return local_file


def get_hf_revision(repo_id: str) -> str | None:
    """Return the current commit SHA of a Hugging Face repo.

    Returns ``None`` when the Hub cannot be reached (e.g. offline) so callers
    can fall back to local files with a staleness warning instead of failing.
    """
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id).sha
    except Exception as exc:  # offline, auth failure, missing repo, ...
        logger.debug("Could not resolve revision for %s: %s", repo_id, exc)
        return None


def write_manifest(
    model_dir: Path,
    repo_id: str,
    files: list[str],
    revision: str | None = None,
) -> Path:
    """Write a manifest after a successful model setup.

    ``revision`` records the upstream commit the files were downloaded from,
    so later runs can detect when the local copy is out of date.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo_id": repo_id,
        "revision": revision,
        "files": sorted(files),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = model_dir / _MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.debug("Wrote manifest to %s", manifest_path)
    return manifest_path


def verify_manifest(model_dir: Path) -> bool:
    """Return True when a manifest exists and every listed file is present."""
    manifest = read_manifest(model_dir)
    if manifest is None:
        return False

    files = manifest.get("files", [])
    if not files:
        return False
    model_dir = Path(model_dir)
    return all((model_dir / filename).exists() for filename in files)


def read_manifest(model_dir: Path) -> dict | None:
    """Read a model manifest, or return None when missing or corrupt."""
    manifest_path = Path(model_dir) / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt manifest at %s", manifest_path)
        return None


class ModelStatus(Enum):
    """Result of comparing a local model copy against its upstream repo."""

    UP_TO_DATE = "up_to_date"
    INCOMPLETE = "incomplete"
    STALE = "stale"


def check_model_status(
    model_dir: Path,
    repo_id: str,
    *,
    files_present: bool,
    revision: str | None,
) -> ModelStatus:
    """Classify the local model copy against the upstream repo.

    Args:
        files_present: Whether the demo's required files already exist locally
            (the caller's own integrity check).
        revision: The upstream commit SHA from :func:`get_hf_revision`, or
            ``None`` when the Hub is unreachable.

    Returns one of:
        ``ModelStatus.STALE``: the upstream revision is known and differs from
            the one recorded in the local manifest (including a local copy that
            predates revision tracking). Clear the directory and re-download.
        ``ModelStatus.INCOMPLETE``: required files are missing; fetch what's absent.
        ``ModelStatus.UP_TO_DATE``: local files are present and current. Also
            returned when the Hub is unreachable and files exist, after logging
            a staleness warning, so the demo can still run offline.
    """
    manifest = read_manifest(model_dir)
    local_revision = manifest.get("revision") if manifest else None

    if revision is not None and local_revision != revision:
        logger.info("Local model files for %s are out of date; re-downloading.", repo_id)
        return ModelStatus.STALE
    if not files_present:
        return ModelStatus.INCOMPLETE
    if revision is None:
        logger.warning(
            "Could not reach Hugging Face to check for updates to %s; using "
            "local files in %s, which may be out of date.",
            repo_id,
            model_dir,
        )
    return ModelStatus.UP_TO_DATE


def clear_model_dir(model_dir: Path) -> None:
    """Remove a model directory and all its contents, if it exists.

    Used before a refresh so stale files are not left behind: ``download_from_hf``
    skips files that already exist, so same-named files with updated content
    (or files dropped from the required set) must be removed first.
    """
    model_dir = Path(model_dir)
    if model_dir.exists():
        shutil.rmtree(model_dir, ignore_errors=True)
        logger.debug("Cleared model directory %s", model_dir)


def base_dir_for(model_dir: Path, repo_id: str) -> Path:
    """Return the models base dir given a model dir laid out as ``base/repo_id``."""
    base = Path(model_dir)
    for _ in Path(repo_id).parts:
        base = base.parent
    return base


def ensure_model(
    model_dir: Path,
    repo_id: str,
    *,
    files_present: bool,
    revision: str | None,
    download: Callable[[], list[str]],
) -> ModelStatus:
    """Refresh ``model_dir`` to ``revision`` when it is stale or incomplete.

    Shared by setup and inference so both apply identical refresh semantics.
    ``download`` fetches the required files and returns the filenames to record
    in the manifest; it is invoked only for stale/incomplete states. Stale dirs
    are cleared first so updated same-named files are not skipped on re-download.

    Returns the :class:`ModelStatus` that was acted on.
    """
    status = check_model_status(
        model_dir, repo_id, files_present=files_present, revision=revision
    )
    if status is ModelStatus.UP_TO_DATE:
        return status
    if status is ModelStatus.STALE:
        clear_model_dir(model_dir)
    files = download()
    write_manifest(model_dir, repo_id, files, revision=revision)
    return status
