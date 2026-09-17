# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Model download and version tracking shared by the demos.

Each demo's model lives in a Hugging Face repo and is, optionally, identified
by a version tag (``vX.Y.Z``) within that repo. Setup downloads the files for
the requested version and records them in a ``.manifest.json`` next to the
model. Subsequent runs (setup or inference) re-resolve the tracked version tag
and refresh the local copy when the tag moved upstream or local files went
missing/corrupt. A model is never upgraded to a newer version on its own: the
version it was set up with is the one it stays on until the user changes it.

Models downloaded with ``--no-update`` have no manifest and are excluded from
the tracking system altogether.
"""

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
    "ModelVersionNotFoundError",
    "HF_CHECK_TIMEOUT",
    "default_models_dir",
    "download_from_url",
    "download_from_hf",
    "get_hf_revision",
    "hf_file_exists",
    "list_hf_files",
    "write_manifest",
    "verify_manifest",
    "read_manifest",
    "ModelStatus",
    "check_model_status",
    "clear_model_dir",
    "base_dir_for",
    "ensure_model",
    "resolve_repo_id",
    "local_model_dir",
]

logger = logging.getLogger(__name__)
_MANIFEST_FILENAME: Final[str] = ".manifest.json"

# Short timeout (seconds) for Hub metadata checks (revision lookups), so
# offline/airgapped runs fail fast instead of hanging on the connection. File
# downloads themselves use the huggingface_hub defaults. Override with
# TORQ_HF_CHECK_TIMEOUT if 5s is too aggressive for your network.
HF_CHECK_TIMEOUT: Final[float] = float(os.getenv("TORQ_HF_CHECK_TIMEOUT", "5"))


class DownloadError(Exception):
    """Raised when setup cannot download required model files."""


class ModelVersionNotFoundError(DownloadError):
    """Raised when a requested model version does not exist in the HF repo.

    Setup fails loudly on this rather than silently falling back to another
    revision.
    """


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
    *,
    revision: str | None = None,
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

    logger.debug("Attempting to download %s from %s (revision=%s)...", filename, repo_id, revision)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(base_dir / repo_id),
        revision=revision,
    )
    logger.debug("Download from HuggingFace completed.")
    return local_file


def get_hf_revision(repo_id: str, *, revision: str | None = None) -> str | None:
    """Return the commit SHA of *revision* (default: HEAD) in *repo_id*.

    Raises:
        ModelVersionNotFoundError: The repo or version tag definitively does
            not exist, so setup should fail loudly instead of silently falling
            back to another revision.

    Returns:
        The commit SHA, or ``None`` when the Hub cannot be reached (offline,
        auth problems, transient errors) so callers can fall back to local
        files with a staleness warning instead of failing.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        HFValidationError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    logger.debug("Resolving revision %r of %s ...", revision, repo_id)
    try:
        return HfApi().model_info(
            repo_id, revision=revision, timeout=HF_CHECK_TIMEOUT
        ).sha
    except (HFValidationError, RepositoryNotFoundError, RevisionNotFoundError) as exc:
        raise ModelVersionNotFoundError(
            f"Model version '{revision}' not found in Hugging Face repo '{repo_id}'."
        ) from exc
    except Exception as exc:  # offline, auth failure, transient Hub errors, ...
        logger.debug("Could not resolve revision for %s (%s): %s", repo_id, revision, exc)
        return None


def hf_file_exists(repo_id: str, filename: str, *, revision: str | None = None) -> bool:
    """Whether ``filename`` exists in the HF repo at *revision* (default HEAD)."""
    from huggingface_hub import HfApi

    return HfApi().file_exists(repo_id=repo_id, filename=filename, revision=revision)


def list_hf_files(
    repo_id: str, *, prefix: str | None = None, revision: str | None = None
) -> list[str]:
    """List file paths in an HF repo (directories excluded), optionally under *prefix*."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, timeout=HF_CHECK_TIMEOUT)
    files = [sibling.rfilename for sibling in info.siblings]
    return [
        path for path in files
        if not path.endswith("/") and (prefix is None or path.startswith(prefix))
    ]


def write_manifest(
    model_dir: Path,
    repo_id: str,
    files: list[str],
    *,
    version: str | None = None,
    revision: str | None = None,
) -> Path:
    """Write a manifest after a successful model setup.

    ``version`` records the version tag the copy tracks (``None`` =
    unversioned, i.e. the repo's latest at download time); ``revision`` records
    the upstream commit the files were downloaded from, so later runs can
    detect when the local copy is out of date.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo_id": repo_id,
        "version": version,
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
    version: str | None,
    revision: str | None,
) -> ModelStatus:
    """Classify the local model copy against its upstream repo.

    Args:
        files_present: Whether the demo's required files already exist locally
            (the caller's own integrity check).
        version: The version tag the copy is supposed to track (e.g.
            ``v2.1.0``), or ``None`` for an unversioned copy.
        revision: The commit SHA *version* currently resolves to from
            :func:`get_hf_revision`, or ``None`` when the Hub is unreachable
            or *version* is ``None``.

    Returns one of:
        ``ModelStatus.STALE``: the tracked version resolves to a different
            commit than the one recorded in the local manifest (including a
            local copy that predates version tracking). Clear the directory
            and re-download.
        ``ModelStatus.INCOMPLETE``: required files are missing; fetch what's
            absent.
        ``ModelStatus.UP_TO_DATE``: local files are present and current. Also
            returned when the Hub is unreachable and files exist, after logging
            a staleness warning, so the demo can still run offline.
    """
    manifest = read_manifest(model_dir)
    local_revision = manifest.get("revision") if manifest else None

    if version is not None and revision is not None and local_revision != revision:
        logger.info(
            "Local model files for %s no longer match version %s; re-downloading.",
            repo_id, version,
        )
        return ModelStatus.STALE
    if not files_present:
        return ModelStatus.INCOMPLETE
    if version is not None and revision is None:
        logger.warning(
            "Could not reach Hugging Face to check the %s version of %s; using "
            "local files in %s, which may be out of date.",
            version, repo_id, model_dir,
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


def base_dir_for(model_dir: Path, repo_id: str) -> Path | None:
    """Return the models base dir given a model dir laid out as ``base/repo_id``.

    Returns ``None`` when *model_dir* does not end in *repo_id*, e.g. a bare
    Hugging Face clone in ``models/<repo name>``. That layout is what maps a
    repo id onto a local path, so without it there is no base dir to download
    into: stripping the parts blindly points at an unrelated directory, where a
    refresh would fetch a second full copy of the model instead of updating the
    one being used.
    """
    model_dir = Path(model_dir)
    repo_parts = tuple(repo_id.split("/"))
    if len(model_dir.parts) < len(repo_parts):
        return None
    if model_dir.parts[-len(repo_parts):] != repo_parts:
        return None
    base_parts = model_dir.parts[:-len(repo_parts)]
    return Path(*base_parts) if base_parts else None


def resolve_repo_id(model: str, repo_map: dict[str, str]) -> str:
    """Resolve a model name to its HF repo id via ``repo_map``.

    Unknown names pass through unchanged so callers can also supply a raw repo id.
    """
    return repo_map.get(model, model)


def local_model_dir(
    model: str,
    repo_map: dict[str, str],
    *,
    base_dir: str | os.PathLike | None = None,
) -> Path | None:
    """Return the local dir for ``model`` if it has a valid manifest, else ``None``.

    The dir is ``base_dir / <resolved repo id>``; ``base_dir`` defaults to
    :func:`default_models_dir`.
    """
    if base_dir is None:
        base_dir = default_models_dir()
    model_dir = Path(base_dir) / resolve_repo_id(model, repo_map)
    return model_dir if verify_manifest(model_dir) else None


def ensure_model(
    model_dir: Path,
    repo_id: str,
    *,
    files_present: bool,
    version: str | None,
    revision: str | None,
    download: Callable[[], list[str]],
    record: bool = True,
) -> ModelStatus:
    """Refresh ``model_dir`` to track ``version`` when it is stale or incomplete.

    Shared by setup and inference so both apply identical refresh semantics.
    ``download`` fetches the required files and returns the filenames to record
    in the manifest; it is invoked only for stale/incomplete states. Stale dirs
    are cleared first so updated same-named files are not skipped on re-download.

    When ``record`` is False (``--no-update``) the copy is untracked: only file
    completeness is verified, no version check is performed, and no manifest is
    written, so the model is excluded from the tracking system altogether.

    Returns the :class:`ModelStatus` that was acted on.
    """
    model_dir = Path(model_dir)
    if record:
        status = check_model_status(
            model_dir, repo_id, files_present=files_present, version=version, revision=revision
        )
        if status is ModelStatus.UP_TO_DATE and version is not None:
            manifest = read_manifest(model_dir)
            if manifest is not None and manifest.get("version") != version:
                # Content is identical (same commit) but the user asked to
                # track a different version name; adopt it so future refreshes
                # follow the version the user last requested, without
                # re-downloading.
                write_manifest(
                    model_dir,
                    repo_id,
                    manifest.get("files", []),
                    version=version,
                    revision=manifest.get("revision") or revision,
                )
    else:
        status = ModelStatus.UP_TO_DATE if files_present else ModelStatus.INCOMPLETE

    if status is ModelStatus.UP_TO_DATE:
        return status
    if status is ModelStatus.STALE:
        clear_model_dir(model_dir)
    files = download()
    if record:
        write_manifest(model_dir, repo_id, files, version=version, revision=revision)
    else:
        logger.info(
            "Untracked model in %s (--no-update): no manifest written, so this "
            "copy will never be checked for updates or refreshed.",
            model_dir,
        )
    return status
