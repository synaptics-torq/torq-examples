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
    resolve_repo_id,
    verify_manifest,
)
from utils.version import parse_model_specs, resolve_model_version

logger = logging.getLogger("object_detection.setup")

_OD_HF_REPO_MAP: Final[dict[str, str]] = {
    "nano": "Synaptics/yolov8-od-nano-320-int8-torq",
}
_BUILTIN_REPOS: Final[frozenset[str]] = frozenset(_OD_HF_REPO_MAP.values())
_MODEL_FILENAME: Final[str] = "yolo_od.vmfb"
_LABELS_FILENAME: Final[str] = "labels.json"
_SAMPLES_PREFIX: Final[str] = "samples/"


def _hf_file_exists(repo_id: str, filename: str, *, revision: str | None = None) -> bool:
    from huggingface_hub import HfApi

    return HfApi().file_exists(repo_id=repo_id, filename=filename, revision=revision)


def _list_sample_files(repo_id: str, *, revision: str | None = None) -> list[str]:
    from huggingface_hub import HfApi

    return [
        path for path in HfApi().list_repo_files(repo_id=repo_id, revision=revision)
        if path.startswith(_SAMPLES_PREFIX) and not path.endswith("/")
    ]


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
        if not _hf_file_exists(repo_id, filename, revision=revision):
            raise FileNotFoundError(f"Required file '{filename}' not found in {repo_id}")
        download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
        manifest_files.append(filename)

    for sample_file in _list_sample_files(repo_id, revision=revision):
        download_from_hf(repo_id, sample_file, base_dir=base_dir, revision=revision)
        manifest_files.append(sample_file)

    return manifest_files


def _refresh_object_detection(
    repo_id: str,
    model_dir: Path,
    base_dir: Path,
    *,
    version: str | None,
    record: bool = True,
) -> ModelStatus:
    files_present = _has_object_detection_files(model_dir)
    revision = None
    if record:
        files_present = verify_manifest(model_dir) and files_present
        if version is not None:
            revision = get_hf_revision(repo_id, revision=version)
    return ensure_model(
        model_dir,
        repo_id,
        files_present=files_present,
        version=version if record else None,
        revision=revision,
        download=lambda: _download_object_detection(repo_id, base_dir, revision=version),
        record=record,
    )


def download_object_detection(
    models: list[str] | None = None,
    *,
    base_dir: str | Path | None = None,
    model_version: str | None = None,
    no_update: bool = False,
) -> dict[str, Path]:
    """Download/refresh the given Yolo models; return ``{name: model_dir}``.

    ``models`` entries may be built-in names, raw HF repo ids, or
    ``name:version`` to pin a specific version for that model. ``model_version``
    applies to every model without its own ``:version``; built-in repos default
    to the torq-examples version and custom repos to their latest (HEAD)
    revision. ``no_update=True`` downloads without writing a manifest, so the
    models are never tracked or refreshed (at your own risk).

    Unlike :func:`setup_object_detection`, this does not check demo requirements, so it
    can be reused by other projects that manage their own environment and models dir.
    """
    if models is None:
        models = ["nano"]
    if base_dir is None:
        base_dir = default_models_dir()
    base_dir = Path(base_dir)

    logger.info("Resolving Yolo models: [%s]", ", ".join(models))
    result: dict[str, Path] = {}
    for name, spec_version in parse_model_specs(models):
        repo_id = resolve_repo_id(name, _OD_HF_REPO_MAP)
        version = resolve_model_version(
            repo_id, spec_version or model_version, builtin_repos=_BUILTIN_REPOS
        )
        model_dir = base_dir / repo_id
        try:
            _refresh_object_detection(
                repo_id, model_dir, base_dir, version=version, record=not no_update
            )
        except Exception as exc:
            raise DownloadError(f"Unable to download Yolo files from {repo_id}") from exc
        result[name] = model_dir
        logger.info(
            "Yolo model files for version %s ready at '%s'",
            version or "latest", model_dir,
        )
    return result


def ensure_object_detection_models(
    model_dir: str | Path,
    *,
    refresh: bool = True,
) -> None:
    """Verify/refresh object detection assets before inference.

    Re-syncs the local copy to the version recorded in its manifest (never to a
    newer one). Untracked models (``--no-update``) have no manifest and are
    left as-is. When ``refresh`` is ``False`` the check is skipped entirely
    for offline/airgapped runs. Refresh failures are logged, not raised, so
    inference can still proceed using local files.
    """
    model_dir = Path(model_dir)
    if not refresh:
        return

    manifest = read_manifest(model_dir)
    if manifest is None:
        logger.info(
            "No manifest in %s; the model is untracked, so skipping the freshness check.",
            model_dir,
        )
        return
    repo_id = manifest.get("repo_id")
    if not repo_id:
        logger.warning(
            "Manifest in %s has no repo_id; cannot verify object detection asset freshness. "
            "Run `python setup_demos.py object_detection` if inference fails.",
            model_dir,
        )
        return
    base_dir = base_dir_for(model_dir, repo_id)
    if base_dir is None:
        logger.warning(
            "%s is not laid out as <models dir>/%s; skipping the freshness check "
            "so a refresh cannot fetch a second copy elsewhere. "
            "Run `python setup_demos.py object_detection` to manage assets.",
            model_dir,
            repo_id,
        )
        return

    try:
        _refresh_object_detection(
            repo_id,
            model_dir,
            base_dir,
            version=manifest.get("version"),
        )
    except Exception as e:
        logger.warning(
            "Could not refresh object detection assets from %s (%s); using local files.",
            repo_id,
            e,
        )


def setup_object_detection(
    model_version: str | None = None,
    no_update: bool = False,
):
    repo_id = _OD_HF_REPO_MAP["nano"]
    version = resolve_model_version(repo_id, model_version, builtin_repos=_BUILTIN_REPOS)
    base_dir = default_models_dir()
    model_dir = base_dir / repo_id

    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info(
        "Setting up object detection demo from %s (version=%s)",
        repo_id, version or "latest",
    )

    try:
        status = _refresh_object_detection(
            repo_id, model_dir, base_dir, version=version, record=not no_update
        )
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

    parser = argparse.ArgumentParser(description="Set up the object detection demo.")
    add_logging_args(parser)
    parser.add_argument(
        "--model-version",
        default=None,
        help=(
            "Model version tag to download (default: the torq-examples version for "
            "built-in repos, the repo's latest revision for custom repos). A "
            "pinned version is kept in sync with its own tag but never upgraded."
        ),
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help=(
            "Download without tracking: no .manifest.json is written, so the model "
            "is never checked for updates or refreshed (at your own risk)."
        ),
    )
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_object_detection(model_version=args.model_version, no_update=args.no_update)
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
