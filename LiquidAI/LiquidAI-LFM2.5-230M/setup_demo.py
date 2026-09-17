# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.deps import MissingRequirementsError, check_requirements
from utils.download import (
    DownloadError,
    ModelStatus,
    default_models_dir,
    download_from_hf,
    ensure_model,
    get_hf_revision,
    verify_manifest,
)
from utils.version import parse_model_specs, resolve_model_version

logger = logging.getLogger("Liquid.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/LiquidAI-LFM2.5-230M",
    "230m": "Synaptics/LiquidAI-LFM2.5-230M",
}
_BUILTIN_REPOS: Final[frozenset[str]] = frozenset(_HF_REPO_MAP.values())
_LIQUID_MODEL_FILENAMES: Final[list[str]] = [
    "body.vmfb",    # decoder minus lm_head
    "lm_head.vmfb", # standalone lm_head (skipped during prefill)
]
_LIQUID_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "token_embeddings.npy",
    "config.json",
    "tokenizer.json",
)


def _hf_file_exists(repo_id: str, filename: str, *, revision: str | None = None) -> bool:
    from huggingface_hub import HfApi

    return HfApi().file_exists(repo_id=repo_id, filename=filename, revision=revision)


def _has_liquid_files(model_dir: Path) -> bool:
    has_model = any(
        (model_dir / filename).exists() for filename in _LIQUID_MODEL_FILENAMES
    )
    has_required = all(
        (model_dir / filename).exists() for filename in _LIQUID_REQUIRED_FILES
    )
    return has_model and has_required


def _download_liquid_model(repo_id: str, base_dir: Path, *, revision: str | None = None) -> list[str]:
    """Download the model vmfbs (fused model.vmfb + the split body/lm_head pair,
    whichever the repo has). Downloads each missing file individually so a
    partially-populated dir is completed rather than skipped."""
    local_dir = base_dir / repo_id
    present: list[str] = []
    for filename in _LIQUID_MODEL_FILENAMES:
        if (local_dir / filename).exists():
            present.append(filename)
            continue
        if _hf_file_exists(repo_id, filename, revision=revision):
            download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
            logger.info("Downloaded %s from %s", filename, repo_id)
            present.append(filename)

    if not present:
        raise FileNotFoundError(f"no model vmfb found in {repo_id}")
    return present


def _download_liquid(repo_id: str, base_dir: Path, *, revision: str | None = None) -> list[str]:
    """Download all Liquid model files; return the manifest file list."""
    manifest_files = _download_liquid_model(repo_id, base_dir, revision=revision)
    for filename in _LIQUID_REQUIRED_FILES:
        download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
        manifest_files.append(filename)
    return manifest_files


def _refresh_liquid(
    repo_id: str,
    model_dir: Path,
    base_dir: Path,
    *,
    version: str | None,
    record: bool = True,
) -> ModelStatus:
    files_present = _has_liquid_files(model_dir)
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
        download=lambda: _download_liquid(repo_id, base_dir, revision=version),
        record=record,
    )


def setup_liquid(
    models: list[str],
    model_version: str | None = None,
    no_update: bool = False,
):
    """Set up the LiquidAI-LFM2.5-230M demo.

    ``models`` entries may be built-in names, raw HF repo ids, or
    ``name:version`` to pin a specific version for that model. ``model_version``
    applies to every model without its own ``:version``; built-in repos default
    to the torq-examples version and custom repos to their latest (HEAD)
    revision. ``no_update=True`` downloads without writing a manifest, so the
    models are never tracked or refreshed (at your own risk).
    """
    logger.info("Setting up LiquidAI-LFM2.5-230M demo with models: [%s]", ", ".join(models))
    base_dir = default_models_dir()
    for name, spec_version in parse_model_specs(models):
        repo_id = _HF_REPO_MAP.get(name, name)
        version = resolve_model_version(
            repo_id, spec_version or model_version, builtin_repos=_BUILTIN_REPOS
        )
        model_dir = base_dir / repo_id
        try:
            status = _refresh_liquid(
                repo_id, model_dir, base_dir, version=version, record=not no_update
            )
        except Exception as e:
            raise DownloadError(f"Unable to download model files from {repo_id}") from e
        if status is ModelStatus.UP_TO_DATE:
            logger.info("Using local liquid model files from %s", model_dir)
        else:
            logger.info("Downloaded liquid model files from %s", repo_id)
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("liquid setup complete.")


if __name__ == "__main__":
    import argparse
    import sys
    from utils.log import add_logging_args, configure_logging

    available_models = ", ".join(
        f"'{model_name}' ({repo_id})" for model_name, repo_id in _HF_REPO_MAP.items()
    )
    parser = argparse.ArgumentParser(
        description="Download LFM2.5 (Liquid) model files.",
    )
    parser.add_argument(
        "models", nargs="*", default=["default"],
        help=f"Model name or HF repo ID, optionally 'name:version' to pin a "
             f"specific model version for that model. Built-in: [{available_models}] (default: %(default)s)",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help=(
            "Model version tag to download for every model without its own "
            "'name:version' (default: the torq-examples version for built-in "
            "repos, the repo's latest revision for custom repos). A pinned "
            "version is kept in sync with its own tag but never upgraded."
        ),
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help=(
            "Download without tracking: no .manifest.json is written, so the "
            "models are never checked for updates or refreshed (at your own risk)."
        ),
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_liquid(args.models, model_version=args.model_version, no_update=args.no_update)
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
