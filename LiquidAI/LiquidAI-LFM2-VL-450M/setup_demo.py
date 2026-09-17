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
from utils.version import parse_model_specs, resolve_model_version

logger = logging.getLogger("LiquidVL.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/LiquidAI-LFM2-VL-450M",
    "LiquidAI-LFM2-VL-450M": "Synaptics/LiquidAI-LFM2-VL-450M",  # accept the demo name as an alias
}
_BUILTIN_REPOS: Final[frozenset[str]] = frozenset(_HF_REPO_MAP.values())

# The full LFM2-VL-450M asset set the demo needs for the one-shot image-prefill
# path with the NPU lm_head (see lfm2-vl-450m-usage.md).
_LFM2VL_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "vision_encoder_256.vmfb",     # SigLIP encoder, 256-res -> 64 image tokens
    "decoder_image_2part_A.vmfb",  # one-shot image-prefill decoder, part A (layers 0-7)
    "decoder_image_2part_B.vmfb",  # one-shot image-prefill decoder, part B (layers 8-15)
    "decoder_nolm.vmfb",           # decode decoder body (hidden output)
    "lm_head.vmfb",                # NPU lm_head (hidden -> logits)
    "token_embeddings.npy",        # CPU embedding LUT / tied lm_head source
    "config.json",
    "tokenizer.json",
    "cats-and-dogs-256.jpg",       # sample 256-res image for the demo command
)


def _has_lfm2vl_files(model_dir: Path) -> bool:
    return all((model_dir / filename).exists() for filename in _LFM2VL_REQUIRED_FILES)


def _download_lfm2vl(repo_id: str, base_dir: Path, *, revision: str | None = None) -> list[str]:
    """Download every required LFM2-VL file; return the manifest file list."""
    for filename in _LFM2VL_REQUIRED_FILES:
        download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
        logger.info("Downloaded %s from %s", filename, repo_id)
    return list(_LFM2VL_REQUIRED_FILES)


def _refresh_lfm2vl(
    repo_id: str,
    model_dir: Path,
    base_dir: Path,
    *,
    version: str | None,
    record: bool = True,
) -> ModelStatus:
    files_present = _has_lfm2vl_files(model_dir)
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
        download=lambda: _download_lfm2vl(repo_id, base_dir, revision=version),
        record=record,
    )


def ensure_lfm2vl_models(model_dir: str | Path, *, refresh: bool = True) -> None:
    """Verify/refresh the LFM2-VL models in ``model_dir`` before inference.

    Re-syncs the local copy to the version recorded in its manifest (never to a
    newer one). Untracked models (``--no-update``) have no manifest and are
    left as-is. When ``refresh`` is ``False`` the check is skipped entirely
    (offline/airgapped runs, e.g. the board). Refresh failures are logged, not
    raised, so inference can still proceed on whatever is available locally.
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
            "Manifest in %s has no repo_id; cannot verify model freshness. "
            "Run `python setup_demos.py LiquidAI-LFM2-VL-450M` if inference fails.",
            model_dir,
        )
        return
    base_dir = base_dir_for(model_dir, repo_id)
    if base_dir is None:
        logger.warning(
            "%s is not laid out as <models dir>/%s; skipping the freshness check "
            "so a refresh cannot fetch a second copy elsewhere. "
            "Run `python setup_demos.py LiquidAI-LFM2-VL-450M` to manage models.",
            model_dir,
            repo_id,
        )
        return
    try:
        _refresh_lfm2vl(repo_id, model_dir, base_dir, version=manifest.get("version"))
    except Exception as e:
        logger.warning(
            "Could not refresh models from %s (%s); using local files.", repo_id, e
        )


def setup_liquidvl(
    models: list[str],
    model_version: str | None = None,
    no_update: bool = False,
):
    """Set up the LiquidAI-LFM2-VL-450M demo.

    ``models`` entries may be built-in names, raw HF repo ids, or
    ``name:version`` to pin a specific version for that model. ``model_version``
    applies to every model without its own ``:version``; built-in repos default
    to the torq-examples version and custom repos to their latest (HEAD)
    revision. ``no_update=True`` downloads without writing a manifest, so the
    models are never tracked or refreshed (at your own risk).
    """
    logger.info("Setting up LiquidAI-LFM2-VL-450M demo with models: [%s]", ", ".join(models))
    base_dir = default_models_dir()
    model_dir = None
    for name, spec_version in parse_model_specs(models):
        repo_id = _HF_REPO_MAP.get(name, name)
        version = resolve_model_version(
            repo_id, spec_version or model_version, builtin_repos=_BUILTIN_REPOS
        )
        model_dir = base_dir / repo_id
        try:
            status = _refresh_lfm2vl(
                repo_id, model_dir, base_dir, version=version, record=not no_update
            )
        except Exception as e:
            raise DownloadError(f"Unable to download model files from {repo_id}") from e
        if status is ModelStatus.UP_TO_DATE:
            logger.info("Using local LiquidAI-LFM2-VL-450M model files from %s", model_dir)
        else:
            logger.info("Downloaded LiquidAI-LFM2-VL-450M model files from %s", repo_id)
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("LiquidAI-LFM2-VL-450M setup complete. Model dir: %s", model_dir)


if __name__ == "__main__":
    import argparse
    import sys
    from utils.log import add_logging_args, configure_logging

    available = ", ".join(f"'{name}' ({repo})" for name, repo in _HF_REPO_MAP.items())
    parser = argparse.ArgumentParser(description="Download LFM2-VL-450M model files.")
    parser.add_argument(
        "models", nargs="*", default=["default"],
        help=f"Model name or HF repo ID, optionally 'name:version' to pin a "
             f"specific model version for that model. Built-in: [{available}] (default: %(default)s)",
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
        setup_liquidvl(args.models, model_version=args.model_version, no_update=args.no_update)
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
