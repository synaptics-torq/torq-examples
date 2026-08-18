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

logger = logging.getLogger("LiquidVL.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/LiquidAI-LFM2-VL-450M",
    "LiquidAI-LFM2-VL-450M": "Synaptics/LiquidAI-LFM2-VL-450M",  # accept the demo name as an alias
}

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


def _download_lfm2vl(repo_id: str, base_dir: Path) -> list[str]:
    """Download every required LFM2-VL file; return the manifest file list."""
    for filename in _LFM2VL_REQUIRED_FILES:
        download_from_hf(repo_id, filename, base_dir=base_dir)
        logger.info("Downloaded %s from %s", filename, repo_id)
    return list(_LFM2VL_REQUIRED_FILES)


def _refresh_lfm2vl(repo_id: str, model_dir: Path, base_dir: Path) -> ModelStatus:
    files_present = verify_manifest(model_dir) and _has_lfm2vl_files(model_dir)
    revision = get_hf_revision(repo_id)
    return ensure_model(
        model_dir,
        repo_id,
        files_present=files_present,
        revision=revision,
        download=lambda: _download_lfm2vl(repo_id, base_dir),
    )


def ensure_lfm2vl_models(model_dir: str | Path, *, refresh: bool = True) -> None:
    """Verify/refresh the LFM2-VL models in ``model_dir`` before inference.

    Reads the repo id from the local manifest and applies the same revision
    check as setup. When ``refresh`` is ``False`` the check is skipped entirely
    (offline/airgapped runs, e.g. the board). Refresh failures are logged, not
    raised, so inference can still proceed on whatever is available locally.
    """
    model_dir = Path(model_dir)
    if not refresh:
        return
    manifest = read_manifest(model_dir)
    repo_id = manifest.get("repo_id") if manifest else None
    if not repo_id:
        logger.warning(
            "No manifest in %s; cannot verify model freshness. "
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
        _refresh_lfm2vl(repo_id, model_dir, base_dir)
    except Exception as e:
        logger.warning(
            "Could not refresh models from %s (%s); using local files.", repo_id, e
        )


def setup_liquidvl(models: list[str]):
    logger.info("Setting up LiquidAI-LFM2-VL-450M demo with models: [%s]", ", ".join(models))
    repos = [_HF_REPO_MAP.get(m, m) for m in models]
    base_dir = default_models_dir()
    for repo_id in repos:
        model_dir = base_dir / repo_id
        try:
            status = _refresh_lfm2vl(repo_id, model_dir, base_dir)
        except Exception as e:
            raise DownloadError(f"Unable to download model files from {repo_id}") from e
        if status is ModelStatus.UP_TO_DATE:
            logger.info("Using local LiquidAI-LFM2-VL-450M model files from %s", model_dir)
        else:
            logger.info("Downloaded LiquidAI-LFM2-VL-450M model files from %s", repo_id)
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("LiquidAI-LFM2-VL-450M setup complete. Model dir: %s", base_dir / repos[0])


if __name__ == "__main__":
    import argparse
    import sys
    from utils.log import add_logging_args, configure_logging

    available = ", ".join(f"'{name}' ({repo})" for name, repo in _HF_REPO_MAP.items())
    parser = argparse.ArgumentParser(description="Download LFM2-VL-450M model files.")
    parser.add_argument(
        "models", nargs="*", default=["default"],
        help=f"Model name or HF repo ID. Built-in: [{available}] (default: %(default)s)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_liquidvl(args.models)
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
