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

logger = logging.getLogger("LFM2VLINT8.setup")

# The W8 vmfbs live in the same HF repo as the bf16 set, with a ``_W8`` name
# suffix; the sidecars (config/tokenizer/embeddings/sample image) are shared.
_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/LiquidAI-LFM2-VL-450M",
    "w8": "Synaptics/LiquidAI-LFM2-VL-450M",
    "int8": "Synaptics/LiquidAI-LFM2-VL-450M",
}

# The all-NPU W8 asset set (INT8 weight-only, bf16 activations).
_LFM2VL_W8_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "vision_encoder_256_W8.vmfb",     # SigLIP encoder, 256-res -> 64 image tokens
    "decoder_image_2part_A_W8.vmfb",  # one-shot image-prefill decoder, part A (layers 0-7)
    "decoder_image_2part_B_W8.vmfb",  # one-shot image-prefill decoder, part B (layers 8-15)
    "decoder_nolm_W8.vmfb",           # decoder body (hidden output)
    "lm_head_W8.vmfb",                # NPU lm_head (hidden -> logits)
    "token_embeddings.npy",           # CPU embedding LUT (bf16; shared with the bf16 set)
    "config.json",                    # shared with the bf16 set
    "tokenizer.json",                 # shared with the bf16 set
    "cats-and-dogs-256.jpg",          # sample 256-res image (shared)
)


def _has_files(model_dir: Path) -> bool:
    return all((model_dir / f).exists() for f in _LFM2VL_W8_REQUIRED_FILES)


def _download(repo_id: str, base_dir: Path) -> list[str]:
    for filename in _LFM2VL_W8_REQUIRED_FILES:
        download_from_hf(repo_id, filename, base_dir=base_dir)
        logger.info("Downloaded %s from %s", filename, repo_id)
    return list(_LFM2VL_W8_REQUIRED_FILES)


def _refresh(repo_id: str, model_dir: Path, base_dir: Path) -> ModelStatus:
    files_present = verify_manifest(model_dir) and _has_files(model_dir)
    revision = get_hf_revision(repo_id)
    return ensure_model(
        model_dir,
        repo_id,
        files_present=files_present,
        revision=revision,
        download=lambda: _download(repo_id, base_dir),
    )


def ensure_lfm2vl_int8_models(model_dir: str | Path, *, refresh: bool = True) -> None:
    """Verify/refresh the W8 model set in ``model_dir`` before inference.

    Refresh failures are logged, not raised, so offline/airgapped runs (e.g. the
    board with a hand-staged model dir) proceed on whatever is local.
    """
    model_dir = Path(model_dir)
    if not refresh:
        return
    manifest = read_manifest(model_dir)
    repo_id = manifest.get("repo_id") if manifest else None
    if not repo_id:
        logger.warning(
            "No manifest in %s; cannot verify model freshness. Local files will be used.",
            model_dir,
        )
        return
    base_dir = base_dir_for(model_dir, repo_id)
    if base_dir is None:
        logger.warning(
            "%s is not laid out as <models dir>/%s; skipping the freshness check.",
            model_dir, repo_id,
        )
        return
    try:
        _refresh(repo_id, model_dir, base_dir)
    except Exception as e:
        logger.warning("Could not refresh models from %s (%s); using local files.", repo_id, e)


def setup_liquidvl_int8(models: list[str]):
    logger.info("Setting up LFM2-VL-450M INT8 demo with models: [%s]", ", ".join(models))
    repos = [_HF_REPO_MAP.get(m, m) for m in models]
    base_dir = default_models_dir()
    for repo_id in repos:
        model_dir = base_dir / repo_id
        try:
            status = _refresh(repo_id, model_dir, base_dir)
        except Exception as e:
            raise DownloadError(f"Unable to download model files from {repo_id}") from e
        if status is ModelStatus.UP_TO_DATE:
            logger.info("Using local LFM2-VL-450M INT8 model files from %s", model_dir)
        else:
            logger.info("Downloaded LFM2-VL-450M INT8 model files from %s", repo_id)
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("LFM2-VL-450M INT8 setup complete. Model dir: %s", base_dir / repos[0])


if __name__ == "__main__":
    import argparse
    import sys
    from utils.log import add_logging_args, configure_logging

    available = ", ".join(f"'{name}' ({repo})" for name, repo in _HF_REPO_MAP.items())
    parser = argparse.ArgumentParser(description="Download LFM2-VL-450M INT8 model files.")
    parser.add_argument(
        "models", nargs="*", default=["default"],
        help=f"Model name or HF repo ID. Built-in: [{available}] (default: %(default)s)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_liquidvl_w8(args.models)
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
