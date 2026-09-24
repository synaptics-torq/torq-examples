# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.download import download_from_hf
from utils.model_setup import demo_main, ensure_demo_models, setup_demo

logger = logging.getLogger("LiquidVL.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/LiquidAI-LFM2-VL-450M",
    "LiquidAI-LFM2-VL-450M": "Synaptics/LiquidAI-LFM2-VL-450M",  # accept the demo name as an alias
}
_DEFAULT_MODELS: Final[list[str]] = ["default"]

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


def ensure_lfm2vl_models(model_dir: str | Path, *, refresh: bool = True) -> None:
    """Verify/refresh the LFM2-VL models in ``model_dir`` before inference.

    Delegates to :func:`utils.model_setup.ensure_demo_models`, which re-syncs
    the local copy to the version recorded in its manifest. Refresh failures
    are logged, not raised, so inference can still proceed on whatever is
    available locally (offline/airgapped runs, e.g. the board).
    """
    ensure_demo_models(
        model_dir,
        refresh=refresh,
        demo_name="LiquidAI-LFM2-VL-450M",
        files_present=_has_lfm2vl_files,
        download=_download_lfm2vl,
    )


def setup_liquidvl(
    models: list[str] | None = None,
    model_version: str | None = None,
    no_update: bool = False,
):
    """Set up the LiquidAI-LFM2-VL-450M demo.

    Version selection, ``name:version`` specs and ``no_update`` are handled by
    :func:`utils.model_setup.setup_demo`, which checks demo requirements first,
    then downloads/refreshes the models.
    """
    if models is None:
        models = _DEFAULT_MODELS
    return setup_demo(
        _HF_REPO_MAP,
        models,
        demo_name="LiquidAI-LFM2-VL-450M",
        requirements=Path(__file__).parent / "requirements.txt",
        files_present=_has_lfm2vl_files,
        download=_download_lfm2vl,
        model_version=model_version,
        no_update=no_update,
    )


if __name__ == "__main__":
    demo_main(
        setup_liquidvl,
        description="Download LFM2-VL-450M model files.",
        default_models=_DEFAULT_MODELS,
        repo_map=_HF_REPO_MAP,
    )
