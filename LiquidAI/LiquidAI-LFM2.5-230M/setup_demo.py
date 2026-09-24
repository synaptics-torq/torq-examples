# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.download import download_from_hf, hf_file_exists
from utils.model_setup import demo_main, setup_demo

logger = logging.getLogger("Liquid.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/LiquidAI-LFM2.5-230M",
    "230m": "Synaptics/LiquidAI-LFM2.5-230M",
}
_DEFAULT_MODELS: Final[list[str]] = ["default"]
_LIQUID_MODEL_FILENAMES: Final[list[str]] = [
    "body.vmfb",    # decoder minus lm_head
    "lm_head.vmfb", # standalone lm_head (skipped during prefill)
]
_LIQUID_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "token_embeddings.npy",
    "config.json",
    "tokenizer.json",
)


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
        if hf_file_exists(repo_id, filename, revision=revision):
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


def setup_liquid(
    models: list[str] | None = None,
    model_version: str | None = None,
    no_update: bool = False,
):
    """Set up the LiquidAI-LFM2.5-230M demo.

    Version selection, ``name:version`` specs and ``no_update`` are handled by
    :func:`utils.model_setup.setup_demo`, which checks demo requirements first,
    then downloads/refreshes the models.
    """
    if models is None:
        models = _DEFAULT_MODELS
    return setup_demo(
        _HF_REPO_MAP,
        models,
        demo_name="LiquidAI-LFM2.5-230M",
        requirements=Path(__file__).parent / "requirements.txt",
        files_present=_has_liquid_files,
        download=_download_liquid,
        model_version=model_version,
        no_update=no_update,
    )


if __name__ == "__main__":
    demo_main(
        setup_liquid,
        description="Download LFM2.5 (Liquid) model files.",
        default_models=_DEFAULT_MODELS,
        repo_map=_HF_REPO_MAP,
    )
