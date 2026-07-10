# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.deps import MissingRequirementsError, check_requirements
from utils.download import (
    DownloadError,
    default_models_dir,
    download_from_hf,
    verify_manifest,
    write_manifest,
)

logger = logging.getLogger("Liquid.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/liquidAI-LFM2p5-230M-LLM",
    "230m": "Synaptics/liquidAI-LFM2p5-230M-LLM",
}
_LIQUID_MODEL_FILENAMES: Final[list[str]] = [
    "model.vmfb",   # fused decoder (monolithic)
    "body.vmfb",    # decoder minus lm_head (split path)
    "lm_head.vmfb", # standalone lm_head (split path)
]
_LIQUID_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "token_embeddings.npy",
    "config.json",
    "tokenizer.json",
)


def _hf_file_exists(repo_id: str, filename: str) -> bool:
    from huggingface_hub import HfApi

    return HfApi().file_exists(repo_id=repo_id, filename=filename)


def _has_liquid_files(model_dir: Path) -> bool:
    has_model = any(
        (model_dir / filename).exists() for filename in _LIQUID_MODEL_FILENAMES
    )
    has_required = all(
        (model_dir / filename).exists() for filename in _LIQUID_REQUIRED_FILES
    )
    return has_model and has_required


def _download_liquid_model(repo_id: str, base_dir: Path) -> list[str]:
    """Download the model vmfbs (fused model.vmfb + the split body/lm_head pair,
    whichever the repo has). Downloads each missing file individually so a
    partially-populated dir is completed rather than skipped."""
    local_dir = base_dir / repo_id
    present: list[str] = []
    for filename in _LIQUID_MODEL_FILENAMES:
        if (local_dir / filename).exists():
            present.append(filename)
            continue
        if _hf_file_exists(repo_id, filename):
            download_from_hf(repo_id, filename, base_dir=base_dir)
            logger.info("Downloaded %s from %s", filename, repo_id)
            present.append(filename)

    if not present:
        raise FileNotFoundError(f"no model vmfb found in {repo_id}")
    return present


def setup_liquid(models: list[str]):
    logger.info("Setting up liquid demo with models: [%s]", ", ".join(models))
    repos = [_HF_REPO_MAP.get(m, m) for m in models]
    base_dir = default_models_dir()
    for repo_id in repos:
        model_dir = base_dir / repo_id
        if verify_manifest(model_dir) and _has_liquid_files(model_dir):
            logger.info("Using local liquid model files from %s", model_dir)
            continue

        try:
            manifest_files = _download_liquid_model(repo_id, base_dir)
            for filename in _LIQUID_REQUIRED_FILES:
                download_from_hf(repo_id, filename, base_dir=base_dir)
                manifest_files.append(filename)

            write_manifest(model_dir, repo_id, manifest_files)
            logger.info("Downloaded liquid model files from %s", repo_id)
        except Exception as e:
            raise DownloadError(f"Unable to download model files from {repo_id}") from e
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
        help=f"Model name or HF repo ID. Built-in: [{available_models}] (default: %(default)s)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_liquid(args.models)
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
