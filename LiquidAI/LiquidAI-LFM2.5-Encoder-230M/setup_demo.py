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

logger = logging.getLogger("LiquidEncoder.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/LiquidAI-LFM2.5-Encoder-230M",
    "230m": "Synaptics/LiquidAI-LFM2.5-Encoder-230M",
}
_MODEL_FILENAMES: Final[list[str]] = [
    "body_s256.vmfb",  # encoder body, static 256-token sequence
]
_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "token_embeddings.npy",
    "config.json",
    "tokenizer.json",
    "encoder_manifest.json",
)


def _has_encoder_files(model_dir: Path) -> bool:
    has_model = any((model_dir / f).exists() for f in _MODEL_FILENAMES)
    has_required = all((model_dir / f).exists() for f in _REQUIRED_FILES)
    return has_model and has_required


def setup_liquid_encoder(models: list[str]):
    logger.info(
        "Setting up LiquidAI-LFM2.5-Encoder-230M demo with models: [%s]",
        ", ".join(models),
    )
    repos = [_HF_REPO_MAP.get(m, m) for m in models]
    base_dir = default_models_dir()
    for repo_id in repos:
        model_dir = base_dir / repo_id
        if verify_manifest(model_dir) and _has_encoder_files(model_dir):
            logger.info("Using local encoder model files from %s", model_dir)
            continue

        try:
            manifest_files: list[str] = []
            for filename in _MODEL_FILENAMES + list(_REQUIRED_FILES):
                download_from_hf(repo_id, filename, base_dir=base_dir)
                manifest_files.append(filename)
            write_manifest(model_dir, repo_id, manifest_files)
            logger.info("Downloaded encoder model files from %s", repo_id)
        except Exception as e:
            raise DownloadError(
                f"Unable to download model files from {repo_id}") from e
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("liquid encoder setup complete.")


if __name__ == "__main__":
    import argparse
    import sys
    from utils.log import add_logging_args, configure_logging

    available_models = ", ".join(
        f"'{name}' ({repo})" for name, repo in _HF_REPO_MAP.items()
    )
    parser = argparse.ArgumentParser(
        description="Download LFM2.5-Encoder (Liquid) model files.",
    )
    parser.add_argument(
        "models", nargs="*", default=["default"],
        help=f"Model name or HF repo ID. Built-in: [{available_models}] "
             "(default: %(default)s)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_liquid_encoder(args.models)
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
