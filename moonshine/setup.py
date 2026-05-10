# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from huggingface_hub import HfApi

from utils.deps import check_requirements
from utils.download import download_from_hf
from utils.errors import SetupError

logger = logging.getLogger("moonshine.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "tiny-en": "Synaptics/moonshine-tiny-bf16-torq",
}


def _download_preprocessor(repo_id: str):
    preproc_vmfb = "preprocessor.vmfb"
    preproc_onnx = "preprocessor.onnx"
    hf_api = HfApi()
    has_vmfb = hf_api.file_exists(repo_id=repo_id, filename=preproc_vmfb)
    has_onnx = hf_api.file_exists(repo_id=repo_id, filename=preproc_onnx)

    if has_vmfb:
        if has_onnx:
            logger.warning(
                "Both optional preprocessor files exist in %s; using %s.",
                repo_id,
                preproc_vmfb,
            )
        download_from_hf(repo_id, preproc_vmfb)
        return

    if has_onnx:
        download_from_hf(repo_id, preproc_onnx)
        return

    logger.info(
        "No preprocessor model found in %s; continuing without it.",
        repo_id,
    )


def setup_moonshine(
    models: list[str],
):
    logger.info("Setting up moonshine demo with models: [%s]", ", ".join(models))
    repos = [_HF_REPO_MAP.get(m, m) for m in models]
    for repo_id in repos:
        try:
            _download_preprocessor(repo_id)
            download_from_hf(repo_id, "encoder.vmfb")
            download_from_hf(repo_id, "decoder.vmfb")
            download_from_hf(repo_id, "decoder_with_past.vmfb")
            download_from_hf(repo_id, "decoder_token_embeddings.npy")
            download_from_hf(repo_id, "tokenizer.json")
            logger.info("Downloaded moonshine model files from %s", repo_id)
        except Exception as e:
            raise SetupError(f"Unable to download model files from {repo_id}") from e
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("moonshine setup complete.")


if __name__ == "__main__":
    import argparse
    import sys
    from utils.log import add_logging_args, configure_logging

    available_models = ", ".join(f"'{model_name}' ({repo_id})" for model_name, repo_id in _HF_REPO_MAP.items())
    parser = argparse.ArgumentParser(
        description="Download Moonshine model files.",
    )
    parser.add_argument(
        "models", nargs="*", default=["tiny-en"],
        help=f"Model name or HF repo ID. Built-in: [{available_models}] (default: %(default)s)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_moonshine(args.models)
    except (SetupError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
