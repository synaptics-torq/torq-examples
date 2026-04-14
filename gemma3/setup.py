# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.deps import check_requirements
from utils.download import download_from_hf
from utils.errors import SetupError

logger = logging.getLogger("Gemma3.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/gemma-3-270m",
    "instruct": "Synaptics/gemma-3-270m-it"
}


def setup_gemma3(
    models: list[str],
):
    logger.info("Setting up gemma3 demo with models: [%s]", ", ".join(models))
    repos = [_HF_REPO_MAP.get(m, m) for m in models]
    for repo_id in repos:
        try:
            download_from_hf(repo_id, "model.vmfb")
            download_from_hf(repo_id, "token_embeddings.npy")
            download_from_hf(repo_id, "config.json")
            download_from_hf(repo_id, "tokenizer.json")
            logger.info("Downloaded gemma3 model files from %s", repo_id)
        except Exception as e:
            raise SetupError(f"Unable to download model files from {repo_id}") from e
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("gemma3 setup complete.")


if __name__ == "__main__":
    import argparse
    import sys
    from utils.log import add_logging_args, configure_logging

    available_models = ", ".join(f"'{model_name}' ({repo_id})" for model_name, repo_id in _HF_REPO_MAP.items())
    parser = argparse.ArgumentParser(
        description="Download Gemma3 model files.",
    )
    parser.add_argument(
        "models", nargs="*", default=["instruct"],
        help=f"Model name or HF repo ID. Built-in: [{available_models}] (default: %(default)s)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_gemma3(args.models)
    except (SetupError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
