# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.deps import check_requirements
from utils.download import download_from_hf, default_models_dir
from utils.errors import SetupError

logger = logging.getLogger("Gemma3.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/gemma-3-270m",
    "instruct": "Synaptics/gemma-3-270m-it"
}
_GEMMA3_MODEL_FILENAMES: Final[list[str]] = [
    "model.vmfb",
    "model.vmfb.trim",
]
_GEMMA3_TRIM_LUT_FILENAME: Final[str] = "token_id_lut.npy"


def _download_gemma3_model(repo_id: str, base_dir: Path) -> None:
    """Download model.vmfb and/or model.vmfb.trim as they exist."""
    local_dir = base_dir / repo_id
    if any((local_dir / filename).exists() for filename in _GEMMA3_MODEL_FILENAMES):
        return

    from huggingface_hub import HfApi

    hf_api = HfApi()
    downloaded_any = False
    for filename in _GEMMA3_MODEL_FILENAMES:
        if hf_api.file_exists(repo_id=repo_id, filename=filename):
            download_from_hf(repo_id, filename, base_dir=base_dir)
            logger.info("Downloaded %s from %s", filename, repo_id)
            downloaded_any = True

    if not downloaded_any:
        raise FileNotFoundError(
            f"Neither model.vmfb nor model.vmfb.trim found in {repo_id}"
        )


def setup_gemma3(
    models: list[str],
):
    logger.info("Setting up gemma3 demo with models: [%s]", ", ".join(models))
    repos = [_HF_REPO_MAP.get(m, m) for m in models]
    base_dir = default_models_dir()
    for repo_id in repos:
        try:
            _download_gemma3_model(repo_id, base_dir)
            download_from_hf(repo_id, "token_embeddings.npy", base_dir=base_dir)
            download_from_hf(repo_id, "config.json", base_dir=base_dir)
            download_from_hf(repo_id, "tokenizer.json", base_dir=base_dir)
            logger.info("Downloaded gemma3 model files from %s", repo_id)
            # Optional: trimmed vocab LUT
            try:
                from huggingface_hub import HfApi

                hf_api = HfApi()
                if hf_api.file_exists(repo_id=repo_id, filename=_GEMMA3_TRIM_LUT_FILENAME):
                    download_from_hf(repo_id, _GEMMA3_TRIM_LUT_FILENAME, base_dir=base_dir)
            except Exception:
                pass
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
