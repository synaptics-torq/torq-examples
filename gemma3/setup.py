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

logger = logging.getLogger("Gemma3.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/gemma-3-270m-torq",
    "instruct": "Synaptics/gemma-3-270m-it-torq"
}
_GEMMA3_MODEL_FILENAMES: Final[list[str]] = [
    "model.vmfb",
    "model.vmfb.trim",
]
_GEMMA3_TRIM_LUT_FILENAME: Final[str] = "token_id_lut.npy"
_GEMMA3_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "token_embeddings.npy",
    "config.json",
    "tokenizer.json",
)


def _hf_file_exists(repo_id: str, filename: str) -> bool:
    from huggingface_hub import HfApi

    return HfApi().file_exists(repo_id=repo_id, filename=filename)


def _has_gemma3_files(model_dir: Path) -> bool:
    has_model = any(
        (model_dir / filename).exists() for filename in _GEMMA3_MODEL_FILENAMES
    )
    has_required = all((model_dir / filename).exists() for filename in _GEMMA3_REQUIRED_FILES)
    return has_model and has_required


def _download_gemma3_model(repo_id: str, base_dir: Path) -> list[str]:
    """Download model.vmfb and/or model.vmfb.trim as they exist."""
    local_dir = base_dir / repo_id
    existing = [
        filename for filename in _GEMMA3_MODEL_FILENAMES
        if (local_dir / filename).exists()
    ]
    if existing:
        return existing

    downloaded: list[str] = []
    for filename in _GEMMA3_MODEL_FILENAMES:
        if _hf_file_exists(repo_id, filename):
            download_from_hf(repo_id, filename, base_dir=base_dir)
            logger.info("Downloaded %s from %s", filename, repo_id)
            downloaded.append(filename)

    if not downloaded:
        raise FileNotFoundError(
            f"Neither model.vmfb nor model.vmfb.trim found in {repo_id}"
        )
    return downloaded


def _download_optional_if_exists(repo_id: str, filename: str, base_dir: Path) -> str | None:
    if not _hf_file_exists(repo_id, filename):
        return None
    download_from_hf(repo_id, filename, base_dir=base_dir)
    return filename


def setup_gemma3(
    models: list[str],
):
    logger.info("Setting up gemma3 demo with models: [%s]", ", ".join(models))
    repos = [_HF_REPO_MAP.get(m, m) for m in models]
    base_dir = default_models_dir()
    for repo_id in repos:
        model_dir = base_dir / repo_id
        if verify_manifest(model_dir) and _has_gemma3_files(model_dir):
            logger.info("Using local gemma3 model files from %s", model_dir)
            continue

        try:
            manifest_files = _download_gemma3_model(repo_id, base_dir)
            for filename in _GEMMA3_REQUIRED_FILES:
                download_from_hf(repo_id, filename, base_dir=base_dir)
                manifest_files.append(filename)

            lut_file = _download_optional_if_exists(
                repo_id, _GEMMA3_TRIM_LUT_FILENAME, base_dir
            )
            if lut_file is not None:
                manifest_files.append(lut_file)

            write_manifest(model_dir, repo_id, manifest_files)
            logger.info("Downloaded gemma3 model files from %s", repo_id)
        except Exception as e:
            raise DownloadError(f"Unable to download model files from {repo_id}") from e
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
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
