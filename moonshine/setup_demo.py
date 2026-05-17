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

logger = logging.getLogger("moonshine.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "tiny-en": "Synaptics/moonshine-tiny-bf16-torq",
}
_MOONSHINE_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "encoder.vmfb",
    "decoder.vmfb",
    "decoder_with_past.vmfb",
    "decoder_token_embeddings.npy",
    "tokenizer.json",
)


def _hf_file_exists(repo_id: str, filename: str) -> bool:
    from huggingface_hub import HfApi

    return HfApi().file_exists(repo_id=repo_id, filename=filename)


def _has_moonshine_files(model_dir: Path) -> bool:
    return all((model_dir / filename).exists() for filename in _MOONSHINE_REQUIRED_FILES)


def _download_preprocessor(repo_id: str, base_dir: Path) -> str | None:
    preproc_vmfb = "preprocessor.vmfb"
    preproc_onnx = "preprocessor.onnx"
    local_dir = base_dir / repo_id
    if (local_dir / preproc_vmfb).exists():
        return preproc_vmfb
    if (local_dir / preproc_onnx).exists():
        return preproc_onnx

    has_vmfb = _hf_file_exists(repo_id, preproc_vmfb)
    has_onnx = _hf_file_exists(repo_id, preproc_onnx)

    if has_vmfb:
        if has_onnx:
            logger.warning(
                "Both optional preprocessor files exist in %s; using %s.",
                repo_id,
                preproc_vmfb,
            )
        download_from_hf(repo_id, preproc_vmfb, base_dir=base_dir)
        return preproc_vmfb

    if has_onnx:
        download_from_hf(repo_id, preproc_onnx, base_dir=base_dir)
        return preproc_onnx

    logger.info(
        "No preprocessor model found in %s; continuing without it.",
        repo_id,
    )
    return None


def setup_moonshine(
    models: list[str],
):
    logger.info("Setting up moonshine demo with models: [%s]", ", ".join(models))
    repos = [_HF_REPO_MAP.get(m, m) for m in models]
    base_dir = default_models_dir()
    for repo_id in repos:
        model_dir = base_dir / repo_id
        if verify_manifest(model_dir) and _has_moonshine_files(model_dir):
            logger.info("Using local moonshine model files from %s", model_dir)
            continue

        try:
            manifest_files: list[str] = []
            preprocessor = _download_preprocessor(repo_id, base_dir)
            if preprocessor is not None:
                manifest_files.append(preprocessor)

            for filename in _MOONSHINE_REQUIRED_FILES:
                download_from_hf(repo_id, filename, base_dir=base_dir)
                manifest_files.append(filename)

            write_manifest(model_dir, repo_id, manifest_files)
            logger.info("Downloaded moonshine model files from %s", repo_id)
        except Exception as e:
            raise DownloadError(f"Unable to download model files from {repo_id}") from e
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
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
