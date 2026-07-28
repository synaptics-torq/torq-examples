# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import ctypes.util
import logging
import subprocess
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

logger = logging.getLogger("moonshine_streaming.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "streaming-tiny-en": "Synaptics/moonshine-streaming-tiny-torq",
}
_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "encoder.vmfb",
    "decoder.vmfb",
    "streaming_config.json",
    "config.json",
    "adapter_pos_emb.npy",
    "decoder_token_embeddings.npy",
    "tokenizer.json",
)
_INSTALL_PORTAUDIO_SCRIPT: Final[Path] = Path(__file__).resolve().parent.parent / "setup" / "install_portaudio.sh"


class PortAudioInstallError(RuntimeError):
    """Raised when PortAudio is missing and cannot be installed automatically."""


def _has_required_files(model_dir: Path) -> bool:
    return all((model_dir / filename).exists() for filename in _REQUIRED_FILES)


def _portaudio_installed() -> bool:
    return ctypes.util.find_library("portaudio") is not None


def _install_portaudio():
    if not _INSTALL_PORTAUDIO_SCRIPT.exists():
        raise PortAudioInstallError(
            f"PortAudio is not installed and the install script was not found at {_INSTALL_PORTAUDIO_SCRIPT}"
        )
    logger.info("PortAudio not found; installing from %s", _INSTALL_PORTAUDIO_SCRIPT)
    try:
        subprocess.run([str(_INSTALL_PORTAUDIO_SCRIPT)], check=True)
    except subprocess.CalledProcessError as e:
        raise PortAudioInstallError("Failed to install PortAudio") from e
    if not _portaudio_installed():
        raise PortAudioInstallError("PortAudio install script ran but libportaudio is still not found")


def setup_moonshine_streaming(
    models: list[str],
):
    logger.info("Setting up moonshine_streaming demo with models: [%s]", ", ".join(models))
    if not _portaudio_installed():
        _install_portaudio()
    repos = [_HF_REPO_MAP.get(m, m) for m in models]
    base_dir = default_models_dir()
    for repo_id in repos:
        model_dir = base_dir / repo_id
        if verify_manifest(model_dir) and _has_required_files(model_dir):
            logger.info("Using local moonshine_streaming model files from %s", model_dir)
            continue

        try:
            manifest_files: list[str] = []
            for filename in _REQUIRED_FILES:
                download_from_hf(repo_id, filename, base_dir=base_dir)
                manifest_files.append(filename)

            write_manifest(model_dir, repo_id, manifest_files)
            logger.info("Downloaded moonshine_streaming model files from %s", repo_id)
        except Exception as e:
            raise DownloadError(f"Unable to download model files from {repo_id}") from e
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("moonshine_streaming setup complete.")


if __name__ == "__main__":
    import argparse
    import sys
    from utils.log import add_logging_args, configure_logging

    available_models = ", ".join(f"'{model_name}' ({repo_id})" for model_name, repo_id in _HF_REPO_MAP.items())
    parser = argparse.ArgumentParser(
        description="Download Moonshine streaming model files.",
    )
    parser.add_argument(
        "models", nargs="*", default=["streaming-tiny-en"],
        help=f"Model name or HF repo ID. Built-in: [{available_models}] (default: %(default)s)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_moonshine_streaming(args.models)
    except (DownloadError, MissingRequirementsError, PortAudioInstallError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
