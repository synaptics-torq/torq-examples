# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download the Piper TTS demo assets from Hugging Face.

Pulls, per voice, partA (the ORT/CPU text-encoder + duration predictor), the five
NSS-only bf16 partB vocoder vmfbs (1/2/4/6/8 s windows) and the voice config that
carries the phoneme->id map, plus the espeak-ng phonemizer assets shared by all
voices, from ``Synaptics/Piper-TTS`` into the shared ``models/`` dir. The espeak
data tarball is unpacked and the phonemizer daemon made executable on first
download.
"""

import logging
import tarfile
from pathlib import Path
from typing import Final

from piper_tts.piper_core.voices import VOICES, Voice, get_voice
from utils.deps import MissingRequirementsError, check_requirements
from utils.download import DownloadError, default_models_dir, download_from_hf

logger = logging.getLogger("piper_tts.setup")

PIPER_REPO_ID: Final[str] = "Synaptics/Piper-TTS"
WINDOWS: Final[tuple[int, ...]] = (1, 2, 4, 6, 8)

# espeak ships the daemon binary plus its dictionaries (unpacked below); one copy
# covers every language, so a voice adds no phonemizer assets.
ESPEAK_FILES: Final[tuple[str, ...]] = ("espeak/phonemizerd", "espeak/espeak-ng-data.tar.gz")


def voice_files(voice: Voice) -> tuple[str, ...]:
    """The per-voice assets: partA (CPU), the partB windows (NPU), the config.

    partA is the 82%-of-nodes half of the VITS graph that ends at the exact frame
    count, so the vocoder window is known before partB runs.
    """
    return (voice.asset("onnx", "partA.onnx"),
            *(voice.asset("vmfb", f"partB_static_{s}s.vmfb") for s in WINDOWS),
            voice.asset("voice", voice.config_name))


def _unpack_espeak(model_dir: Path) -> None:
    """Unpack espeak-ng-data and mark the phonemizer daemon executable."""
    tarball, data_dir = model_dir / "espeak" / "espeak-ng-data.tar.gz", model_dir / "espeak" / "espeak-ng-data"
    if tarball.exists() and not data_dir.exists():
        logger.debug("Unpacking %s", tarball)
        with tarfile.open(tarball) as tf:
            tf.extractall(tarball.parent)
    daemon = model_dir / "espeak" / "phonemizerd"
    if daemon.exists():
        daemon.chmod(0o755)


def download_piper(base_dir: str | Path | None = None, voices: tuple[Voice, ...] | None = None) -> Path:
    """Download the assets for ``voices`` (default: all); return the model dir.

    Files already present are not re-downloaded (see
    :func:`utils.download.download_from_hf`).
    """
    base_dir = Path(base_dir) if base_dir is not None else default_models_dir()
    model_dir = base_dir / PIPER_REPO_ID
    voices = voices if voices is not None else tuple(VOICES.values())
    unpacked = (model_dir / "espeak" / "espeak-ng-data").exists()
    wanted = (*(f for v in voices for f in voice_files(v)), *ESPEAK_FILES)
    for filename in wanted:
        if unpacked and filename.endswith(".tar.gz"):
            continue  # espeak dictionaries already unpacked; no need for the archive
        download_from_hf(PIPER_REPO_ID, filename, base_dir=base_dir)
    _unpack_espeak(model_dir)
    return model_dir


def ensure_piper_models(model_dir: str | Path | None = None, *, refresh: bool = True,
                        voice: Voice | str | None = None) -> Path:
    """Ensure one voice's Piper assets are present; return the model dir.

    ``model_dir`` may be the ``.../Synaptics/Piper-TTS`` dir (as passed by
    ``infer.py``) or ``None`` to use the shared ``models/`` dir. Only the
    requested voice is fetched, so speaking English never downloads the Spanish
    vocoder. Already-present files are never re-downloaded.

    ``refresh=False`` skips Hugging Face entirely and runs against whatever is on
    disk, which is what makes locally built assets usable; anything missing is
    named in the error rather than being fetched.
    """
    if model_dir is None:
        base_dir = default_models_dir()
    else:
        base_dir = Path(model_dir)
        for _ in Path(PIPER_REPO_ID).parts:  # strip Synaptics/Piper-TTS -> base
            base_dir = base_dir.parent
    if not isinstance(voice, Voice):
        voice = get_voice(voice)
    if not refresh:
        # What the demo actually needs to run: partA, the config, the phonemizer,
        # and at least one window. A partial set of windows is legitimate — the
        # pipeline picks from whichever vmfbs are present.
        local = base_dir / PIPER_REPO_ID
        required = (voice.asset("onnx", "partA.onnx"), voice.asset("voice", voice.config_name),
                    "espeak/phonemizerd", "espeak/espeak-ng-data")
        missing = [f for f in required if not (local / f).exists()]
        if not list((local / voice.asset("vmfb")).glob("partB_static_*s.vmfb")):
            missing.append(voice.asset("vmfb", "partB_static_*s.vmfb"))
        if missing:
            raise DownloadError(f"--no-refresh but {len(missing)} asset(s) missing under {local}: "
                                + ", ".join(missing))
        return local
    return download_piper(base_dir, voices=(voice,))


def setup_piper() -> None:
    """``setup_demos.py`` entry point: verify deps + download assets."""
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("Setting up Piper TTS demo from %s", PIPER_REPO_ID)
    try:
        model_dir = download_piper()
    except Exception as e:  # noqa: BLE001 - surface as a DownloadError to setup_demos
        raise DownloadError(f"Unable to download Piper assets from {PIPER_REPO_ID}") from e
    logger.info("Piper assets ready at %s", model_dir)


if __name__ == "__main__":
    import argparse
    import sys

    from utils.log import add_logging_args, configure_logging

    parser = argparse.ArgumentParser(description="Download the Piper TTS demo assets.")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)
    try:
        setup_piper()
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
