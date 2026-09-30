# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download the Piper TTS demo assets from Hugging Face.

Pulls, per voice, partA (the ORT/CPU text-encoder + duration predictor), the
NSS-only bf16 partB vocoder vmfbs (one per window in ``Voice.windows``) and the voice config that
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
from utils.deps import check_requirements
from utils.download import DownloadError, base_dir_for, default_models_dir, download_from_hf, read_manifest
from utils.model_setup import demo_main, download_models, ensure_demo_models
from utils.version import parse_model_specs

logger = logging.getLogger("piper_tts.setup")

PIPER_REPO_ID: Final[str] = "Synaptics/Piper-TTS"
# Every voice is a "model" in the one repo: each has its own file check and
# download, and the shared manifest records the union of the voices fetched.
PIPER_REPO_MAP: Final[dict[str, str]] = {key: PIPER_REPO_ID for key in VOICES}

# espeak ships the daemon binary plus its dictionaries (unpacked below); one copy
# covers every language, so a voice adds no phonemizer assets.
ESPEAK_FILES: Final[tuple[str, ...]] = ("espeak/phonemizerd", "espeak/espeak-ng-data.tar.gz")


def voice_files(voice: Voice) -> tuple[str, ...]:
    """The per-voice assets: partA (CPU), the partB windows (NPU), the config.

    partA is the 82%-of-nodes half of the VITS graph that ends at the exact frame
    count, so the vocoder window is known before partB runs.
    """
    return (voice.asset("onnx", "partA.onnx"),
            *(voice.asset("vmfb", f"partB_static_{s:g}s.vmfb") for s in voice.windows),
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


def _voice_hooks(voice: Voice) -> dict:
    """The ``utils.model_setup`` hooks for one voice (plus the shared espeak assets)."""
    wanted = (*voice_files(voice), *ESPEAK_FILES)

    def files_present(model_dir: Path) -> bool:
        return all((Path(model_dir) / f).exists() for f in wanted)

    def tracked_files_present(model_dir: Path) -> bool:
        manifest = read_manifest(model_dir)
        return manifest is not None and set(wanted) <= set(manifest.get("files", ())) and files_present(model_dir)

    def download(repo_id: str, base_dir: Path, *, revision: str | None = None) -> list[str]:
        model_dir = Path(base_dir) / repo_id
        have = [f for f in (read_manifest(model_dir) or {}).get("files", ()) if (model_dir / f).exists()]
        for filename in wanted:
            download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
        _unpack_espeak(model_dir)
        return sorted(set(have) | set(wanted))   # keep voices fetched earlier tracked

    return {"files_present": files_present, "download": download,
            "tracked_files_present": tracked_files_present}


def download_piper(voices: list[str] | None = None, *, base_dir: str | Path | None = None,
                   model_version: str | None = None, no_update: bool = False) -> Path:
    """Download/refresh ``voices`` (keys or ``key:version`` specs; default all); return the model dir.

    Versions follow :func:`utils.model_setup.download_models`: by default the
    files tagged with the torq-examples version.
    """
    model_dir = (Path(base_dir) if base_dir is not None else default_models_dir()) / PIPER_REPO_ID
    for key, version in parse_model_specs(voices or list(VOICES)):
        voice = get_voice(key)
        download_models({voice.key: PIPER_REPO_ID}, [f"{key}:{version}" if version else key],
                        label="piper_tts", base_dir=base_dir, model_version=model_version,
                        no_update=no_update, **_voice_hooks(voice))
        _unpack_espeak(model_dir)
    return model_dir


def ensure_piper_models(model_dir: str | Path | None = None, *, refresh: bool = True,
                        voice: Voice | str | None = None) -> Path:
    """Ensure one voice's Piper assets are present and current; return the model dir.

    ``model_dir`` may be the ``.../Synaptics/Piper-TTS`` dir (as passed by
    ``infer.py``) or ``None`` to use the shared ``models/`` dir. Only the
    requested voice is fetched, so speaking English never downloads the Spanish
    vocoder. A tracked copy is kept in sync with the version in its manifest; a
    first run with nothing on disk downloads the torq-examples version. If the
    Hub can't be reached, whatever is on disk is used.

    ``refresh=False`` skips Hugging Face entirely and runs against whatever is on
    disk, which is what makes locally built assets usable; anything missing is
    named in the error rather than being fetched.
    """
    local = Path(model_dir) if model_dir is not None else default_models_dir() / PIPER_REPO_ID
    if not isinstance(voice, Voice):
        voice = get_voice(voice)
    hooks = _voice_hooks(voice)
    if refresh:
        base_dir = base_dir_for(local, PIPER_REPO_ID)
        if read_manifest(local) is not None:
            ensure_demo_models(local, refresh=True, demo_name="piper_tts", **hooks)
        elif base_dir is not None and not hooks["files_present"](local):
            try:
                download_piper([voice.key], base_dir=base_dir)
            except DownloadError as e:
                logger.warning("Could not download %s (%s); using local files.", voice.key, e.__cause__ or e)
    _unpack_espeak(local)
    # What the demo actually needs to run: partA, the config, the phonemizer,
    # and at least one window. A partial set of windows is legitimate — the
    # pipeline picks from whichever vmfbs are present.
    required = (voice.asset("onnx", "partA.onnx"), voice.asset("voice", voice.config_name),
                "espeak/phonemizerd", "espeak/espeak-ng-data")
    missing = [f for f in required if not (local / f).exists()]
    if not list((local / voice.asset("vmfb")).glob("partB_static_*s.vmfb")):
        missing.append(voice.asset("vmfb", "partB_static_*s.vmfb"))
    if missing:
        raise DownloadError(f"{len(missing)} Piper asset(s) missing under {local}: " + ", ".join(missing))
    return local


def setup_piper(models: list[str] | None = None, model_version: str | None = None,
                no_update: bool = False) -> dict[str, Path]:
    """Set up the piper_tts demo (check requirements, then download/refresh voices)."""
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("Setting up piper_tts demo from %s", PIPER_REPO_ID)
    model_dir = download_piper(models, model_version=model_version, no_update=no_update)
    logger.info("Piper assets ready at %s", model_dir)
    return {key: model_dir for key, _ in parse_model_specs(models or list(VOICES))}


if __name__ == "__main__":
    demo_main(setup_piper, description="Download the Piper TTS voices.",
              default_models=list(VOICES), repo_map=PIPER_REPO_MAP)
