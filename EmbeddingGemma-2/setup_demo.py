# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.download import default_models_dir, download_from_hf
from utils.model_setup import demo_main, setup_demo

logger = logging.getLogger("EmbeddingGemma2.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/Google-EmbeddingGemma-2",
    "EmbeddingGemma-2": "Synaptics/Google-EmbeddingGemma-2",
}
_DEFAULT_MODELS: Final[list[str]] = ["default"]

# Weight builds on Hugging Face, chosen with --model-version (or `default:<version>`):
#   q8      branch `q8` (the default): 8-bit weights (int8 block-32, bf16 activations); much
#           closer to the original model than 4-bit at the same or better NPU speed
#   main    branch `main` or the version tag (e.g. v2.2.0): 4-bit weights (q4)
#   bf16    branch `bf16`: bf16 weights
# All builds use the same file names and go to the same model directory; switching
# replaces the files.
#
# The build downloaded when no --model-version is given: the 8-bit build on the `q8` branch.
# Once a release tag itself holds the 8-bit build, set it back to None (the version tag).
_DEFAULT_MODEL_VERSION: Final[str | None] = "q8"

_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "vision_384x384.vmfb",            # vision encoder, 384x384 -> 64 soft tokens
    "text_body_s128.vmfb",            # text body, 128 tokens (queries, images)
    "text_body_s512.vmfb",            # text body, 512 tokens (documents, video up to 7 frames, audio)
    "audio_140.vmfb",                 # audio encoder, 1.4 s -> up to 35 soft tokens
    "audio_280.vmfb",                 # audio encoder, 2.8 s -> up to 70 soft tokens
    "audio_560.vmfb",                 # audio encoder, 5.6 s -> up to 140 soft tokens
    "audio_1120.vmfb",                # audio encoder, 11.2 s -> up to 280 soft tokens
    "mel_filters.npy",                # audio front end: mel filter bank [257, 128]
    "mel_window.npy",                 # audio front end: analysis window [320]
    "token_embeddings.npy",           # host token-embedding LUT (bf16, mmap'd)
    "tokenizer.json",
    "config.json",
    "embeddinggemma2_manifest.json",
)

# Example media for `src/infer.py --examples`: CC0 / public-domain photos, sounds and videos with a
# description each (credits in examples/README.md). They live on the `main` branch whichever weight
# build --model-version selects, so they are always downloaded from there.
_EXAMPLES_REVISION: Final[str] = "main"
_EXAMPLE_FILES: Final[tuple[str, ...]] = tuple(f"examples/{f}" for f in (
    "examples.json", "README.md",
    "cat_tabby.jpg", "cat_meowing.wav", "dog_labrador.jpg", "dog_barking.wav",
    "bird_sparrow.jpg", "birds_chirping.wav", "piano_keyboard.jpg", "piano_arpeggio.wav",
    "train_locomotive.jpg", "train_passing.wav", "police_car.jpg", "police_siren.wav",
    "waterfall.ogv", "sea_turtle.ogv",
))


def _has_model_files(model_dir: Path) -> bool:
    return all((model_dir / f).exists() for f in _REQUIRED_FILES)


def _has_files(model_dir: Path) -> bool:
    return _has_model_files(model_dir) and all((model_dir / f).exists() for f in _EXAMPLE_FILES)


def local_embeddinggemma2_model_dir(base_dir: str | Path | None = None) -> Path | None:
    """The downloaded model directory to default ``--model-dir`` to, if it is complete."""
    base_dir = Path(base_dir) if base_dir is not None else default_models_dir()
    for repo_id in dict.fromkeys(_HF_REPO_MAP.values()):
        if _has_model_files(base_dir / repo_id):
            return base_dir / repo_id
    return None


def _download(repo_id: str, base_dir: Path, *, revision: str | None = None) -> list[str]:
    for f in _REQUIRED_FILES:
        download_from_hf(repo_id, f, base_dir=base_dir, revision=revision)
        logger.info("Downloaded %s from %s", f, repo_id)
    for f in _EXAMPLE_FILES:
        download_from_hf(repo_id, f, base_dir=base_dir, revision=_EXAMPLES_REVISION)
        logger.info("Downloaded %s from %s", f, repo_id)
    return list(_REQUIRED_FILES + _EXAMPLE_FILES)


def setup_embeddinggemma2(models: list[str] | None = None, model_version: str | None = None,
                          no_update: bool = False):
    """Set up the EmbeddingGemma-2 demo (download/refresh the model files)."""
    model_version = model_version or _DEFAULT_MODEL_VERSION
    return setup_demo(
        _HF_REPO_MAP,
        models or _DEFAULT_MODELS,
        demo_name="EmbeddingGemma-2",
        requirements=Path(__file__).parent / "requirements.txt",
        files_present=_has_files,
        download=_download,
        model_version=model_version,
        no_update=no_update,
    )


if __name__ == "__main__":
    demo_main(
        setup_embeddinggemma2,
        description="Download EmbeddingGemma-2 model files.",
        default_models=_DEFAULT_MODELS,
        repo_map=_HF_REPO_MAP,
    )
