# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.download import download_from_hf
from utils.model_setup import demo_main, download_models, ensure_demo_models, setup_demo

logger = logging.getLogger("moonshine_streaming.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "streaming-tiny-en": "Synaptics/moonshine-streaming-tiny-torq",
}
_DEFAULT_MODELS: Final[list[str]] = ["streaming-tiny-en"]
_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "encoder.vmfb",
    "decoder.vmfb",
    "streaming_config.json",
    "config.json",
    "adapter_pos_emb.npy",
    "decoder_token_embeddings.npy",
    "tokenizer.json",
)


def _has_required_files(model_dir: Path) -> bool:
    return all((model_dir / filename).exists() for filename in _REQUIRED_FILES)


def _download(repo_id: str, base_dir: Path, *, revision: str | None = None) -> list[str]:
    """Download all required streaming files; return the manifest file list."""
    for filename in _REQUIRED_FILES:
        download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
    return list(_REQUIRED_FILES)


def download_moonshine_streaming(
    models: list[str] | None = None,
    *,
    base_dir: str | Path | None = None,
    model_version: str | None = None,
    no_update: bool = False,
) -> dict[str, Path]:
    """Download/refresh the given streaming models; return ``{name: model_dir}``.

    Version selection, ``name:version`` specs and ``no_update`` are handled by
    :func:`utils.model_setup.download_models`. Unlike
    :func:`setup_moonshine_streaming`, this does not check demo requirements,
    so it can be reused by other projects that manage their own environment
    and models dir.
    """
    if models is None:
        models = _DEFAULT_MODELS
    return download_models(
        _HF_REPO_MAP,
        models,
        files_present=_has_required_files,
        download=_download,
        label="moonshine_streaming",
        base_dir=base_dir,
        model_version=model_version,
        no_update=no_update,
    )


def ensure_moonshine_streaming_models(
    model_dir: str | Path, *, refresh: bool = True
) -> None:
    """Verify/refresh the streaming models in ``model_dir`` before inference.

    Delegates to :func:`utils.model_setup.ensure_demo_models`, which re-syncs
    the local copy to the version recorded in its manifest. Refresh failures
    are logged, not raised, so inference can still proceed on whatever is
    available locally.
    """
    ensure_demo_models(
        model_dir,
        refresh=refresh,
        demo_name="moonshine_streaming",
        files_present=_has_required_files,
        download=_download,
    )


def setup_moonshine_streaming(
    models: list[str] | None = None,
    model_version: str | None = None,
    no_update: bool = False,
):
    """Set up the moonshine_streaming demo (requirements, then models)."""
    if models is None:
        models = _DEFAULT_MODELS
    return setup_demo(
        _HF_REPO_MAP,
        models,
        demo_name="moonshine_streaming",
        requirements=Path(__file__).parent / "requirements.txt",
        files_present=_has_required_files,
        download=_download,
        model_version=model_version,
        no_update=no_update,
    )


if __name__ == "__main__":
    demo_main(
        setup_moonshine_streaming,
        description="Download Moonshine streaming model files.",
        default_models=_DEFAULT_MODELS,
        repo_map=_HF_REPO_MAP,
    )
