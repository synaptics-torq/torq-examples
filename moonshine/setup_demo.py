# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.download import default_models_dir, download_from_hf, resolve_repo_id
from utils.model_setup import demo_main, download_models, ensure_demo_models, setup_demo

logger = logging.getLogger("moonshine.setup")

MOONSHINE_HF_REPO_MAP: Final[dict[str, str]] = {
    "tiny-en": "Synaptics/moonshine-tiny-bf16-torq",
}
_DEFAULT_MODELS: Final[list[str]] = ["tiny-en"]
_MOONSHINE_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "encoder.vmfb",
    "decoder.vmfb",
    "decoder_token_embeddings.npy",
    "tokenizer.json",
)


def _has_moonshine_files(model_dir: Path) -> bool:
    return all((model_dir / filename).exists() for filename in _MOONSHINE_REQUIRED_FILES)


def _download_moonshine(
    repo_id: str,
    base_dir: Path,
    *,
    revision: str | None = None,
) -> list[str]:
    """Download all required Moonshine files; return the manifest file list."""
    for filename in _MOONSHINE_REQUIRED_FILES:
        download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
    return list(_MOONSHINE_REQUIRED_FILES)


def download_moonshine(
    models: list[str] | None = None,
    *,
    base_dir: str | Path | None = None,
    model_version: str | None = None,
    no_update: bool = False,
) -> dict[str, Path]:
    """Download/refresh the given Moonshine models; return ``{name: model_dir}``.

    Version selection, ``name:version`` specs and ``no_update`` are handled by
    :func:`utils.model_setup.download_models`. Unlike :func:`setup_moonshine`,
    this does not check demo requirements, so it can be reused by other
    projects that manage their own environment and models dir.
    """
    if models is None:
        models = _DEFAULT_MODELS
    return download_models(
        MOONSHINE_HF_REPO_MAP,
        models,
        files_present=_has_moonshine_files,
        download=_download_moonshine,
        label="moonshine",
        base_dir=base_dir,
        model_version=model_version,
        no_update=no_update,
    )


def local_moonshine_model_dir(
    model: str | None = None,
    *,
    base_dir: str | Path | None = None,
) -> Path | None:
    """Return the local Moonshine model dir to default ``-m``/``--model-dir`` to.

    The runner takes a model *directory*, so this returns the dir (not a file).
    Looks for the complete required file set; with no ``model``, tries the
    built-in repos in repo-map order.
    """
    if base_dir is None:
        base_dir = default_models_dir()
    if model is not None:
        repo_ids = [resolve_repo_id(model, MOONSHINE_HF_REPO_MAP)]
    else:
        repo_ids = list(dict.fromkeys(MOONSHINE_HF_REPO_MAP.values()))
    for repo_id in repo_ids:
        model_dir = Path(base_dir) / repo_id
        if _has_moonshine_files(model_dir):
            return model_dir
    return None


def ensure_moonshine_models(
    model_dir: str | Path,
    *,
    refresh: bool = True,
) -> None:
    """Verify/refresh the Moonshine models in ``model_dir`` before inference.

    Delegates to :func:`utils.model_setup.ensure_demo_models`, which re-syncs
    the local copy to the version recorded in its manifest. Refresh failures
    are logged, not raised, so inference can still proceed on whatever is
    available locally.
    """
    ensure_demo_models(
        model_dir,
        refresh=refresh,
        demo_name="moonshine",
        files_present=_has_moonshine_files,
        download=_download_moonshine,
    )


def setup_moonshine(
    models: list[str] | None = None,
    model_version: str | None = None,
    no_update: bool = False,
):
    """Set up the moonshine demo (check requirements, then download/refresh)."""
    if models is None:
        models = _DEFAULT_MODELS
    return setup_demo(
        MOONSHINE_HF_REPO_MAP,
        models,
        demo_name="moonshine",
        requirements=Path(__file__).parent / "requirements.txt",
        files_present=_has_moonshine_files,
        download=_download_moonshine,
        model_version=model_version,
        no_update=no_update,
    )


if __name__ == "__main__":
    demo_main(
        setup_moonshine,
        description="Download Moonshine model files.",
        default_models=_DEFAULT_MODELS,
        repo_map=MOONSHINE_HF_REPO_MAP,
    )
