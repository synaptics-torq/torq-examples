# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.download import download_from_hf, hf_file_exists, read_manifest
from utils.model_setup import demo_main, ensure_demo_models, setup_demo

logger = logging.getLogger("Liquid.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/LiquidAI-LFM2.5-230M",
    "230m": "Synaptics/LiquidAI-LFM2.5-230M",
    "350m": "Synaptics/LiquidAI-LFM2.5-350M",
}
_DEFAULT_MODELS: Final[list[str]] = ["default"]
# Ordered most-preferred first; the first set fully available in the repo is
# used. The new exporter emits transformer.vmfb + lm_head.vmfb (+ an optional
# transformer_prefill.vmfb, downloaded separately); the legacy sets keep old
# deployments working.
_LIQUID_MODEL_FILENAMES: Final[tuple[tuple[str, ...], ...]] = (
    ("transformer.vmfb", "lm_head.vmfb"),
    ("body.vmfb", "lm_head.vmfb"),
    ("model.vmfb",),
)
_LIQUID_PREFILL_FILENAME: Final[str] = "transformer_prefill.vmfb"
_LIQUID_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "token_embeddings.npy",
    "config.json",
    "tokenizer.json",
)


def _has_liquid_files(model_dir: Path) -> bool:
    has_model = any(
        all((model_dir / filename).exists() for filename in filenames)
        for filenames in _LIQUID_MODEL_FILENAMES
    )
    has_required = all(
        (model_dir / filename).exists() for filename in _LIQUID_REQUIRED_FILES
    )
    return has_model and has_required


def _local_liquid_model_files(local_dir: Path) -> list[str] | None:
    for filenames in _LIQUID_MODEL_FILENAMES:
        if all((local_dir / filename).exists() for filename in filenames):
            return list(filenames)
    return None


def _format_liquid_model_file_sets() -> str:
    return " or ".join(
        " + ".join(filenames) for filenames in _LIQUID_MODEL_FILENAMES
    )


def _download_liquid_model(
    repo_id: str, base_dir: Path, *, revision: str | None = None
) -> list[str]:
    """Download the first supported Liquid model file set available."""
    local_dir = base_dir / repo_id
    existing = _local_liquid_model_files(local_dir)
    if existing is not None:
        return existing

    available_cache: dict[str, bool] = {}

    def is_available(filename: str) -> bool:
        if (local_dir / filename).exists():
            return True
        if filename not in available_cache:
            available_cache[filename] = hf_file_exists(repo_id, filename, revision=revision)
        return available_cache[filename]

    for filenames in _LIQUID_MODEL_FILENAMES:
        if not all(is_available(filename) for filename in filenames):
            continue
        for filename in filenames:
            if (local_dir / filename).exists():
                continue
            download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
            logger.info("Downloaded %s from %s", filename, repo_id)
        return list(filenames)

    raise FileNotFoundError(
        f"No supported Liquid model file set found in {repo_id}; expected "
        f"{_format_liquid_model_file_sets()}"
    )


def _download_optional_if_exists(
    repo_id: str, filename: str, base_dir: Path, *, revision: str | None = None
) -> str | None:
    if not hf_file_exists(repo_id, filename, revision=revision):
        return None
    download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
    return filename


def _download_liquid(
    repo_id: str, base_dir: Path, *, revision: str | None = None
) -> list[str]:
    """Download all Liquid files; return the manifest file list."""
    manifest_files = _download_liquid_model(repo_id, base_dir, revision=revision)
    for filename in _LIQUID_REQUIRED_FILES:
        download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
        manifest_files.append(filename)

    prefill_file = _download_optional_if_exists(
        repo_id, _LIQUID_PREFILL_FILENAME, base_dir, revision=revision
    )
    if prefill_file is not None:
        manifest_files.append(prefill_file)
    return manifest_files


def _liquid_files_present(model_dir: Path) -> bool:
    """Whether a *tracked* copy is complete: manifest consistent with local files.

    The manifest records the model files that were downloaded, but any set in
    :data:`_LIQUID_MODEL_FILENAMES` is a valid local set: a run using
    ``body.vmfb`` is complete even when the manifest recorded
    ``transformer.vmfb``. So the model files are checked against those sets
    (via :func:`_has_liquid_files`) rather than against the recorded names,
    which are only verified for the non-model files.
    """
    if not _has_liquid_files(model_dir):
        return False
    manifest = read_manifest(model_dir)
    if manifest is None:
        return False
    files = manifest.get("files", [])
    if not files:
        return False
    model_filenames = {name for names in _LIQUID_MODEL_FILENAMES for name in names}
    return all(
        (model_dir / filename).exists()
        for filename in files
        if filename not in model_filenames
    )


def ensure_liquid_models(model_dir: str | Path, *, refresh: bool = True) -> None:
    """Verify/refresh the Liquid models in ``model_dir`` before inference.

    Delegates to :func:`utils.model_setup.ensure_demo_models`, which re-syncs
    the local copy to the version recorded in its manifest. Refresh failures
    are logged, not raised, so inference can still proceed on whatever is
    available locally (offline/airgapped runs, e.g. the board).
    """
    ensure_demo_models(
        model_dir,
        refresh=refresh,
        demo_name="LiquidAI-LFM2.5",
        files_present=_has_liquid_files,
        download=_download_liquid,
        tracked_files_present=_liquid_files_present,
    )


def setup_liquid(
    models: list[str] | None = None,
    model_version: str | None = None,
    no_update: bool = False,
):
    """Set up the LiquidAI-LFM2.5 demo.

    Version selection, ``name:version`` specs and ``no_update`` are handled by
    :func:`utils.model_setup.setup_demo`, which checks demo requirements first,
    then downloads/refreshes the models.
    """
    if models is None:
        models = _DEFAULT_MODELS
    return setup_demo(
        _HF_REPO_MAP,
        models,
        demo_name="LiquidAI-LFM2.5",
        requirements=Path(__file__).parent / "requirements.txt",
        files_present=_has_liquid_files,
        download=_download_liquid,
        model_version=model_version,
        no_update=no_update,
        tracked_files_present=_liquid_files_present,
    )


if __name__ == "__main__":
    demo_main(
        setup_liquid,
        description="Download LFM2.5 (Liquid) model files.",
        default_models=_DEFAULT_MODELS,
        repo_map=_HF_REPO_MAP,
    )
