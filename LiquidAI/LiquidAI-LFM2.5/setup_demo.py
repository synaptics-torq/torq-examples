# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.download import (
    download_from_hf,
    hf_file_exists,
    read_manifest,
    resolve_repo_id,
)
from utils.model_setup import (
    demo_main,
    ensure_demo_models,
    setup_demo,
    sync_prefill_tracking,
)

logger = logging.getLogger("Liquid.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/LiquidAI-LFM2.5-230M",
    "230m": "Synaptics/LiquidAI-LFM2.5-230M",
    "350m": "Synaptics/LiquidAI-LFM2.5-350M",
    "230m-w8a8": "Synaptics/LiquidAI-LFM2.5-230M-w8a8-torq",
    "350m-w8a8": "Synaptics/LiquidAI-LFM2.5-350M-w8a8-torq",
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
    repo_id: str,
    base_dir: Path,
    *,
    revision: str | None = None,
    enable_prefill: bool = False,
) -> list[str]:
    """Download all Liquid files; return the manifest file list.

    The optional batched prefill model (``transformer_prefill.vmfb``) is
    downloaded only when ``enable_prefill`` is set, or when the existing
    manifest already tracks it, so repairing a previously enabled copy does
    not silently drop the prefill build.
    """
    manifest_files = _download_liquid_model(repo_id, base_dir, revision=revision)
    for filename in _LIQUID_REQUIRED_FILES:
        download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
        manifest_files.append(filename)

    manifest = read_manifest(Path(base_dir) / repo_id)
    tracks_prefill = (
        bool(manifest) and _LIQUID_PREFILL_FILENAME in manifest.get("files", [])
    )
    if enable_prefill or tracks_prefill:
        local_dir = Path(base_dir) / repo_id
        if (local_dir / _LIQUID_PREFILL_FILENAME).exists():
            manifest_files.append(_LIQUID_PREFILL_FILENAME)
        elif _download_optional_if_exists(
            repo_id, _LIQUID_PREFILL_FILENAME, base_dir, revision=revision
        ):
            manifest_files.append(_LIQUID_PREFILL_FILENAME)
        elif enable_prefill:
            logger.warning(
                "--with-prefill: no %s in %s; continuing without a batched "
                "prefill model.", _LIQUID_PREFILL_FILENAME, repo_id,
            )
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
    enable_prefill: bool = False,
):
    """Set up the LiquidAI-LFM2.5 demo.

    Version selection, ``name:version`` specs and ``no_update`` are handled by
    :func:`utils.model_setup.setup_demo`, which checks demo requirements first,
    then downloads/refreshes the models. ``enable_prefill`` additionally
    downloads and tracks the optional batched prefill model
    (``transformer_prefill.vmfb``): an extra model, so it uses more memory,
    with a fixed 64-token prompt chunk size. Without it, a tracked prefill
    model is removed from the manifest (the local file is kept but left
    untracked), so it stays excluded on later runs.
    """
    if models is None:
        models = _DEFAULT_MODELS

    def _tracked_complete(model_dir: Path) -> bool:
        """A complete copy — plus, with ``--with-prefill``, a prefill model that is
        tracked or at least present locally. The download hook is incremental
        (``download_from_hf`` skips existing files), so marking a complete copy
        incomplete only fetches the missing prefill build, or warns when the
        repo has none."""
        if not _liquid_files_present(model_dir):
            return False
        if enable_prefill:
            manifest = read_manifest(model_dir)
            files = manifest.get("files", []) if manifest else []
            if (
                _LIQUID_PREFILL_FILENAME not in files
                and not (model_dir / _LIQUID_PREFILL_FILENAME).exists()
            ):
                return False
        return True

    result = setup_demo(
        _HF_REPO_MAP,
        models,
        demo_name="LiquidAI-LFM2.5",
        requirements=Path(__file__).parent / "requirements.txt",
        files_present=_has_liquid_files,
        download=lambda repo_id, base_dir, *, revision=None: _download_liquid(
            repo_id, base_dir, revision=revision, enable_prefill=enable_prefill
        ),
        model_version=model_version,
        no_update=no_update,
        tracked_files_present=_tracked_complete,
    )
    for name, model_dir in result.items():
        sync_prefill_tracking(
            model_dir,
            resolve_repo_id(name, _HF_REPO_MAP),
            _LIQUID_PREFILL_FILENAME,
            enable_prefill=enable_prefill,
        )
    return result


if __name__ == "__main__":
    demo_main(
        setup_liquid,
        description="Download LFM2.5 (Liquid) model files.",
        default_models=_DEFAULT_MODELS,
        repo_map=_HF_REPO_MAP,
        supports_prefill=True,
    )
