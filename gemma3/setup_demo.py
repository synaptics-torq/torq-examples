# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.deps import MissingRequirementsError, check_requirements
from utils.download import (
    DownloadError,
    ModelStatus,
    base_dir_for,
    default_models_dir,
    download_from_hf,
    ensure_model,
    get_hf_revision,
    read_manifest,
    resolve_repo_id,
)

logger = logging.getLogger("Gemma3.setup")

GEMMA3_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/gemma-3-270m-torq",
    "instruct": "Synaptics/gemma-3-270m-it-torq"
}
_GEMMA3_MODEL_FILENAMES: Final[tuple[tuple[str, ...], ...]] = (
    ("transformer.vmfb", "lm_head.vmfb.trim"),
    ("transformer.vmfb", "lm_head.vmfb"),
    ("model.vmfb.trim",),
    ("model.vmfb",),
)
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
        all((model_dir / filename).exists() for filename in filenames)
        for filenames in _GEMMA3_MODEL_FILENAMES
    )
    has_required = all((model_dir / filename).exists() for filename in _GEMMA3_REQUIRED_FILES)
    return has_model and has_required


def _local_gemma3_model_files(local_dir: Path) -> list[str] | None:
    for filenames in _GEMMA3_MODEL_FILENAMES:
        if all((local_dir / filename).exists() for filename in filenames):
            return list(filenames)
    return None


def _format_gemma3_model_file_sets() -> str:
    return " or ".join(
        " + ".join(filenames) for filenames in _GEMMA3_MODEL_FILENAMES
    )


def _download_gemma3_model(repo_id: str, base_dir: Path) -> list[str]:
    """Download the first supported Gemma3 model file set available."""
    local_dir = base_dir / repo_id
    existing = _local_gemma3_model_files(local_dir)
    if existing is not None:
        return existing

    available_cache: dict[str, bool] = {}

    def is_available(filename: str) -> bool:
        if (local_dir / filename).exists():
            return True
        if filename not in available_cache:
            available_cache[filename] = _hf_file_exists(repo_id, filename)
        return available_cache[filename]

    for filenames in _GEMMA3_MODEL_FILENAMES:
        if not all(is_available(filename) for filename in filenames):
            continue
        for filename in filenames:
            if (local_dir / filename).exists():
                continue
            download_from_hf(repo_id, filename, base_dir=base_dir)
            logger.info("Downloaded %s from %s", filename, repo_id)
        return list(filenames)

    raise FileNotFoundError(
        f"No supported Gemma3 model file set found in {repo_id}; expected "
        f"{_format_gemma3_model_file_sets()}"
    )


def _download_optional_if_exists(repo_id: str, filename: str, base_dir: Path) -> str | None:
    if not _hf_file_exists(repo_id, filename):
        return None
    download_from_hf(repo_id, filename, base_dir=base_dir)
    return filename


def _download_gemma3(repo_id: str, base_dir: Path) -> list[str]:
    """Download all Gemma3 files; return the manifest file list."""
    manifest_files = _download_gemma3_model(repo_id, base_dir)
    for filename in _GEMMA3_REQUIRED_FILES:
        download_from_hf(repo_id, filename, base_dir=base_dir)
        manifest_files.append(filename)

    lut_file = _download_optional_if_exists(repo_id, _GEMMA3_TRIM_LUT_FILENAME, base_dir)
    if lut_file is not None:
        manifest_files.append(lut_file)
    return manifest_files


def _gemma3_files_present(model_dir: Path) -> bool:
    """Whether everything needed to run is present in ``model_dir``.

    The manifest records the model files that were downloaded, but any set in
    :data:`_GEMMA3_MODEL_FILENAMES` is a valid local set: a run using
    ``lm_head.vmfb`` is complete even when the manifest recorded
    ``lm_head.vmfb.trim``. So the model files are checked against those sets
    (via :func:`_has_gemma3_files`) rather than against the recorded names,
    which are only verified for the non-model files.
    """
    if not _has_gemma3_files(model_dir):
        return False
    manifest = read_manifest(model_dir)
    if manifest is None:
        return False
    files = manifest.get("files", [])
    if not files:
        return False
    model_filenames = {name for names in _GEMMA3_MODEL_FILENAMES for name in names}
    return all(
        (model_dir / filename).exists()
        for filename in files
        if filename not in model_filenames
    )


def _refresh_gemma3(repo_id: str, model_dir: Path, base_dir: Path) -> ModelStatus:
    files_present = _gemma3_files_present(model_dir)
    revision = get_hf_revision(repo_id)
    return ensure_model(
        model_dir,
        repo_id,
        files_present=files_present,
        revision=revision,
        download=lambda: _download_gemma3(repo_id, base_dir),
    )


def local_gemma3_model_path(
    model: str = "instruct",
    *,
    base_dir: str | Path | None = None,
) -> Path | None:
    """Return the local ``model.vmfb.trim`` path for ``model`` if it exists."""
    if base_dir is None:
        base_dir = default_models_dir()
    repo_id = resolve_repo_id(model, GEMMA3_HF_REPO_MAP)
    model_path = Path(base_dir) / repo_id / "model.vmfb.trim"
    return model_path if model_path.exists() else None


def download_gemma3(
    models: list[str] | None = None,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Download/refresh the given Gemma3 models; return ``{name: model_dir}``.

    Unlike :func:`setup_gemma3`, this does not check demo requirements, so it can
    be reused by other projects that manage their own environment and models dir.
    """
    if models is None:
        models = ["instruct"]
    if base_dir is None:
        base_dir = default_models_dir()
    base_dir = Path(base_dir)

    logger.info("Resolving Gemma3 models: [%s]", ", ".join(models))
    result: dict[str, Path] = {}
    for name in models:
        repo_id = resolve_repo_id(name, GEMMA3_HF_REPO_MAP)
        model_dir = base_dir / repo_id
        try:
            _refresh_gemma3(repo_id, model_dir, base_dir)
        except Exception as exc:
            raise DownloadError(f"Unable to download Gemma3 files from {repo_id}") from exc
        result[name] = model_dir
        logger.info("Gemma3 model files ready at '%s'", model_dir)
    return result


def ensure_gemma3_models(model_dir: str | Path, *, refresh: bool = True) -> None:
    """Verify/refresh the Gemma3 models in ``model_dir`` before inference.

    Reads the repo id from the local manifest and applies the same revision
    check as setup. When ``refresh`` is ``False`` the check is skipped entirely
    (offline/airgapped runs). Refresh failures are logged, not raised, so
    inference can still proceed on whatever is available locally.
    """
    model_dir = Path(model_dir)
    if not refresh:
        return
    manifest = read_manifest(model_dir)
    repo_id = manifest.get("repo_id") if manifest else None
    if not repo_id:
        logger.warning(
            "No manifest in %s; cannot verify model freshness. "
            "Run `python setup_demos.py gemma3` if inference fails.",
            model_dir,
        )
        return
    base_dir = base_dir_for(model_dir, repo_id)
    if base_dir is None:
        logger.warning(
            "%s is not laid out as <models dir>/%s; skipping the freshness check "
            "so a refresh cannot fetch a second copy elsewhere. "
            "Run `python setup_demos.py gemma3` to manage models.",
            model_dir,
            repo_id,
        )
        return
    try:
        _refresh_gemma3(repo_id, model_dir, base_dir)
    except Exception as e:
        logger.warning(
            "Could not refresh models from %s (%s); using local files.", repo_id, e
        )


def setup_gemma3(
    models: list[str],
):
    logger.info("Setting up gemma3 demo with models: [%s]", ", ".join(models))
    download_gemma3(models)
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("gemma3 setup complete.")


if __name__ == "__main__":
    import argparse
    import sys
    from utils.log import add_logging_args, configure_logging

    available_models = ", ".join(f"'{model_name}' ({repo_id})" for model_name, repo_id in GEMMA3_HF_REPO_MAP.items())
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
