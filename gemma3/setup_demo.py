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
    verify_manifest,
)

logger = logging.getLogger("Gemma3.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
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


def _refresh_gemma3(repo_id: str, model_dir: Path, base_dir: Path) -> ModelStatus:
    files_present = verify_manifest(model_dir) and _has_gemma3_files(model_dir)
    revision = get_hf_revision(repo_id)
    return ensure_model(
        model_dir,
        repo_id,
        files_present=files_present,
        revision=revision,
        download=lambda: _download_gemma3(repo_id, base_dir),
    )


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
    try:
        _refresh_gemma3(repo_id, model_dir, base_dir_for(model_dir, repo_id))
    except Exception as e:
        logger.warning(
            "Could not refresh models from %s (%s); using local files.", repo_id, e
        )


def setup_gemma3(
    models: list[str],
):
    logger.info("Setting up gemma3 demo with models: [%s]", ", ".join(models))
    repos = [_HF_REPO_MAP.get(m, m) for m in models]
    base_dir = default_models_dir()
    for repo_id in repos:
        model_dir = base_dir / repo_id
        try:
            status = _refresh_gemma3(repo_id, model_dir, base_dir)
        except Exception as e:
            raise DownloadError(f"Unable to download model files from {repo_id}") from e
        if status is ModelStatus.UP_TO_DATE:
            logger.info("Using local gemma3 model files from %s", model_dir)
        else:
            logger.info("Downloaded gemma3 model files from %s", repo_id)
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("gemma3 setup complete.")


if __name__ == "__main__":
    import argparse
    import sys
    from utils.log import add_logging_args, configure_logging

    available_models = ", ".join(f"'{model_name}' ({repo_id})" for model_name, repo_id in _HF_REPO_MAP.items())
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
