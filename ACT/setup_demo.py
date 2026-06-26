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

logger = logging.getLogger("ACT.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/ACT",
}
DEFAULT_REPO: Final[str] = _HF_REPO_MAP["default"]

# The three NSS modules of the ACT pipeline (all required).
_ACT_VMFB_FILES: Final[tuple[str, ...]] = (
    "backbone_fused.vmfb",
    "enc4_dattn.vmfb",
    "decoder_ffn.vmfb",
)
# Pre-encoder host-glue constants + action denorm stats (required for run.py).
_ACT_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "glue_params.npz",
)
# A bundled sample frame so the demo runs out of the box, and optional metadata.
_ACT_OPTIONAL_FILES: Final[tuple[str, ...]] = (
    "sample_image_i16.bin",
    "sample_state.bin",
    "config.json",
)


def _required_files() -> tuple[str, ...]:
    return _ACT_VMFB_FILES + _ACT_REQUIRED_FILES


def _hf_file_exists(repo_id: str, filename: str) -> bool:
    from huggingface_hub import HfApi

    return HfApi().file_exists(repo_id=repo_id, filename=filename)


def _has_act_files(model_dir: Path) -> bool:
    return all((model_dir / filename).exists() for filename in _required_files())


def _download_act(repo_id: str, base_dir: Path) -> list[str]:
    """Download all ACT files; return the manifest file list."""
    manifest_files: list[str] = []
    for filename in _required_files():
        download_from_hf(repo_id, filename, base_dir=base_dir)
        logger.info("Downloaded %s from %s", filename, repo_id)
        manifest_files.append(filename)
    for filename in _ACT_OPTIONAL_FILES:
        if _hf_file_exists(repo_id, filename):
            download_from_hf(repo_id, filename, base_dir=base_dir)
            manifest_files.append(filename)
    return manifest_files


def _refresh_act(repo_id: str, model_dir: Path, base_dir: Path) -> ModelStatus:
    files_present = verify_manifest(model_dir) and _has_act_files(model_dir)
    revision = get_hf_revision(repo_id)
    return ensure_model(
        model_dir,
        repo_id,
        files_present=files_present,
        revision=revision,
        download=lambda: _download_act(repo_id, base_dir),
    )


def ensure_act_models(model_dir: str | Path, *, refresh: bool = True) -> None:
    """Verify/refresh the ACT model files in ``model_dir`` before inference.

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
            "Run `python setup_demos.py ACT` if inference fails.",
            model_dir,
        )
        return
    try:
        _refresh_act(repo_id, model_dir, base_dir_for(model_dir, repo_id))
    except Exception as e:
        logger.warning(
            "Could not refresh models from %s (%s); using local files.", repo_id, e
        )


def setup_act(models: list[str]):
    logger.info("Setting up ACT demo with models: [%s]", ", ".join(models))
    repos = [_HF_REPO_MAP.get(m, m) for m in models]
    base_dir = default_models_dir()
    for repo_id in repos:
        model_dir = base_dir / repo_id
        try:
            status = _refresh_act(repo_id, model_dir, base_dir)
        except Exception as e:
            raise DownloadError(f"Unable to download model files from {repo_id}") from e
        if status is ModelStatus.UP_TO_DATE:
            logger.info("Using local ACT model files from %s", model_dir)
        else:
            logger.info("Downloaded ACT model files from %s", repo_id)
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("ACT setup complete.")


if __name__ == "__main__":
    import argparse
    import sys
    from utils.log import add_logging_args, configure_logging

    available_models = ", ".join(f"'{model_name}' ({repo_id})" for model_name, repo_id in _HF_REPO_MAP.items())
    parser = argparse.ArgumentParser(
        description="Download ACT model files.",
    )
    parser.add_argument(
        "models", nargs="*", default=["default"],
        help=f"Model name or HF repo ID. Built-in: [{available_models}] (default: %(default)s)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_act(args.models)
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
