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

logger = logging.getLogger("moonshine.setup")

_HF_REPO_MAP: Final[dict[str, str]] = {
    "tiny-en": "Synaptics/moonshine-tiny-bf16-torq",
}
_MOONSHINE_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "encoder.vmfb",
    "decoder.vmfb",
    "decoder_token_embeddings.npy",
    "tokenizer.json",
)


def _has_moonshine_files(model_dir: Path) -> bool:
    return all((model_dir / filename).exists() for filename in _MOONSHINE_REQUIRED_FILES)


def _download_moonshine(repo_id: str, base_dir: Path) -> list[str]:
    """Download all required Moonshine files; return the manifest file list."""
    for filename in _MOONSHINE_REQUIRED_FILES:
        download_from_hf(repo_id, filename, base_dir=base_dir)
    return list(_MOONSHINE_REQUIRED_FILES)


def _refresh_moonshine(repo_id: str, model_dir: Path, base_dir: Path) -> ModelStatus:
    files_present = verify_manifest(model_dir) and _has_moonshine_files(model_dir)
    revision = get_hf_revision(repo_id)
    return ensure_model(
        model_dir,
        repo_id,
        files_present=files_present,
        revision=revision,
        download=lambda: _download_moonshine(repo_id, base_dir),
    )


def ensure_moonshine_models(model_dir: str | Path, *, refresh: bool = True) -> None:
    """Verify/refresh the Moonshine models in ``model_dir`` before inference.

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
            "Run `python setup_demos.py moonshine` if inference fails.",
            model_dir,
        )
        return
    try:
        _refresh_moonshine(repo_id, model_dir, base_dir_for(model_dir, repo_id))
    except Exception as e:
        logger.warning(
            "Could not refresh models from %s (%s); using local files.", repo_id, e
        )


def setup_moonshine(
    models: list[str],
):
    logger.info("Setting up moonshine demo with models: [%s]", ", ".join(models))
    repos = [_HF_REPO_MAP.get(m, m) for m in models]
    base_dir = default_models_dir()
    for repo_id in repos:
        model_dir = base_dir / repo_id
        try:
            status = _refresh_moonshine(repo_id, model_dir, base_dir)
        except Exception as e:
            raise DownloadError(f"Unable to download model files from {repo_id}") from e
        if status is ModelStatus.UP_TO_DATE:
            logger.info("Using local moonshine model files from %s", model_dir)
        else:
            logger.info("Downloaded moonshine model files from %s", repo_id)
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("moonshine setup complete.")


if __name__ == "__main__":
    import argparse
    import sys
    from utils.log import add_logging_args, configure_logging

    available_models = ", ".join(f"'{model_name}' ({repo_id})" for model_name, repo_id in _HF_REPO_MAP.items())
    parser = argparse.ArgumentParser(
        description="Download Moonshine model files.",
    )
    parser.add_argument(
        "models", nargs="*", default=["tiny-en"],
        help=f"Model name or HF repo ID. Built-in: [{available_models}] (default: %(default)s)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_moonshine(args.models)
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
