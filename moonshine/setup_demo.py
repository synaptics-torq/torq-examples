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
    verify_manifest,
)

logger = logging.getLogger("moonshine.setup")

MOONSHINE_HF_REPO_MAP: Final[dict[str, str]] = {
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


def download_moonshine(
    models: list[str] | None = None,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Download/refresh the given Moonshine models; return ``{name: model_dir}``.

    Unlike :func:`setup_moonshine`, this does not check demo requirements, so it
    can be reused by other projects that manage their own environment and models dir.
    """
    if models is None:
        models = ["tiny-en"]
    if base_dir is None:
        base_dir = default_models_dir()
    base_dir = Path(base_dir)

    logger.info("Resolving Moonshine models: [%s]", ", ".join(models))
    result: dict[str, Path] = {}
    for name in models:
        repo_id = resolve_repo_id(name, MOONSHINE_HF_REPO_MAP)
        model_dir = base_dir / repo_id
        try:
            _refresh_moonshine(repo_id, model_dir, base_dir)
        except Exception as exc:
            raise DownloadError(f"Unable to download Moonshine files from {repo_id}") from exc
        result[name] = model_dir
        logger.info("Moonshine model files ready at '%s'", model_dir)
    return result


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
    download_moonshine(models)
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("moonshine setup complete.")


if __name__ == "__main__":
    import argparse
    import sys
    from utils.log import add_logging_args, configure_logging

    available_models = ", ".join(f"'{model_name}' ({repo_id})" for model_name, repo_id in MOONSHINE_HF_REPO_MAP.items())
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
