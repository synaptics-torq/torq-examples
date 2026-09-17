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
from utils.version import parse_model_specs, resolve_model_version

logger = logging.getLogger("moonshine.setup")

MOONSHINE_HF_REPO_MAP: Final[dict[str, str]] = {
    "tiny-en": "Synaptics/moonshine-tiny-bf16-torq",
}
_BUILTIN_REPOS: Final[frozenset[str]] = frozenset(MOONSHINE_HF_REPO_MAP.values())
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


def _refresh_moonshine(
    repo_id: str,
    model_dir: Path,
    base_dir: Path,
    *,
    version: str | None,
    record: bool = True,
) -> ModelStatus:
    files_present = _has_moonshine_files(model_dir)
    revision = None
    if record:
        files_present = verify_manifest(model_dir) and files_present
        if version is not None:
            revision = get_hf_revision(repo_id, revision=version)
    return ensure_model(
        model_dir,
        repo_id,
        files_present=files_present,
        version=version if record else None,
        revision=revision,
        download=lambda: _download_moonshine(repo_id, base_dir, revision=version),
        record=record,
    )


def download_moonshine(
    models: list[str] | None = None,
    *,
    base_dir: str | Path | None = None,
    model_version: str | None = None,
    no_update: bool = False,
) -> dict[str, Path]:
    """Download/refresh the given Moonshine models; return ``{name: model_dir}``.

    ``models`` entries may be built-in names, raw HF repo ids, or
    ``name:version`` to pin a specific version for that model. ``model_version``
    applies to every model without its own ``:version``; built-in repos default
    to the torq-examples version and custom repos to their latest (HEAD)
    revision. ``no_update=True`` downloads without writing a manifest, so the
    models are never tracked or refreshed (at your own risk).

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
    for name, spec_version in parse_model_specs(models):
        repo_id = resolve_repo_id(name, MOONSHINE_HF_REPO_MAP)
        version = resolve_model_version(
            repo_id, spec_version or model_version, builtin_repos=_BUILTIN_REPOS
        )
        model_dir = base_dir / repo_id
        try:
            _refresh_moonshine(
                repo_id, model_dir, base_dir, version=version, record=not no_update
            )
        except Exception as exc:
            raise DownloadError(f"Unable to download Moonshine files from {repo_id}") from exc
        result[name] = model_dir
        logger.info(
            "Moonshine model files for version %s ready at '%s'",
            version or "latest", model_dir,
        )
    return result


def ensure_moonshine_models(
    model_dir: str | Path,
    *,
    refresh: bool = True,
) -> None:
    """Verify/refresh the Moonshine models in ``model_dir`` before inference.

    Re-syncs the local copy to the version recorded in its manifest (never to a
    newer one). Untracked models (``--no-update``) have no manifest and are
    left as-is. When ``refresh`` is ``False`` the check is skipped entirely
    (offline/airgapped runs). Refresh failures are logged, not raised, so
    inference can still proceed on whatever is available locally.
    """
    model_dir = Path(model_dir)
    if not refresh:
        return
    manifest = read_manifest(model_dir)
    if manifest is None:
        logger.info(
            "No manifest in %s; the model is untracked, so skipping the freshness check.",
            model_dir,
        )
        return
    repo_id = manifest.get("repo_id")
    if not repo_id:
        logger.warning(
            "Manifest in %s has no repo_id; cannot verify model freshness. "
            "Run `python setup_demos.py moonshine` if inference fails.",
            model_dir,
        )
        return
    base_dir = base_dir_for(model_dir, repo_id)
    if base_dir is None:
        logger.warning(
            "%s is not laid out as <models dir>/%s; skipping the freshness check "
            "so a refresh cannot fetch a second copy elsewhere. "
            "Run `python setup_demos.py moonshine` to manage models.",
            model_dir,
            repo_id,
        )
        return
    try:
        _refresh_moonshine(
            repo_id,
            model_dir,
            base_dir,
            version=manifest.get("version"),
        )
    except Exception as e:
        logger.warning(
            "Could not refresh models from %s (%s); using local files.", repo_id, e
        )


def setup_moonshine(
    models: list[str],
    model_version: str | None = None,
    no_update: bool = False,
):
    logger.info("Setting up moonshine demo with models: [%s]", ", ".join(models))
    download_moonshine(models, model_version=model_version, no_update=no_update)
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
        help=f"Model name or HF repo ID, optionally 'name:version' to pin a "
             f"specific model version for that model. Built-in: [{available_models}] (default: %(default)s)",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help=(
            "Model version tag to download for every model without its own "
            "'name:version' (default: the torq-examples version for built-in "
            "repos, the repo's latest revision for custom repos). A pinned "
            "version is kept in sync with its own tag but never upgraded."
        ),
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help=(
            "Download without tracking: no .manifest.json is written, so the "
            "models are never checked for updates or refreshed (at your own risk)."
        ),
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    try:
        setup_moonshine(args.models, model_version=args.model_version, no_update=args.no_update)
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
