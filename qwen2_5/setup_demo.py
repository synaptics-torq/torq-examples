# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from typing import Final

from utils.download import (
    default_models_dir,
    download_from_hf,
    read_manifest,
    resolve_repo_id,
)
from utils.model_setup import (
    demo_main,
    ensure_demo_models,
    setup_demo,
)

logger = logging.getLogger("Qwen2.5.setup")

QWEN25_HF_REPO_MAP: Final[dict[str, str]] = {
    "default": "Synaptics/Qwen2.5-0.5B-Instruct-GPTQ-int4-torq",
    "instruct": "Synaptics/Qwen2.5-0.5B-Instruct-GPTQ-int4-torq",
}
_DEFAULT_MODELS: Final[list[str]] = ["instruct"]
# The Qwen repo is not released under torq-examples version tags yet, so the
# built-in default (the VERSION tag) cannot resolve; track its main branch.
_DEFAULT_MODEL_VERSION: Final[str] = "main"
_QWEN25_FILES: Final[tuple[str, ...]] = (
    "transformer.vmfb",
    "lm_head.vmfb",
    "token_embeddings.npy",
    "token_id_lut.npy",
    "config.json",
    "tokenizer.json",
)


def _has_qwen25_files(model_dir: Path) -> bool:
    return all((model_dir / filename).exists() for filename in _QWEN25_FILES)


def _qwen25_files_present(model_dir: Path) -> bool:
    """Whether a *tracked* copy is complete: all files plus a manifest."""
    if not _has_qwen25_files(model_dir):
        return False
    manifest = read_manifest(model_dir)
    return bool(manifest and manifest.get("files"))


def _download_qwen25(
    repo_id: str, base_dir: Path, *, revision: str | None = None
) -> list[str]:
    for filename in _QWEN25_FILES:
        download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
        logger.info("Downloaded %s from %s", filename, repo_id)
    return list(_QWEN25_FILES)


def local_qwen25_model_path(
    model: str | None = None,
    *,
    base_dir: str | Path | None = None,
) -> Path | None:
    """Return the local ``transformer.vmfb`` to default ``-m``/``--model`` to.

    The sibling ``lm_head.vmfb`` is auto-discovered by the runner.
    """
    if base_dir is None:
        base_dir = default_models_dir()
    base_dir = Path(base_dir)
    if model is not None:
        repo_ids = [resolve_repo_id(model, QWEN25_HF_REPO_MAP)]
    else:
        repo_ids = list(dict.fromkeys(QWEN25_HF_REPO_MAP.values()))
    for repo_id in repo_ids:
        model_dir = base_dir / repo_id
        if _has_qwen25_files(model_dir):
            return model_dir / "transformer.vmfb"
    return None


def ensure_qwen25_models(model_dir: str | Path, *, refresh: bool = True) -> None:
    """Verify/refresh the Qwen2.5 models in ``model_dir`` before inference.

    Refresh failures are logged, not raised, so inference can still proceed on
    whatever is available locally (offline/airgapped runs, e.g. the board).
    """
    ensure_demo_models(
        model_dir,
        refresh=refresh,
        demo_name="qwen2_5",
        files_present=_has_qwen25_files,
        download=_download_qwen25,
        tracked_files_present=_qwen25_files_present,
    )


def setup_qwen25(
    models: list[str] | None = None,
    model_version: str | None = None,
    no_update: bool = False,
):
    """Set up the Qwen2.5 demo.

    The model repo is private while the export is experimental: log in with
    ``huggingface-cli login`` (an account with access to the Synaptics org) first.
    """
    if models is None:
        models = _DEFAULT_MODELS
    if model_version is None and not no_update:
        model_version = _DEFAULT_MODEL_VERSION

    return setup_demo(
        QWEN25_HF_REPO_MAP,
        models,
        demo_name="qwen2_5",
        requirements=Path(__file__).parent / "requirements.txt",
        files_present=_has_qwen25_files,
        download=_download_qwen25,
        model_version=model_version,
        no_update=no_update,
        tracked_files_present=_qwen25_files_present,
    )


if __name__ == "__main__":
    demo_main(
        setup_qwen25,
        description="Download Qwen2.5-0.5B-Instruct (GPTQ INT4) model files.",
        default_models=_DEFAULT_MODELS,
        repo_map=QWEN25_HF_REPO_MAP,
    )
