# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.download import (
    default_models_dir,
    download_from_hf,
    hf_file_exists,
    read_manifest,
    resolve_repo_id,
)
from utils.model_setup import (
    demo_main,
    ensure_demo_models,
    setup_demo,
)

logger = logging.getLogger("Qwen3.setup")

QWEN3_HF_REPO_MAP: Final[dict[str, str]] = {
    "instruct": "Synaptics/Qwen3-0.6B-GPTQ-int4-torq",
}
_DEFAULT_MODELS: Final[list[str]] = ["instruct"]
_QWEN3_FILES: Final[tuple[str, ...]] = (
    "transformer.vmfb",
    "token_embeddings.npy",
    "token_id_lut.npy",
    "config.json",
    "tokenizer.json",
)
#: LM head file names in download preference order. The trimmed-vocabulary
#: LM head is named ``lm_head.vmfb.trim``; older uploads use ``lm_head.vmfb``.
_LM_HEAD_FILENAMES: Final[tuple[str, ...]] = ("lm_head.vmfb.trim", "lm_head.vmfb")


def _has_qwen3_files(model_dir: Path) -> bool:
    has_lm_head = any((model_dir / name).exists() for name in _LM_HEAD_FILENAMES)
    return has_lm_head and all((model_dir / filename).exists() for filename in _QWEN3_FILES)


def _qwen3_files_present(model_dir: Path) -> bool:
    """Whether a *tracked* copy is complete: all files plus a manifest."""
    if not _has_qwen3_files(model_dir):
        return False
    manifest = read_manifest(model_dir)
    return bool(manifest and manifest.get("files"))


def _download_qwen3(
    repo_id: str, base_dir: Path, *, revision: str | None = None
) -> list[str]:
    lm_head = next(
        (name for name in _LM_HEAD_FILENAMES if hf_file_exists(repo_id, name, revision=revision)),
        None,
    )
    if lm_head is None:
        raise FileNotFoundError(
            f"No LM head ({' or '.join(_LM_HEAD_FILENAMES)}) in {repo_id}"
        )
    files = [*_QWEN3_FILES, lm_head]
    for filename in files:
        download_from_hf(repo_id, filename, base_dir=base_dir, revision=revision)
        logger.info("Downloaded %s from %s", filename, repo_id)
    return files


def local_qwen3_model_path(
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
        repo_ids = [resolve_repo_id(model, QWEN3_HF_REPO_MAP)]
    else:
        repo_ids = list(dict.fromkeys(QWEN3_HF_REPO_MAP.values()))
    for repo_id in repo_ids:
        model_dir = base_dir / repo_id
        if _has_qwen3_files(model_dir):
            return model_dir / "transformer.vmfb"
    return None


def ensure_qwen3_models(model_dir: str | Path, *, refresh: bool = True) -> None:
    """Verify/refresh the Qwen3 models in ``model_dir`` before inference.

    Refresh failures are logged, not raised, so inference can still proceed on
    whatever is available locally (offline/airgapped runs, e.g. the board).
    """
    ensure_demo_models(
        model_dir,
        refresh=refresh,
        demo_name="qwen3",
        files_present=_has_qwen3_files,
        download=_download_qwen3,
        tracked_files_present=_qwen3_files_present,
    )


def setup_qwen3(
    models: list[str] | None = None,
    model_version: str | None = None,
    no_update: bool = False,
    enable_prefill: bool = False,
):
    """Set up the Qwen3 demo."""
    if models is None:
        models = _DEFAULT_MODELS

    return setup_demo(
        QWEN3_HF_REPO_MAP,
        models,
        demo_name="qwen3",
        requirements=Path(__file__).parent / "requirements.txt",
        files_present=_has_qwen3_files,
        download=_download_qwen3,
        model_version=model_version,
        no_update=no_update,
        enable_prefill=enable_prefill,
        tracked_files_present=_qwen3_files_present,
    )


if __name__ == "__main__":
    demo_main(
        setup_qwen3,
        description="Download Qwen3-0.6B (GPTQ INT4) model files.",
        default_models=_DEFAULT_MODELS,
        supports_prefill=True,
        repo_map=QWEN3_HF_REPO_MAP,
    )
