# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Set up FunctionGemma using the shared Gemma export file layout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma3.setup_demo import (
    _download_gemma3,
    _gemma3_files_present,
    _has_gemma3_files,
    download_gemma3,
    local_gemma3_model_path,
)
from utils.download import resolve_repo_id
from utils.model_setup import demo_main, ensure_demo_models, setup_demo

DEFAULT_MODEL = "default"
FUNCTIONGEMMA_HF_REPO_MAP = {DEFAULT_MODEL: "Synaptics/functiongemma-270m-it-torq"}


def local_functiongemma_model_path(model=DEFAULT_MODEL, *, base_dir=None):
    return local_gemma3_model_path(
        resolve_repo_id(model, FUNCTIONGEMMA_HF_REPO_MAP), base_dir=base_dir
    )


def download_functiongemma(*, base_dir=None):
    return download_gemma3(list(FUNCTIONGEMMA_HF_REPO_MAP.values()), base_dir=base_dir)


def ensure_functiongemma_models(model_dir, *, refresh=True):
    ensure_demo_models(
        model_dir, refresh=refresh, demo_name="functiongemma",
        files_present=_has_gemma3_files, download=_download_gemma3,
        tracked_files_present=_gemma3_files_present,
    )


def setup_functiongemma(models=None, model_version=None, no_update=False, enable_prefill=False):
    return setup_demo(
        FUNCTIONGEMMA_HF_REPO_MAP,
        [DEFAULT_MODEL] if models is None else models,
        demo_name="functiongemma",
        requirements=Path(__file__).parent / "requirements.txt",
        files_present=_has_gemma3_files,
        download=_download_gemma3,
        model_version=model_version, no_update=no_update,
        tracked_files_present=_gemma3_files_present,
        enable_prefill=enable_prefill,
    )


if __name__ == "__main__":
    demo_main(
        setup_functiongemma, description="Download FunctionGemma model files.",
        default_models=[DEFAULT_MODEL], repo_map=FUNCTIONGEMMA_HF_REPO_MAP,
        supports_prefill=True,
    )
