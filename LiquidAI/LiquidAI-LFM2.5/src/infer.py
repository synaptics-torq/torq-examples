# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from runner import LiquidStatic
from utils.tasks.text_generation import infer_main, llm_kwargs

# The model-refresh helpers live one level up (the demo dir's setup_demo.py). The demo
# dir name has a hyphen, so it is not importable as a package; add it to the path and
# import the module directly. Guarded so a missing setup_demo never breaks inference.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from setup_demo import (
        download_liquid,
        ensure_liquid_models,
        local_liquid_model_path,
    )
except Exception:
    download_liquid = None
    ensure_liquid_models = None
    local_liquid_model_path = None


def make_runner(args, runtime):
    return LiquidStatic(
        args.model, runtime=runtime,
        instruct_model=args.instruct_model, **llm_kwargs(args),
    )


if __name__ == "__main__":
    infer_main(
        make_runner,
        description="Run LFM2.5 (Liquid) VMFB inference.",
        demo="LiquidAI-LFM2.5",
        # The guarded imports above may be None: fall back to no-ops, which
        # yield the standard no-local-model error.
        local_model=local_liquid_model_path or (lambda: None),
        ensure_models=ensure_liquid_models or (lambda model_dir, **kwargs: None),
        download=download_liquid,
    )
