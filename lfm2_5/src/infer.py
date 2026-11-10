# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from lfm2_5.setup_demo import (
    download_liquid,
    ensure_liquid_models,
    local_liquid_model_path,
)
from lfm2_5.src.runner import LiquidStatic
from utils.tasks.text_generation import infer_main, llm_kwargs


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
        local_model=local_liquid_model_path,
        ensure_models=ensure_liquid_models,
        download=download_liquid,
    )
