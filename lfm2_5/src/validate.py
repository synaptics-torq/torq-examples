# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from lfm2_5.setup_demo import (
    download_liquid,
    ensure_liquid_models,
    local_liquid_model_path,
)
from lfm2_5.src.runner import LiquidStatic
from utils.tasks import demo_validation
from utils.tasks.text_generation import llm_kwargs, validate_main

MIN_ACCURACY = 2.0  # DUMMY until a measured baseline exists: above 1.0, so the accuracy check always FAILS.


def make_runner(args, runtime):
    return LiquidStatic(
        args.model, runtime=runtime,
        instruct_model=args.instruct_model, **llm_kwargs(args),
    )


if __name__ == "__main__":
    validate_main(
        make_runner,
        demo="LiquidAI-LFM2.5",
        validation=demo_validation(__file__),
        min_accuracy=MIN_ACCURACY,
        local_model=local_liquid_model_path,
        ensure_models=ensure_liquid_models,
        download=download_liquid,
    )
