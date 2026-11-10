# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from qwen2_5.setup_demo import ensure_qwen25_models, local_qwen25_model_path
from qwen2_5.src.runner import QwenStatic
from utils.tasks import demo_validation
from utils.tasks.text_generation import llm_kwargs, validate_main

MIN_ACCURACY = 2.0  # DUMMY until a measured baseline exists: above 1.0, so the accuracy check always FAILS.


def make_runner(args, runtime):
    return QwenStatic(
        args.model, runtime=runtime,
        instruct_model=args.instruct_model, **llm_kwargs(args),
    )


if __name__ == "__main__":
    validate_main(
        make_runner,
        demo="qwen2_5",
        validation=demo_validation(__file__),
        min_accuracy=MIN_ACCURACY,
        local_model=local_qwen25_model_path,
        ensure_models=ensure_qwen25_models,
    )
