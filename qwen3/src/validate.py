# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from qwen3.setup_demo import ensure_qwen3_models, local_qwen3_model_path
from qwen3.src.runner import Qwen3Static
from utils.tasks import demo_validation
from utils.tasks.text_generation import llm_kwargs, validate_main

def make_runner(args, runtime):
    thinking = getattr(args, "thinking", True)  # the demo's default
    kwargs = llm_kwargs(args)
    if thinking:
        # In thinking mode the sliding window would drop the question and the
        # reasoning, so generation stops when the context is full.
        kwargs["cache_keep_n"] = None
    return Qwen3Static(
        args.model, runtime=runtime,
        instruct_model=args.instruct_model, thinking=thinking, **kwargs,
    )


if __name__ == "__main__":
    validate_main(
        make_runner,
        demo="qwen3",
        validation=demo_validation(__file__),
        local_model=local_qwen3_model_path,
        ensure_models=ensure_qwen3_models,
    )
