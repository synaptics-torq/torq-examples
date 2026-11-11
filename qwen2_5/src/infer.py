# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from qwen2_5.setup_demo import ensure_qwen25_models, local_qwen25_model_path
from qwen2_5.src.runner import QwenStatic
from utils.tasks.text_generation import infer_main, llm_kwargs


def make_runner(args, runtime):
    return QwenStatic(
        args.model, runtime=runtime,
        instruct_model=args.instruct_model, **llm_kwargs(args),
    )


if __name__ == "__main__":
    infer_main(
        make_runner,
        description="Run Qwen2.5 VMFB inference.",
        demo="qwen2_5",
        local_model=local_qwen25_model_path,
        ensure_models=ensure_qwen25_models,
    )
