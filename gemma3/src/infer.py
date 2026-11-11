# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from gemma3.setup_demo import (
    download_gemma3,
    ensure_gemma3_models,
    local_gemma3_model_path,
)
from gemma3.src.runner import Gemma3Static
from utils.tasks.text_generation import infer_main, llm_kwargs


def make_runner(args, runtime):
    return Gemma3Static(
        args.model, runtime=runtime,
        instruct_model=args.instruct_model, **llm_kwargs(args),
    )


if __name__ == "__main__":
    infer_main(
        make_runner,
        description="Run Gemma3 VMFB inference.",
        demo="gemma3",
        local_model=local_gemma3_model_path,
        ensure_models=ensure_gemma3_models,
        download=download_gemma3,
    )
