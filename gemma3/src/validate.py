# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Validate the Gemma 3 model.

``prompts`` mode (the default) scores the validation dataset's questions
(``info.json``: PIQA by default); ``bleu`` and ``teacher-forced`` modes score
OPUS-100 English to Spanish translation (``--config es-en`` for the other
direction).
"""

from gemma3.setup_demo import (
    download_gemma3,
    ensure_gemma3_models,
    local_gemma3_model_path,
)
from gemma3.src.runner import Gemma3Static
from utils.tasks import demo_validation
from utils.tasks.text_generation import llm_kwargs, validate_main

MIN_ACCURACY = 2.0  # DUMMY until a measured baseline exists: above 1.0, so the accuracy check always FAILS.


def make_runner(args, runtime):
    return Gemma3Static(
        args.model, runtime=runtime,
        instruct_model=args.instruct_model, **llm_kwargs(args),
    )


if __name__ == "__main__":
    validate_main(
        make_runner,
        demo="gemma3",
        validation=demo_validation(__file__),
        min_accuracy=MIN_ACCURACY,
        local_model=local_gemma3_model_path,
        ensure_models=ensure_gemma3_models,
        download=download_gemma3,
    )
