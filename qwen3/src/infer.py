# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse

from qwen3.setup_demo import ensure_qwen3_models, local_qwen3_model_path
from qwen3.src.runner import Qwen3Static, hide_thinking
from utils.inference import print_llm_stats, run_chat_loop
from utils.tasks.text_generation import infer_main, llm_kwargs


def make_runner(args, runtime):
    kwargs = llm_kwargs(args)
    if args.thinking:
        # In thinking mode the sliding window would drop the question and the
        # reasoning, so generation stops when the context is full.
        kwargs["cache_keep_n"] = None
    return Qwen3Static(
        args.model, runtime=runtime,
        instruct_model=args.instruct_model, thinking=args.thinking, **kwargs,
    )


def add_thinking_args(parser):
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=True,
        help="Let the model reason in a <think> block before it answers "
             "(default: on; --no-thinking answers at once)",
    )
    parser.add_argument(
        "--show-thinking", action="store_true", default=False,
        help="Print the <think> block too (default: only the answer)",
    )


def chat(runner, args, m):
    hide = args.thinking and not args.show_thinking
    run_chat_loop(
        lambda text, should_stop: "".join(
            hide_thinking([runner.run(text, should_stop=should_stop)])
        ) if hide else runner.run(text, should_stop=should_stop),
        lambda text, should_stop=None: hide_thinking(
            runner.run_stream(text, should_stop=should_stop)
        ) if hide else runner.run_stream(text, should_stop=should_stop),
        lambda: print_llm_stats(runner, m),
        debug=args.logging.upper() == "DEBUG",
    )


if __name__ == "__main__":
    infer_main(
        make_runner,
        description="Run Qwen3 VMFB inference.",
        demo="qwen3",
        local_model=local_qwen3_model_path,
        ensure_models=ensure_qwen3_models,
        add_args=add_thinking_args,
        chat=chat,
    )
