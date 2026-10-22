# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runner import Qwen3Static, hide_thinking
from qwen3.setup_demo import ensure_qwen3_models, local_qwen3_model_path
from utils.inference import add_llm_inference_args, print_llm_stats, run_chat_loop
from utils.log import configure_logging
from utils.runtime import (
    build_runtime_flags,
    cleanup_npu_after_inference,
    setup_npu_for_inference,
)


def main(args: argparse.Namespace):

    configure_logging(args.logging)
    logging.getLogger("Qwen3").info("Starting assistant...")
    ensure_qwen3_models(Path(args.model).parent, refresh=not args.no_refresh)

    setup_npu_for_inference()

    qwen = Qwen3Static(
        args.model,
        args.max_seq_len,
        max_prompt_tokens=args.max_inp_len,
        n_threads=args.threads,
        instruct_model=args.instruct_model,
        # In thinking mode the sliding window would drop the question and the
        # reasoning, so generation stops when the context is full.
        cache_keep_n=(
            None if args.no_kv_cache_window or args.thinking else args.kv_cache_window
        ),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_gen_tokens=args.max_gen_tokens,
        runtime_flags=build_runtime_flags(args.tda, args.runtime_flags),
        device_io=args.device_io,
        lm_head_path=args.lm_head,
        disable_lm_head=args.no_lm_head,
        prefill_model_path=args.batch_prefill_model,
        disable_prefill=args.no_batch_prefill_model,
        thinking=args.thinking,
    )
    hide = args.thinking and not args.show_thinking
    try:
        run_chat_loop(
            lambda text, should_stop: "".join(
                hide_thinking([qwen.run(text, should_stop=should_stop)])
            ) if hide else qwen.run(text, should_stop=should_stop),
            lambda text, should_stop=None: hide_thinking(
                qwen.run_stream(text, should_stop=should_stop)
            ) if hide else qwen.run_stream(text, should_stop=should_stop),
            lambda: print_llm_stats(qwen),
            debug=args.logging.upper() == "DEBUG",
        )
    finally:
        cleanup_npu_after_inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen3 VMFB inference.")
    add_llm_inference_args(parser)
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=True,
        help="Let the model reason in a <think> block before it answers "
             "(default: on; --no-thinking answers at once)",
    )
    parser.add_argument(
        "--show-thinking", action="store_true", default=False,
        help="Print the <think> block too (default: only the answer)",
    )
    args = parser.parse_args()
    if args.model is None:
        local_model = local_qwen3_model_path()
        if local_model is None:
            parser.error(
                "no local Qwen3 model found; pass -m/--model or run "
                "`python setup_demos.py qwen3` from torq-examples root"
            )
        args.model = str(local_model)
    main(args)
