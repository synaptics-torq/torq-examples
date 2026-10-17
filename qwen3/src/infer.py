# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import logging
from pathlib import Path

from runner import Qwen3Static
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
        cache_keep_n=None if args.no_kv_cache_window else args.kv_cache_window,
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
    )
    try:
        run_chat_loop(
            lambda text, should_stop: qwen.run(text, should_stop=should_stop),
            qwen.run_stream,
            lambda: print_llm_stats(qwen),
            debug=args.logging.upper() == "DEBUG",
        )
    finally:
        cleanup_npu_after_inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen3 VMFB inference.")
    add_llm_inference_args(parser)
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
