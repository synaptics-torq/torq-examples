# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import logging
import sys

from split_runner import LiquidSplitStatic, InferenceInterrupted
from utils.log import add_logging_args, configure_logging
from utils.terminal import InferenceStopInput

YELLOW = "\033[33m"
RESET = "\033[0m"


def _finish_interrupted_output(started_output: bool) -> None:
    marker = f"{YELLOW}[Interrupt]{RESET}"
    if started_output:
        sys.stdout.write(f" {marker} \n")
    else:
        sys.stdout.write("\r" + " " * 80 + f"\r{marker} \n")
    sys.stdout.flush()


def _print_inference_stats(liquid: LiquidSplitStatic) -> None:
    decode_ms = liquid.last_infer_time - liquid.time_to_first_token
    tps = liquid.generated_tokens / decode_ms * 1000 if decode_ms > 0 else 0
    print(
        f"  ({liquid.last_infer_time:.0f} ms, "
        f"TTFT: {liquid.time_to_first_token:.0f} ms, "
        f"{tps:.1f} tok/s)\n"
    )


def main(args: argparse.Namespace):
    configure_logging(args.logging)
    logging.getLogger("Liquid").info("Starting split assistant...")
    liquid = LiquidSplitStatic(
        args.models_dir,
        args.max_seq_len,
        max_prompt_tokens=args.max_inp_len,
        n_threads=args.threads,
        instruct_model=args.instruct_model,
        cache_keep_n=None if args.no_kv_cache_window else args.kv_cache_window,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        runtime_flags=args.runtime_flags,
    )
    if args.prompt:
        # Single-prompt mode
        out = liquid.run(args.prompt)
        sys.stdout.write(f"Agent: {out}")
        _print_inference_stats(liquid)
        return

    try:
        while True:
            try:
                inp = input("You (type 'exit' or 'quit' to stop): ").strip()
            except EOFError:
                break
            if not inp:
                continue
            if inp.lower() in ("exit", "quit"):
                break
            sys.stdout.write('\033[2m[thinking...]\033[0m')
            sys.stdout.flush()
            first = True
            started = False
            try:
                with InferenceStopInput(sys.stdin) as should_stop:
                    for chunk in liquid.run_stream(inp, should_stop=should_stop):
                        if first:
                            sys.stdout.write('\r' + ' ' * 40 + '\rAgent: ')
                            first = False
                            started = True
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
            except (InferenceInterrupted, KeyboardInterrupt):
                _finish_interrupted_output(started)
                _print_inference_stats(liquid)
                continue
            _print_inference_stats(liquid)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LFM2.5 chained-split VMFB inference.")
    parser.add_argument(
        "-d", "--models-dir", type=str, required=True,
        help="Directory containing split_0..split_7.vmfb + head.vmfb + token_embeddings.npy + config.json + tokenizer.json",
    )
    parser.add_argument(
        "-p", "--prompt", type=str, default=None,
        help="Single-prompt mode: run this prompt once and exit (no REPL).",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=None,
        help="Auto-detected from the splits if omitted.",
    )
    parser.add_argument("--max-inp-len", type=int)
    parser.add_argument("--instruct-model", action="store_true", default=False)
    parser.add_argument("-j", "--threads", type=int)
    runtime_group = parser.add_argument_group("runtime")
    add_logging_args(parser)
    inf = parser.add_argument_group("inference")
    inf.add_argument("--kv-cache-window", type=int, default=2)
    inf.add_argument("--no-kv-cache-window", action="store_true", default=False)
    inf.add_argument("--temperature", type=float, default=0.0)
    inf.add_argument("--top-p", type=float, default=1.0)
    inf.add_argument("--top-k", type=int, default=64)
    runtime_group.add_argument("--runtime-flags", nargs=argparse.REMAINDER, default=None)
    main(parser.parse_args())
