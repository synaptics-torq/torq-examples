# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import logging
import sys

from runner import Gemma3Static
from utils.log import add_logging_args, configure_logging


def main(args: argparse.Namespace):

    configure_logging(args.logging)
    logging.getLogger("Gemma3").info("Starting assistant...")
    gemma3 = Gemma3Static(
        args.model,
        args.max_seq_len,
        max_prompt_tokens=args.max_inp_len,
        n_threads=args.threads,
        instruct_model=args.instruct_model,
        cache_keep_n=None if args.no_kv_cache_window else args.kv_cache_window,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
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

            if args.logging.upper() == "DEBUG":
                answer = gemma3.run(inp, args.max_seq_len)
                sys.stdout.write(f"Agent: {answer}")
            else:
                sys.stdout.write('\033[2m[thinking...]\033[0m')
                sys.stdout.flush()
                first = True
                for chunk in gemma3.run_stream(inp, args.max_seq_len):
                    if first:
                        sys.stdout.write('\r' + ' ' * 40 + '\rAgent: ')
                        first = False
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
            decode_ms = gemma3.last_infer_time - gemma3.time_to_first_token
            tps = gemma3.generated_tokens / decode_ms * 1000 if decode_ms > 0 else 0
            print(f"  ({gemma3.last_infer_time:.0f} ms, TTFT: {gemma3.time_to_first_token:.0f} ms, {tps:.1f} tok/s)\n")
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemma3 VMFB inference.")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to VMFB model"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=None,
        help="Maximum sequence length (prompt + generation); auto-detected from model if omitted",
    )
    parser.add_argument(
        "--max-inp-len", type=int, help="Maximum input length"
    )
    parser.add_argument(
        "--instruct-model", action="store_true", default=False,
        help="Is instruct model",
    )
    parser.add_argument(
        "-j", "--threads", type=int,
        help="Number of cores to use for CPU execution (default: all)",
    )
    add_logging_args(parser)
    inference_group = parser.add_argument_group("inference")
    inference_group.add_argument(
        "--kv-cache-window",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Enable sliding-window KV cache: when the cache is full, keep the most "
            "recent N entries and discard older ones before continuing generation "
            "(default: %(default)s)"
        ),
    )
    inference_group.add_argument(
        "--no-kv-cache-window",
        action="store_true",
        default=False,
        help=(
            "Disable sliding-window KV cache behavior. "
            "Once the KV cache reaches its maximum length, no further tokens can be generated."
        ),
    )
    inference_group.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0.0 = greedy) (default: %(default)s)",
    )
    inference_group.add_argument(
        "--top-p", type=float, default=1.0,
        help="Top-p (nucleus) sampling threshold (default: %(default)s)",
    )
    inference_group.add_argument(
        "--top-k", type=int, default=64,
        help="Top-k pre-filter size for sampling (default: %(default)s)",
    )
    main(parser.parse_args())
