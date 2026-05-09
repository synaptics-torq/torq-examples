# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import logging
import os

import numpy as np
import soundfile as sf
from tokenizers import Tokenizer

from runner import MoonshineRunner
from utils.log import add_logging_args, configure_logging

GREEN = "\033[32m"
RESET = "\033[0m"


def _transcribe(
    wav: str | os.PathLike,
    runner: MoonshineRunner,
    tokenizer: Tokenizer,
) -> str:
    data, _ = sf.read(wav, dtype="float32")
    audio = data.astype(np.float32)[np.newaxis, :]
    tokens = runner.run(audio)
    return tokenizer.decode_batch(tokens.tolist(), skip_special_tokens=True)[0]


def _print_result(text: str, runner: MoonshineRunner) -> None:
    ttft = runner.time_to_first_token
    total = runner.last_infer_time
    decode_ms = total - ttft
    tps = runner.generated_tokens / decode_ms * 1000 if decode_ms > 0 else 0
    print(
        f"{GREEN}Transcribed: {text}{RESET}"
        f"  ({total:.0f} ms, TTFT: {ttft:.0f} ms, {tps:.1f} tok/s)"
    )


def main(args: argparse.Namespace):
    configure_logging(args.logging)
    logger = logging.getLogger("Moonshine")
    logger.info("Starting demo...")

    runner = MoonshineRunner(
        args.model_dir,
        n_threads=args.threads,
        runtime_flags=args.runtime_flags,
    )

    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        candidate = os.path.join(args.model_dir, "tokenizer.json")
        tokenizer_path = candidate if os.path.isfile(candidate) else "tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    try:
        for wav in args.inputs:
            text = _transcribe(wav, runner, tokenizer)
            _print_result(text, runner)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Moonshine speech-to-text inference.")
    parser.add_argument(
        "inputs",
        type=str,
        metavar="WAV",
        nargs="+",
        help="WAV files for inference",
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Path to Moonshine model directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to tokenizer.json (default: auto-detect in model dir)",
    )
    parser.add_argument(
        "-j",
        "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)",
    )
    add_logging_args(parser)
    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument(
        "--runtime-flags",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="FLAG",
        help=(
            "[Advanced] Extra flags for the Torq runtime. "
            "Must be specified last; all remaining arguments are forwarded."
        ),
    )
    main(parser.parse_args())
