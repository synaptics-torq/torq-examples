# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import logging
import os
from typing import NamedTuple

import numpy as np
import soundfile as sf
from tokenizers import Tokenizer

from runner import MoonshineRunner
from utils.log import add_logging_args, configure_logging

GREEN = "\033[32m"
RESET = "\033[0m"


class TranscriptionResult(NamedTuple):
    text: str
    audio_duration_ms: float


def _transcribe(
    wav: str | os.PathLike,
    runner: MoonshineRunner,
    tokenizer: Tokenizer,
) -> TranscriptionResult:
    data, sample_rate = sf.read(wav, dtype="float32")
    runner.input_freq = sample_rate
    audio_duration_ms = data.shape[0] / sample_rate * 1000 if sample_rate > 0 else 0
    audio = data.astype(np.float32)[np.newaxis, :]
    tokens = runner.run(audio)
    text = tokenizer.decode_batch(tokens.tolist(), skip_special_tokens=True)[0]
    return TranscriptionResult(text, audio_duration_ms)


def _print_result(result: TranscriptionResult, runner: MoonshineRunner) -> None:
    ttft = runner.time_to_first_token
    total = runner.last_infer_time
    decode_ms = total - ttft
    decode_tokens = max(0, runner.generated_tokens - 1)
    tps = decode_tokens / decode_ms * 1000 if decode_ms > 0 else 0
    rtf = total / result.audio_duration_ms if result.audio_duration_ms > 0 else 0
    print(
        f"{GREEN}Transcribed: {result.text}{RESET}"
        f"  ({total:.0f} ms, TTFT: {ttft:.0f} ms, "
        f"{tps:.1f} tok/s, RTF: {rtf:.2f})"
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
            result = _transcribe(wav, runner, tokenizer)
            _print_result(result, runner)
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
