# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
import os
from typing import NamedTuple

import numpy as np
import soundfile as sf
from tokenizers import Tokenizer

from moonshine.setup_demo import (
    download_moonshine,
    ensure_moonshine_models,
    local_moonshine_model_dir,
)
from moonshine.src.runner import MoonshineRunner
from utils.runtime import add_common_inference_args
from utils.tasks.automatic_speech_recognition import infer_main

GREEN = "\033[32m"
RESET = "\033[0m"


class TranscriptionResult(NamedTuple):
    text: str
    audio_duration_ms: float


def transcribe_wav(
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


def _print_result(result: TranscriptionResult, runner: MoonshineRunner, metrics=None) -> None:
    ttft = runner.time_to_first_token
    total = runner.last_infer_time
    decode_ms = total - ttft
    decode_tokens = max(0, runner.generated_tokens - 1)
    tps = decode_tokens / decode_ms * 1000 if decode_ms > 0 else 0
    rtf = total / result.audio_duration_ms if result.audio_duration_ms > 0 else 0
    if metrics is not None:
        metrics.add("latency", int(total * 1e6))
        metrics.add("real_time_factor", rtf * 100)
    print(
        f"{GREEN}Transcribed: {result.text}{RESET}"
        f"  ({total:.0f} ms, TTFT: {ttft:.0f} ms, "
        f"{tps:.1f} tok/s, RTF: {rtf:.2f})"
    )


def add_args(parser) -> None:
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
        default=None,
        metavar="DIR",
        help="Path to Moonshine model directory (default: the one setup_demo.py downloaded)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to tokenizer.json (default: auto-detect in model dir)",
    )
    add_common_inference_args(parser)


def make_runner(args, runtime):
    model = MoonshineRunner(args.model_dir, runtime=runtime)
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        candidate = os.path.join(args.model_dir, "tokenizer.json")
        tokenizer_path = candidate if os.path.isfile(candidate) else "tokenizer.json"
    return model, Tokenizer.from_file(tokenizer_path)


def loop(runner, args, metrics) -> None:
    logging.getLogger("Moonshine").info("Starting demo...")
    model, tokenizer = runner
    try:
        for wav in args.inputs:
            result = transcribe_wav(wav, model, tokenizer)
            _print_result(result, model, metrics)
    except KeyboardInterrupt:
        logging.getLogger("Moonshine").info("Stopped by user.")


if __name__ == "__main__":
    infer_main(
        make_runner,
        loop,
        description="Run Moonshine speech-to-text inference.",
        demo="moonshine",
        local_model=local_moonshine_model_dir,
        ensure_models=ensure_moonshine_models,
        download=download_moonshine,
        add_args=add_args,
        message=(
            "no local Moonshine model found; pass -m/--model-dir or run "
            "`python setup_demos.py moonshine` from torq-examples root"
        ),
    )
