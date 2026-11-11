# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Validate the Moonshine model.

Transcribes the speech of the validation dataset (``info.json``: LibriSpeech
test-clean by default) and scores each transcript against its reference with
the word error rate.
"""

from moonshine.setup_demo import (
    download_moonshine,
    ensure_moonshine_models,
    local_moonshine_model_dir,
)
from moonshine.src.infer import make_runner, transcribe_wav
from utils.runtime import add_common_inference_args
from utils.tasks import demo_validation
from utils.tasks.automatic_speech_recognition import validate_main

def add_args(parser) -> None:
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


def transcribe(runner, item, args, metrics) -> str:
    model, tokenizer = runner
    result = transcribe_wav(item["audio"], model, tokenizer)
    total = model.last_infer_time
    rtf = total / result.audio_duration_ms if result.audio_duration_ms > 0 else 0
    metrics.add("latency", int(total * 1e6))
    metrics.add("real_time_factor", rtf * 100)
    return result.text


if __name__ == "__main__":
    validate_main(
        make_runner,
        transcribe,
        demo="moonshine",
        validation=demo_validation(__file__),
        local_model=local_moonshine_model_dir,
        ensure_models=ensure_moonshine_models,
        download=download_moonshine,
        add_args=add_args,
        message=(
            "no local Moonshine model found; pass -m/--model-dir or run "
            "`python setup_demos.py moonshine` from torq-examples root"
        ),
    )
