# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Validate the Moonshine streaming model.

Runs the demo's VAD/streaming pipeline over each clip of the validation
dataset (``info.json``: LibriSpeech test-clean by default), as ``infer.py``
does with ``--wav``, and scores the final transcript against its reference
with the word error rate.
"""

from moonshine_streaming.setup_demo import (
    download_moonshine_streaming,
    ensure_moonshine_streaming_models,
    local_moonshine_streaming_model_dir,
)
from moonshine_streaming.src.infer import make_runner, pipeline_args, transcribe_streaming
from utils.runtime import add_common_inference_args
from utils.tasks import demo_validation
from utils.tasks.automatic_speech_recognition import validate_main

def add_args(parser) -> None:
    parser.add_argument(
        "-m", "--model-dir", type=str, default=None, metavar="DIR",
        help="Path to the flat moonshine-streaming-tiny model dir (default: the one setup_demo.py downloaded)",
    )
    pipeline_args(parser)
    add_common_inference_args(parser, with_threads=False)


def transcribe(runner, item, args, metrics) -> str:
    model, tokenizer = runner
    return transcribe_streaming(model, tokenizer, args, item["audio"], metrics)


if __name__ == "__main__":
    validate_main(
        make_runner,
        transcribe,
        demo="moonshine_streaming",
        validation=demo_validation(__file__),
        local_model=local_moonshine_streaming_model_dir,
        ensure_models=ensure_moonshine_streaming_models,
        download=download_moonshine_streaming,
        add_args=add_args,
        message=(
            "no local Moonshine streaming model found; pass -m/--model-dir or run "
            "`python setup_demos.py moonshine_streaming` from torq-examples root"
        ),
    )
