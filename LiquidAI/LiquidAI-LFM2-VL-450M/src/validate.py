# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Validate the LFM2-VL-450M model.

Encodes each image of the validation dataset (``info.json``: MMStar by
default) and checks the option letter the model answers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# The default-model-dir helper lives one level up (the demo dir's
# setup_demo.py). The demo dir name has a hyphen, so it is not importable as
# a package; add it to the path and import the module directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup_demo import (  # noqa: E402
    download_lfm2vl_int8,
    ensure_lfm2vl_int8_models,
    local_lfm2vl_model_dir,
)

from utils.runtime import add_common_inference_args  # noqa: E402
from utils.tasks import demo_validation
from utils.tasks.image_text_to_text import validate_main  # noqa: E402
from infer import gen_args, load_bgr, make_runner, model_args, print_answer_stats  # noqa: E402

MIN_ACCURACY = 2.0  # DUMMY until a measured baseline exists: above 1.0, so the accuracy check always FAILS.


def add_args(parser) -> None:
    model_args(parser)
    add_common_inference_args(parser)
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Decoder KV cache length (auto-detected from the vmfb if omitted)")
    gen_args(parser)


def ask(vl, item, args, metrics) -> str:
    vl.encode_image(load_bgr(item["image"]))
    answer = vl.infer(item["question"])
    print_answer_stats(vl, metrics)
    return answer


if __name__ == "__main__":
    validate_main(
        make_runner,
        ask,
        demo="LiquidAI-LFM2-VL-450M",
        validation=demo_validation(__file__),
        min_accuracy=MIN_ACCURACY,
        local_model=local_lfm2vl_model_dir,
        ensure_models=ensure_lfm2vl_int8_models,
        download=download_lfm2vl_int8,
        add_args=add_args,
        message=(
            "no local LFM2-VL-450M model set found; pass --model-dir or run "
            "`python setup_demos.py LiquidAI-LFM2-VL-450M` from torq-examples root"
        ),
    )
