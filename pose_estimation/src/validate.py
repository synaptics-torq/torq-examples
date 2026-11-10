# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Validate the pose estimation demo.

Detects persons in each image of the validation dataset (``info.json``: COCO
val2017 keypoints by default) and scores them against the labelled persons: a
person matches by box IoU at or above 0.5, and the accuracy is the fraction
of labelled keypoints within 5% of the box size of the predicted keypoint.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pose_estimation.setup_demo import (
    download_pose_estimation,
    ensure_pose_estimation_models,
    local_pose_model_path,
)
from utils.preprocess import preprocess_image
from utils.runtime import Model, add_common_inference_args
from utils.tasks import demo_validation
from utils.tasks.keypoint_detection import LOADERS, aggregate, score_item, validate_main
from utils.pose_estimation import dequantize_out, postprocess_pose
from infer import OUT_SCALE, OUT_ZP

MIN_ACCURACY = 2.0  # DUMMY until a measured baseline exists: above 1.0, so the accuracy check always FAILS.


def add_args(parser) -> None:
    parser.add_argument(
        "--model", default=None,
        help="Path to the model VMFB (default: the one setup_demo.py downloaded)",
    )
    parser.add_argument("--device", default="torq")
    add_common_inference_args(parser)


def make_runner(args, runtime):
    return Model(args.model, runtime)


def detect(runner, item, args, metrics):
    model = runner
    input_data, pad_info, orig_shape = preprocess_image(item["image"])
    raw_out = model.run(input_data)[0]
    latency_ns = int(model.infer_time_ms * 1e6)
    metrics.add("latency", latency_ns)
    metrics.add("frames_per_second", 1e9 / latency_ns)
    outputs = dequantize_out(raw_out, OUT_SCALE, OUT_ZP, int8=True)
    return postprocess_pose(outputs, orig_shape, pad_info)


if __name__ == "__main__":
    validate_main(
        make_runner,
        detect,
        score_item,
        aggregate,
        demo="pose_estimation",
        validation=demo_validation(__file__),
        loaders=LOADERS,
        min_accuracy=MIN_ACCURACY,
        local_model=lambda args: local_pose_model_path(),
        ensure_models=ensure_pose_estimation_models,
        download=lambda args: download_pose_estimation(),
        add_args=add_args,
        message=(
            "no local pose model found; pass --model or run "
            "`python setup_demos.py pose_estimation` from torq-examples root"
        ),
    )
