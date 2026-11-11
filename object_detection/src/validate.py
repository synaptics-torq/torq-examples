# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Validate the object detection demo.

Detects objects in each image of the validation dataset (``info.json``: the
COCO val2017 mini-500 subset by default) and scores the detections against
the labelled objects: greedy one-to-one matching with
the same label and IoU at or above 0.5.
"""

from pathlib import Path

from object_detection.setup_demo import (
    VARIANT_MODEL_FILES,
    download_object_detection,
    ensure_object_detection_models,
)
from object_detection.src.infer import load_labels, local_model, make_runner as _make_runner, message
from utils.runtime import add_common_inference_args
from utils.tasks import demo_validation
from utils.tasks.object_detection import LOADERS, aggregate, score_item, validate_main
from utils.vision import dequantize_out
from utils.object_detection import VARIANTS, postprocess
from utils.preprocess import preprocess_image

def add_args(parser) -> None:
    parser.add_argument(
        "--model", default=None,
        help="Path to the model VMFB (default: the one setup_demo.py downloads for --variant)",
    )
    parser.add_argument("--labels")
    parser.add_argument(
        "--variant", choices=sorted(VARIANTS), default="yolov8",
        help="Model head variant (default: %(default)s)",
    )
    parser.add_argument("--device", default="torq")
    add_common_inference_args(parser)


def make_runner(args, runtime):
    model, labels, variant = _make_runner(args, runtime)
    if not labels:
        # The labels the setup downloads next to the model keep the expected
        # spec labels and the demo's class names in sync.
        labels_path = Path(args.model).parent / "labels.json"
        if labels_path.is_file():
            labels = load_labels(str(labels_path))
    return model, labels, variant


def detect(runner, item, args, metrics):
    model, labels, variant = runner
    input_data, pad_info, orig_shape = preprocess_image(item["image"])
    raw_out = model.run(input_data)[0]
    latency_ns = int(model.infer_time_ms * 1e6)
    metrics.add("latency", latency_ns)
    metrics.add("frames_per_second", 1e9 / latency_ns)
    outputs = dequantize_out(raw_out, variant["out_scale"], variant["out_zp"], int8=True)
    return postprocess(outputs, orig_shape, pad_info, labels)


if __name__ == "__main__":
    validate_main(
        make_runner,
        detect,
        score_item,
        aggregate,
        demo="object_detection",
        validation=demo_validation(__file__),
        loaders=LOADERS,
        local_model=local_model,
        ensure_models=ensure_object_detection_models,
        download=lambda args: download_object_detection([VARIANT_MODEL_FILES[args.variant][0]]),
        add_args=add_args,
        message=message,
    )
