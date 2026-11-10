"""Validate the Face ID detector.

Detects faces in each photo of the validation dataset (``info.json``: Labeled
Faces in the Wild by default, one face centred per photo); a photo passes
when a detected face box contains its centre.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Face_ID.src.postprocess import decode_face_outputs
from Face_ID.setup_demo import download_face_detector, local_face_id_model_path
from utils.runtime import Model, add_common_inference_args
from utils.tasks import demo_validation
from utils.tasks.face_recognition import LOADERS, aggregate, score_item, validate_main
from infer import (
    BOX1_SCALE,
    BOX1_ZERO_POINT,
    BOX2_SCALE,
    BOX2_ZERO_POINT,
    IOU_THRESHOLD,
    MODEL_HEIGHT,
    MODEL_WIDTH,
    SCORE_SCALE,
    SCORE_ZERO_POINT,
    map_boxes,
    preprocess_image,
    resolve_input_path,
)

MIN_ACCURACY = 2.0  # DUMMY until a measured baseline exists: above 1.0, so the accuracy check always FAILS.


def add_args(parser) -> None:
    parser.add_argument(
        "--model", default=None,
        help="Path to face_detection.vmfb (default: the one setup_demo.py downloaded)",
    )
    parser.add_argument("--device", default="torq")
    add_common_inference_args(parser, with_threads=False, with_no_refresh=False)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)


def make_runner(args, runtime):
    return Model(str(resolve_input_path(args.model)), runtime)


def detect(runner, item, args, metrics):
    model = runner
    image_path = resolve_input_path(str(item["image"]))
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Unable to read image: {image_path}")

    input_tensor, letterbox = preprocess_image(image)
    outputs = model.run(input_tensor)
    latency_ns = int(model.infer_time_ms * 1e6)
    metrics.add("latency", latency_ns)
    metrics.add("frames_per_second", 1e9 / latency_ns)
    detections = decode_face_outputs(
        outputs,
        box1_scale=BOX1_SCALE,
        box1_zero_point=BOX1_ZERO_POINT,
        box2_scale=BOX2_SCALE,
        box2_zero_point=BOX2_ZERO_POINT,
        score_scale=SCORE_SCALE,
        score_zero_point=SCORE_ZERO_POINT,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=IOU_THRESHOLD,
        image_width=MODEL_WIDTH,
        image_height=MODEL_HEIGHT,
    )
    return map_boxes(detections, letterbox, image.shape)


if __name__ == "__main__":
    validate_main(
        make_runner,
        detect,
        score_item,
        aggregate,
        demo="face_id",
        validation=demo_validation(__file__),
        loaders=LOADERS,
        min_accuracy=MIN_ACCURACY,
        local_model=lambda args: local_face_id_model_path(),
        download=lambda args: download_face_detector(),
        add_args=add_args,
        message=(
            "no local Face ID model found; pass --model or run "
            "`python setup_demos.py face_id` from torq-examples root"
        ),
    )
