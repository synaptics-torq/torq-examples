# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import json
from pathlib import Path

from object_detection.setup_demo import (
    default_model_path,
    download_object_detection,
    ensure_object_detection_models,
    VARIANT_MODEL_FILES,
)
from utils.runtime import Model, RuntimeConfig, npu_clock, run_profile_if_requested
from utils.video import build_video_argparser, run_video_inference_loop, setup_display_and_camera
from utils.object_detection import (
    VARIANTS,
    annotate_frame,
    dequantize_out,
    postprocess,
    preprocess_frame_cv,
)
from utils.log import configure_logging
from utils.metrics import MetricsRun
from utils.model_setup import resolve_default_model




def make_process_fn(model, labels, variant):
    def process_fn(bgr_frame):
        input_data, pad_info, orig_shape = preprocess_frame_cv(bgr_frame)
        raw_out = model.run(input_data)[0]
        outputs = dequantize_out(raw_out, variant["out_scale"], variant["out_zp"])
        detections = postprocess(outputs, orig_shape, pad_info, labels)
        annotated, frame_detections = annotate_frame(bgr_frame, detections)
        log_str = " ".join(f"{lbl} {conf:.2f}" for lbl, conf, _ in detections)
        return annotated, frame_detections, model.infer_time_ms, log_str
    return process_fn


def main():
    parser = build_video_argparser(
        "Run YOLO object detection on video, RTSP, or camera input.",
        default_json_results="detection_results.json",
        model_required=False,
        model_help="Path to the model VMFB (default: the one setup_demo.py downloads for --variant)",
    )
    parser.add_argument("--labels")
    parser.add_argument(
        "--variant", choices=sorted(VARIANTS), default="yolov8",
        help="Model head variant (default: %(default)s)",
    )
    args = parser.parse_args()
    configure_logging(args.logging)

    if args.model is None:
        expected = default_model_path(args.variant)
        args.model = str(
            resolve_default_model(
                parser,
                lambda: expected if expected.exists() else None,
                download=lambda: download_object_detection([VARIANT_MODEL_FILES[args.variant][0]]),
                no_refresh=args.no_refresh,
                message=(
                    f"default model {expected} not found; pass --model or run "
                    f"`python setup_demos.py object_detection` from torq-examples root"
                ),
            )
        )

    ensure_object_detection_models(Path(args.model).parent, refresh=not args.no_refresh)
    runtime = RuntimeConfig.from_args(args)

    if run_profile_if_requested(args, runtime):
        return

    labels = {}
    if args.labels:
        with open(args.labels, encoding="utf-8") as handle:
            data = json.load(handle)
            labels = {str(k): v for k, v in data["names"].items()} if "names" in data else data

    with npu_clock():
        setup_display_and_camera(args)
        model = Model(args.model, runtime)
        with MetricsRun(args.metrics_out) as m:
            run_video_inference_loop(
                args, make_process_fn(model, labels, VARIANTS[args.variant]),
                "Object Detection", metrics=m,
            )


if __name__ == "__main__":
    main()
