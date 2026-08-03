# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import json
from pathlib import Path

from object_detection.setup_demo import ensure_object_detection_models
from utils.runtime import (
    build_runtime_flags,
    cleanup_npu_after_inference,
    run_profile_if_requested,
    setup_npu_and_runner,
)
from utils.video import build_video_argparser, run_video_inference_loop
from utils.object_detection import (
    annotate_frame,
    dequantize_out,
    postprocess,
    preprocess_frame_cv,
)




def make_process_fn(runner, labels):
    def process_fn(bgr_frame):
        input_data, pad_info, orig_shape = preprocess_frame_cv(bgr_frame)
        raw_out = runner.infer(input_data)
        outputs = dequantize_out(raw_out, 0.004194467328488827, -128)
        detections = postprocess(outputs, orig_shape, pad_info, labels)
        annotated, frame_detections = annotate_frame(bgr_frame, detections)
        log_str = " ".join(f"{lbl} {conf:.2f}" for lbl, conf, _ in detections)
        return annotated, frame_detections, runner.infer_time_ms, log_str
    return process_fn


def main():
    parser = build_video_argparser(
        "Run YOLOv8 object detection on video, RTSP, or camera input.",
        default_json_results="detection_results.json",
    )
    parser.add_argument("--labels")
    args = parser.parse_args()

    ensure_object_detection_models(Path(args.model).parent, refresh=not args.no_refresh)
    runtime_flags = build_runtime_flags(args.tda, args.runtime_flags)

    if run_profile_if_requested(args, runtime_flags):
        return

    runner = setup_npu_and_runner(args, runtime_flags)

    labels = {}
    if args.labels:
        with open(args.labels, encoding="utf-8") as handle:
            data = json.load(handle)
            labels = {str(k): v for k, v in data["names"].items()} if "names" in data else data

    try:
        run_video_inference_loop(args, make_process_fn(runner, labels), "Object Detection")
    finally:
        cleanup_npu_after_inference()


if __name__ == "__main__":
    main()
