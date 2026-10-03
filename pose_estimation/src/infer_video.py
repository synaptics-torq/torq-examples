# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from pathlib import Path

from pose_estimation.setup_demo import ensure_pose_estimation_models, local_pose_model_path
from utils.runtime import (
    build_runtime_flags,
    cleanup_npu_after_inference,
    run_profile_if_requested,
    setup_npu_and_runner,
)
from utils.video import build_video_argparser, run_video_inference_loop
from utils.pose_estimation import (
    annotate_pose_frame,
    dequantize_out,
    postprocess_pose,
    preprocess_frame_cv,
)


def make_process_fn(runner):
    def process_fn(bgr_frame):
        input_data, pad_info, orig_shape = preprocess_frame_cv(bgr_frame)
        raw_out = runner.infer(input_data)
        outputs = dequantize_out(raw_out, 0.0056150914169847965, -117)
        detections = postprocess_pose(outputs, orig_shape, pad_info)
        annotated, frame_detections = annotate_pose_frame(bgr_frame, detections)
        log_str = f"Poses: {len(detections)}"
        return annotated, frame_detections, runner.infer_time_ms, log_str
    return process_fn


def main():
    parser = build_video_argparser(
        "Run YOLOv8 Pose estimation on video, RTSP, or camera input.",
        default_json_results="pose_results.json",
        model_required=False,
        model_help="Path to the model VMFB (default: the one setup_demo.py downloaded)",
    )
    args = parser.parse_args()

    if args.model is None:
        local_model = local_pose_model_path()
        if local_model is None:
            parser.error(
                "no local pose model found; pass --model or run "
                "`python setup_demos.py pose_estimation` from torq-examples root"
            )
        args.model = str(local_model)

    ensure_pose_estimation_models(Path(args.model).parent, refresh=not args.no_refresh)
    runtime_flags = build_runtime_flags(args.tda, args.runtime_flags)

    if run_profile_if_requested(args, runtime_flags):
        return

    runner = setup_npu_and_runner(args, runtime_flags)
    try:
        run_video_inference_loop(args, make_process_fn(runner), "Pose Estimation")
    finally:
        cleanup_npu_after_inference()


if __name__ == "__main__":
    main()
