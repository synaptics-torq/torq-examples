# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import os
import sys

from pose_estimation.setup_demo import (
    download_pose_estimation,
    ensure_pose_estimation_models,
    local_pose_model_path,
)
from utils.preprocess import preprocess_image
from utils.runtime import Model, add_common_inference_args
from utils.tasks.keypoint_detection import infer_main
from utils.pose_estimation import (
    dequantize_out,
    postprocess_pose,
    render_annotated_pose_image,
)
from utils.draw import display_image_gst
from utils.video import setup_display_and_camera

# Dequantization of the fused 1x56x2100 int8 head.
OUT_SCALE = 0.0056150914169847965
OUT_ZP = -117


def maybe_save_and_display(args, results):
    if not (args.save_image or args.display) or not results:
        return

    orientation = os.environ.get("ORIENTATION", "landscape")
    disp_w = int(os.environ.get("DISPLAY_WIDTH", 800 if orientation == "landscape" else 480))
    disp_h = int(os.environ.get("DISPLAY_HEIGHT", 480 if orientation == "landscape" else 800))

    print("\n[5/5] Saving result image...")
    try:
        img = render_annotated_pose_image(args.image, results, (disp_w, disp_h))
        print(f"Resized image to {disp_w}x{disp_h} (letterboxed).")

        out_img = "output_pose.jpg"
        img.save(out_img)
        print(f"Result image saved to: {out_img}")

        if args.display:
            display_image_gst(out_img, disp_w, disp_h)
    except Exception as exc:
        print(f"Failed to save result image: {exc}")


def add_args(parser) -> None:
    parser.add_argument(
        "--model", default=None,
        help="Path to the model VMFB (default: the one setup_demo.py downloaded)",
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="torq")
    add_common_inference_args(parser)
    parser.add_argument("--save-image", action="store_true", help="If set, output annotated image")
    parser.add_argument("--display", action="store_true", help="Display annotated frame")


def make_runner(args, runtime):
    setup_display_and_camera(args)
    return Model(args.model, runtime)


def loop(runner, args, metrics) -> None:
    model = runner

    print("\n[1/4] Preprocessing...")
    try:
        input_data, pad_info, orig_shape = preprocess_image(args.image)
    except Exception as exc:
        print(exc)
        sys.exit(1)
    print("\n[2/4] Inference...")
    try:
        raw_out = model.run(input_data)[0]
        print(f"Time: {model.infer_time_ms:.3f}ms")
        latency_ns = int(model.infer_time_ms * 1e6)
        metrics.add("latency", latency_ns)
        metrics.add("frames_per_second", 1e9 / latency_ns)
    except Exception as exc:
        print(f"Inference failed: {exc}")
        sys.exit(1)

    print("\n[3/4] Processing...")
    if raw_out.shape != (1, 56, 2100):
        print(f"Warning: Output shape {raw_out.shape} doesn't match expected (1, 56, 2100).")

    outputs = dequantize_out(raw_out, OUT_SCALE, OUT_ZP, int8=True)
    results = postprocess_pose(outputs, orig_shape, pad_info)

    print("\n[4/4] Detections:")
    if not results:
        print("No poses detected.")

    for i, (bbox, conf, keypoints) in enumerate(results):
        print(f"  Person {i + 1:<3} Conf: {conf:.4f}  Box: {bbox.astype(int)}")

    maybe_save_and_display(args, results)


if __name__ == "__main__":
    infer_main(
        make_runner,
        loop,
        description="Run YOLOv8 Pose estimation on an image.",
        demo="pose_estimation",
        local_model=lambda args: local_pose_model_path(),
        ensure_models=ensure_pose_estimation_models,
        download=lambda args: download_pose_estimation(),
        add_args=add_args,
        message=(
            "no local pose model found; pass --model or run "
            "`python setup_demos.py pose_estimation` from torq-examples root"
        ),
    )
