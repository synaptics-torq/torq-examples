# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import os
import sys
from pathlib import Path

from pose_estimation.setup_demo import ensure_pose_estimation_models
from utils.preprocess import preprocess_image
from utils.runtime import build_runtime_flags, cleanup_npu_after_inference, setup_npu_and_runner
from utils.pose_estimation import (
    dequantize_out,
    postprocess_pose,
    render_annotated_pose_image,
)
from utils.draw import display_image_gst


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


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 Pose estimation on an image.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="torq")
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        default=False,
        help="Skip the model freshness check (offline/airgapped runs)",
    )
    parser.add_argument(
        "--tda", type=str, choices=["cpu", "dmabuf"], default="dmabuf",
        help="Allocator backing Torq device buffers (default: %(default)s)",
    )
    parser.add_argument(
        "--device-io", action="store_true",
        help="Preallocate inputs and keep outputs as device arrays",
    )
    parser.add_argument("--save-image", action="store_true", help="If set, output annotated image")
    parser.add_argument("--display", action="store_true", help="Display annotated frame")
    args = parser.parse_args()

    ensure_pose_estimation_models(Path(args.model).parent, refresh=not args.no_refresh)

    runtime_flags = build_runtime_flags(args.tda)
    runner = setup_npu_and_runner(args, runtime_flags)

    try:
        print("\n[1/4] Preprocessing...")
        try:
            input_data, pad_info, orig_shape = preprocess_image(args.image)
        except Exception as exc:
            print(exc)
            sys.exit(1)
        print("\n[2/4] Inference...")
        try:
            raw_out = runner.infer(input_data)
        except Exception as exc:
            print(f"Inference failed: {exc}")
            sys.exit(1)
        print(f"Time: {runner.infer_time_ms:.3f}ms")

        print("\n[3/4] Processing...")
        if raw_out.shape != (1, 56, 2100):
            print(f"Warning: Output shape {raw_out.shape} doesn't match expected (1, 56, 2100).")

        out_scale = 0.0056150914169847965
        out_zp = -117
        outputs = dequantize_out(raw_out, out_scale, out_zp, int8=True)

        results = postprocess_pose(outputs, orig_shape, pad_info)

        print("\n[4/4] Detections:")
        if not results:
            print("No poses detected.")

        for i, (bbox, conf, keypoints) in enumerate(results):
            print(f"  Person {i + 1:<3} Conf: {conf:.4f}  Box: {bbox.astype(int)}")

        maybe_save_and_display(args, results)
    finally:
        cleanup_npu_after_inference()


if __name__ == "__main__":
    main()
