# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import json
import os
import sys
from pathlib import Path

from object_detection.setup_demo import ensure_object_detection_models
from utils.vision import dequantize_out
from utils.preprocess import preprocess_image
from utils.runtime import build_runtime_flags, cleanup_npu_after_inference, setup_npu_and_runner
from utils.object_detection import (
    VARIANTS,
    postprocess,
    render_annotated_image,
)
from utils.draw import display_image_gst


def load_labels(labels_path):
    if not labels_path:
        return {}

    with open(labels_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if "names" in data:
        return {str(key): value for key, value in data["names"].items()}
    return data


def maybe_save_and_display(args, results):
    if not (args.save_image or args.display) or not results:
        return

    orientation = os.environ.get("ORIENTATION", "landscape")
    disp_w = int(os.environ.get("DISPLAY_WIDTH", 800 if orientation == "landscape" else 480))
    disp_h = int(os.environ.get("DISPLAY_HEIGHT", 480 if orientation == "landscape" else 800))

    print("\n[5/5] Saving result image...")
    try:
        img = render_annotated_image(args.image, results, (disp_w, disp_h))
        print(f"Resized image to {disp_w}x{disp_h} (letterboxed).")

        out_img = "output_yolo.jpg"
        img.save(out_img)
        print(f"Result image saved to: {out_img}")

        if args.display:
            display_image_gst(out_img, disp_w, disp_h)
    except Exception as exc:
        print(f"Failed to save result image: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Run YOLO object detection on an image.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels")
    parser.add_argument(
        "--variant", choices=sorted(VARIANTS), default="yolov8",
        help="Model head variant (default: %(default)s)",
    )
    parser.add_argument("--device", default="torq")
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        default=False,
        help="Skip the Hugging Face check for updated models (offline/airgapped runs)",
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

    ensure_object_detection_models(Path(args.model).parent, refresh=not args.no_refresh)

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
        if raw_out.shape != (1, 84, 2100):
            print(f"Warning: Output shape {raw_out.shape} doesn't match expected (1, 84, 2100). Metadata might be needed.")

        variant = VARIANTS[args.variant]
        outputs = dequantize_out(raw_out, variant["out_scale"], variant["out_zp"], int8=True)

        labels = load_labels(args.labels)
        results = postprocess(outputs, orig_shape, pad_info, labels)

        print("\n[4/4] Detections:")
        if not results:
            print("No objects detected.")

        for label, conf, box in results:
            print(f"  {label:<15} Conf: {conf:.4f}  Box: {box.astype(int)}")

        maybe_save_and_display(args, results)
    finally:
        cleanup_npu_after_inference()


if __name__ == "__main__":
    main()
