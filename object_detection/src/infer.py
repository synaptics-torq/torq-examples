# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from object_detection.setup_demo import (
    VARIANT_MODEL_FILES,
    default_model_path,
    download_object_detection,
    ensure_object_detection_models,
)
from utils.runtime import Model, add_common_inference_args
from utils.tasks.object_detection import infer_main
from utils.vision import dequantize_out
from utils.object_detection import (
    VARIANTS,
    postprocess,
    render_annotated_image,
)
from utils.draw import display_image_gst
from utils.preprocess import preprocess_image
from utils.video import setup_display_and_camera


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


def add_args(parser) -> None:
    parser.add_argument(
        "--model", default=None,
        help="Path to the model VMFB (default: the one setup_demo.py downloads for --variant)",
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels")
    parser.add_argument(
        "--variant", choices=sorted(VARIANTS), default="yolov8",
        help="Model head variant (default: %(default)s)",
    )
    parser.add_argument("--device", default="torq")
    add_common_inference_args(parser)
    parser.add_argument("--save-image", action="store_true", help="If set, output annotated image")
    parser.add_argument("--display", action="store_true", help="Display annotated frame")


def local_model(args):
    expected = default_model_path(args.variant)
    return expected if expected.exists() else None


def message(args):
    return (
        f"default model {default_model_path(args.variant)} not found; pass --model or run "
        f"`python setup_demos.py object_detection` from torq-examples root"
    )


def make_runner(args, runtime):
    setup_display_and_camera(args)
    model = Model(args.model, runtime)
    labels = load_labels(args.labels)
    return model, labels, VARIANTS[args.variant]


def loop(runner, args, metrics) -> None:
    model, labels, variant = runner

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
    if raw_out.shape != (1, 84, 2100):
        print(f"Warning: Output shape {raw_out.shape} doesn't match expected (1, 84, 2100). Metadata might be needed.")

    outputs = dequantize_out(raw_out, variant["out_scale"], variant["out_zp"], int8=True)
    results = postprocess(outputs, orig_shape, pad_info, labels)

    print("\n[4/4] Detections:")
    if not results:
        print("No objects detected.")

    for label, conf, box in results:
        print(f"  {label:<15} Conf: {conf:.4f}  Box: {box.astype(int)}")

    maybe_save_and_display(args, results)


if __name__ == "__main__":
    infer_main(
        make_runner,
        loop,
        description="Run YOLO object detection on an image.",
        demo="object_detection",
        local_model=local_model,
        ensure_models=ensure_object_detection_models,
        download=lambda args: download_object_detection([VARIANT_MODEL_FILES[args.variant][0]]),
        add_args=add_args,
        message=message,
    )
