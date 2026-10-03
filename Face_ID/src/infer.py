"""Run the Face ID detector on one image file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from torq.runtime import VMFBInferenceRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Face_ID.src.postprocess import decode_face_outputs
from Face_ID.setup_demo import local_face_id_model_path
from utils.runtime import build_runtime_flags, cleanup_npu_after_inference

MODEL_WIDTH = 1280
MODEL_HEIGHT = 704
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_input_path(value: str) -> Path:
    """Resolve a path from the current directory, then the repository root."""
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    return path


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = min(MODEL_WIDTH / width, MODEL_HEIGHT / height)
    resized = cv2.resize(gray, (int(width * scale), int(height * scale)))
    canvas = np.zeros((MODEL_HEIGHT, MODEL_WIDTH), dtype=np.uint8)
    offset_x = (MODEL_WIDTH - resized.shape[1]) // 2
    offset_y = (MODEL_HEIGHT - resized.shape[0]) // 2
    canvas[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized
    tensor = np.clip(canvas.astype(np.int16) - 128, -128, 127).astype(np.int8)
    return tensor[None, :, :, None], (offset_x, offset_y, resized.shape[1], resized.shape[0], scale)


def map_boxes(detections, letterbox, image_shape):
    offset_x, offset_y, _width, _height, scale = letterbox
    image_height, image_width = image_shape[:2]
    mapped = []
    for label, confidence, box in detections:
        x, y, width, height = box
        x = (x - offset_x) / scale
        y = (y - offset_y) / scale
        width /= scale
        height /= scale
        x = float(np.clip(x, 0, image_width - 1))
        y = float(np.clip(y, 0, image_height - 1))
        width = float(np.clip(width, 0, image_width - x))
        height = float(np.clip(height, 0, image_height - y))
        mapped.append((label, confidence, [x, y, width, height]))
    return mapped


def prepare_outputs(outputs):
    """Copy device-backed model outputs to host arrays for postprocessing."""
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return [output.to_host() if hasattr(output, "to_host") else output for output in outputs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect faces in an image with a Face ID VMFB.")
    parser.add_argument(
        "--model", default=None,
        help="Path to face_detection.vmfb (default: the one setup_demo.py downloaded)",
    )
    parser.add_argument("--image", required=True, help="Input image file")
    parser.add_argument("--output", default="face_detection.jpg", help="Annotated output image")
    parser.add_argument("--json-results", default="face_detection_results.json")
    parser.add_argument("--device", default="torq")
    parser.add_argument(
        "--tda",
        choices=("cpu", "dmabuf"),
        default="dmabuf",
        help="Allocator backing Torq buffers (default: %(default)s)",
    )
    parser.add_argument(
        "--device-io",
        action="store_true",
        help="Allocate inputs and keep model outputs device-backed (enabled automatically with --tda dmabuf)",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    args = parser.parse_args()
    if args.model is None:
        local_model = local_face_id_model_path()
        if local_model is None:
            parser.error(
                "no local Face ID model found; pass --model or run "
                "`python setup_demos.py face_id` from torq-examples root"
            )
        args.model = str(local_model)
    device_io = args.device_io or args.tda == "dmabuf"

    model_path = resolve_input_path(args.model)
    image_path = resolve_input_path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Unable to read image: {image_path}")

    input_tensor, letterbox = preprocess_image(image)
    runner = VMFBInferenceRunner(
        str(model_path),
        device_uri=args.device,
        function="main",
        runtime_flags=build_runtime_flags(args.tda),
        device_outputs=device_io,
    )
    try:
        runner_input = runner.allocate_device_array(input_tensor) if device_io else input_tensor
        outputs = prepare_outputs(runner.infer([runner_input]))
        detections = decode_face_outputs(
            outputs,
            box1_scale=6.7147956,
            box1_zero_point=-61,
            box2_scale=6.6746836,
            box2_zero_point=-128,
            score_scale=1.0 / 256.0,
            score_zero_point=-128,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=0.4,
            image_width=MODEL_WIDTH,
            image_height=MODEL_HEIGHT,
        )
    finally:
        cleanup_npu_after_inference()

    results = map_boxes(detections, letterbox, image.shape)
    for _label, confidence, box in results:
        x, y, width, height = [int(value) for value in box]
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(image, f"face {confidence:.2f}", (x, max(y - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(args.output, image)
    Path(args.json_results).write_text(json.dumps([
        {"label": label, "confidence": confidence, "box": box}
        for label, confidence, box in results
    ], indent=2) + "\n")
    print(f"Detected {len(results)} face(s); wrote {args.output} and {args.json_results}")


if __name__ == "__main__":
    main()
