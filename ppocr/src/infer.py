# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""PP-OCRv6-tiny OCR: DBNet text detection + CTC text recognition on the Torq NPU.

Detection runs at one static shape; recognition uses one vmfb per width bucket,
so each detected line is padded to the narrowest width that fits it.
"""

import argparse
import os
import sys
from pathlib import Path

from ppocr.setup_demo import ensure_ppocr_models
from utils.draw import display_image_gst
from utils.ppocr import (
    BucketTextRecognizer,
    NPUBackend,
    ORTBackend,
    TextDetector,
    TextRecognizer,
    load_char_dict,
    render_annotated_ocr_image,
    run_ocr,
)
from utils.runtime import build_runtime_flags, cleanup_npu_after_inference, setup_npu_for_inference

DEFAULT_DET = "ppocr_det_800x608.vmfb"
DEFAULT_REC_YML = "ppocr_rec.yml"
DEFAULT_REC_BUCKET_DIR = "rec_buckets"
DEFAULT_BUCKETS = (320, 640, 1280, 2432)


def build_detector(args, runtime_flags):
    """Detection backend: NPU vmfb (static H×W) or fp32 ONNX Runtime."""
    if args.det_backend == "ort":
        if not args.det_onnx:
            sys.exit("--det-backend ort requires --det-onnx")
        return TextDetector(ORTBackend(args.det_onnx))

    det_vmfb = args.det_vmfb or (Path(args.models) / DEFAULT_DET)
    if not Path(det_vmfb).exists():
        sys.exit(f"Detection vmfb not found: {det_vmfb}")
    backend = NPUBackend(det_vmfb, device_uri=args.device, runtime_flags=runtime_flags,
                         device_io=args.device_io)
    return TextDetector(backend, static_hw=tuple(args.det_hw))


def build_recognizer(args, char_dict, runtime_flags):
    """Recognition backend: bucketed NPU vmfbs, a single NPU vmfb, or ONNX Runtime."""
    if args.rec_backend == "ort":
        if not args.rec_onnx:
            sys.exit("--rec-backend ort requires --rec-onnx")
        return TextRecognizer(char_dict, ORTBackend(args.rec_onnx))

    if args.rec_vmfb:  # single static-width recognizer
        backend = NPUBackend(args.rec_vmfb, device_uri=args.device, runtime_flags=runtime_flags,
                             device_io=args.device_io)
        return TextRecognizer(char_dict, backend, static_width=args.rec_width)

    bucket_dir = Path(args.rec_bucket_dir or (Path(args.models) / DEFAULT_REC_BUCKET_DIR))
    backends = {}
    for width in args.rec_buckets:
        vmfb = bucket_dir / f"rec_w{width}.vmfb"
        if not vmfb.exists():
            sys.exit(f"Recognition vmfb not found: {vmfb}")
        backends[width] = NPUBackend(vmfb, device_uri=args.device, runtime_flags=runtime_flags,
                                     device_io=args.device_io)
    return BucketTextRecognizer(char_dict, backends)


def maybe_save_and_display(args, results):
    if not (args.save_image or args.display) or not results:
        return

    orientation = os.environ.get("ORIENTATION", "landscape")
    disp_w = int(os.environ.get("DISPLAY_WIDTH", 800 if orientation == "landscape" else 480))
    disp_h = int(os.environ.get("DISPLAY_HEIGHT", 480 if orientation == "landscape" else 800))

    print("\n[5/5] Saving result image...")
    try:
        img = render_annotated_ocr_image(args.image, results, (disp_w, disp_h))
        out_img = "output_ocr.jpg"
        img.save(out_img)
        print(f"Result image saved to: {out_img}")

        if args.display:
            display_image_gst(out_img, disp_w, disp_h)
    except Exception as exc:
        print(f"Failed to save result image: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Run PP-OCRv6-tiny OCR on an image.")
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--models", default=".",
        help="Directory holding the PP-OCR assets (default: %(default)s)",
    )
    parser.add_argument("--device", default="torq")
    parser.add_argument(
        "--no-refresh", action="store_true", default=False,
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
    # Detection
    parser.add_argument("--det-vmfb", help=f"Detection vmfb (default: <models>/{DEFAULT_DET})")
    parser.add_argument("--det-onnx", help="Detection fp32 ONNX, for --det-backend ort")
    parser.add_argument(
        "--det-hw", type=int, nargs=2, metavar=("H", "W"), default=[800, 608],
        help="Static input the detection vmfb was compiled for (default: %(default)s)",
    )
    parser.add_argument("--det-backend", choices=["npu", "ort"], default="npu")
    # Recognition
    parser.add_argument("--rec-yml", help=f"Char dict (default: <models>/{DEFAULT_REC_YML})")
    parser.add_argument("--rec-bucket-dir", help="Directory of rec_w<W>.vmfb bucket models")
    parser.add_argument(
        "--rec-buckets", type=int, nargs="+", default=list(DEFAULT_BUCKETS),
        help="Static widths available as bucket vmfbs (default: %(default)s)",
    )
    parser.add_argument("--rec-vmfb", help="Single recognition vmfb instead of buckets")
    parser.add_argument("--rec-width", type=int, default=320, help="Width of a single --rec-vmfb")
    parser.add_argument("--rec-onnx", help="Recognition fp32 ONNX, for --rec-backend ort")
    parser.add_argument("--rec-backend", choices=["npu", "ort"], default="npu")
    # Output
    parser.add_argument("--drop-score", type=float, default=0.5,
                        help="Minimum recognition confidence to keep a line")
    parser.add_argument("--save-image", action="store_true", help="If set, output annotated image")
    parser.add_argument("--display", action="store_true", help="Display annotated frame")
    args = parser.parse_args()

    ensure_ppocr_models(args.models, refresh=not args.no_refresh)

    rec_yml = args.rec_yml or (Path(args.models) / DEFAULT_REC_YML)
    if not Path(rec_yml).exists():
        sys.exit(f"Recognition char dict not found: {rec_yml}")

    runtime_flags = build_runtime_flags(args.tda)
    uses_npu = "npu" in (args.det_backend, args.rec_backend)
    if uses_npu:
        setup_npu_for_inference()

    try:
        import cv2

        print("\n[1/5] Loading image and models...")
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"Could not read image '{args.image}'")
        char_dict = load_char_dict(rec_yml)
        detector = build_detector(args, runtime_flags)
        recognizer = build_recognizer(args, char_dict, runtime_flags)
        print(f"Image {img.shape[1]}x{img.shape[0]}, vocabulary {len(char_dict)} chars")

        print(f"\n[2/5] Detection ({args.det_backend})...")
        print(f"[3/5] Recognition ({args.rec_backend})...")
        results, stats = run_ocr(img, detector, recognizer, drop_score=args.drop_score)
        print(f"Time: detection {stats['det_ms']:.1f}ms, recognition {stats['rec_ms']:.1f}ms "
              f"({stats['n_boxes']} boxes detected)")

        print("\n[4/5] Text:")
        if not results:
            print("No text detected.")
        for i, (_box, text, score) in enumerate(results, 1):
            print(f"  {i:<3} [{score:.3f}] {text}")

        maybe_save_and_display(args, results)
    finally:
        if uses_npu:
            cleanup_npu_after_inference()


if __name__ == "__main__":
    main()
