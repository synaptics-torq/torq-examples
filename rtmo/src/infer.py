# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""RTMO tiny multi-person pose demo on Torq.

Chains the three NSS-only hybrid vmfbs — int8 conv backbone -> bf16 AIFI
transformer neck -> int8 detection head — requantizing at the seams host-side,
decodes the eight head tensors (NMS + DCC/SimCC pose) on the host, and draws
bounding boxes + 17-keypoint skeletons on the image. Assets download from
Hugging Face (``Synaptics/RTMO_pose``) on first run.
"""

import argparse
from pathlib import Path

from rtmo.rtmo_core.draw import predictions as draw_predictions
from rtmo.rtmo_core.hybrid import HybridRunner
from rtmo.rtmo_core.postprocess import model_postprocess
from rtmo.rtmo_core.preprocess import image_preprocess
from rtmo.rtmo_core.quant import (
    BACKBONE_TFLITE,
    HEAD_TFLITE,
    dequantize_heads,
    quantize_input,
    read_hybrid_quant,
)
from rtmo.setup_demo import ensure_rtmo_models
from utils.npu import enable_npu_clock

BACKBONE_VMFB = "rtmo_hyb_backbone_int8.vmfb"
TRANSFORMER_VMFB = "rtmo_hyb_transformer_bf16.vmfb"
HEAD_VMFB = "rtmo_hyb_head_int8.vmfb"


def main():
    parser = argparse.ArgumentParser(
        description="Run RTMO tiny multi-person pose estimation on an image."
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Dir with the RTMO assets (default: models/Synaptics/RTMO_pose).",
    )
    parser.add_argument(
        "--image", default=None,
        help="Input image (default: the downloaded calib/person.jpg sample).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output image path (default: <input-stem>_rtmo.jpg).",
    )
    parser.add_argument(
        "--device", default="torq",
        help="IREE device URI to run the vmfbs on (default: %(default)s).",
    )
    parser.add_argument(
        "--no-refresh", action="store_true",
        help="Skip the Hugging Face check for updated assets (offline runs).",
    )
    args = parser.parse_args()

    model_dir = ensure_rtmo_models(args.model_dir, refresh=not args.no_refresh)

    ok, message = enable_npu_clock()
    print(f"[NPU] {message}")

    vmfb_dir = model_dir / "vmfb"
    tflite_dir = model_dir / "tflite"
    # Quant params (input scale, seam scales, head scales) are read from the
    # TFLite parts, not hardcoded — a compiled vmfb does not expose them.
    params = read_hybrid_quant(tflite_dir / BACKBONE_TFLITE, tflite_dir / HEAD_TFLITE)
    runner = HybridRunner(
        str(vmfb_dir / BACKBONE_VMFB),
        str(vmfb_dir / TRANSFORMER_VMFB),
        str(vmfb_dir / HEAD_VMFB),
        params["seams"],
        device_uri=args.device,
    )

    image = args.image or str(model_dir / "calib" / "person.jpg")
    output = args.output or f"{Path(image).stem}_rtmo.jpg"

    print("\n" + "=" * 70)
    print("RTMO POSE ESTIMATION  (int8 backbone + bf16 transformer + int8 head)")
    print("=" * 70)
    print(f"{'Image':<14} {image}")
    print(f"{'Model dir':<14} {vmfb_dir}")

    tensor, meta = image_preprocess(image)
    quantized = quantize_input(tensor, params["in_scale"], params["in_zp"], params["in_dtype"])

    print("\nRunning inference (three chained vmfbs on the NPU)...")
    heads = dequantize_heads(runner.infer([quantized]), params["head_quant"])
    print(
        f"Inference took {runner.infer_time_ms:.2f} ms "
        f"(backbone {runner.part_ms['backbone']:.1f} + "
        f"transformer {runner.part_ms['transformer']:.1f} + "
        f"head {runner.part_ms['head']:.1f} ms)"
    )

    dets, keypoints = model_postprocess(heads)
    path, num_people = draw_predictions(image, dets, keypoints, meta, output)

    print("-" * 70)
    print(f"{'People drawn':<14} {num_people}")
    print(f"{'Output image':<14} {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
