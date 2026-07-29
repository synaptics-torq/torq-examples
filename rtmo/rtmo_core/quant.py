"""
Quantized (int8 / int16) model support for the RTMO demo.

The bf16 export and the quantized TFLite-derived exports differ in two ways:

  * layout  - bf16 is NCHW; the TFLite exports are NHWC, so their heads come
              back as (1, H, W, C) instead of (1, C, H, W).
  * scaling - the TFLite exports have quantized I/O, so the input must be
              quantized and every head dequantized with its own
              (scale, zero_point).

This module adapts the quantized models to the float NCHW form the existing
postprocess already expects, so rtmo.py stays common to all three variants.

Quantization parameters are read straight from the TFLite exports:
    rtmo_int16x8_i16io.tflite   int16 I/O, zero_point 0 everywhere
    rtmo_int8.tflite            int8 I/O, non-zero zero_points
"""

import numpy as np

# name -> vmfb file, entry function, input quantization.
#
# The bf16 export comes from the torch path (entry "torch_jit"); the quantized
# ones come from TFLite -> TOSA, whose entry point is "main".
MODELS = {
    "bf16": {
        "vmfb": "rtmo_tiny_bf16.vmfb",
        "function": "torch_jit",
        "quantized": False,
    },
    # Hybrid: int8 conv backbone + bf16 AIFI transformer + int8 head, chained as
    # three NSS-only vmfbs (no CSS ops). int8 speed with the transformer's higher
    # precision, which removes the full-int8 false positives. The backbone image
    # input is int8; heads come back int8 with their own scales.
    "hybrid": {
        "vmfb": ("rtmo_hyb_backbone_int8.vmfb",
                 "rtmo_hyb_transformer_bf16.vmfb",
                 "rtmo_hyb_head_int8.vmfb"),
        "function": "main",
        "quantized": True,
        "hybrid": True,
        "in_dtype": np.int8,
        "in_scale": 1.0,
        "in_zp": -128,
        # seam (scale, zero_point): backbone outputs -> {dequant} -> transformer
        # (bf16) -> {requant} -> head inputs. P3/P4 are skip connections.
        "seams": {
            "p3_shape": (1, 40, 40, 96),  "p4_shape": (1, 20, 20, 192), "p5_shape": (1, 10, 10, 256),
            "bb_p3": (0.026187874376773834, -117), "hd_p3": (0.026187879964709282, -117),
            "bb_p4": (0.048588335514068604, -122), "hd_p4": (0.048588335514068604, -122),
            "bb_p5": (0.037300221621990204,   -7), "hd_p5": (0.06488647311925888,     2),
        },
    },
}

# NHWC output shape -> (head name, scale, zero_point).
# Shapes are unique per head, so outputs are matched by shape rather than by
# position - the same approach the bf16 path uses in utils.unpack_heads.
HEAD_QUANT = {
    # int8 head part of the hybrid (its own PTQ calibration -> own scales).
    "hybrid": {
        (1, 20, 20, 1):   ("cls_scores_s16", 0.102855027, 95),
        (1, 10, 10, 1):   ("cls_scores_s32", 0.152402967, 100),
        (1, 20, 20, 4):   ("bbox_preds_s16", 0.0596383289, 3),
        (1, 10, 10, 4):   ("bbox_preds_s32", 0.0294422898, 0),
        (1, 20, 20, 17):  ("kpt_vis_s16",    0.0980607048, 21),
        (1, 10, 10, 17):  ("kpt_vis_s32",    0.103052862, 22),
        (1, 20, 20, 192): ("pose_feats_s16", 0.01536687,  -5),
        (1, 10, 10, 192): ("pose_feats_s32", 0.0186605658, 10),
    },
}


def quantize_input(tensor_nchw, model):
    """float32 NCHW [0,255] tensor -> quantized NHWC tensor for the vmfb."""
    spec = MODELS[model]
    nhwc = np.transpose(tensor_nchw, (0, 2, 3, 1))
    q = np.rint(nhwc / spec["in_scale"]) + spec["in_zp"]
    info = np.iinfo(spec["in_dtype"])
    return np.clip(q, info.min, info.max).astype(spec["in_dtype"])


def dequantize_heads(outputs, model):
    """
    Quantized NHWC vmfb outputs -> {head name: float32 NCHW array}.

    Produces exactly the same dict shape as utils.unpack_heads does for bf16,
    so the postprocess is identical for all three model variants.
    """
    table = HEAD_QUANT[model]
    heads = {}

    for o in outputs:
        arr = np.asarray(o)
        key = tuple(arr.shape)
        entry = table.get(key)
        if entry is None:
            raise ValueError(
                f"unexpected output shape {key} for model {model!r}; "
                f"expected one of {sorted(table)}"
            )
        name, scale, zero_point = entry
        if name in heads:
            raise ValueError(f"duplicate output for {name}")

        real = (arr.astype(np.float32) - zero_point) * scale
        heads[name] = np.transpose(real, (0, 3, 1, 2))  # NHWC -> NCHW

    missing = {n for n, _, _ in table.values()} - set(heads)
    if missing:
        raise ValueError(f"missing outputs: {sorted(missing)}")

    return heads
