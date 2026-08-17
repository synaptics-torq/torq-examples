"""Quantization support for the RTMO hybrid demo.

A compiled vmfb does not expose its quantization — IREE returns raw int8
tensors. The (scale, zero_point) params live only in the source TFLite parts,
so they are read from the files shipped alongside the vmfbs: the backbone
image-input quant, the P3/P4/P5 seam scales (backbone outputs / head inputs,
for host-side requant between the chained parts), and the eight head output
scales. Everything is matched by (unique) NHWC shape, not position.
"""

import numpy as np

BACKBONE_TFLITE = "rtmo_hybrid_backbone_int8.tflite"
TRANSFORMER_TFLITE = "rtmo_hybrid_transformer_bf16.tflite"
HEAD_TFLITE = "rtmo_hybrid_head_int8.tflite"

# NHWC head-output shape -> canonical head name (shapes are unique per head).
HEAD_NAMES = {
    (1, 20, 20, 1): "cls_scores_s16",   (1, 10, 10, 1): "cls_scores_s32",
    (1, 20, 20, 4): "bbox_preds_s16",   (1, 10, 10, 4): "bbox_preds_s32",
    (1, 20, 20, 17): "kpt_vis_s16",     (1, 10, 10, 17): "kpt_vis_s32",
    (1, 20, 20, 192): "pose_feats_s16", (1, 10, 10, 192): "pose_feats_s32",
}
# FPN seam feature-map shapes (NHWC): P3/P4 skips + transformer-carried P5.
P3_SHAPE = (1, 40, 40, 96)
P4_SHAPE = (1, 20, 20, 192)
P5_SHAPE = (1, 10, 10, 256)


def _tflite_io_quant(tflite_path):
    """Return (inputs, outputs), each a list of (shape, dtype, scale, zp)."""
    try:
        import ai_edge_litert.interpreter as lite
    except ImportError:  # same fallback order as runners.py
        import tensorflow.lite as lite

    interp = lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()

    def rows(details):
        return [(tuple(int(x) for x in d["shape"]), d["dtype"], float(d["quantization"][0]), int(d["quantization"][1])) for d in details]

    return rows(interp.get_input_details()), rows(interp.get_output_details())


def read_hybrid_quant(backbone_tflite, head_tflite):
    """Read quant params from the backbone + head TFLite parts.

    Returns ``{in_scale, in_zp, in_dtype, seams, head_quant}``; ``seams`` is the
    dict :class:`.hybrid.HybridRunner` expects, ``head_quant`` maps
    ``nhwc_shape -> (name, scale, zero_point)``.
    """
    bb_in, bb_out = _tflite_io_quant(backbone_tflite)
    hd_in, hd_out = _tflite_io_quant(head_tflite)
    _, in_dtype, in_scale, in_zp = bb_in[0]  # single image input

    def sz(rows, shape):
        for sh, _dtype, scale, zp in rows:
            if sh == shape:
                return (scale, zp)
        raise ValueError(f"shape {shape} not in TFLite I/O {[r[0] for r in rows]}")

    seams = {
        "p3_shape": P3_SHAPE, "p4_shape": P4_SHAPE, "p5_shape": P5_SHAPE,
        "bb_p3": sz(bb_out, P3_SHAPE), "hd_p3": sz(hd_in, P3_SHAPE),
        "bb_p4": sz(bb_out, P4_SHAPE), "hd_p4": sz(hd_in, P4_SHAPE),
        "bb_p5": sz(bb_out, P5_SHAPE), "hd_p5": sz(hd_in, P5_SHAPE),
    }

    head_quant = {}
    for shape, _dtype, scale, zp in hd_out:
        name = HEAD_NAMES.get(shape)
        if name is None:
            raise ValueError(f"unexpected head output shape {shape}")
        head_quant[shape] = (name, scale, zp)
    missing = set(HEAD_NAMES.values()) - {n for n, _, _ in head_quant.values()}
    if missing:
        raise ValueError(f"missing head outputs in TFLite: {sorted(missing)}")

    return {"in_scale": in_scale, "in_zp": in_zp, "in_dtype": in_dtype, "seams": seams, "head_quant": head_quant}


def quantize_input(tensor_nchw, in_scale, in_zp, in_dtype=np.int8):
    """float32 NCHW [0,255] tensor -> quantized NHWC tensor for the backbone."""
    q = np.rint(np.transpose(tensor_nchw, (0, 2, 3, 1)) / in_scale) + in_zp
    info = np.iinfo(in_dtype)
    return np.clip(q, info.min, info.max).astype(in_dtype)


def dequantize_heads(outputs, head_quant):
    """Quantized NHWC vmfb outputs -> {head name: float32 NCHW array}, matched by shape."""
    heads = {}
    for o in outputs:
        arr = np.asarray(o)
        entry = head_quant.get(tuple(arr.shape))
        if entry is None:
            raise ValueError(f"unexpected output shape {arr.shape}; expected one of {sorted(head_quant)}")
        name, scale, zero_point = entry
        if name in heads:
            raise ValueError(f"duplicate output for {name}")
        heads[name] = np.transpose((arr.astype(np.float32) - zero_point) * scale, (0, 3, 1, 2))

    missing = {n for n, _, _ in head_quant.values()} - set(heads)
    if missing:
        raise ValueError(f"missing outputs: {sorted(missing)}")
    return heads
