#!/usr/bin/env python3
"""
Preprocess an image for the RTMO NPU export.

Returns a 1 x 3 x 320 x 320 NCHW RGB tensor with values in [0, 255].
Default dtype is float32 (for the ONNX reference path); pass dtype="bf16"
for the vmfb / torq-run-module path.

Per the model's preprocessor config:
    do_convert_rgb : true   -> BGR (cv2 default) becomes RGB
    do_rescale     : false  -> values stay in 0..255, NOT divided by 255
    do_normalize   : false  -> no mean/std subtraction
    do_resize      : true   -> bilinear
    resample       : 2      -> PIL BILINEAR

Resize modes:
  stretch    Resize straight to 320x320, ignoring aspect ratio.
  letterbox  Scale the long side to 320 and pad the short side, preserving
             aspect ratio. Returns scale/padding so detections can be mapped
             back to original image coordinates.
"""
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required:  pip install opencv-python")

import ml_dtypes
BF16 = ml_dtypes.bfloat16


def letterbox(img, size):
    """Aspect-preserving resize with padding. Returns (image, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def preprocess(img_bgr, size=320, mode="letterbox"):
    """
    BGR uint8 HWC image -> (1, 3, size, size) float32 NCHW RGB in [0, 255],
    plus the geometry needed to invert the resize.
    """
    if mode == "letterbox":
        proc, scale, pad_x, pad_y = letterbox(img_bgr, size)
    elif mode == "stretch":
        h, w = img_bgr.shape[:2]
        proc = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
        scale, pad_x, pad_y = (size / w, size / h), 0, 0
    else:
        raise ValueError(f"unknown mode {mode!r}")

    rgb = proc[:, :, ::-1]                       # BGR -> RGB
    chw = np.transpose(rgb, (2, 0, 1))           # HWC -> CHW
    tensor = np.ascontiguousarray(chw, dtype=np.float32)[None, ...]
    return tensor, {"scale": scale, "pad_x": pad_x, "pad_y": pad_y, "mode": mode}


def to_original_coords(xy, meta):
    """Map coordinates from network space back to the original image."""
    xy = np.asarray(xy, dtype=np.float32).copy()
    if meta["mode"] == "letterbox":
        xy[..., 0] = (xy[..., 0] - meta["pad_x"]) / meta["scale"]
        xy[..., 1] = (xy[..., 1] - meta["pad_y"]) / meta["scale"]
    else:
        sx, sy = meta["scale"]
        xy[..., 0] /= sx
        xy[..., 1] /= sy
    return xy


def image_preprocess(image, size=320, mode="letterbox", npy=None, out=None, dtype="f32"):
    img = cv2.imread(image)
    if img is None:
        raise FileNotFoundError(f"Could not read '{image}'.")

    tensor, meta = preprocess(img, size, mode)

    print("\nPreprocessing")
    print("-" * 60)
    print(f"{'Input image':<18}: {img.shape[1]} x {img.shape[0]}")
    print(f"{'Input tensor':<18}: {tensor.shape}")
    print(f"{'Data type':<18}: {tensor.dtype}")
    print(f"{'Value range':<18}: [{tensor.min():.1f}, {tensor.max():.1f}]")
    print(f"{'Resize mode':<18}: {mode}")
    print(f"{'Scale':<18}: {meta['scale']:.4f}")
    print(f"{'Padding':<18}: left/right={meta['pad_x']}, top/bottom={meta['pad_y']}")

    if dtype == "bf16":
        tensor = tensor.astype(BF16)
        print(f"{'Output type':<18}: bf16")

    if npy:
        np.save(npy, tensor)
        print(f"{'Saved .npy':<18}: {npy}")

    if out:
        with open(out, "wb") as fh:
            fh.write(tensor.astype(BF16).tobytes())
        print(f"{'Saved binary':<18}: {out} ({tensor.size * 2:,} bytes)")

    return tensor, meta