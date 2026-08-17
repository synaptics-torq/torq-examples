"""Preprocess an image for the RTMO NPU export: 1x3xSxS NCHW RGB in [0,255]
(per the model config: RGB, no rescale, no normalise, bilinear). ``letterbox``
(default) preserves aspect ratio and returns the geometry to invert it;
``stretch`` resizes straight to SxS. Pass ``dtype="bf16"`` for the vmfb path.
"""

import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required:  pip install opencv-python")

import ml_dtypes


def letterbox(img, size):
    """Aspect-preserving resize with padding -> (image, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def preprocess(img_bgr, size=320, mode="letterbox"):
    """BGR uint8 HWC -> ((1,3,S,S) float32 NCHW RGB [0,255], resize-geometry meta)."""
    if mode == "letterbox":
        proc, scale, pad_x, pad_y = letterbox(img_bgr, size)
    elif mode == "stretch":
        h, w = img_bgr.shape[:2]
        proc = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
        scale, pad_x, pad_y = (size / w, size / h), 0, 0
    else:
        raise ValueError(f"unknown mode {mode!r}")

    chw = np.transpose(proc[:, :, ::-1], (2, 0, 1))  # BGR->RGB, HWC->CHW
    tensor = np.ascontiguousarray(chw, dtype=np.float32)[None, ...]
    return tensor, {"scale": scale, "pad_x": pad_x, "pad_y": pad_y, "mode": mode}


def image_preprocess(image, size=320, mode="letterbox", dtype="f32"):
    img = cv2.imread(image)
    if img is None:
        raise FileNotFoundError(f"Could not read '{image}'.")
    tensor, meta = preprocess(img, size, mode)

    print("\nPreprocessing")
    print("-" * 60)
    print(f"{'Input image':<18}: {img.shape[1]} x {img.shape[0]}")
    print(f"{'Input tensor':<18}: {tensor.shape}")
    print(f"{'Resize mode':<18}: {mode}")
    print(f"{'Scale':<18}: {meta['scale']:.4f}")
    print(f"{'Padding':<18}: left/right={meta['pad_x']}, top/bottom={meta['pad_y']}")

    if dtype == "bf16":
        tensor = tensor.astype(ml_dtypes.bfloat16)
    return tensor, meta
