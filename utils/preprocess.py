# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared image preprocessing for YOLOv8-style vision models."""

from __future__ import annotations

import numpy as np
from PIL import Image

from utils.vision import IN_SCALE, IN_ZERO_POINT, PADDING_COLOR


def preprocess_image(image_path, target_size=(320, 320)):
    try:
        img = Image.open(image_path)
    except Exception as exc:
        raise RuntimeError(f"Error opening image {image_path}: {exc}") from exc

    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    new_width, new_height = target_size
    scale = min(new_width / width, new_height / height)

    new_unpad = (int(round(width * scale)), int(round(height * scale)))
    pad_width = (new_width - new_unpad[0]) / 2
    pad_height = (new_height - new_unpad[1]) / 2

    img_resized = img.resize(new_unpad, Image.BILINEAR)
    padded_img = Image.new("RGB", target_size, PADDING_COLOR)

    top = int(round(pad_height - 0.1))
    left = int(round(pad_width - 0.1))
    padded_img.paste(img_resized, (left, top))

    input_data = np.array(padded_img, dtype=np.float32)
    input_data /= 255.0

    input_data = input_data / IN_SCALE + IN_ZERO_POINT
    input_data = np.clip(input_data, -128, 127).astype(np.int8)
    input_data = np.expand_dims(input_data, axis=0)

    pad_info = (top / new_height, left / new_width)
    return input_data, pad_info, (height, width)


def preprocess_frame_cv(bgr_frame, target_size=(320, 320)):
    import cv2

    new_width, new_height = target_size
    height, width = bgr_frame.shape[:2]
    scale = min(new_width / width, new_height / height)
    new_unpad = (int(round(width * scale)), int(round(height * scale)))

    resized = cv2.resize(bgr_frame, new_unpad, interpolation=cv2.INTER_LINEAR)

    padded = np.full((new_height, new_width, 3), PADDING_COLOR, dtype=np.uint8)
    top = int(round((new_height - new_unpad[1]) / 2 - 0.1))
    left = int(round((new_width - new_unpad[0]) / 2 - 0.1))
    padded[top:top + new_unpad[1], left:left + new_unpad[0]] = resized

    rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
    input_data = np.clip(rgb / IN_SCALE + IN_ZERO_POINT, -128, 127).astype(np.int8)
    input_data = np.expand_dims(input_data, axis=0)

    pad_info = (top / new_height, left / new_width)
    return input_data, pad_info, (height, width)
