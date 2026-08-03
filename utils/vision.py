# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared math utilities and constants for YOLOv8-style vision models (object detection, pose estimation)."""

from __future__ import annotations

import numpy as np
from typing import List

# Quantization constants (shared across all models)
IN_SCALE = 0.003921568859368563
IN_ZERO_POINT = -128
PADDING_COLOR = (114, 114, 114)

# Postprocessing thresholds
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def dequantize_out(output_data: np.ndarray, out_scale: float, out_zero_point: int, int8: bool = True) -> np.ndarray:
    """Dequantize output tensor from int8 to float32."""
    if int8:
        return (output_data.astype(np.float32) - out_zero_point) * out_scale
    return output_data


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """
    Non-Maximum Suppression using numpy.
    
    Args:
        boxes: (N, 4) array in format [x1, y1, w, h] (top-left x, top-left y, width, height)
        scores: (N,) confidence scores
        iou_threshold: IOU threshold for suppression
    
    Returns:
        List of indices to keep after NMS
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        index = order[0]
        keep.append(index)

        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])

        width = np.maximum(0.0, xx2 - xx1)
        height = np.maximum(0.0, yy2 - yy1)
        inter = width * height

        overlap = inter / (areas[index] + areas[order[1:]] - inter)
        remaining = np.where(overlap <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


def decode_yolo_boxes(boxes: np.ndarray, pad_info: tuple, orig_shape: tuple) -> np.ndarray:
    """Decode YOLOv8 normalized center-xywh boxes to pixel-space top-left-xywh.

    Shared by object detection and pose estimation postprocessing.

    Args:
        boxes: (N, 4) array in normalized center-xywh format
        pad_info: (pad_h_ratio, pad_w_ratio) letterbox padding fractions
        orig_shape: (h, w) original image dimensions

    Returns:
        (N, 4) array in pixel top-left-xywh format (in-place modification)
    """
    max_dim = max(orig_shape)
    boxes[:, 0] = (boxes[:, 0] - pad_info[1]) * max_dim  # x
    boxes[:, 1] = (boxes[:, 1] - pad_info[0]) * max_dim  # y
    boxes[:, 2] *= max_dim                                # w
    boxes[:, 3] *= max_dim                                # h
    boxes[:, 0] -= boxes[:, 2] / 2                        # center → top-left x
    boxes[:, 1] -= boxes[:, 3] / 2                        # center → top-left y
    return boxes
