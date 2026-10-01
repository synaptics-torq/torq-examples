"""Postprocess the quantized Face ID detector outputs."""

from __future__ import annotations

import numpy as np


def _dequantize(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (values.astype(np.float32) - zero_point) * scale


def decode_face_outputs(
    outputs,
    *,
    box1_scale: float,
    box1_zero_point: int,
    box2_scale: float,
    box2_zero_point: int,
    score_scale: float,
    score_zero_point: int,
    confidence_threshold: float,
    iou_threshold: float,
    image_width: int,
    image_height: int,
):
    if len(outputs) != 3:
        raise ValueError(f"expected three detector outputs, got {len(outputs)}")

    box1 = _dequantize(np.asarray(outputs[0]).reshape(-1, 2), box1_scale, box1_zero_point)
    scores = _dequantize(np.asarray(outputs[1]).reshape(-1), score_scale, score_zero_point)
    box2 = _dequantize(np.asarray(outputs[2]).reshape(-1, 2), box2_scale, box2_zero_point)
    if box1.shape != box2.shape or len(box1) != len(scores):
        raise ValueError("detector output tensors have incompatible shapes")

    boxes = np.column_stack((box1[:, 0], box1[:, 1], box2[:, 0], box2[:, 1]))
    valid = (scores >= confidence_threshold) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], scores[valid]
    if len(boxes) == 0:
        return []

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height - 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(boxes[index, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[index, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[index, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[index, 3], boxes[order[1:], 3])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        denominator = areas[index] + areas[order[1:]] - intersection
        overlap = np.divide(intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0)
        order = order[np.where(overlap <= iou_threshold)[0] + 1]

    return [
        ("face", float(scores[index]), np.array([boxes[index, 0], boxes[index, 1], boxes[index, 2] - boxes[index, 0], boxes[index, 3] - boxes[index, 1]]))
        for index in keep
    ]
