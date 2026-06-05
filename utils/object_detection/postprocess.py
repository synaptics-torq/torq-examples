# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import numpy as np


def dequantize_out(output_data, out_scale, out_zero_point, int8=True):
    if int8:
        return (output_data.astype(np.float32) - out_zero_point) * out_scale
    return output_data


def nms_numpy(boxes, scores, iou_threshold):
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


def postprocess(outputs, orig_shape, pad_info, labels=None):
    outputs = np.squeeze(outputs)
    outputs = outputs.transpose()

    if outputs.shape[1] < 5:
        print(f"Error: Output shape {outputs.shape} too small")
        return []

    boxes = outputs[:, :4]
    scores_data = outputs[:, 4:]

    class_ids = np.argmax(scores_data, axis=1)
    scores = np.max(scores_data, axis=1)

    confidence_threshold = 0.25
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    boxes[:, 0] -= pad_info[1]
    boxes[:, 1] -= pad_info[0]
    boxes[:, :4] *= max(orig_shape)
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2

    indices = nms_numpy(boxes, scores, 0.45)

    results = []
    for index in indices[:10]:
        class_id = class_ids[index]
        label = labels.get(str(class_id), f"Class {class_id}") if labels else f"Class {class_id}"
        results.append((label, scores[index], boxes[index]))

    return results
