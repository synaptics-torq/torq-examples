# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import numpy as np
from utils.vision import nms_numpy, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, decode_yolo_boxes

# Per-variant output dequantization for the fused 1x84x2100 int8 head.
# YOLO26 has an NMS-free one-to-one head, but its int8 output quantization
# saturates confidences near 0.5, producing tied duplicate boxes; the NMS
# pass below then acts as the required IoU dedupe.
VARIANTS = {
    "yolov8": {"out_scale": 0.004194467328488827, "out_zp": -128},
    "yolo26": {"out_scale": 0.00423651235178113, "out_zp": -128},
}


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

    mask = scores > CONFIDENCE_THRESHOLD
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    decode_yolo_boxes(boxes, pad_info, orig_shape)

    indices = nms_numpy(boxes, scores, IOU_THRESHOLD)

    results = []
    for index in indices[:10]:
        class_id = class_ids[index]
        label = labels.get(str(class_id), f"Class {class_id}") if labels else f"Class {class_id}"
        results.append((label, scores[index], boxes[index]))

    return results
