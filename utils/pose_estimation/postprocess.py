# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from utils.vision import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, nms_numpy, decode_yolo_boxes


# YOLOv8 Pose skeleton connections (17 keypoints COCO format)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),    # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                         # torso
    (11, 13), (13, 15), (12, 14), (14, 16)              # legs
]


def postprocess_pose(
    outputs: np.ndarray,
    orig_shape: Tuple[int, int],
    pad_info: Tuple[float, float],
    conf_threshold: float = None,
    iou_threshold: float = None,
    max_detections: int = 10,
) -> List[Tuple[np.ndarray, float, np.ndarray]]:
    """
    YOLOv8 Pose postprocessing.

    Args:
        outputs: (1, 56, 2100) where 56 = 4(bbox) + 1(conf) + 17*3(keypoints)
        orig_shape: (h, w) original image shape
        pad_info: (pad_h_ratio, pad_w_ratio)
        conf_threshold: confidence threshold (defaults to shared CONFIDENCE_THRESHOLD)
        iou_threshold: IOU threshold for NMS (defaults to shared IOU_THRESHOLD)
        max_detections: maximum detections to return

    Returns:
        List of (bbox, conf, keypoints_array) tuples
        where bbox = [x1, y1, w, h], keypoints_array = (17, 3) with (x, y, conf)
    """
    if conf_threshold is None:
        conf_threshold = CONFIDENCE_THRESHOLD
    if iou_threshold is None:
        iou_threshold = IOU_THRESHOLD

    outputs = np.squeeze(outputs)   # (56, 2100)
    outputs = outputs.transpose()   # (2100, 56)

    if outputs.shape[1] < 56:
        print(f"Error: Output shape {outputs.shape} incompatible with pose model")
        return []

    boxes = outputs[:, :4]           # (2100, 4) - bbox center-wh format
    conf_scores = outputs[:, 4]      # (2100,) - object confidence
    keypoints_raw = outputs[:, 5:56] # (2100, 51) - 17 keypoints * 3 (x, y, conf)

    mask = conf_scores > conf_threshold
    boxes = boxes[mask]
    conf_scores = conf_scores[mask]
    keypoints_raw = keypoints_raw[mask]

    if len(boxes) == 0:
        return []

    decode_yolo_boxes(boxes, pad_info, orig_shape)
    max_dim = max(orig_shape)

    indices = nms_numpy(boxes, conf_scores, iou_threshold)

    results = []
    for i in indices[:max_detections]:
        bbox = boxes[i]
        conf = conf_scores[i]

        kpts = keypoints_raw[i].reshape(17, 3)
        kpts[:, 0] = (kpts[:, 0] - pad_info[1]) * max_dim
        kpts[:, 1] = (kpts[:, 1] - pad_info[0]) * max_dim
        kpts[:, 2] *= conf

        results.append((bbox, conf, kpts))

    return results
