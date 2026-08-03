# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from utils.draw import letterbox_pil_image
from utils.pose_estimation.postprocess import SKELETON

# COCO keypoint names
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

_VISIBILITY_THRESHOLD = 0.6


def annotate_pose_frame(
    bgr_frame: np.ndarray,
    detections: List[Tuple[np.ndarray, float, np.ndarray]],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Annotate a BGR frame with bounding boxes, skeleton lines, and keypoint dots.

    Returns:
        (annotated_frame, frame_detections_json)
    """
    import cv2

    annotated = bgr_frame.copy()
    frame_detections = []

    for bbox, conf, keypoints in detections:
        x1, y1, w_box, h_box = [int(round(x)) for x in bbox]

        cv2.rectangle(annotated, (x1, y1), (x1 + w_box, y1 + h_box), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"Pose: {conf:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        for start_idx, end_idx in SKELETON:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                x_s, y_s, conf_s = keypoints[start_idx]
                x_e, y_e, conf_e = keypoints[end_idx]
                if conf_s > _VISIBILITY_THRESHOLD and conf_e > _VISIBILITY_THRESHOLD:
                    cv2.line(
                        annotated,
                        (int(x_s), int(y_s)),
                        (int(x_e), int(y_e)),
                        (0, 255, 255),
                        2,
                    )

        for _kpt_idx, (kx, ky, kconf) in enumerate(keypoints):
            if kconf > _VISIBILITY_THRESHOLD:
                cv2.circle(annotated, (int(kx), int(ky)), 3, (0, 255, 0), -1)

        kpts_list = [
            {
                "keypoint_id": kpt_idx,
                "keypoint_name": KEYPOINT_NAMES[kpt_idx] if kpt_idx < len(KEYPOINT_NAMES) else f"kpt_{kpt_idx}",
                "x": float(kx),
                "y": float(ky),
                "confidence": float(kconf),
            }
            for kpt_idx, (kx, ky, kconf) in enumerate(keypoints)
        ]

        frame_detections.append({
            "person_confidence": float(conf),
            "bounding_box": {
                "origin": {"x": x1, "y": y1},
                "size": {"w": w_box, "h": h_box},
            },
            "keypoints": kpts_list,
        })

    return annotated, frame_detections


def render_annotated_pose_image(
    image_path: str,
    results: List[Tuple[np.ndarray, float, np.ndarray]],
    display_size: Tuple[int, int],
):
    """
    Render pose results onto a letterboxed PIL image (for saving/display).
    Mirrors object detection's render_annotated_image.
    """
    from PIL import ImageDraw

    canvas, scale, offset_x, offset_y = letterbox_pil_image(image_path, display_size)
    draw = ImageDraw.Draw(canvas)

    for bbox, conf, keypoints in results:
        x1, y1, w_box, h_box = [float(v) for v in bbox]
        sx1 = x1 * scale + offset_x
        sy1 = y1 * scale + offset_y
        sx2 = (x1 + w_box) * scale + offset_x
        sy2 = (y1 + h_box) * scale + offset_y
        draw.rectangle([sx1, sy1, sx2, sy2], outline="green", width=2)

        for start_idx, end_idx in SKELETON:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                x_s, y_s, conf_s = keypoints[start_idx]
                x_e, y_e, conf_e = keypoints[end_idx]
                if conf_s > _VISIBILITY_THRESHOLD and conf_e > _VISIBILITY_THRESHOLD:
                    draw.line(
                        [
                            (x_s * scale + offset_x, y_s * scale + offset_y),
                            (x_e * scale + offset_x, y_e * scale + offset_y),
                        ],
                        fill="cyan",
                        width=2,
                    )

        r = 4
        for kx, ky, kconf in keypoints:
            if kconf > _VISIBILITY_THRESHOLD:
                sx = kx * scale + offset_x
                sy = ky * scale + offset_y
                draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill="lime")

    return canvas
