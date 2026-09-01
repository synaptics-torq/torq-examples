"""Map RTMO detections back to original-image coordinates and render them via
the shared pose annotator (utils.pose_estimation.draw.annotate_pose_frame)."""

import cv2
import numpy as np

from utils.pose_estimation.draw import annotate_pose_frame


def predictions(image_path, dets, keypoints, meta, output_path="rtmo_output.jpg", score_threshold=0.30):
    """Draw RTMO detections and poses on the original input image; return (path, n_drawn)."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read '{image_path}'.")
    scale = float(meta["scale"])
    pad = np.array([meta.get("pad_x", 0), meta.get("pad_y", 0)], np.float32)
    image_h, image_w = image.shape[:2]

    detections = []
    for det, pose in zip(np.asarray(dets[0], np.float32), np.asarray(keypoints[0], np.float32)):
        score = float(det[4])
        if not np.isfinite(score) or score < score_threshold:
            continue
        # Undo letterbox: model coords -> original image coords, clipped.
        box = (det[:4].reshape(2, 2) - pad) / scale
        (x1, y1), (x2, y2) = np.clip(box, 0, [image_w - 1, image_h - 1]).round()
        if x2 <= x1 or y2 <= y1:
            continue
        pose = pose.copy()
        pose[:, :2] = np.clip((pose[:, :2] - pad) / scale, 0, [image_w - 1, image_h - 1])
        detections.append(((x1, y1, x2 - x1, y2 - y1), score, pose))

    image, _ = annotate_pose_frame(image, detections)
    output_path = str(output_path)
    if not cv2.imwrite(output_path, image):
        raise OSError(f"Could not write output image '{output_path}'.")
    return output_path, len(detections)
