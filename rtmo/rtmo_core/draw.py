import cv2
import numpy as np

# COCO 17-keypoint skeleton (nose/eyes/ears, arms, torso, legs).
COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def predictions(image_path, dets, keypoints, meta, output_path="rtmo_output.jpg", score_threshold=0.30, keypoint_threshold=0.30):
    """Draw RTMO detections and poses on the original input image; return (path, n_drawn)."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read '{image_path}'.")
    scale = float(meta["scale"])
    pad_x, pad_y = float(meta.get("pad_x", 0)), float(meta.get("pad_y", 0))
    image_h, image_w = image.shape[:2]

    drawn = 0
    for detection, pose in zip(np.asarray(dets[0], np.float32), np.asarray(keypoints[0], np.float32)):
        score = float(detection[4])
        if not np.isfinite(score) or score < score_threshold:
            continue

        # Undo letterbox: model coords -> original image coords.
        x1, y1, x2, y2 = detection[:4]
        x1 = int(np.clip(round((x1 - pad_x) / scale), 0, image_w - 1))
        y1 = int(np.clip(round((y1 - pad_y) / scale), 0, image_h - 1))
        x2 = int(np.clip(round((x2 - pad_x) / scale), 0, image_w - 1))
        y2 = int(np.clip(round((y2 - pad_y) / scale), 0, image_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        pose = pose.copy()
        pose[:, 0] = np.clip((pose[:, 0] - pad_x) / scale, 0, image_w - 1)
        pose[:, 1] = np.clip((pose[:, 1] - pad_y) / scale, 0, image_h - 1)

        # Box + confidence label.
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"person {score:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(image, (x1, max(0, y1 - label_h - baseline - 6)), (min(image_w - 1, x1 + label_w + 8), y1), (0, 220, 0), -1)
        cv2.putText(image, label, (x1 + 4, max(label_h + 1, y1 - baseline - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Skeleton lines (both endpoints visible) + keypoint dots.
        for start, end in COCO_SKELETON:
            if pose[start, 2] >= keypoint_threshold and pose[end, 2] >= keypoint_threshold:
                p1 = tuple(np.rint(pose[start, :2]).astype(int))
                p2 = tuple(np.rint(pose[end, :2]).astype(int))
                cv2.line(image, p1, p2, (255, 180, 0), 2, cv2.LINE_AA)
        for x, y, visibility in pose:
            if visibility >= keypoint_threshold:
                cv2.circle(image, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1, cv2.LINE_AA)
        drawn += 1

    output_path = str(output_path)
    if not cv2.imwrite(output_path, image):
        raise OSError(f"Could not write output image '{output_path}'.")
    return output_path, drawn
