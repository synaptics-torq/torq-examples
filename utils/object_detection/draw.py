# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def letterbox_frame(frame, target_size):
    import cv2

    height, width = frame.shape[:2]
    target_width, target_height = target_size
    scale = min(target_width / width, target_height / height)
    new_width, new_height = int(width * scale), int(height * scale)

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_height, target_width, 3), dtype=frame.dtype)
    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas, (top, left, new_width, new_height)


def draw_ui(canvas, title, stats, video_rect):
    import cv2

    target_height, target_width = canvas.shape[:2]
    top, _left, _video_width, video_height = video_rect
    font = cv2.FONT_HERSHEY_SIMPLEX

    if top > 40:
        (text_width, text_height), _ = cv2.getTextSize(title, font, 1.1, 2)
        text_x = (target_width - text_width) // 2
        text_y = (top // 2) + (text_height // 2)
        cv2.putText(canvas, title, (text_x + 1, text_y + 1), font, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, title, (text_x, text_y), font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    bottom_y_start = top + video_height
    if target_height - bottom_y_start > 60:
        y_cursor = bottom_y_start + 40
        cv2.putText(canvas, f"FPS: {stats['fps']:.1f}", (30, y_cursor), font, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"NPU: {stats['npu']:.1f} ms", (30, y_cursor + 35), font, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

        count_text = f"DETECTIONS: {stats['count']}"
        (count_width, _), _ = cv2.getTextSize(count_text, font, 0.7, 2)
        cv2.putText(canvas, count_text, (target_width - count_width - 30, y_cursor + 15), font, 0.7, (0, 255, 100), 2, cv2.LINE_AA)


def annotate_frame(frame, detections):
    import cv2

    annotated = frame.copy()
    frame_detections = []
    for label, confidence, box in detections:
        x1, y1, width, height = [float(v) for v in box]
        x2, y2 = x1 + width, y1 + height
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        text = f"{label} {confidence:.2f}"
        text_y = int(y1) - 8 if int(y1) - 8 > 10 else int(y1) + 18
        cv2.putText(annotated, text, (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        frame_detections.append({
            "label": label,
            "confidence": float(confidence),
            "bounding_box": {
                "origin": {"x": int(round(x1)), "y": int(round(y1))},
                "size": {"x": int(round(width)), "y": int(round(height))},
            },
        })
    return annotated, frame_detections


def render_annotated_image(image_path, results, display_size):
    try:
        image = Image.open(image_path)
    except Exception as exc:
        raise RuntimeError(f"Error opening image {image_path}: {exc}") from exc

    if image.mode != "RGB":
        image = image.convert("RGB")

    target_width, target_height = display_size
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    new_width, new_height = int(width * scale), int(height * scale)

    resized = image.resize((new_width, new_height), Image.BILINEAR)
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    canvas.paste(resized, (offset_x, offset_y))

    draw = ImageDraw.Draw(canvas)
    font = _load_font()

    for label, _confidence, box in results:
        x1 = box[0] * scale + offset_x
        y1 = box[1] * scale + offset_y
        box_width = box[2] * scale
        box_height = box[3] * scale
        x2 = x1 + box_width
        y2 = y1 + box_height
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        text = f"{label}"
        text_pos = [x1, y1 - 45]
        if text_pos[1] < 0:
            text_pos[1] = y1 + 5

        try:
            left, top, right, bottom = draw.textbbox((text_pos[0], text_pos[1]), text, font=font)
            draw.rectangle((left - 5, top - 5, right + 5, bottom + 5), fill="red")
        except AttributeError:
            draw.rectangle((text_pos[0], text_pos[1], text_pos[0] + len(text) * 20, text_pos[1] + 40), fill="red")

        draw.text((text_pos[0], text_pos[1]), text, fill="white", font=font)

    return canvas


def _load_font():
    font_path = "/usr/share/fonts/ttf/LiberationSans-Regular.ttf"
    fallback_paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]

    try:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, 35)
        for path in fallback_paths:
            if os.path.exists(path):
                return ImageFont.truetype(path, 40)
    except Exception:
        pass

    return ImageFont.load_default()
