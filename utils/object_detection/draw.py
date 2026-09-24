# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

from utils.draw import letterbox_frame, draw_ui, load_font, letterbox_pil_image

# Re-export shared draw utilities so existing OD imports keep working
__all__ = [
    "annotate_frame",
    "draw_ui",
    "letterbox_frame",
    "render_annotated_image",
]

def annotate_frame(frame, detections, show_labels=True):
    import cv2

    annotated = frame.copy()
    frame_detections = []
    for label, confidence, box in detections:
        x1, y1, width, height = [float(v) for v in box]
        x2, y2 = x1 + width, y1 + height
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        if show_labels:
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
    from PIL import ImageDraw

    canvas, scale, offset_x, offset_y = letterbox_pil_image(image_path, display_size)
    draw = ImageDraw.Draw(canvas)
    font = load_font()

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
