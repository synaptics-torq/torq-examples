# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import numpy as np

from utils.draw import letterbox_pil_image

_BOX_COLOR = (0, 220, 0)
_TEXT_COLOR = (255, 255, 80)
_TEXT_BG = (0, 0, 0)


def annotate_ocr_frame(bgr_frame: np.ndarray, results) -> np.ndarray:
    """Draw detected quads and their recognized text onto a BGR frame.

    ``results`` is the ``[(box, text, score), ...]`` list returned by ``run_ocr``.
    """
    import cv2

    frame = bgr_frame.copy()
    for box, text, _score in results:
        cv2.polylines(frame, [box.astype(np.int32).reshape(-1, 1, 2)],
                      isClosed=True, color=_BOX_COLOR, thickness=2)
        if not text:
            continue
        x, y = int(box[:, 0].min()), int(box[:, 1].min())
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Label above the box when there is room, otherwise just below it.
        ty = y - 4 if y - th - 6 >= 0 else int(box[:, 1].max()) + th + 4
        cv2.rectangle(frame, (x - 2, ty - th - 4), (x + tw + 4, ty + 2), _TEXT_BG, -1)
        cv2.putText(frame, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1,
                    cv2.LINE_AA)
    return frame


# Below this on-screen line height a label would be unreadable and would just
# bury the page, so dense documents get boxes only (the text goes to stdout).
_MIN_LABEL_HEIGHT_PX = 14


def render_annotated_ocr_image(image_path, results, display_size):
    """Letterbox ``image_path`` to ``display_size`` and draw the OCR results on it.

    Every detected quad is outlined. A text label is drawn only where the line is
    tall enough on screen to read it, so a dense page stays legible.
    """
    from PIL import ImageDraw

    canvas, scale, offset_x, offset_y = letterbox_pil_image(image_path, display_size)
    draw = ImageDraw.Draw(canvas)

    for box, text, _score in results:
        pts = [(float(x) * scale + offset_x, float(y) * scale + offset_y) for x, y in box]
        draw.line([*pts, pts[0]], fill=_BOX_COLOR, width=2)

        ys = [p[1] for p in pts]
        if not text or (max(ys) - min(ys)) < _MIN_LABEL_HEIGHT_PX:
            continue

        tx = min(p[0] for p in pts)
        left, top, right, bottom = draw.textbbox((tx, min(ys)), text)
        text_h = bottom - top
        # Label above the box when there is room, otherwise just below it.
        ty = min(ys) - text_h if min(ys) - text_h >= 0 else max(ys)
        left, top, right, bottom = draw.textbbox((tx, ty), text)
        draw.rectangle([left - 1, top - 1, right + 1, bottom + 1], fill=_TEXT_BG)
        draw.text((tx, ty), text, fill=_TEXT_COLOR)

    return canvas
