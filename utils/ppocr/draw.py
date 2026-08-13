# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import numpy as np

from utils.draw import letterbox_pil_image, load_font

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

# Labels are sized from the line they annotate, so they stay proportionate at
# any output resolution. PIL's built-in font is a ~11px bitmap whatever the
# canvas, which is illegible once the output is more than a few hundred px wide.
_LABEL_HEIGHT_RATIO = 0.62
_MIN_LABEL_FONT_PX = 13
_MAX_LABEL_FONT_PX = 34
_LABEL_PAD_PX = 2

_font_cache: dict[int, object] = {}


def _label_font(line_height: float):
    size = round(line_height * _LABEL_HEIGHT_RATIO)
    size = int(min(max(size, _MIN_LABEL_FONT_PX), _MAX_LABEL_FONT_PX))
    if size not in _font_cache:
        _font_cache[size] = load_font(size)
    return _font_cache[size]


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
        line_h = max(ys) - min(ys)
        if not text or line_h < _MIN_LABEL_HEIGHT_PX:
            continue

        font = _label_font(line_h)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top

        # Keep the label inside the canvas rather than letting a long line run
        # off the right edge.
        tx = min(p[0] for p in pts)
        tx = max(min(tx, canvas.width - text_w - _LABEL_PAD_PX), _LABEL_PAD_PX)
        # Label above the box when there is room, otherwise just below it.
        ty = min(ys) - text_h - _LABEL_PAD_PX
        if ty < 0:
            ty = max(ys) + _LABEL_PAD_PX

        left, top, right, bottom = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle(
            [left - _LABEL_PAD_PX, top - _LABEL_PAD_PX,
             right + _LABEL_PAD_PX, bottom + _LABEL_PAD_PX],
            fill=_TEXT_BG,
        )
        draw.text((tx, ty), text, fill=_TEXT_COLOR, font=font)

    return canvas
