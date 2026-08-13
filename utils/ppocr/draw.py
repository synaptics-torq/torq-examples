# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

from utils.draw import letterbox_pil_image, load_font

_BOX_COLOR = (0, 220, 0)
_TEXT_COLOR = (255, 255, 80)
_TEXT_BG = (0, 0, 0)

# Labels are sized from the line they annotate; PIL's built-in font is a fixed ~11px bitmap.
# Below _MIN_LABEL_HEIGHT_PX a label is unreadable anyway, so dense pages get boxes only.
_MIN_LABEL_HEIGHT_PX = 14
_LABEL_HEIGHT_RATIO = 0.62
_MIN_LABEL_FONT_PX, _MAX_LABEL_FONT_PX = 13, 34
_LABEL_PAD_PX = 2

_font_cache: dict[int, object] = {}


def _label_font(line_height: float):
    size = int(min(max(round(line_height * _LABEL_HEIGHT_RATIO), _MIN_LABEL_FONT_PX), _MAX_LABEL_FONT_PX))
    if size not in _font_cache:
        _font_cache[size] = load_font(size)
    return _font_cache[size]


def render_annotated_ocr_image(image_path, results, display_size):
    """Letterbox ``image_path`` to ``display_size`` and draw the OCR results on it."""
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
        # Keep the label inside the canvas, and above the box unless it would clip.
        tx = max(min(min(p[0] for p in pts), canvas.width - text_w - _LABEL_PAD_PX), _LABEL_PAD_PX)
        ty = min(ys) - text_h - _LABEL_PAD_PX
        if ty < 0:
            ty = max(ys) + _LABEL_PAD_PX
        left, top, right, bottom = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle([left - _LABEL_PAD_PX, top - _LABEL_PAD_PX, right + _LABEL_PAD_PX, bottom + _LABEL_PAD_PX], fill=_TEXT_BG)
        draw.text((tx, ty), text, fill=_TEXT_COLOR, font=font)

    return canvas
