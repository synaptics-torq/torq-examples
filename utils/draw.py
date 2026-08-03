# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared display drawing utilities for vision inference scripts."""

from __future__ import annotations

import os

import numpy as np


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


def load_font(size=35):
    from PIL import ImageFont

    font_path = "/usr/share/fonts/ttf/LiberationSans-Regular.ttf"
    fallback_paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]

    try:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)
        for path in fallback_paths:
            if os.path.exists(path):
                return ImageFont.truetype(path, size + 5)
    except Exception:
        pass

    return ImageFont.load_default()


def letterbox_pil_image(image_path, display_size):
    """Open an image, letterbox it to display_size on a black canvas.

    Shared by render_annotated_image and render_annotated_pose_image.

    Returns:
        (canvas, scale, offset_x, offset_y)
    """
    from PIL import Image

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
    return canvas, scale, offset_x, offset_y


def display_image_gst(image_path, disp_w, disp_h):
    """Display a JPEG image on the Wayland display via gst-launch-1.0 for 5 seconds.

    Shared by object detection and pose estimation image inference.
    """
    import shutil
    import subprocess
    import time

    if not shutil.which("gst-launch-1.0"):
        return

    try:
        print("Attempting to display image...")
        print("Found gst-launch-1.0. Displaying with waylandsink for 5 seconds...")
        print(f"Using display resolution {disp_w}x{disp_h}")

        command = [
            "gst-launch-1.0",
            "filesrc", f"location={image_path}", "!",
            "jpegdec", "!",
            "videoconvert", "!",
            "imagefreeze", "!",
            "videoscale", "!",
            f"video/x-raw,width={disp_w},height={disp_h}", "!",
            "waylandsink",
        ]
        proc = subprocess.Popen(command)
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            pass
        proc.terminate()
        proc.wait()
        print("Display closed.")
    except Exception as exc:
        print(f"GStreamer failed: {exc}")
