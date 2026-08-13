# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from utils.ppocr.backends import NPUBackend, ORTBackend, entry_function
from utils.ppocr.draw import render_annotated_ocr_image
from utils.ppocr.pipeline import BucketTextRecognizer, TextDetector, TextRecognizer, get_rotate_crop, load_char_dict, run_ocr, sort_boxes
from utils.runtime import build_runtime_flags

__all__ = [
    "BucketTextRecognizer", "NPUBackend", "ORTBackend", "TextDetector", "TextRecognizer",
    "build_runtime_flags", "entry_function", "get_rotate_crop",
    "load_char_dict", "render_annotated_ocr_image", "run_ocr", "sort_boxes",
]
