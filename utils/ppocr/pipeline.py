# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""PP-OCRv6-tiny pipeline: DBNet text detection + CTC text recognition.

Pre/post-processing (DB decode, box warp, CTC decode) is ported from PaddleOCR so results match
the reference. Each stage takes a backend, so detection and recognition can run on the NPU or on
ONNX Runtime independently.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pyclipper
import yaml
from shapely.geometry import Polygon

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


class TextDetector:
    """DBNet detector: image -> list of 4-point text-box quads in image coords.

    ``static_hw``/``static_size`` pin the input to the shape a Torq vmfb was compiled for; leave
    both unset for a dynamic (ONNX Runtime) backend, which then follows DetResizeForTest.
    """

    def __init__(self, backend, *, static_size=None, static_hw=None, limit_side_len=960, limit_type="max", thresh=0.2, box_thresh=0.4, unclip_ratio=1.4, max_candidates=1000, min_size=3):
        self.backend = backend
        self.static_size, self.static_hw = static_size, static_hw
        self.limit_side_len, self.limit_type = limit_side_len, limit_type
        self.thresh, self.box_thresh = thresh, box_thresh
        self.unclip_ratio = unclip_ratio
        self.max_candidates, self.min_size = max_candidates, min_size

    def _resize(self, img):
        if self.static_hw:
            height, width = self.static_hw
            return cv2.resize(img, (width, height))
        if self.static_size:
            return cv2.resize(img, (self.static_size, self.static_size))
        h, w = img.shape[:2]
        longest, shortest = max(h, w), min(h, w)
        if self.limit_type == "max":
            ratio = float(self.limit_side_len) / longest if longest > self.limit_side_len else 1.0
        else:
            ratio = float(self.limit_side_len) / shortest if shortest < self.limit_side_len else 1.0
        resize_h = max(int(round(h * ratio / 32) * 32), 32)
        resize_w = max(int(round(w * ratio / 32) * 32), 32)
        return cv2.resize(img, (resize_w, resize_h))

    def _preprocess(self, img):
        x = (self._resize(img).astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)

    @staticmethod
    def _mini_box(contour):
        rect = cv2.minAreaRect(contour)
        pts = sorted(cv2.boxPoints(rect).tolist(), key=lambda p: p[0])
        i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
        i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
        return np.array([pts[i1], pts[i2], pts[i3], pts[i4]], np.float32), min(rect[1])

    @staticmethod
    def _box_score(prob, box):
        h, w = prob.shape[:2]
        b = box.copy()
        xmin = np.clip(int(np.floor(b[:, 0].min())), 0, w - 1)
        xmax = np.clip(int(np.ceil(b[:, 0].max())), 0, w - 1)
        ymin = np.clip(int(np.floor(b[:, 1].min())), 0, h - 1)
        ymax = np.clip(int(np.ceil(b[:, 1].max())), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
        b[:, 0] -= xmin
        b[:, 1] -= ymin
        cv2.fillPoly(mask, [b.astype(np.int32)], 1)
        return cv2.mean(prob[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def _unclip(self, box):
        poly = Polygon(box)
        if poly.length == 0:
            return None
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(poly.area * self.unclip_ratio / poly.length)
        if not expanded:
            return None
        return np.array(max(expanded, key=len), np.float32).reshape(-1, 1, 2)

    def _boxes_from_bitmap(self, prob, bitmap, src_w, src_h):
        h, w = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours[: self.max_candidates]:
            points, side = self._mini_box(contour)
            if side < self.min_size or self._box_score(prob, points) < self.box_thresh:
                continue
            expanded = self._unclip(points)
            if expanded is None or len(expanded) < 4:
                continue
            box, side = self._mini_box(expanded)
            if side < self.min_size + 2:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] / w * src_w), 0, src_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / h * src_h), 0, src_h)
            boxes.append(box.astype(np.int32))
        return boxes

    def __call__(self, img):
        src_h, src_w = img.shape[:2]
        prob = self.backend.run(self._preprocess(img))[0, 0]
        return sort_boxes(self._boxes_from_bitmap(prob, prob > self.thresh, src_w, src_h))


def sort_boxes(boxes):
    """Order boxes top-to-bottom, then left-to-right within a line."""
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: (b[0][1], b[0][0]))
    for i in range(len(ordered) - 1):
        for j in range(i, -1, -1):
            same_line = abs(ordered[j + 1][0][1] - ordered[j][0][1]) < 10
            if not (same_line and ordered[j + 1][0][0] < ordered[j][0][0]):
                break
            ordered[j], ordered[j + 1] = ordered[j + 1], ordered[j]
    return ordered


def get_rotate_crop(img, box):
    """Perspective-warp a quad out of the image into an upright text-line crop."""
    box = box.astype(np.float32)
    width = max(int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3]))), 1)
    height = max(int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2]))), 1)
    transform = cv2.getPerspectiveTransform(box, np.float32([[0, 0], [width, 0], [width, height], [0, height]]))
    crop = cv2.warpPerspective(img, transform, (width, height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    return np.rot90(crop) if crop.shape[0] * 1.0 / crop.shape[1] >= 1.5 else crop  # tall crop: rotated line


def load_char_dict(rec_yml):
    """Read the recognizer character dictionary from a PaddleOCR inference.yml."""
    with open(rec_yml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["PostProcess"]["character_dict"]


def _build_charset(char_dict, num_classes):
    """CTC vocabulary = [blank] + dict (+ space when the model has one more class)."""
    character = ["blank"] + list(char_dict)
    if not isinstance(num_classes, int) or num_classes == len(character) + 1:
        character.append(" ")
    return character


def _resize_norm(img, img_h, width):
    """Resize a crop to ``img_h`` keeping aspect, then right-pad to ``width``."""
    h, w = img.shape[:2]
    resized_w = min(int(math.ceil(img_h * (w / float(h)))), width)
    resized = cv2.resize(img, (resized_w, img_h)).astype(np.float32).transpose(2, 0, 1) / 255.0
    padded = np.zeros((3, img_h, width), np.float32)
    padded[:, :, :resized_w] = (resized - 0.5) / 0.5
    return padded


def _ctc_decode(preds, character):
    """Greedy CTC decode of ``[T, num_classes]`` into (text, mean confidence)."""
    idx, prob = preds.argmax(1), preds.max(1)
    keep = np.ones(len(idx), bool)
    keep[1:] = idx[1:] != idx[:-1]  # collapse repeats
    keep &= idx != 0                # drop blanks
    conf = prob[keep]
    return "".join(character[i] for i in idx[keep]), float(conf.mean()) if conf.size else 0.0


class TextRecognizer:
    """CTC recognizer against a single backend (one static width, or dynamic)."""

    def __init__(self, char_dict, backend, *, img_h=48, static_width=None, batch_size=6):
        self.backend = backend
        self.img_h, self.batch_size = img_h, batch_size
        self.static_width = static_width
        self.base_ratio = (static_width or 320) / img_h
        self.character = _build_charset(char_dict, getattr(backend, "num_classes", None))

    def __call__(self, crops):
        n = len(crops)
        # Group crops of similar aspect so each batch pads to a similar width.
        order = sorted(range(n), key=lambda i: crops[i].shape[1] / float(crops[i].shape[0]))
        results = [("", 0.0)] * n
        for start in range(0, n, self.batch_size):
            batch_idx = order[start:start + self.batch_size]
            max_ratio = max([self.base_ratio] + [crops[i].shape[1] / float(crops[i].shape[0]) for i in batch_idx])
            width = self.static_width or int(self.img_h * max_ratio)
            preds = self.backend.run(np.stack([_resize_norm(crops[i], self.img_h, width) for i in batch_idx]))
            for k, i in enumerate(batch_idx):
                results[i] = _ctc_decode(preds[k], self.character)
        return results


class BucketTextRecognizer:
    """Recognizer routing each crop to the narrowest static-width vmfb that fits.

    Mirrors PaddleOCR's dynamic per-batch width using a fixed set of static widths (one vmfb
    each). ``backends`` maps width -> backend.
    """

    def __init__(self, char_dict, backends, *, img_h=48):
        self.img_h = img_h
        self.backends = backends
        self.buckets = sorted(backends)
        self.character = _build_charset(char_dict, getattr(backends[self.buckets[0]], "num_classes", None))

    def _bucket_for(self, crop):
        h, w = crop.shape[:2]
        needed = self.img_h * (w / float(h))
        return next((width for width in self.buckets if needed <= width), self.buckets[-1])

    def __call__(self, crops):
        results = [("", 0.0)] * len(crops)
        by_bucket: dict[int, list[int]] = {}
        for i, crop in enumerate(crops):
            by_bucket.setdefault(self._bucket_for(crop), []).append(i)
        for width, idxs in by_bucket.items():
            preds = self.backends[width].run(np.stack([_resize_norm(crops[i], self.img_h, width) for i in idxs]))
            for k, i in enumerate(idxs):
                results[i] = _ctc_decode(preds[k], self.character)
        return results


def run_ocr(img, detector, recognizer, drop_score=0.5):
    """Detect then recognize; returns ``(kept, stats)`` where kept is ``(box, text, score)``."""
    t0 = time.time()
    boxes = detector(img)
    t1 = time.time()
    crops = [get_rotate_crop(img, box) for box in boxes]
    results = recognizer(crops) if crops else []
    t2 = time.time()
    kept = [(box, text, score) for box, (text, score) in zip(boxes, results) if score >= drop_score and text]
    return kept, {"det_ms": (t1 - t0) * 1e3, "rec_ms": (t2 - t1) * 1e3, "n_boxes": len(boxes)}
