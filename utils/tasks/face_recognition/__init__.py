# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""The ``face_recognition`` task category (CONTRIBUTING.md).

``infer_main`` and ``validate_main`` (shared with the other image-based
categories, see ``utils.tasks.vision``) are the shared command-line mains of
the demo's ``src/infer.py`` and ``src/validate.py`` scripts: the demo's own
flags, default-model resolution and refresh (``--model`` is the model VMFB
file), one NPU-boosted session, and the demo's per-image loop.
``validate_main`` scores a Hugging Face dataset of face-centred photos
(``LOADERS``): an image passes when a detected face box contains the image
centre, and accuracy is the fraction of passing images. Both record the
CONTRIBUTING.md metrics; validate prints PASS/FAIL and exits 1 on FAIL.
"""

from pathlib import Path

from utils.datasets import fetch_rows
from utils.tasks.vision import infer_main, round_box, validate_main

__all__ = [
    "infer_main",
    "validate_main",
    "score_item",
    "aggregate",
    "LOADERS",
]


def score_item(item: dict, predictions: list) -> tuple[float, tuple[int, int], list[str]]:
    """Score one item against the demo's ``(label, confidence, box)`` results.

    Returns ``(1.0 if a box contains the image centre else 0.0, (hit, 1), lines)``,
    where *lines* print on FAIL.
    """
    cx, cy = item["expected"]["center"]
    hit = any(x <= cx <= x + w and y <= cy <= y + h for _label, _conf, (x, y, w, h) in predictions)
    lines = [
        f"no face box contains the image centre {[cx, cy]}",
        f"predicted: {[round_box(box) for _label, _conf, box in predictions]}",
    ]
    return float(hit), (int(hit), 1), lines


def aggregate(counts: list[tuple[int, int]]) -> float:
    """Accuracy over all items: the fraction of images with a centred face found."""
    return sum(count[0] for count in counts) / max(1, sum(count[1] for count in counts))


def _centred_faces(spec: dict) -> list[dict]:
    """Rows of face-centred photos (LFW crops: one person's face in the middle)."""
    import cv2

    rows = fetch_rows(spec["dataset"], config=spec["config"], split=spec["split"], samples=spec["samples"])
    items = []
    for row in rows:
        height, width = cv2.imread(row["image"]).shape[:2]
        items.append({"name": row.get("filename", row["image"]), "image": Path(row["image"]),
                      "expected": {"center": [width / 2, height / 2]}})
    return items


#: Supported datasets: rows -> ``{"name", "image", "expected": {"center"}}`` items.
LOADERS = {"bitmind/lfw": _centred_faces}
