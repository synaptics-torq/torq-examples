# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""The ``object_detection`` task category (CONTRIBUTING.md).

``infer_main`` and ``validate_main`` (shared with the other image-based
categories, see ``utils.tasks.vision``) are the shared command-line mains of
the demo's ``src/infer.py`` and ``src/validate.py`` scripts: the demo's own
flags, default-model resolution and refresh (``--model`` is the model VMFB
file), one NPU-boosted session, and the demo's per-image loop.
``validate_main`` scores a Hugging Face dataset of images with their labelled
objects (``LOADERS``): greedy one-to-one matching of detections to expected objects
with the same label and IoU at or above 0.5, precision and recall over all
items, and accuracy as their F1. Both record the CONTRIBUTING.md metrics;
validate also records its accuracy.
"""

import json
from collections import defaultdict

from utils.datasets import fetch_file
from utils.tasks.vision import (
    f1_score,
    infer_main,
    iou,
    match_boxes,
    round_box,
    validate_main,
)

__all__ = [
    "infer_main",
    "validate_main",
    "iou",
    "match_boxes",
    "f1_score",
    "score_item",
    "aggregate",
    "LOADERS",
]


def score_item(item: dict, predictions: list) -> tuple[float, tuple[int, int, int], list[str]]:
    """Score one item against the demo's ``(label, confidence, box)`` results.

    Returns ``(item F1, (tp, n_predicted, n_expected), lines)``, where
    *lines* print next to the expected values on FAIL.
    """
    expected = item["expected"]
    predicted = [{"label": label, "box": round_box(box)} for label, _conf, box in predictions]
    tp = len(match_boxes(predicted, expected, label="label"))
    f1 = f1_score(tp, len(predicted), len(expected))
    lines = [
        f"expected:  {expected}",
        f"predicted: {predicted}",
    ]
    return f1, (tp, len(predicted), len(expected)), lines


def aggregate(counts: list[tuple[int, int, int]]) -> float:
    """Accuracy over all items: the F1 of the summed precision and recall."""
    tp = sum(count[0] for count in counts)
    n_predicted = sum(count[1] for count in counts)
    n_expected = sum(count[2] for count in counts)
    return f1_score(tp, n_predicted, n_expected)


def _coco_files(spec: dict) -> list[dict]:
    """A dataset repo of raw COCO files: ``annotations/instances_val2017_mini500.json``
    and ``images/val2017/<file>``; the lowest image ids first, crowd regions skipped."""
    ann = json.loads(fetch_file(spec["dataset"], "annotations/instances_val2017_mini500.json").read_text(encoding="utf-8"))
    names = {category["id"]: category["name"] for category in ann["categories"]}
    expected = defaultdict(list)
    for a in ann["annotations"]:
        if not a.get("iscrowd"):
            expected[a["image_id"]].append({"label": names[a["category_id"]], "box": round_box(a["bbox"])})
    images = sorted(ann["images"], key=lambda image: image["id"])[spec["offset"]: spec["offset"] + spec["samples"]]
    return [
        {
            "name": image["file_name"],
            "image": fetch_file(spec["dataset"], f"images/val2017/{image['file_name']}"),
            "expected": expected[image["id"]],
        }
        for image in images
    ]


#: Supported datasets: -> ``{"name", "image", "expected": [{"label", "box"}]}`` items.
LOADERS = {"LibreYOLO/coco-val2017-mini500": _coco_files}
