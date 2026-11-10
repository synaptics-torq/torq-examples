# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""The ``keypoint_detection`` task category (CONTRIBUTING.md).

``infer_main`` and ``validate_main`` (shared with the other image-based
categories, see ``utils.tasks.vision``) are the shared command-line mains of
the demo's ``src/infer.py`` and ``src/validate.py`` scripts: the demo's own
flags, default-model resolution and refresh (``--model`` is the model VMFB
file), one NPU-boosted session, and the demo's per-image loop.
``validate_main`` scores a Hugging Face dataset of images with their labelled
persons (``LOADERS``): a person matches by box IoU at or above 0.5, and accuracy is the
fraction of expected keypoints within 5% of the box size of the predicted
keypoint at the same index (only the dataset's labelled keypoints count).
Both record the CONTRIBUTING.md metrics; validate prints PASS/FAIL and exits
1 on FAIL.
"""

from pathlib import Path

from utils.datasets import fetch_rows
from utils.tasks.vision import (
    infer_main,
    match_boxes,
    round_box,
    validate_main,
)

__all__ = [
    "infer_main",
    "validate_main",
    "match_boxes",
    "score_item",
    "aggregate",
    "LOADERS",
]

#: A keypoint counts as a hit within this fraction of ``max(box w, h)``.
KEYPOINT_TOLERANCE = 0.05


def _person(bbox, keypoints) -> dict:
    return {
        "box": round_box(bbox),
        "keypoints": [
            [i, int(round(float(x))), int(round(float(y)))]
            for i, (x, y, _conf) in enumerate(keypoints)
        ],
    }


def score_item(item: dict, predictions: list) -> tuple[float, tuple[int, int], list[str]]:
    """Score one item against the demo's ``(bbox, conf, keypoints)`` results.

    Returns ``(keypoint fraction, (hits, total), lines)``, where *lines*
    print next to the expected values on FAIL.
    """
    expected = item["expected"]
    persons = [_person(bbox, keypoints) for bbox, _conf, keypoints in predictions]
    total = sum(len(person["keypoints"]) for person in expected)
    hits = 0
    for p_idx, e_idx in match_boxes(persons, expected):
        person = expected[e_idx]
        tolerance = KEYPOINT_TOLERANCE * max(person["box"][2], person["box"][3])
        keypoints = persons[p_idx]["keypoints"]
        for index, x, y in person["keypoints"]:
            if (keypoints[index][1] - x) ** 2 + (keypoints[index][2] - y) ** 2 <= tolerance**2:
                hits += 1
    lines = [
        f"expected:  {expected}",
        f"predicted: {persons}",
    ]
    return (hits / total if total else 0.0), (hits, total), lines


def aggregate(counts: list[tuple[int, int]]) -> float:
    """Accuracy over all items: the summed keypoint hit fraction."""
    hits = sum(count[0] for count in counts)
    total = sum(count[1] for count in counts)
    return hits / total if total else 0.0


def _coco_pose(spec: dict) -> list[dict]:
    """COCO keypoints rows: ``objects.bbox`` per person and, in the same order,
    ``keypoints`` as flat ``[x, y, visibility] * 17`` (visibility 0: unlabelled)."""
    rows = fetch_rows(spec["dataset"], config=spec["config"], split=spec["split"], samples=spec["samples"])
    items = []
    for index, row in enumerate(rows):
        persons = []
        for box, person in zip(row["objects"]["bbox"], row["keypoints"]):
            flat = person["keypoints"]
            persons.append({
                "box": round_box(box),
                "keypoints": [
                    [k, int(round(flat[3 * k])), int(round(flat[3 * k + 1]))]
                    for k in range(len(flat) // 3)
                    if flat[3 * k + 2] > 0
                ],
            })
        items.append({"name": f"{spec['dataset']} #{index}", "image": Path(row["image"]), "expected": persons})
    return items


#: Supported datasets: rows -> ``{"name", "image", "expected": [{"box", "keypoints"}]}`` items.
LOADERS = {"pictograph/coco-2017-pose": _coco_pose}
