# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Host tests for the ``keypoint_detection`` task category: the person
matching and keypoint accuracy, and the COCO keypoints loader run through
``validate_main`` with fake dataset rows, runner and detect hook in place of
the download and the model (the shared box math and validate flow are tested
with ``object_detection``)."""

import json
import sys
from unittest import mock

import numpy as np
import pytest

from utils.tasks import keypoint_detection as pose


# ── box math ─────────────────────────────────────────────────────────────────


def test_match_boxes_one_to_one():
    # Two expected persons on the same spot, one predicted person:
    # the predicted person can only match one of them.
    expected = [{"box": [0, 0, 10, 10]}, {"box": [0, 0, 10, 10]}]
    predicted = [{"box": [0, 0, 10, 10]}]
    assert len(pose.match_boxes(predicted, expected)) == 1


# ── keypoint scoring ─────────────────────────────────────────────────────────


def _kpts(positions=None):
    """A (17, 3) keypoint array with *positions* {index: (x, y)} at conf 0.9."""
    kpts = np.zeros((17, 3))
    for index, (x, y) in (positions or {}).items():
        kpts[index] = (x, y, 0.9)
    return kpts


def _person(x, y, w=10.0, h=20.0):
    return (np.array([x, y, w, h]), 0.9, _kpts())


def test_score_item_keypoint_accuracy():
    # Tolerance is 0.05 * max(100, 200) = 10 pixels.
    item = {"expected": [{"box": [0, 0, 100, 200], "keypoints": [[0, 50, 100], [1, 48, 102]]}]}
    kpts = _kpts({0: (50, 100), 1: (60, 100)})  # 0: on the nose; 1: ~12 px away
    predictions = [(np.array([0.0, 0.0, 100.0, 200.0]), 0.9, kpts)]
    score, (hits, total), lines = pose.score_item(item, predictions)
    assert (hits, total) == (1, 2)
    assert score == pytest.approx(0.5)
    assert "expected:" in lines[0] and "predicted:" in lines[1]


def test_score_item_keypoint_hit_at_tolerance_boundary():
    item = {"expected": [{"box": [0, 0, 100, 200], "keypoints": [[0, 50, 100]]}]}
    kpts = _kpts({0: (60, 100)})  # exactly 10 px away: a hit
    predictions = [(np.array([0.0, 0.0, 100.0, 200.0]), 0.9, kpts)]
    score, (hits, total), _ = pose.score_item(item, predictions)
    assert (hits, total) == (1, 1)
    assert score == 1.0


def test_score_item_unmatched_person_counts_all_keypoints_as_misses():
    item = {"expected": [{"box": [500, 500, 10, 10], "keypoints": [[0, 505, 505], [1, 506, 507]]}]}
    predictions = [_person(0, 0)]
    score, (hits, total), _ = pose.score_item(item, predictions)
    assert (hits, total) == (0, 2)
    assert score == 0.0


def test_aggregate_sums_keypoint_hits_over_items():
    assert pose.aggregate([(1, 2), (2, 2)]) == pytest.approx(0.75)
    assert pose.aggregate([(0, 0)]) == 0.0


# ── validate_main end to end ─────────────────────────────────────────────────

VALIDATION = [{"dataset": "pictograph/coco-2017-pose", "config": "default", "split": "validation", "samples": 10}]


def _row(image, people):
    """A COCO keypoints row: *people* is ``[(box, {index: (x, y)})]``; other keypoints unlabelled."""
    keypoints = []
    for _box, labelled in people:
        flat = [0.0] * 51
        for k, (x, y) in labelled.items():
            flat[3 * k: 3 * k + 3] = [x, y, 2.0]
        keypoints.append({"category": "person", "keypoints": flat})
    return {"image": image, "objects": {"bbox": [box for box, _ in people]}, "keypoints": keypoints}


def _add_args(parser):
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--logging", default="INFO")
    parser.add_argument("--metrics-out", default=None)


def _run(module, tmp_path, rows, predictions, argv=()):
    metrics_out = tmp_path / "metrics.json"
    queue = [list(item) for item in predictions]
    seen = []

    def detect(runner, item, args, metrics):
        seen.append(item)
        metrics.add("latency", 123_000)
        metrics.add("frames_per_second", 8000.0)
        return queue.pop(0)

    def fetch_rows(dataset, *, config, split, samples, offset=0):
        assert (dataset, config, split) == (VALIDATION[0]["dataset"], VALIDATION[0]["config"], VALIDATION[0]["split"])
        return rows[offset:offset + samples]

    with mock.patch.object(module, "fetch_rows", fetch_rows), mock.patch.object(sys, "argv", [
        "validate.py",
        "--model", str(tmp_path / "model.vmfb"),
        "--no-refresh",
        "--metrics-out", str(metrics_out),
        *argv,
    ]):
        try:
            module.validate_main(
                lambda args, runtime: "fake",
                detect,
                module.score_item,
                module.aggregate,
                demo="demo",
                validation=VALIDATION,
                loaders=module.LOADERS,
                local_model=lambda args: None,
                ensure_models=lambda model_dir, **kwargs: None,
                add_args=_add_args,
            )
            code = 0
        except SystemExit as e:
            code = e.code
    return code, metrics_out, seen


def test_loader_keeps_only_labelled_keypoints(tmp_path):
    rows = [_row(str(tmp_path / "a.jpg"), [([0.4, 0.0, 99.6, 200.0], {0: (50.2, 100.0), 5: (10.0, 20.0)})])]
    code, metrics_out, seen = _run(pose, tmp_path, rows, [[]])
    assert seen[0]["image"] == tmp_path / "a.jpg"
    assert seen[0]["name"] == "pictograph/coco-2017-pose #0"
    assert seen[0]["expected"] == [{"box": [0, 0, 100, 200], "keypoints": [[0, 50, 100], [5, 10, 20]]}]
    assert code == 0
    assert json.loads(metrics_out.read_text())["measurements"]["accuracy"] == 0.0  # nothing detected


