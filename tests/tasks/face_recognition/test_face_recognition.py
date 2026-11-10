# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Host tests for the ``face_recognition`` task category: the box helpers it
re-exports, the centred-face score, the LFW loader, and the ``validate_main``
flow end to end (exit code, printed lines, metrics JSON) with fake dataset
rows, runner and detect hook in place of the download and the model."""

import json
import sys
from unittest import mock

import numpy as np
import pytest

from utils.tasks import face_recognition as face


# ── box math ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("a,b,expected", [
    ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),  # identical
    ((0, 0, 10, 10), (20, 0, 10, 10), 0.0),  # disjoint
    ((0, 0, 10, 10), (5, 0, 10, 10), 1 / 3),  # half overlap: 50 / 150
    ((0, 0, 10, 10), (5, 5, 10, 10), 25 / 175),  # corner overlap
    ((0, 0, 10, 10), (0, 0, 0, 10), 0.0),  # zero-area box
])
def test_iou(a, b, expected):
    assert face.iou(a, b) == pytest.approx(expected)


def test_match_boxes_greedy_one_to_one():
    expected = [{"box": [0, 0, 10, 10]}, {"box": [50, 50, 10, 10]}]
    predicted = [
        {"box": [51, 50, 10, 10]},  # best match for the second face
        {"box": [0, 0, 10, 10]},  # best match for the first face
        {"box": [1, 1, 10, 10]},  # duplicate: stays unmatched
    ]
    assert sorted(face.match_boxes(predicted, expected)) == [(0, 1), (1, 0)]


def test_match_boxes_below_the_iou_threshold_does_not_match():
    assert face.match_boxes([{"box": [5, 5, 10, 10]}], [{"box": [0, 0, 10, 10]}]) == []


@pytest.mark.parametrize("tp,n_predicted,n_expected,expected", [
    (1, 1, 1, 1.0),
    (0, 0, 0, 0.0),  # nothing detected, nothing expected
    (1, 2, 2, 0.5),  # precision 0.5, recall 0.5
    (1, 1, 2, 2 / 3),  # precision 1.0, recall 0.5
    (0, 1, 1, 0.0),  # no matches
])
def test_f1_score(tp, n_predicted, n_expected, expected):
    assert face.f1_score(tp, n_predicted, n_expected) == pytest.approx(expected)


def test_score_item_needs_a_box_over_the_image_centre():
    item = {"expected": {"center": [125.0, 125.0]}}
    hit, counts, _ = face.score_item(item, [("face", 0.9, [100, 90, 50, 60]), ("face", 0.8, [0, 0, 5, 5])])
    assert (hit, counts) == (1.0, (1, 1))
    miss, counts, lines = face.score_item(item, [("face", 0.9, [0.4, 0.2, 9.6, 9.9])])
    assert (miss, counts) == (0.0, (0, 1))
    assert lines == ["no face box contains the image centre [125.0, 125.0]", "predicted: [[0, 0, 10, 10]]"]


def test_aggregate_is_the_hit_fraction():
    assert face.aggregate([(1, 1), (0, 1), (1, 1), (1, 1)]) == pytest.approx(0.75)
    assert face.aggregate([]) == 0.0


# ── validate_main end to end ─────────────────────────────────────────────────

VALIDATION = [{"dataset": "bitmind/lfw", "config": "default", "split": "train", "samples": 10}]


def _rows(tmp_path, n=1):
    import cv2

    rows = []
    for i in range(n):
        path = tmp_path / f"Person_{i}_0001.jpg"
        cv2.imwrite(str(path), np.zeros((200, 250, 3), dtype=np.uint8))  # 250 wide, 200 high
        rows.append({"image": str(path), "filename": path.name})
    return rows


def _add_args(parser):
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--logging", default="INFO")
    parser.add_argument("--metrics-out", default=None)


def _run(module, tmp_path, rows, min_accuracy, predictions, argv=()):
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
        return rows[:samples]

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
                min_accuracy=min_accuracy,
                local_model=lambda args: None,
                ensure_models=lambda model_dir, **kwargs: None,
                add_args=_add_args,
            )
            code = 0
        except SystemExit as e:
            code = e.code
    return code, metrics_out, seen


def test_loader_centre_from_the_image_size(tmp_path):
    code, _, seen = _run(face, tmp_path, _rows(tmp_path), 0.0, [[]])
    assert seen[0]["name"] == "Person_0_0001.jpg"
    assert seen[0]["expected"] == {"center": [125.0, 100.0]}
    assert code == 0


def test_validate_fail_exits_1_and_writes_metrics(tmp_path, capsys):
    code, metrics_out, _ = _run(face, tmp_path, _rows(tmp_path), 1.0, [[("face", 0.9, [0, 0, 50, 50])]])
    out = capsys.readouterr().out
    assert code == 1
    assert "  [FAIL] Person_0_0001.jpg" in out
    assert "no face box contains the image centre [125.0, 100.0]" in out
    assert "accuracy=0.0000 min=1.0" in out
    doc = json.loads(metrics_out.read_text())
    assert doc["measurements"]["latency"] == 123_000.0


def test_validate_pass_exits_0(tmp_path, capsys):
    code, _, _ = _run(face, tmp_path, _rows(tmp_path, 2), 1.0, [[("face", 0.9, [80, 60, 90, 90])]] * 2)
    out = capsys.readouterr().out
    assert code == 0
    assert "  [PASS] Person_1_0001.jpg" in out
    assert "accuracy=1.0000 min=1.0" in out


def test_validate_dummy_min_accuracy_always_fails(tmp_path, capsys):
    code, _, _ = _run(face, tmp_path, _rows(tmp_path), 2.0, [[("face", 0.9, [80, 60, 90, 90])]])
    out = capsys.readouterr().out
    assert code == 1
    assert "accuracy=1.0000 min=2.0" in out
