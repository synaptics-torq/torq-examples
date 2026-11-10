# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Host tests for the ``object_detection`` task category: the IoU, the
label-aware greedy matching and the F1, the raw-COCO-files loader, and the
``validate_main`` flow end to end (exit code, printed lines, metrics JSON)
with a fake dataset download, runner and detect hook in place of the model."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from utils.tasks import object_detection as od


# ── box math ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("a,b,expected", [
    ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),  # identical
    ((0, 0, 10, 10), (20, 0, 10, 10), 0.0),  # disjoint
    ((0, 0, 10, 10), (5, 0, 10, 10), 1 / 3),  # half overlap: 50 / 150
    ((0, 0, 10, 10), (5, 5, 10, 10), 25 / 175),  # corner overlap
    ((0, 0, 10, 10), (0, 0, 5, 5), 0.25),  # containment: 25 / 100
    ((0, 0, 0, 10), (0, 0, 0, 10), 0.0),  # zero-area boxes
])
def test_iou(a, b, expected):
    assert od.iou(a, b) == pytest.approx(expected)


def test_match_boxes_greedy_one_to_one_with_labels():
    expected = [
        {"label": "dog", "box": [0, 0, 10, 10]},
        {"label": "cat", "box": [50, 50, 10, 10]},
    ]
    predicted = [
        {"label": "cat", "box": [51, 50, 10, 10]},  # best match for the cat
        {"label": "dog", "box": [0, 0, 10, 10]},  # best match for the dog
        {"label": "dog", "box": [1, 1, 10, 10]},  # duplicate: stays unmatched
    ]
    assert sorted(od.match_boxes(predicted, expected, label="label")) == [(0, 1), (1, 0)]


def test_match_boxes_skips_label_mismatch_and_low_iou():
    expected = [{"label": "dog", "box": [0, 0, 10, 10]}]
    predicted = [
        {"label": "cat", "box": [0, 0, 10, 10]},  # same box, wrong label
        {"label": "dog", "box": [90, 0, 10, 10]},  # right label, disjoint
    ]
    assert od.match_boxes(predicted, expected, label="label") == []


def test_match_boxes_accepts_the_iou_threshold():
    # IoU of (0,0,10,10) and (5,5,10,10) is 25/175 ≈ 0.14: below 0.5...
    assert od.match_boxes(
        [{"box": [5, 5, 10, 10]}], [{"box": [0, 0, 10, 10]}],
    ) == []
    # ...and (0,0,10,10) and (3,0,10,10) is 70/130 ≈ 0.54: at or above 0.5.
    assert od.match_boxes([{"box": [3, 0, 10, 10]}], [{"box": [0, 0, 10, 10]}]) == [(0, 0)]


@pytest.mark.parametrize("tp,n_predicted,n_expected,expected", [
    (2, 2, 2, 1.0),
    (0, 0, 0, 0.0),  # nothing detected, nothing expected
    (1, 2, 2, 0.5),  # precision 0.5, recall 0.5
    (1, 1, 2, 2 / 3),  # precision 1.0, recall 0.5
    (0, 3, 1, 0.0),  # no matches
    (1, 0, 1, 0.0),  # no predictions
    (1, 1, 0, 0.0),  # no expectations
])
def test_f1_score(tp, n_predicted, n_expected, expected):
    assert od.f1_score(tp, n_predicted, n_expected) == pytest.approx(expected)


def test_score_item_counts_precision_and_recall():
    item = {"expected": [{"label": "dog", "box": [0, 0, 10, 10]}]}
    predictions = [("dog", 0.9, [0, 0, 10, 10]), ("cat", 0.8, [50, 50, 10, 10])]
    f1, counts, lines = od.score_item(item, predictions)
    assert counts == (1, 2, 1)
    assert f1 == pytest.approx(2 / 3)
    assert "expected:" in lines[0] and "predicted:" in lines[1]


def test_aggregate_uses_totals_not_per_item_f1s():
    # Two items: (tp, n_predicted, n_expected) — the per-item F1s differ
    # from the F1 of the totals.
    counts = [(1, 2, 1), (1, 1, 1)]
    assert od.aggregate(counts) == pytest.approx(od.f1_score(2, 3, 2))


# ── validate_main end to end ─────────────────────────────────────────────────

VALIDATION = [{"dataset": "LibreYOLO/coco-val2017-mini500", "samples": 10}]


def _fake_fetch_file(tmp_path: Path, expected: list):
    """Fake raw COCO files: image ``i`` (id ``i + 1``) holds the ``expected[i]`` objects."""
    names = {}
    annotations = []
    for i, objects in enumerate(expected):
        for obj in objects:
            category = names.setdefault(obj["label"], len(names) + 1)
            annotations.append({"image_id": i + 1, "category_id": category, "bbox": obj["box"], "iscrowd": 0})
    annotations.append({"image_id": 1, "category_id": 99, "bbox": [0, 0, 500, 500], "iscrowd": 1})
    coco = {
        # listed out of id order: the loader sorts by id
        "images": [{"id": i + 1, "file_name": f"img{i}.jpg"} for i in reversed(range(len(expected)))],
        "annotations": annotations,
        "categories": [{"id": c, "name": n} for n, c in names.items()] + [{"id": 99, "name": "crowd"}],
    }
    ann = tmp_path / "instances.json"
    ann.write_text(json.dumps(coco), encoding="utf-8")

    def fetch_file(dataset, filename, *, revision=None):
        assert dataset == "LibreYOLO/coco-val2017-mini500"
        if filename == "annotations/instances_val2017_mini500.json":
            return ann
        assert filename.startswith("images/val2017/")
        return tmp_path / Path(filename).name

    return fetch_file


def _add_args(parser):
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--logging", default="INFO")
    parser.add_argument("--metrics-out", default=None)


def _run_validate_main(tmp_path, expected, min_accuracy, predictions, argv=()):
    metrics_out = tmp_path / "metrics.json"
    queue = [list(item) for item in predictions]
    seen = []

    def detect(runner, item, args, metrics):
        seen.append(item["image"])
        metrics.add("latency", 123_000)
        metrics.add("frames_per_second", 8000.0)
        return queue.pop(0)

    with mock.patch.object(od, "fetch_file", _fake_fetch_file(tmp_path, expected)), mock.patch.object(sys, "argv", [
        "validate.py",
        "--model", str(tmp_path / "model.vmfb"),
        "--no-refresh",
        "--metrics-out", str(metrics_out),
        *argv,
    ]):
        try:
            od.validate_main(
                lambda args, runtime: "fake",
                detect,
                od.score_item,
                od.aggregate,
                demo="demo",
                validation=VALIDATION,
                loaders=od.LOADERS,
                min_accuracy=min_accuracy,
                local_model=lambda args: None,
                ensure_models=lambda model_dir, **kwargs: None,
                add_args=_add_args,
            )
            code = 0
        except SystemExit as e:
            code = e.code
    return code, metrics_out, seen


def test_validate_fail_exits_1_and_writes_metrics(tmp_path, capsys):
    code, metrics_out, seen = _run_validate_main(
        tmp_path, [[{"label": "dog", "box": [0, 0, 10, 10]}]], 1.0, [[("cat", 0.9, [0, 0, 10, 10])]],
    )
    out = capsys.readouterr().out
    assert code == 1
    assert seen == [tmp_path / "img0.jpg"]
    assert "  [FAIL] img0.jpg" in out
    assert "expected:  [{'label': 'dog', 'box': [0, 0, 10, 10]}]" in out  # the crowd region is skipped
    assert "predicted: [{'label': 'cat', 'box': [0, 0, 10, 10]}]" in out
    assert "accuracy=0.0000 min=1.0" in out
    assert out.rstrip().endswith("FAIL")
    doc = json.loads(metrics_out.read_text())
    assert doc["measurements"]["latency"] == 123_000.0
    assert doc["measurements"]["frames_per_second"] == 8000.0
    assert "cpu_load" in doc["measurements"]


def test_validate_pass_exits_0(tmp_path, capsys):
    code, _, _ = _run_validate_main(
        tmp_path, [[{"label": "dog", "box": [0.4, 0.2, 9.6, 9.9]}]], 1.0, [[("dog", 0.9, [0.4, 0.2, 9.6, 9.9])]],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "  [PASS] img0.jpg" in out
    assert "accuracy=1.0000 min=1.0" in out
    assert out.rstrip().endswith("PASS")


def test_validate_scores_the_f1_of_the_totals(tmp_path, capsys):
    # Item 0: one match plus one extra detection (per-item F1 2/3);
    # item 1: one match (per-item F1 1.0). Totals: F1 of tp=2, n=3, e=2.
    code, _, _ = _run_validate_main(
        tmp_path,
        [[{"label": "dog", "box": [0, 0, 10, 10]}], [{"label": "cat", "box": [50, 50, 10, 10]}]],
        0.8,
        [
            [("dog", 0.9, [0, 0, 10, 10]), ("bird", 0.8, [90, 90, 5, 5])],
            [("cat", 0.9, [50, 50, 10, 10])],
        ],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "  [FAIL] img0.jpg" in out
    assert "  [PASS] img1.jpg" in out
    assert "accuracy=0.8000 min=0.8" in out


def test_validate_samples_takes_the_lowest_image_ids(tmp_path):
    code, _, seen = _run_validate_main(
        tmp_path, [[], []], 1.0,
        [[("dog", 0.9, [0, 0, 10, 10])], [("unused", 0.9, [0, 0, 10, 10])]],
        argv=["--samples", "1"],
    )
    assert seen == [tmp_path / "img0.jpg"]
    assert code == 1  # item 0 has no expected objects, so nothing matches
