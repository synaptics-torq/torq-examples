# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Host tests for the ``face_recognition`` task category: the centred-face
score and the LFW loader, run through ``validate_main`` with fake dataset rows,
runner and detect hook in place of the download and the model (the shared
box math and validate flow are tested with ``object_detection``)."""

import sys
import types
from unittest import mock

import numpy as np
import pytest

from utils.tasks import face_recognition as face


# ── box math ─────────────────────────────────────────────────────────────────


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


@pytest.fixture(autouse=True)
def _fake_cv2(monkeypatch):
    """The loader reads image sizes with OpenCV (a demo dependency, not a test
    one): every photo is 250 wide and 200 high."""
    fake = types.ModuleType("cv2")
    fake.imread = lambda path: np.zeros((200, 250, 3), dtype=np.uint8)
    monkeypatch.setitem(sys.modules, "cv2", fake)


def _rows(tmp_path, n=1):
    return [{"image": str(tmp_path / f"Person_{i}_0001.jpg"), "filename": f"Person_{i}_0001.jpg"} for i in range(n)]


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


