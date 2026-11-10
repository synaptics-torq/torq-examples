# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Host tests for the ``image_text_to_text`` task category: the MMStar loader
and the ``validate_main`` flow end to end (exit code, printed accuracy,
metrics JSON, greedy decoding forced) with fake dataset rows, runner and ask
hook in place of the download and the model."""

import json
import sys
from pathlib import Path
from unittest import mock

from utils.tasks import image_text_to_text as vqa

VALIDATION = [{"dataset": "Lin-Chen/MMStar", "config": "val", "split": "val", "samples": 10}]


def _fake_fetch_rows(tmp_path: Path, answers: list[str]):
    rows = [
        {"index": i, "image": str(tmp_path / f"{i}.jpg"),
         "question": f"Question {i}?\nOptions: A: cat, B: dog, C: car, D: tree", "answer": answer}
        for i, answer in enumerate(answers)
    ]

    def fetch_rows(dataset, *, config, split, samples, offset=0):
        assert (dataset, config, split) == ("Lin-Chen/MMStar", "val", "val")
        return rows[:samples]

    return fetch_rows


def _add_args(parser):
    parser.add_argument("-m", "--model-dir", default=None)
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--logging", default="INFO")
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)


def _run_validate_main(tmp_path, expected, answers, min_accuracy, argv=()):
    metrics_out = tmp_path / "metrics.json"
    queue = list(answers)
    made = {}
    asked = []

    def make_runner(args, runtime):
        made.update(model_dir=args.model_dir, temperature=args.temperature)
        return "fake"

    def ask(runner, item, args, metrics):
        asked.append((item["image"], item["question"]))
        metrics.add("time_to_first_token", 456_000_000)
        metrics.add("tokens_per_second", 7.5)
        return queue.pop(0)

    with mock.patch.object(vqa, "fetch_rows", _fake_fetch_rows(tmp_path, expected)), mock.patch.object(sys, "argv", [
        "validate.py",
        "-m", str(tmp_path / "model"),
        "--no-refresh",
        "--metrics-out", str(metrics_out),
        *argv,
    ]):
        try:
            vqa.validate_main(
                make_runner,
                ask,
                demo="demo",
                validation=VALIDATION,
                min_accuracy=min_accuracy,
                local_model=lambda: None,
                ensure_models=lambda model_dir, **kwargs: None,
                add_args=_add_args,
            )
            code = 0
        except SystemExit as e:
            code = e.code
    return code, metrics_out, made, asked


def test_validate_fail_exits_1_and_writes_metrics(tmp_path, capsys):
    code, metrics_out, made, asked = _run_validate_main(tmp_path, ["A"], ["B"], 1.0)
    out = capsys.readouterr().out
    assert code == 1
    assert made == {"model_dir": str(tmp_path / "model"), "temperature": 0.0}  # greedy forced
    assert asked[0][0] == tmp_path / "0.jpg"
    assert asked[0][1].endswith("Answer with the option's letter from the given choices directly.")
    assert "  [FAIL] Lin-Chen/MMStar #0" in out
    assert "answer: 'B' (expected 'A')" in out
    assert "accuracy=0.0000 min=1.0" in out
    assert out.rstrip().endswith("FAIL")
    doc = json.loads(metrics_out.read_text())
    assert doc["measurements"]["time_to_first_token"] == 456_000_000.0
    assert doc["measurements"]["tokens_per_second"] == 7.5


def test_validate_pass_exits_0(tmp_path, capsys):
    code, _, _, _ = _run_validate_main(tmp_path, ["C"], ["C: car"], 1.0)
    out = capsys.readouterr().out
    assert code == 0
    assert "  [PASS] Lin-Chen/MMStar #0" in out
    assert out.rstrip().endswith("PASS")


