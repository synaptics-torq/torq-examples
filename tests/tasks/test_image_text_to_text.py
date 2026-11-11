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

from utils.inference import ContextOverflow
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
        return rows[offset:offset + samples]

    return fetch_rows


def _add_args(parser):
    parser.add_argument("-m", "--model-dir", default=None)
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--logging", default="INFO")
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)


def _run_validate_main(tmp_path, expected, answers, argv=(), too_long=()):
    metrics_out = tmp_path / "metrics.json"
    queue = list(answers)
    made = {}
    asked = []

    def make_runner(args, runtime):
        made.update(model_dir=args.model_dir, temperature=args.temperature)
        return "fake"

    def ask(runner, item, args, metrics):
        if item["name"] in too_long:
            raise ContextOverflow("question too long for the KV cache (256)")
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
                local_model=lambda: None,
                ensure_models=lambda model_dir, **kwargs: None,
                add_args=_add_args,
            )
            code = 0
        except SystemExit as e:
            code = e.code
    return code, metrics_out, made, asked


def test_validate_wrong_answer_scores_zero_and_writes_metrics(tmp_path, capsys):
    code, metrics_out, made, asked = _run_validate_main(tmp_path, ["A"], ["B"])
    out = capsys.readouterr().out
    assert code == 0  # no pass mark: the accuracy is reported, never a verdict
    assert made == {"model_dir": str(tmp_path / "model"), "temperature": 0.0}  # greedy forced
    assert asked[0][0] == tmp_path / "0.jpg"
    assert asked[0][1].endswith("Answer with the option's letter from the given choices directly.")
    assert "  [FAIL] Lin-Chen/MMStar #0" in out
    assert "answer: 'B' (expected 'A')" in out
    assert out.rstrip().endswith("accuracy=0.0000 (0/1 items, 0 skipped)")
    doc = json.loads(metrics_out.read_text())
    assert doc["measurements"]["accuracy"] == 0.0
    assert doc["measurements"]["time_to_first_token"] == 456_000_000.0
    assert doc["measurements"]["tokens_per_second"] == 7.5


def test_validate_right_answer_counts(tmp_path, capsys):
    code, metrics_out, _, _ = _run_validate_main(tmp_path, ["C", "A"], ["C: car", "B"])
    out = capsys.readouterr().out
    assert code == 0
    assert "  [PASS] Lin-Chen/MMStar #0" in out
    assert out.rstrip().endswith("accuracy=0.5000 (1/2 items, 0 skipped)")
    assert json.loads(metrics_out.read_text())["measurements"]["accuracy"] == 50.0


def test_validate_skips_a_question_that_does_not_fit(tmp_path, capsys):
    code, metrics_out, _, asked = _run_validate_main(
        tmp_path, ["C", "A"], ["A"], too_long=("Lin-Chen/MMStar #0",),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "  [SKIP] Lin-Chen/MMStar #0 (question too long for the KV cache (256))" in out
    assert len(asked) == 1
    assert out.rstrip().endswith("accuracy=1.0000 (1/1 items, 1 skipped)")  # no row left to replace it
    assert json.loads(metrics_out.read_text())["validation"][0]["rows"] == [1]


def test_validate_tries_at_most_twice_samples_rows(tmp_path, capsys):
    names = tuple(f"Lin-Chen/MMStar #{i}" for i in range(30))
    code, _, _, asked = _run_validate_main(tmp_path, ["A"] * 30, [], argv=["--samples", "5"], too_long=names)
    out = capsys.readouterr().out
    assert out.count("[SKIP]") == 10
    assert asked == []
    assert code == "No validation item fits the model's context"
