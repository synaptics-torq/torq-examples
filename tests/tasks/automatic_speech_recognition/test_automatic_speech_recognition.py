# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Host tests for the ``automatic_speech_recognition`` task category: the word error
rate, the LibriSpeech loader, and the ``validate_main`` flow end to end
(exit code, printed WER, metrics JSON) with fake dataset rows, runner and
transcribe hook in place of the download and the model."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from utils.tasks import automatic_speech_recognition as sr


# ── word error rate ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("hypothesis,reference,expected", [
    ("the cat sat on the mat", "the cat sat on the mat", 0.0),
    ("The CAT sat on the mat", "the cat sat on the mat", 0.0),  # case-folded
    ("Hello, world!", "hello world", 0.0),  # punctuation stripped
    ("the cat sat on the rug", "the cat sat on the mat", 1 / 6),  # a substitution
    ("the cat sat on the", "the cat sat on the mat", 1 / 6),  # a deletion
    ("the cat sat on the mat pad", "the cat sat on the mat", 1 / 6),  # an insertion
    ("", "a b", 1.0),  # an empty hypothesis: all insertions
    ("a b", "", 2.0),  # an empty reference counts as one word
    ("b a", "a b", 1.0),  # a transposition costs two word edits
])
def test_word_error_rate(hypothesis, reference, expected):
    assert sr.word_error_rate(hypothesis, reference) == pytest.approx(expected)


# ── validate_main end to end ─────────────────────────────────────────────────

VALIDATION = [{"dataset": "openslr/librispeech_asr", "config": "clean", "split": "test", "samples": 10}]


def _fake_fetch_rows(tmp_path: Path, references: list[str]):
    """Fake viewer rows: one LibriSpeech utterance per reference transcript."""
    rows = [{"id": f"utt-{i}", "audio": str(tmp_path / f"{i}.wav"), "text": ref} for i, ref in enumerate(references)]

    def fetch_rows(dataset, *, config, split, samples, offset=0):
        assert (dataset, config, split) == ("openslr/librispeech_asr", "clean", "test")
        return rows[:samples]

    return fetch_rows


def _add_args(parser):
    parser.add_argument("-m", "--model-dir", default=None)
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--logging", default="INFO")
    parser.add_argument("--metrics-out", default=None)


def _run_validate_main(tmp_path, references, transcripts, argv=()):
    metrics_out = tmp_path / "metrics.json"
    answers = list(transcripts)
    made = {}

    def make_runner(args, runtime):
        made["model_dir"] = args.model_dir
        return "fake"

    def transcribe(runner, item, args, metrics):
        assert item["audio"] == tmp_path / f"{item['name'][4:]}.wav"
        metrics.add("latency", 123_000)
        metrics.add("real_time_factor", 50.0)
        return answers.pop(0)

    with mock.patch.object(sr, "fetch_rows", _fake_fetch_rows(tmp_path, references)), mock.patch.object(sys, "argv", [
        "validate.py",
        "-m", str(tmp_path / "model"),
        "--no-refresh",
        "--metrics-out", str(metrics_out),
        *argv,
    ]):
        try:
            sr.validate_main(
                make_runner,
                transcribe,
                demo="demo",
                validation=VALIDATION,
                local_model=lambda: None,
                ensure_models=lambda model_dir, **kwargs: None,
                add_args=_add_args,
            )
            code = 0
        except SystemExit as e:
            code = e.code
    return code, metrics_out, made, answers


def test_validate_perfect_transcript_records_zero_wer(tmp_path, capsys):
    ref = "NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER"
    code, metrics_out, made, _ = _run_validate_main(tmp_path, [ref], [ref.lower()])
    out = capsys.readouterr().out
    assert code == 0  # no pass mark: the WER is reported, never a verdict
    assert made["model_dir"] == str(tmp_path / "model")
    assert "  utt-0 (wer=0.0000)" in out
    assert "reference:" not in out  # transcripts only print for items with errors
    assert out.rstrip().endswith("wer=0.0000")
    doc = json.loads(metrics_out.read_text())
    assert doc["measurements"]["word_error_rate"] == 0.0
    # what the score was measured on, for comparing runs
    assert doc["validation"] == [{"dataset": "openslr/librispeech_asr", "config": "clean", "split": "test", "samples": 10, "items": 1}]
    assert doc["measurements"]["latency"] == 123_000.0
    assert doc["measurements"]["real_time_factor"] == 50.0
    assert "cpu_load" in doc["measurements"]


def test_validate_wer_is_the_mean_over_items(tmp_path, capsys):
    # One perfect transcript (wer 0) and one with a substitution and an
    # insertion (wer 1/3): the mean is 1/6.
    code, metrics_out, _, _ = _run_validate_main(
        tmp_path,
        ["the cat sat on the mat", "the cat sat on the mat"],
        ["the cat sat on the mat", "the cat sat on the rug pad"],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "  utt-0 (wer=0.0000)" in out
    assert "  utt-1 (wer=0.3333)" in out
    assert "reference:  'the cat sat on the mat'" in out
    assert out.rstrip().endswith("wer=0.1667")
    assert json.loads(metrics_out.read_text())["measurements"]["word_error_rate"] == pytest.approx(100 / 6)


def test_validate_unknown_dataset_exits(tmp_path):
    with pytest.raises(SystemExit, match="No loader for dataset 'someone/else'"):
        with mock.patch.object(sys, "argv", ["validate.py", "-m", str(tmp_path), "--no-refresh", "--dataset", "someone/else"]):
            sr.validate_main(
                lambda args, runtime: "fake",
                lambda runner, item, args, metrics: "",
                demo="demo",
                validation=VALIDATION,
                local_model=lambda: None,
                ensure_models=lambda model_dir, **kwargs: None,
                add_args=_add_args,
            )
