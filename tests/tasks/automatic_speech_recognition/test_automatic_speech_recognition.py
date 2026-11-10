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


def _run_validate_main(tmp_path, references, transcripts, max_wer, argv=()):
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
                max_wer=max_wer,
                local_model=lambda: None,
                ensure_models=lambda model_dir, **kwargs: None,
                add_args=_add_args,
            )
            code = 0
        except SystemExit as e:
            code = e.code
    return code, metrics_out, made, answers


def test_validate_wer_fail_exits_1_and_writes_metrics(tmp_path, capsys):
    ref = "NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER"
    code, metrics_out, made, _ = _run_validate_main(
        tmp_path, [ref], [ref.lower()], max_wer=-1.0,  # the DUMMY baseline: always FAILS
    )
    out = capsys.readouterr().out
    assert code == 1
    assert made["model_dir"] == str(tmp_path / "model")
    assert "  [FAIL] utt-0 (wer=0.0000)" in out
    assert f"reference:  {ref!r}" in out
    assert "wer=0.0000 max=-1.0" in out
    assert out.rstrip().endswith("FAIL")
    doc = json.loads(metrics_out.read_text())
    assert doc["measurements"]["latency"] == 123_000.0
    assert doc["measurements"]["real_time_factor"] == 50.0
    assert "cpu_load" in doc["measurements"]


def test_validate_wer_pass_exits_0(tmp_path, capsys):
    ref = "the cat sat on the mat"
    code, _, _, _ = _run_validate_main(tmp_path, [ref], [ref], max_wer=0.0)
    out = capsys.readouterr().out
    assert code == 0
    assert "  [PASS] utt-0 (wer=0.0000)" in out
    assert "wer=0.0000 max=0.0" in out
    assert out.rstrip().endswith("PASS")


def test_validate_wer_scores_the_mean_over_items(tmp_path, capsys):
    # One perfect transcript (wer 0) and one with a substitution and an
    # insertion (wer 1/3): the mean is 1/6, which passes max_wer=0.2
    # although item 1 fails it.
    code, _, _, _ = _run_validate_main(
        tmp_path,
        ["the cat sat on the mat", "the cat sat on the mat"],
        ["the cat sat on the mat", "the cat sat on the rug pad"],
        max_wer=0.2,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "  [PASS] utt-0 (wer=0.0000)" in out
    assert "  [FAIL] utt-1 (wer=0.3333)" in out
    assert "wer=0.1667 max=0.2" in out
    assert out.rstrip().endswith("PASS")


def test_validate_wer_samples(tmp_path):
    ref = "the cat sat on the mat"
    code, _, _, unused = _run_validate_main(
        tmp_path, [ref, ref], [ref, "never used"], max_wer=0.0, argv=["--samples", "1"]
    )
    assert code == 0
    assert unused == ["never used"]


def test_validate_unknown_dataset_exits(tmp_path):
    with pytest.raises(SystemExit, match="No loader for dataset 'someone/else'"):
        with mock.patch.object(sys, "argv", ["validate.py", "-m", str(tmp_path), "--no-refresh", "--dataset", "someone/else"]):
            sr.validate_main(
                lambda args, runtime: "fake",
                lambda runner, item, args, metrics: "",
                demo="demo",
                validation=VALIDATION,
                max_wer=0.0,
                local_model=lambda: None,
                ensure_models=lambda model_dir, **kwargs: None,
                add_args=_add_args,
            )
