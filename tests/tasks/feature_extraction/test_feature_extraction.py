# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Host tests for the ``feature_extraction`` task category: the cosine helper,
the caption-dataset loaders, and the ``validate_main`` flow end to end
(retrieval ranking, PASS/FAIL, exit code, metrics JSON and ``--modalities``
filtering) with fake dataset rows, runner and encode hook in place of the
download and the model."""

import json
import sys
from unittest import mock

import numpy as np
import pytest

from utils.tasks import feature_extraction as emb


# ── cosine ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("a,b,expected", [
    ([1, 0, 0], [1, 0, 0], 1.0),
    ([1, 0, 0], [0, 1, 0], 0.0),
    ([1, 0, 0], [-1, 0, 0], -1.0),
    ([3, 0, 0], [1, 0, 0], 1.0),  # the norms are computed, not assumed
])
def test_cosine(a, b, expected):
    assert emb.cosine(a, b) == pytest.approx(expected)


# ── validate_main end to end ─────────────────────────────────────────────────

VALIDATION = [
    {"dataset": "nlphuji/flickr30k", "config": "TEST", "split": "test", "samples": 10},
    {"dataset": "CLAPv2/Clotho", "config": "default", "split": "validation", "samples": 10},
]
# Three orthogonal captions, one per item; each media embedding leans on its
# own caption, so the retrieval check passes for every item.
QUERIES = {"a dog": [1.0, 0.0, 0.0], "a cat": [0.0, 1.0, 0.0], "a bark": [0.0, 0.0, 1.0]}
ROWS = {
    "nlphuji/flickr30k": [
        {"filename": "dog.jpg", "image": "/data/dog.jpg", "caption": ["a dog", "a puppy"]},
        {"filename": "cat.jpg", "image": "/data/cat.jpg", "caption": ["a cat", "a kitten"]},
    ],
    "CLAPv2/Clotho": [{"index": "bark", "audio": "/data/bark.wav", "text": "a bark. A dog barks twice."}],
}
MEDIA = {"/data/dog.jpg": [0.99, 0.01, 0.01], "/data/cat.jpg": [0.01, 0.99, 0.01], "/data/bark.wav": [0.01, 0.01, 0.99]}


def _fetch_rows(dataset, *, config, split, samples, offset=0):
    return ROWS[dataset][offset:offset + samples]


def _add_args(parser):
    parser.add_argument("-m", "--model-dir", default=None)
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--logging", default="INFO")
    parser.add_argument("--metrics-out", default=None)


def _run_validate_main(tmp_path, media, *, argv=()):
    metrics_out = tmp_path / "metrics.json"
    made = {}

    def make_runner(args, runtime, items):
        made.update(modalities=args.modalities, items=[(it["kind"], it["name"], it["query"]) for it in items])
        return "fake"

    def encode(runner, item, args, metrics):
        # queries are text, like the demo's encode hook
        modality = "text" if item["kind"] == "query" else item["kind"]
        metrics.add(f"latency_{modality}", 123_000)
        if item["kind"] == "query":
            return np.array(QUERIES[item["query"]])
        return np.array(media[str(item["file"])])

    argv = ["validate.py", "-m", str(tmp_path / "model"), "--no-refresh", "--metrics-out", str(metrics_out), *argv]
    with mock.patch.object(emb, "fetch_rows", _fetch_rows), mock.patch.object(sys, "argv", argv):
        try:
            emb.validate_main(
                make_runner,
                encode,
                demo="demo",
                validation=VALIDATION,
                local_model=lambda: None,
                ensure_models=lambda model_dir, **kwargs: None,
                add_args=_add_args,
            )
            code = 0
        except SystemExit as e:
            code = e.code
    return code, metrics_out, made


def test_loaders_take_the_first_caption(tmp_path):
    _, _, made = _run_validate_main(tmp_path, dict(MEDIA))
    assert made["items"] == [("image", "dog.jpg", "a dog"), ("image", "cat.jpg", "a cat"), ("audio", "bark", "a bark")]


def test_validate_all_retrieved_scores_one(tmp_path, capsys):
    code, metrics_out, made = _run_validate_main(tmp_path, dict(MEDIA))
    out = capsys.readouterr().out
    assert code == 0
    assert "  [PASS] image dog.jpg" in out and "  [PASS] audio bark" in out
    assert out.rstrip().endswith("accuracy=1.0000 (3/3 items)")
    doc = json.loads(metrics_out.read_text())
    assert doc["measurements"]["latency_image"] == 123_000.0
    assert doc["measurements"]["accuracy"] == 100.0
    assert doc["measurements"]["latency_text"] == 123_000.0


def test_validate_retrieval_fail(tmp_path, capsys):
    media = dict(MEDIA, **{"/data/cat.jpg": [0.99, 0.01, 0.01]})  # the cat looks like the dog
    code, metrics_out, _ = _run_validate_main(tmp_path, media)
    out = capsys.readouterr().out
    assert code == 0  # no pass mark: the accuracy is reported, never a verdict
    assert json.loads(metrics_out.read_text())["measurements"]["accuracy"] == pytest.approx(200 / 3)
    assert "  [FAIL] image cat.jpg" in out
    assert "top-1 'a dog' (expected 'a cat')" in out


def test_modalities_filter(tmp_path, capsys):
    code, _, made = _run_validate_main(tmp_path, dict(MEDIA), argv=["--modalities", "text,audio"])
    out = capsys.readouterr().out
    assert code == 0
    assert made["items"] == [("audio", "bark", "a bark")]
    assert "image" not in out.split("accuracy")[0]


def test_modalities_unknown_errors(tmp_path, capsys):
    code, _, _ = _run_validate_main(tmp_path, dict(MEDIA), argv=["--modalities", "text,smell"])
    assert code == 2
    assert "unknown modalities: smell" in capsys.readouterr().err


def test_modalities_all_filtered_errors(tmp_path, capsys):
    code, _, _ = _run_validate_main(tmp_path, dict(MEDIA), argv=["--modalities", "text,video"])
    assert code == 2
    assert "no demo validation items for --modalities text,video" in capsys.readouterr().err


def test_samples(tmp_path):
    _, metrics_out, made = _run_validate_main(tmp_path, dict(MEDIA), argv=["--samples", "1"])
    assert [name for _, name, _ in made["items"]] == ["dog.jpg", "bark"]
    assert [(d["dataset"], d["samples"], d["rows"]) for d in json.loads(metrics_out.read_text())["validation"]] == [
        ("nlphuji/flickr30k", 1, [0]), ("CLAPv2/Clotho", 1, [0]),
    ]
