# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""The ``automatic_speech_recognition`` task category (CONTRIBUTING.md).

``infer_main`` (the shared main of the other model-dir categories too, see
``utils.tasks``) is the shared command-line main of the speech demos'
``src/infer.py`` scripts: the demo's own flags, default-model resolution and
refresh, one NPU-boosted session, and the demo's per-input loop — a WAV-file
transcription loop, or the demo's own streaming pipeline. ``validate_main``
is the shared main of the ``src/validate.py`` scripts: the same model setup,
a Hugging Face dataset of speech with reference transcripts (``LOADERS``),
and a word error rate score. Both record the CONTRIBUTING.md metrics; validate
also records the mean word error rate.
"""

import argparse
import re
from collections.abc import Callable
from pathlib import Path

from utils.log import configure_logging
from utils.metrics import MetricsRun
from utils.runtime import RuntimeConfig, npu_clock
from utils.datasets import fetch_rows
from utils.tasks import add_dataset_args, infer_main, load_items, resolve_and_ensure

__all__ = ["infer_main", "validate_main", "word_error_rate", "LOADERS"]


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def word_error_rate(hypothesis: str, reference: str) -> float:
    """Word error rate of *hypothesis* against *reference*.

    Words are case-folded and punctuation is stripped; the score is the
    word-level Levenshtein distance divided by the reference word count (an
    empty reference counts as one word).
    """
    hyp, ref = _words(hypothesis), _words(reference)
    dist = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        prev, dist[0] = dist[0], i
        for j, r in enumerate(ref, 1):
            prev, dist[j] = dist[j], min(prev + (h != r), dist[j] + 1, dist[j - 1] + 1)
    return dist[-1] / max(1, len(ref))


def _librispeech(spec: dict) -> list[dict]:
    rows = fetch_rows(spec["dataset"], config=spec["config"], split=spec["split"], samples=spec["samples"])
    return [{"name": row["id"], "audio": Path(row["audio"]), "reference": row["text"]} for row in rows]


#: Supported datasets: rows -> ``{"name", "audio" (local file), "reference"}`` items.
LOADERS: dict[str, Callable[[dict], list[dict]]] = {
    "openslr/librispeech_asr": _librispeech,
    "hf-internal-testing/librispeech_asr_dummy": _librispeech,
}


def validate_main(
    make_runner: Callable,
    transcribe: Callable,
    *,
    demo: str,
    validation: list[dict],
    local_model: Callable[[], Path | None] | None,
    ensure_models: Callable,
    download: Callable | None = None,
    add_args: Callable[[argparse.ArgumentParser], None],
    message: str | None = None,
) -> None:
    """Shared main of the speech demos' ``src/validate.py``.

    Same model setup as :func:`infer_main`, plus the validation dataset
    flags (defaults: *validation*, the demo's ``info.json`` list).
    *transcribe(runner, item, args, metrics)* returns the transcript of
    ``item["audio"]`` (a local file) and records the demo's metrics; each
    transcript is scored against ``item["reference"]`` with
    :func:`word_error_rate`. Prints one line per item, then
    ``wer=<mean>``, recorded as the ``word_error_rate`` metric (percent).
    """
    parser = argparse.ArgumentParser(description=f"Validate the {demo} demo.")
    add_args(parser)
    add_dataset_args(parser, validation)
    args = parser.parse_args()
    configure_logging(args.logging)
    resolve_and_ensure(
        parser, args, demo=demo, local_model=local_model,
        ensure_models=ensure_models, download=download, message=message,
    )
    items, datasets = load_items(args, validation, LOADERS)
    runtime = RuntimeConfig.from_args(args)
    with npu_clock():
        runner = make_runner(args, runtime)
        with MetricsRun(args.metrics_out, info={"validation": datasets}) as m:
            wers = []
            for item in items:
                text = transcribe(runner, item, args, m)
                wer = word_error_rate(text, item["reference"])
                wers.append(wer)
                print(f"  {item['name']} (wer={wer:.4f})")
                if wer > 0:
                    print(f"         transcript: {text.strip()[:200]!r}")
                    print(f"         reference:  {item['reference'][:200]!r}")
            mean_wer = sum(wers) / len(wers)
            m.add("word_error_rate", mean_wer * 100)
    print(f"wer={mean_wer:.4f}")
