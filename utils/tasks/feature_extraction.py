# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""The ``feature_extraction`` task category (CONTRIBUTING.md).

``infer_main`` (the shared main of the other model-dir categories too, see
``utils.tasks``) is the shared command-line main of the embedding demos'
``src/infer.py`` scripts: the demo's own flags, default-model resolution, one
NPU-boosted session, and the demo's per-input loop. ``validate_main`` is the
shared main of the ``src/validate.py`` scripts: the same model setup,
Hugging Face caption datasets (``LOADERS``) of media items each with the
caption it should match, and a retrieval check per item — its own caption
ranks first among the run's captions by cosine similarity. ``--modalities``
restricts which encoders the session loads and which items are validated (a
board with little RAM runs the modalities in separate sessions). Both record
the CONTRIBUTING.md metrics; validate also records its accuracy.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from utils.datasets import fetch_rows
from utils.log import configure_logging
from utils.metrics import MetricsRun
from utils.runtime import RuntimeConfig, npu_clock
from utils.tasks import add_dataset_args, infer_main, load_items, resolve_and_ensure

__all__ = ["infer_main", "validate_main", "cosine", "LOADERS"]

_MODALITIES = ("text", "image", "video", "audio")


def cosine(a, b) -> float:
    """Cosine similarity of two vectors (the norms are computed, not assumed)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _rows(spec: dict) -> list[dict]:
    return fetch_rows(spec["dataset"], config=spec["config"], split=spec["split"],
                      samples=spec["samples"], offset=spec["offset"])


def _flickr30k(spec: dict) -> list[dict]:
    return [
        {"kind": "image", "name": row["filename"], "file": Path(row["image"]), "query": row["caption"][0]}
        for row in _rows(spec)
    ]


def _clotho(spec: dict) -> list[dict]:
    # ``text`` joins the clip's captions; the first sentence is one caption.
    return [
        {"kind": "audio", "name": row["index"], "file": Path(row["audio"]),
         "query": row["text"].split(". ")[0].rstrip(".")}
        for row in _rows(spec)
    ]


#: Supported datasets: rows -> ``{"kind", "name", "file" (local file), "query"}`` items.
LOADERS: dict[str, Callable[[dict], list[dict]]] = {
    "nlphuji/flickr30k": _flickr30k,
    "CLAPv2/Clotho": _clotho,
}


def validate_main(
    make_runner: Callable,
    encode: Callable,
    *,
    demo: str,
    validation: list[dict],
    local_model: Callable[[], Path | None] | None,
    ensure_models: Callable | None = None,
    download: Callable | None = None,
    add_args: Callable[[argparse.ArgumentParser], None],
    message: str | None = None,
    loaders: dict[str, Callable[[dict], list[dict]]] | None = None,
) -> None:
    """Shared main of the embedding demos' ``src/validate.py``.

    Same model setup as :func:`infer_main`, plus the validation dataset
    flags (defaults: *validation*, the demo's ``info.json`` list; *loaders*
    adds demo-specific datasets to :data:`LOADERS`) and ``--modalities
    text,image,video,audio``. *make_runner(args, runtime, items)* gets the
    selected items (e.g. to size an audio window). *encode(runner, item,
    args, metrics)* embeds one item — a media file (``item["file"]``) or the
    pseudo-item ``{"kind": "query", "query": ...}`` — and records the demo's
    ``latency_<modality>`` metric. An item passes when its own caption ranks
    first among the run's captions (by :func:`cosine`); the accuracy is the
    fraction of passing items, printed as ``accuracy=<x>`` and recorded as
    the ``accuracy`` metric (percent).
    """
    parser = argparse.ArgumentParser(description=f"Validate the {demo} demo.")
    add_args(parser)
    add_dataset_args(parser, validation)
    parser.add_argument(
        "--modalities", type=str, default=",".join(_MODALITIES),
        help="Comma-separated modalities to load and validate: "
             "text,image,video,audio (default: %(default)s)",
    )
    args = parser.parse_args()
    configure_logging(args.logging)
    mods = {m.strip() for m in args.modalities.split(",") if m.strip()}
    unknown = mods - set(_MODALITIES)
    if unknown:
        parser.error(f"unknown modalities: {', '.join(sorted(unknown))} "
                     f"(want a subset of {', '.join(_MODALITIES)})")
    mods.add("text")  # the captions are text: the text encoder is always loaded
    resolve_and_ensure(
        parser, args, demo=demo, local_model=local_model,
        ensure_models=ensure_models, download=download, message=message,
    )
    items, datasets = load_items(args, validation, {**LOADERS, **(loaders or {})})
    items = [it for it in items if it["kind"] in mods]
    # The metrics list only the datasets, and rows, actually scored.
    for spec in datasets:
        spec["rows"] = [it["row"] for it in items if it["spec"] is spec]
    datasets = [spec for spec in datasets if spec["rows"]]
    if not items:
        parser.error(f"no {demo} validation items for --modalities {args.modalities}")
    queries = list(dict.fromkeys(it["query"] for it in items))
    runtime = RuntimeConfig.from_args(args)
    with npu_clock():
        runner = make_runner(args, runtime, items)
        with MetricsRun(args.metrics_out, info={"validation": datasets}) as m:
            query_emb = {q: encode(runner, {"kind": "query", "query": q}, args, m) for q in queries}
            passed = 0
            for it in items:
                emb = encode(runner, it, args, m)
                sims = {q: cosine(emb, e) for q, e in query_emb.items()}
                top = max(sims, key=sims.get)
                ok = top == it["query"]
                passed += ok
                print(f"  [{'PASS' if ok else 'FAIL'}] {it['kind']} {it['name']}")
                if not ok:
                    print(f"         top-1 {top!r} (expected {it['query']!r})")
            accuracy = passed / len(items)
            m.add("accuracy", accuracy * 100)
    print(f"accuracy={accuracy:.4f} ({passed}/{len(items)} items)")
