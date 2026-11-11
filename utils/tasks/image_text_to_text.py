# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""The ``image_text_to_text`` task category (CONTRIBUTING.md).

``infer_main`` (the shared main of the other model-dir categories too, see
``utils.tasks``) is the shared command-line main of the VQA demos'
``src/infer.py`` scripts: the demo's own flags, default-model resolution and
refresh, one NPU-boosted session, and the demo's image + chat loop.
``validate_main`` is the shared main of the ``src/validate.py`` scripts: the
same model setup with greedy decoding forced, a Hugging Face dataset of
image + question items (``LOADERS``), and answer scoring (the option letter
for multiple-choice benchmarks). Both record the CONTRIBUTING.md
metrics; validate also records its accuracy.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from utils.inference import ContextOverflow
from utils.log import configure_logging
from utils.metrics import MetricsRun
from utils.runtime import RuntimeConfig, npu_clock
from utils.datasets import fetch_rows
from utils.tasks import add_dataset_args, infer_main, load_items, resolve_and_ensure, scored, skip_item
from utils.tasks.text_generation import choice_check

__all__ = ["infer_main", "validate_main", "LOADERS"]


def _mmstar(spec: dict) -> list[dict]:
    rows = fetch_rows(spec["dataset"], config=spec["config"], split=spec["split"],
                      samples=spec["samples"], offset=spec["offset"])
    return [
        {
            "name": f"{spec['dataset']} #{row['index']}",
            "image": Path(row["image"]),
            "question": row["question"] + "\nAnswer with the option's letter from the given choices directly.",
            "expected": row["answer"],
            "check": choice_check(row["answer"], "ABCD"),
        }
        for row in rows
    ]


#: Supported datasets: rows -> ``{"name", "image" (local file), "question", "expected", "check"}`` items.
LOADERS: dict[str, Callable[[dict], list[dict]]] = {"Lin-Chen/MMStar": _mmstar}


def validate_main(
    make_runner: Callable,
    ask: Callable,
    *,
    demo: str,
    validation: list[dict],
    local_model: Callable[[], Path | None] | None,
    ensure_models: Callable,
    download: Callable | None = None,
    add_args: Callable[[argparse.ArgumentParser], None],
    message: str | None = None,
) -> None:
    """Shared main of the VQA demos' ``src/validate.py``.

    Same model setup as :func:`infer_main` (with greedy decoding forced)
    plus the validation dataset flags (defaults: *validation*, the demo's
    ``info.json`` list). *ask(runner, item, args, metrics)* encodes
    ``item["image"]`` (a local file) and answers ``item["question"]``; an
    item passes when ``item["check"](answer)`` holds. Prints one line per
    item, then ``accuracy=<x>``, recorded as the ``accuracy`` metric (percent).
    """
    parser = argparse.ArgumentParser(description=f"Validate the {demo} demo.")
    add_args(parser)
    add_dataset_args(parser, validation)
    args = parser.parse_args()
    configure_logging(args.logging)
    if hasattr(args, "temperature"):
        args.temperature = 0.0  # greedy decoding for reproducibility
    resolve_and_ensure(
        parser, args, demo=demo, local_model=local_model,
        ensure_models=ensure_models, download=download, message=message,
    )
    items, datasets = load_items(args, validation, LOADERS)
    runtime = RuntimeConfig.from_args(args)
    with npu_clock():
        runner = make_runner(args, runtime)
        with MetricsRun(args.metrics_out, info={"validation": datasets}) as m:
            passed = skipped = 0
            for item in items:
                try:
                    answer = ask(runner, item, args, m)
                except ContextOverflow as exc:
                    skip_item(items, item, exc)
                    skipped += 1
                    continue
                ok = item["check"](answer)
                passed += ok
                print(f"  [{'PASS' if ok else 'FAIL'}] {item['name']}")
                if not ok:
                    print(f"         answer: {answer.strip()[:200]!r} (expected {item['expected']!r})")
            n = scored(items, skipped)
            accuracy = passed / n
            m.add("accuracy", accuracy * 100)
    print(f"accuracy={accuracy:.4f} ({passed}/{n} items, {skipped} skipped)")
