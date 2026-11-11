# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared plumbing for the image-based task categories (CONTRIBUTING.md).

``infer_main`` is the shared command-line main of the ``object_detection``,
``keypoint_detection`` and ``face_recognition`` demos' ``src/infer.py`` scripts:
the demo's own flags, default-model resolution and refresh (``--model`` is a
model VMFB file, unlike the model-dir categories' ``-m/--model-dir``), one
NPU-boosted session, and the demo's per-image loop. ``validate_main`` is the
shared main of their ``src/validate.py`` scripts: the same model setup, a
Hugging Face dataset of images with their labels (the category's loaders),
and per-item scoring; the accuracy is printed and recorded in the metrics JSON.
``iou``, ``match_boxes`` and ``f1_score`` are the box-matching helpers the
categories' scoring is built from.
"""

import argparse
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

from utils.log import configure_logging
from utils.metrics import MetricsRun
from utils.model_setup import resolve_default_model
from utils.runtime import RuntimeConfig, npu_clock
from utils.tasks import add_dataset_args, load_items

__all__ = ["infer_main", "validate_main", "iou", "match_boxes", "f1_score", "round_box"]



def _message(demo: str) -> str:
    return (
        f"no local {demo} model found; pass --model or run "
        f"`python setup_demos.py {demo}` from torq-examples root"
    )


def _resolve_model(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    demo: str,
    local_model: Callable[[argparse.Namespace], Path | None],
    download: Callable[[argparse.Namespace], object] | None,
    message: str | Callable[[argparse.Namespace], str] | None,
) -> None:
    """Default ``args.model`` to the demo's local VMFB file (and download it).

    *local_model(args)* finds the demo's downloaded model file, *download*
    (optional) fetches it on a first run without one, *message* is the
    no-local-model error (a string or *message(args)*).
    """
    if message is None:
        message = _message(demo)
    message = message(args) if callable(message) else message
    if args.model is None:
        args.model = str(
            resolve_default_model(
                parser,
                partial(local_model, args),
                download=partial(download, args) if download is not None else None,
                no_refresh=getattr(args, "no_refresh", False),
                message=message,
            )
        )


def _ensure_model_dir(
    args: argparse.Namespace,
    ensure_models: Callable | None,
) -> None:
    """Re-sync the model dir containing ``args.model`` (if the demo tracks it)."""
    if ensure_models is not None:
        ensure_models(Path(args.model).parent, refresh=not getattr(args, "no_refresh", False))


def infer_main(
    make_runner: Callable,
    loop: Callable,
    *,
    description: str,
    demo: str,
    local_model: Callable[[argparse.Namespace], Path | None],
    ensure_models: Callable | None = None,
    download: Callable[[argparse.Namespace], object] | None = None,
    add_args: Callable[[argparse.ArgumentParser], None],
    message: str | Callable[[argparse.Namespace], str] | None = None,
) -> None:
    """Shared main of the image-based demos' ``src/infer.py``.

    *add_args(parser)* adds the demo's flags (including ``--model`` and the
    runtime group a ``RuntimeConfig`` is built from); *make_runner(args,
    runtime)* builds the demo's model; *loop(runner, args, metrics)* is the
    demo's per-image loop.
    """
    parser = argparse.ArgumentParser(description=description)
    add_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)
    _resolve_model(
        parser, args, demo=demo, local_model=local_model,
        download=download, message=message,
    )
    _ensure_model_dir(args, ensure_models)
    runtime = RuntimeConfig.from_args(args)
    with npu_clock():
        runner = make_runner(args, runtime)
        with MetricsRun(args.metrics_out) as m:
            loop(runner, args, m)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``[x, y, w, h]`` boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_boxes(
    predictions: list[dict],
    expected: list[dict],
    *,
    iou_threshold: float = 0.5,
    label: str | None = None,
) -> list[tuple[int, int]]:
    """Greedy one-to-one matching of *expected* boxes to *prediction* boxes.

    Each expected entry (in order) claims the still-unmatched prediction with
    the best IoU at or above *iou_threshold*; when *label* is given, only
    predictions with the same value for that key are eligible. Returns the
    ``(prediction_index, expected_index)`` pairs.
    """
    pairs: list[tuple[int, int]] = []
    taken = [False] * len(predictions)
    for e_idx, exp in enumerate(expected):
        best, best_iou = -1, -1.0
        for p_idx, pred in enumerate(predictions):
            if taken[p_idx] or (label is not None and pred[label] != exp[label]):
                continue
            score = iou(pred["box"], exp["box"])
            if score >= iou_threshold and score > best_iou:
                best, best_iou = p_idx, score
        if best >= 0:
            taken[best] = True
            pairs.append((best, e_idx))
    return pairs


def f1_score(tp: int, n_predicted: int, n_expected: int) -> float:
    """F1 of *tp* matched boxes out of *n_predicted* and *n_expected*."""
    precision = tp / n_predicted if n_predicted else 0.0
    recall = tp / n_expected if n_expected else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def round_box(box: Sequence[float]) -> list[int]:
    """A ``[x, y, w, h]`` box as rounded ints."""
    return [int(round(float(v))) for v in box]


def validate_main(
    make_runner: Callable,
    detect: Callable,
    score_item: Callable,
    aggregate: Callable,
    *,
    demo: str,
    validation: list[dict],
    loaders: dict[str, Callable[[dict], list[dict]]],
    local_model: Callable[[argparse.Namespace], Path | None],
    ensure_models: Callable | None = None,
    download: Callable[[argparse.Namespace], object] | None = None,
    add_args: Callable[[argparse.ArgumentParser], None],
    message: str | Callable[[argparse.Namespace], str] | None = None,
) -> None:
    """Shared main of the image-based demos' ``src/validate.py``.

    Same model setup as :func:`infer_main`, plus the validation dataset
    flags (defaults: *validation*, the demo's ``info.json`` list; *loaders*
    are the category's, turning dataset rows into ``{"name", "image" (local
    file), "expected"}`` items). *detect(runner, item, args, metrics)*
    returns the demo's results for ``item["image"]`` and records the demo's
    metrics. *score_item(item, predictions)* returns ``(item_score, counts,
    lines)`` — the item passes at ``item_score == 1.0``, and *lines* print on
    FAIL — then *aggregate(counts over all items)* is the accuracy, printed
    as ``accuracy=<x>`` and recorded as the ``accuracy`` metric (percent).
    """
    parser = argparse.ArgumentParser(description=f"Validate the {demo} demo.")
    add_args(parser)
    add_dataset_args(parser, validation)
    args = parser.parse_args()
    configure_logging(args.logging)
    _resolve_model(
        parser, args, demo=demo, local_model=local_model,
        download=download, message=message,
    )
    _ensure_model_dir(args, ensure_models)
    items, datasets = load_items(args, validation, loaders)
    runtime = RuntimeConfig.from_args(args)
    with npu_clock():
        runner = make_runner(args, runtime)
        with MetricsRun(args.metrics_out, info={"validation": datasets}) as m:
            counts = []
            for item in items:
                predictions = detect(runner, item, args, m)
                item_score, item_counts, lines = score_item(item, predictions)
                counts.append(item_counts)
                print(f"  [{'PASS' if item_score == 1.0 else 'FAIL'}] {item['name']}")
                if item_score != 1.0:
                    for line in lines:
                        print(f"         {line}")
            accuracy = aggregate(counts)
            m.add("accuracy", accuracy * 100)
    print(f"accuracy={accuracy:.4f}")
