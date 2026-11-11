# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared plumbing for the task categories (CONTRIBUTING.md).

``infer_main`` is the shared command-line main of the model-dir demos'
``src/infer.py`` scripts (the ``automatic_speech_recognition``,
``image_text_to_text`` and ``feature_extraction`` categories re-export it):
the demo's own
flags, default-model resolution and refresh, one NPU-boosted session, and
the demo's per-input loop. The categories' ``validate_main`` functions
share ``resolve_and_ensure`` for the same model setup, and
``add_dataset_args`` / ``load_items`` for their validation data: Hugging Face
datasets (``utils.datasets``) whose defaults are the ``validation`` list in
the demo's ``info.json``, turned into scoring items by the category's
per-dataset loaders.
"""

import argparse
import json
from collections.abc import Callable
from itertools import groupby
from pathlib import Path

from utils.log import configure_logging
from utils.metrics import MetricsRun
from utils.model_setup import resolve_default_model
from utils.runtime import RuntimeConfig, npu_clock

__all__ = ["infer_main", "resolve_and_ensure", "demo_validation", "add_dataset_args", "load_items", "skip_item", "scored"]


def _message(demo: str) -> str:
    return (
        f"no local {demo} model found; pass -m/--model-dir or run "
        f"`python setup_demos.py {demo}` from torq-examples root"
    )


def resolve_and_ensure(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    demo: str,
    local_model: Callable[[], Path | None] | None,
    ensure_models: Callable | None = None,
    download: Callable | None = None,
    message: str | None = None,
) -> None:
    """Default ``args.model_dir`` to the demo's local model dir, then refresh it.

    *local_model* (``() -> Path | None``) finds the demo's downloaded model
    dir; *download* (optional) fetches it on a first run without one.
    *ensure_models* (``(model_dir, refresh=)``) re-syncs the local copy;
    ``None`` skips the refresh (the demo does not track its models at
    runtime).
    """
    if message is None:
        message = _message(demo)
    no_refresh = getattr(args, "no_refresh", False)
    if args.model_dir is None:
        if local_model is None:
            parser.error(message)
        args.model_dir = str(
            resolve_default_model(
                parser, local_model,
                download=download, no_refresh=no_refresh, message=message,
            )
        )
    if ensure_models is not None:
        ensure_models(args.model_dir, refresh=not no_refresh)


def demo_validation(script: str | Path) -> list[dict]:
    """The default validation datasets of the demo that owns *script* (its ``src/`` file).

    Each entry of the ``validation`` list in the demo's ``info.json`` names a
    Hugging Face ``dataset``, its ``config`` and ``split`` (for datasets read
    through the viewer API) and the number of ``samples`` to score.
    """
    info = Path(script).resolve().parents[1] / "info.json"
    return json.loads(info.read_text(encoding="utf-8"))["validation"]


def add_dataset_args(parser: argparse.ArgumentParser, validation: list[dict]) -> None:
    """``--dataset``, ``--config``, ``--split`` and ``--samples``, defaulting to
    *validation*, or ``--rows`` to replay a previous run's rows."""
    group = parser.add_argument_group("validation data (Hugging Face datasets)")
    group.add_argument(
        "--dataset", default=None,
        help=f"Dataset repo id (default: {', '.join(v['dataset'] for v in validation)})",
    )
    group.add_argument("--config", default=None, help="Dataset config (default: the dataset's default in info.json)")
    group.add_argument("--split", default=None, help="Dataset split (default: the dataset's default in info.json)")
    group.add_argument("--samples", type=int, default=None, help="Rows to score per dataset (default: info.json)")
    group.add_argument(
        "--rows", metavar="METRICS_JSON", default=None,
        help="Score exactly the datasets and rows recorded in a previous run's metrics JSON "
             "(instead of --dataset/--config/--split/--samples; skipped rows are not replaced)",
    )


def load_items(
    args: argparse.Namespace, validation: list[dict], loaders: dict[str, Callable]
) -> tuple[list[dict], list[dict]]:
    """The scoring items of the selected datasets, and the datasets as run.

    ``--dataset`` replaces the defaults with one dataset (its missing
    ``--config`` / ``--split`` / ``--samples`` come from the matching default,
    if any); ``--samples`` applies to every dataset. *loaders* maps each
    supported dataset to ``loader(spec) -> list[dict]``: one item per row,
    in order, for rows ``spec["offset"]`` to ``offset + samples - 1`` of the
    resolved ``{"dataset", "config", "split", "samples"}`` (fewer at the end
    of a split). The datasets as run are those specs plus ``rows``, the
    indices of the rows scored; validate records them in the metrics JSON,
    so a score says exactly what it was measured on. ``--rows`` replays such
    a list: the same datasets and rows. Each item carries its ``row``, its
    ``spec`` (dataset as run) and its ``loader`` (None when replaying: the
    rows are fixed, so a skipped one is not replaced).
    """
    if args.rows:
        if args.dataset or args.config or args.split or args.samples is not None:
            raise SystemExit("--rows cannot be combined with --dataset, --config, --split or --samples")
        try:
            specs = json.loads(Path(args.rows).read_text(encoding="utf-8"))["validation"]
            recorded = [spec.pop("rows") for spec in specs]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SystemExit(f"--rows {args.rows}: no recorded validation rows ({exc!r})") from None
    elif args.dataset:
        base = next((dict(v) for v in validation if v["dataset"] == args.dataset), {"dataset": args.dataset})
        for key in ("config", "split"):
            if getattr(args, key):
                base[key] = getattr(args, key)
        specs = [base]
    else:
        specs = [dict(v) for v in validation]
    items = []
    for i, spec in enumerate(specs):
        if args.samples is not None:
            spec["samples"] = args.samples
        spec.setdefault("samples", validation[0]["samples"])
        loader = loaders.get(spec["dataset"])
        if loader is None:
            raise SystemExit(
                f"No loader for dataset {spec['dataset']!r}; supported: {', '.join(sorted(loaders))}"
            )
        spec["rows"] = []
        if not args.rows:
            items += _load(loader, spec, 0, spec["samples"])
            continue
        # One load per run of consecutive rows (the viewer API pages ranges).
        for _, run in groupby(enumerate(recorded[i]), lambda pair: pair[1] - pair[0]):
            rows = [row for _, row in run]
            items += _load(loader, spec, rows[0], len(rows), refill=False)
    if not items:
        raise SystemExit("No validation items")
    return items, specs


def _load(loader: Callable, spec: dict, offset: int, samples: int, *, refill: bool = True) -> list[dict]:
    """Rows *offset* to *offset + samples - 1* of *spec* as tagged items."""
    loaded = loader(dict(spec, offset=offset, samples=samples))
    for row, item in enumerate(loaded, offset):
        item.update(row=row, spec=spec, loader=loader if refill else None)
        spec["rows"].append(row)
    return loaded


def skip_item(items: list[dict], item: dict, reason: Exception) -> None:
    """Leave *item* out of the score: it cannot run on this model (e.g. it
    does not fit the context). The dataset's next row is appended to
    *items* (the list being scored) in its place, so the item count stays
    the same; at most ``2 * samples`` rows of a dataset are tried, so a
    model that fits none does not walk the whole split. Replayed rows
    (``--rows``) are not replaced."""
    spec = item["spec"]
    spec["rows"].remove(item["row"])
    print(f"  [SKIP] {item['name']} ({reason})")
    tried = [it["row"] for it in items if it["spec"] is spec]
    if item["loader"] is not None and len(tried) < 2 * spec["samples"]:
        items += _load(item["loader"], spec, max(tried) + 1, 1)


def scored(items: list[dict], skipped: int) -> int:
    """The number of items scored; exits when every item was skipped."""
    if skipped == len(items):
        raise SystemExit("No validation item fits the model's context")
    return len(items) - skipped


def infer_main(
    make_runner: Callable,
    loop: Callable,
    *,
    description: str,
    demo: str,
    local_model: Callable[[], Path | None] | None,
    ensure_models: Callable | None = None,
    download: Callable | None = None,
    add_args: Callable[[argparse.ArgumentParser], None],
    message: str | None = None,
) -> None:
    """Shared main of the model-dir demos' ``src/infer.py`` (re-exported by
    the ``automatic_speech_recognition``, ``image_text_to_text`` and
    ``feature_extraction`` categories).

    *add_args(parser)* adds the demo's flags, including the runtime group a
    ``RuntimeConfig`` is built from; *make_runner(args, runtime)* builds the
    demo's model(s); *loop(runner, args, metrics)* is the per-input loop —
    the demo's file loop or its own streaming pipeline.
    """
    parser = argparse.ArgumentParser(description=description)
    add_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)
    resolve_and_ensure(
        parser, args, demo=demo, local_model=local_model,
        ensure_models=ensure_models, download=download, message=message,
    )
    runtime = RuntimeConfig.from_args(args)
    with npu_clock():
        runner = make_runner(args, runtime)
        with MetricsRun(args.metrics_out) as m:
            loop(runner, args, m)
