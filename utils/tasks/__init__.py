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
from pathlib import Path

from utils.log import configure_logging
from utils.metrics import MetricsRun
from utils.model_setup import resolve_default_model
from utils.runtime import RuntimeConfig, npu_clock

__all__ = ["infer_main", "resolve_and_ensure", "demo_validation", "add_dataset_args", "load_items"]


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
    """``--dataset``, ``--config``, ``--split`` and ``--samples``, defaulting to *validation*."""
    group = parser.add_argument_group("validation data (Hugging Face datasets)")
    group.add_argument(
        "--dataset", default=None,
        help=f"Dataset repo id (default: {', '.join(v['dataset'] for v in validation)})",
    )
    group.add_argument("--config", default=None, help="Dataset config (default: the dataset's default in info.json)")
    group.add_argument("--split", default=None, help="Dataset split (default: the dataset's default in info.json)")
    group.add_argument("--samples", type=int, default=None, help="Rows to score per dataset (default: info.json)")


def load_items(
    args: argparse.Namespace, validation: list[dict], loaders: dict[str, Callable]
) -> tuple[list[dict], list[dict]]:
    """The scoring items of the selected datasets, and the datasets as run.

    ``--dataset`` replaces the defaults with one dataset (its missing
    ``--config`` / ``--split`` / ``--samples`` come from the matching default,
    if any); ``--samples`` applies to every dataset. *loaders* maps each
    supported dataset to ``loader(spec) -> list[dict]``, where *spec* is the
    resolved ``{"dataset", "config", "split", "samples"}``. The datasets as
    run are those specs plus ``items``, the rows each one gave (fewer than
    ``samples`` at the end of a split); validate records them in the metrics
    JSON, so a score says what it was measured on.
    """
    if args.dataset:
        base = next((dict(v) for v in validation if v["dataset"] == args.dataset), {"dataset": args.dataset})
        for key in ("config", "split"):
            if getattr(args, key):
                base[key] = getattr(args, key)
        specs = [base]
    else:
        specs = [dict(v) for v in validation]
    items = []
    for spec in specs:
        if args.samples is not None:
            spec["samples"] = args.samples
        spec.setdefault("samples", validation[0]["samples"])
        loader = loaders.get(spec["dataset"])
        if loader is None:
            raise SystemExit(
                f"No loader for dataset {spec['dataset']!r}; supported: {', '.join(sorted(loaders))}"
            )
        loaded = loader(spec)
        spec["items"] = len(loaded)
        items += loaded
    if not items:
        raise SystemExit("No validation items")
    return items, specs


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
