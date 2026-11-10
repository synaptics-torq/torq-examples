# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Validate the EmbeddingGemma-2 model.

Embeds the media items of the validation datasets (``info.json``: Flickr30k
images, Clotho sounds, and the clips of the ``media_search_eval.json`` set)
and checks that each item's own caption ranks first among the run's captions.

The board loads at most two modalities at once (2 GB RAM), so run them in
separate sessions:

    python src/validate.py --modalities text,image
    python src/validate.py --modalities text,audio
    python src/validate.py --modalities text,video
"""

import json
import logging
from pathlib import Path

from embeddinggemma2.setup_demo import local_embeddinggemma2_model_dir
from embeddinggemma2.src.infer import _audio_window, _total_ms
from embeddinggemma2.src.runner import EmbeddingGemma2
from utils.datasets import fetch_file
from utils.runtime import add_common_inference_args
from utils.tasks import demo_validation
from utils.tasks.feature_extraction import validate_main

MIN_ACCURACY = 2.0  # DUMMY until a measured baseline exists: above 1, so the check always FAILS.


def add_args(parser) -> None:
    parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Path to the EmbeddingGemma-2 model directory (default: the one setup_demo.py downloaded)",
    )
    add_common_inference_args(parser, with_threads=False, with_no_refresh=False)


def make_runner(args, runtime, items):
    logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
    mods = {"text"} | {m.strip() for m in args.modalities.split(",") if m.strip() in ("image", "video", "audio")}
    audio_frames = _audio_window(args.model_dir, "auto", [it["file"] for it in items if it["kind"] == "audio"])
    return EmbeddingGemma2(args.model_dir, modalities=tuple(mods), audio_frames=audio_frames, runtime=runtime)


def _eval_set_videos(spec: dict) -> list[dict]:
    """The clips of ``media_search_eval.json`` (hosted in its ``media_repo``) with their captions."""
    videos = json.loads((Path(__file__).with_name("media_search_eval.json")).read_text(encoding="utf-8"))["videos"]
    return [
        {"kind": "video", "name": v["file"], "file": fetch_file(spec["dataset"], v["file"]), "query": v["query"]}
        for v in videos[: spec["samples"]]
    ]


def encode(runner, item, args, metrics):
    """Embed one item: a query (text) or a media file, recording its latency."""
    kind = item["kind"]
    if kind == "query":
        emb = runner.encode_text(item["query"], "query")
        metrics.add("latency_text", int(_total_ms(runner.timings) * 1e6))
    elif kind == "image":
        emb = runner.encode_image(item["file"])
        metrics.add("latency_image", int(_total_ms(runner.timings) * 1e6))
    elif kind == "audio":
        emb = runner.encode_audio(item["file"])
        metrics.add("latency_audio", int(_total_ms(runner.timings) * 1e6))
    elif kind == "video":
        emb = runner.encode_video(item["file"])
        metrics.add("latency_video", int(_total_ms(runner.timings) * 1e6))
    else:
        raise ValueError(f"unknown item kind {kind!r}")
    return emb


if __name__ == "__main__":
    validate_main(
        make_runner,
        encode,
        demo="EmbeddingGemma-2",
        validation=demo_validation(__file__),
        min_accuracy=MIN_ACCURACY,
        local_model=local_embeddinggemma2_model_dir,
        add_args=add_args,
        loaders={"Xenova/transformers.js-docs": _eval_set_videos},
    )
