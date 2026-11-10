# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""EmbeddingGemma-2: embed your own text, images, videos and sounds into one 768-d space and
rank them against each other.

    python src/infer.py \
        --queries "a fluffy tabby and white cat with bright blue eyes" "a dog barking and growling" \
        --images cat.jpg --audios dog.wav

For each query the images, videos, sounds and documents are listed best match first (cosine
similarity). With only ``--queries`` the queries are compared with each other.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

from embeddinggemma2.setup_demo import local_embeddinggemma2_model_dir
from embeddinggemma2.src.runner import EmbeddingGemma2
from utils.runtime import add_common_inference_args
from utils.tasks.feature_extraction import infer_main


def _fmt_timings(t: dict) -> str:
    return ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items())


def _total_ms(t: dict) -> float:
    """The sum of the ms values _fmt_timings prints for one encode (frames is not ms)."""
    return sum(v for v in t.values() if isinstance(v, float))


def _audio_window(model_dir: str, audio_window_arg: str, audio_paths) -> int | None:
    """The encoder window in frames of 10 ms: ``--audio-window``, or (auto) the smallest
    window that holds the longest clip. None without audio: the runner's default (largest)."""
    if not audio_paths:
        return None
    manifest = json.loads((Path(model_dir) / "embeddinggemma2_manifest.json").read_text())
    windows = sorted((manifest.get("audio") or {}).get("windows", [1120]))
    if audio_window_arg != "auto":
        return int(round(float(audio_window_arg) * 100))
    import preprocess as pp

    longest = max(len(pp.load_audio(p)) for p in audio_paths) / pp.AUDIO_SAMPLE_RATE
    return next((w for w in windows if w * 0.01 >= longest), windows[-1])


def add_args(parser) -> None:
    parser.add_argument(
        "-m", "--model-dir", help="Directory with the vmfbs/onnx graphs and assets (default: the files setup_demos.py downloaded)",
    )
    parser.add_argument("--backend", choices=["torq", "ort"], default="torq")
    parser.add_argument("--ort-threads", type=int, default=0)
    parser.add_argument("--queries", nargs="*", default=[], help="Text queries")
    parser.add_argument("--prompt", default="query", help="Task prompt name for the queries (default: %(default)s)")
    parser.add_argument("--documents", nargs="*", default=[], help="Text documents (prompt 'document')")
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--videos", nargs="*", default=[])
    parser.add_argument("--audios", nargs="*", default=[], help="16 kHz WAV files (resampled if needed)")
    parser.add_argument("--audio-window", default="auto",
                        help="Audio encoder window in seconds (1.4, 2.8, 5.6, 11.2) or 'auto': the smallest "
                             "window that holds the longest clip (default: %(default)s)")
    parser.add_argument("--dim", type=int, default=768, choices=[768, 512, 256, 128], help="Matryoshka dimension")
    parser.add_argument("--top", type=int, default=5, help="Matches listed per query (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true")
    add_common_inference_args(parser, with_threads=False, with_no_refresh=False)


def make_runner(args, runtime):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s", force=True)
    modalities = {"text"} | ({"image"} if args.images else set()) | ({"video"} if args.videos else set()) \
        | ({"audio"} if args.audios else set())
    audio_frames = _audio_window(args.model_dir, args.audio_window, args.audios)
    if audio_frames is not None:
        print(f"[audio window] {audio_frames * 0.01:.1f} s ({audio_frames // 4} tokens max)")
    if args.audios and (args.images or args.videos) and args.backend == "torq":
        # The SL2619 NPU driver maps at most ~1.5 GB of networks at a time and releasing a
        # network inside a process is unreliable, so a session keeps all of its networks
        # mapped; check that the requested networks fit together.
        spans = json.loads((Path(args.model_dir) / "embeddinggemma2_manifest.json").read_text()).get("xram_span_mb", {})
        need = sum(spans.get(n, 0) for n in ("vision_384x384", f"audio_{audio_frames}", "text_body_s128", "text_body_s512"))
        if need > EmbeddingGemma2.NPU_BUDGET_MB:
            print(f"error: these networks need {need:.0f} MB of NPU mappings (limit {EmbeddingGemma2.NPU_BUDGET_MB} MB): "
                  "run --audios in a separate session from --images/--videos; embeddings from both "
                  "sessions share one space and can be compared directly", file=sys.stderr)
            sys.exit(2)
    return EmbeddingGemma2(args.model_dir, backend=args.backend, ort_threads=args.ort_threads,
                           modalities=tuple(modalities), audio_frames=audio_frames,
                           runtime=runtime)


def loop(runner, args, metrics) -> None:
    Q = []
    for q in args.queries:
        Q.append(runner.encode_text(q, args.prompt, args.dim))
        metrics.add("latency_text", int(_total_ms(runner.timings) * 1e6))
        print(f"[text]  {q!r:40.40}  {_fmt_timings(runner.timings)}")
    items, names = [], []
    for d in args.documents:
        items.append(runner.encode_text(d, "document", args.dim)); names.append(f"doc:{d[:30]}")
        metrics.add("latency_text", int(_total_ms(runner.timings) * 1e6))
        print(f"[doc]   {d!r:40.40}  {_fmt_timings(runner.timings)}")
    for p in args.images:
        items.append(runner.encode_image(p, args.dim)); names.append(f"image:{Path(p).name}")
        metrics.add("latency_image", int(_total_ms(runner.timings) * 1e6))
        print(f"[image] {Path(p).name:40.40}  {_fmt_timings(runner.timings)}")
    for p in args.audios:
        items.append(runner.encode_audio(p, args.dim)); names.append(f"audio:{Path(p).name}")
        metrics.add("latency_audio", int(_total_ms(runner.timings) * 1e6))
        print(f"[audio] {Path(p).name:40.40}  {_fmt_timings(runner.timings)}")
    for p in args.videos:
        items.append(runner.encode_video(p, dim=args.dim)); names.append(f"video:{Path(p).name}")
        metrics.add("latency_video", int(_total_ms(runner.timings) * 1e6))
        print(f"[video] {Path(p).name:40.40}  {_fmt_timings(runner.timings)}")

    if not Q:
        return
    Q = np.stack(Q)
    if not items:
        # queries only: compare them with each other
        items, names = list(Q), [f"text:{q}" for q in args.queries]
    sims = Q @ np.stack(items).T
    for qi, q in enumerate(args.queries):
        print(f"\nquery: {q!r}")
        order = [j for j in np.argsort(-sims[qi]) if names[j] != f"text:{q}"]
        for j in order[: args.top]:
            print(f"  {sims[qi, j]:6.3f}  {names[j]}")


if __name__ == "__main__":
    infer_main(
        make_runner,
        loop,
        description=__doc__,
        demo="EmbeddingGemma-2",
        local_model=local_embeddinggemma2_model_dir,
        add_args=add_args,
    )
