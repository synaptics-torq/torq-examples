# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""EmbeddingGemma-2 demo: embed text queries, images and videos into one 768-d space
and rank them against each other.

    python src/infer.py \
        --queries "cats sleeping on a couch" "a turtle swimming in the ocean" \
        --images cats.jpg --videos sea-turtle.mp4

Images and videos are scored against every query (cosine similarity). With only
``--queries`` the queries are compared with each other (sentence similarity).
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_dir import resolve_model_dir  # noqa: E402
from runner import EmbeddingGemma2  # noqa: E402


def _fmt_timings(t: dict) -> str:
    return ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model-dir", help="Directory with the vmfbs/onnx graphs and assets (default: the files setup_demos.py downloaded)")
    ap.add_argument("--backend", choices=["torq", "ort"], default="torq")
    ap.add_argument("--ort-threads", type=int, default=0)
    ap.add_argument("--queries", nargs="*", default=[], help="Text queries")
    ap.add_argument("--prompt", default="query", help="Task prompt name for the queries (default: %(default)s)")
    ap.add_argument("--documents", nargs="*", default=[], help="Text documents (prompt 'document')")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--audios", nargs="*", default=[], help="16 kHz WAV files (resampled if needed)")
    ap.add_argument("--audio-window", default="auto",
                    help="Audio encoder window in seconds (1.4, 2.8, 5.6, 11.2) or 'auto': the smallest "
                         "window that holds the longest clip (default: %(default)s)")
    ap.add_argument("--dim", type=int, default=768, choices=[768, 512, 256, 128], help="Matryoshka dimension")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    args.model_dir = resolve_model_dir(ap, args.model_dir)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    modalities = {"text"} | ({"image"} if args.images else set()) | ({"video"} if args.videos else set()) \
        | ({"audio"} if args.audios else set())
    audio_frames = None
    if args.audios:
        import json
        manifest = json.loads((Path(args.model_dir) / "embeddinggemma2_manifest.json").read_text())
        windows = sorted((manifest.get("audio") or {}).get("windows", [1120]))
        if args.audio_window == "auto":
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import preprocess as pp
            longest = max(len(pp.load_audio(p)) for p in args.audios) / pp.AUDIO_SAMPLE_RATE
            audio_frames = next((w for w in windows if w * 0.01 >= longest), windows[-1])
        else:
            audio_frames = int(round(float(args.audio_window) * 100))
        print(f"[audio window] {audio_frames * 0.01:.1f} s ({audio_frames // 4} tokens max)")
    if args.audios and (args.images or args.videos) and args.backend == "torq":
        # The SL2619 NPU driver maps at most ~1.5 GB of networks at a time and releasing a
        # network inside a process is unreliable, so a session keeps all of its networks
        # mapped. The 4-bit build fits the vision and audio encoders with both text buckets;
        # the bf16 build (~1.7 GB) does not, and then audio runs in its own session.
        import json
        spans = json.loads((Path(args.model_dir) / "embeddinggemma2_manifest.json").read_text()).get("xram_span_mb", {})
        need = sum(spans.get(n, 0) for n in ("vision_384x384", f"audio_{audio_frames}", "text_body_s128", "text_body_s512"))
        if need > EmbeddingGemma2.NPU_BUDGET_MB:
            ap.error(f"these networks need {need:.0f} MB of NPU mappings (limit {EmbeddingGemma2.NPU_BUDGET_MB} MB): "
                     "run --audios in a separate session from --images/--videos; embeddings from both "
                     "sessions share one space and can be compared directly")
    model = EmbeddingGemma2(args.model_dir, backend=args.backend, ort_threads=args.ort_threads,
                            modalities=tuple(modalities), audio_frames=audio_frames)

    Q = []
    for q in args.queries:
        Q.append(model.encode_text(q, args.prompt, args.dim))
        print(f"[text]  {q!r:50.50}  {_fmt_timings(model.timings)}")
    items, names = [], []
    for d in args.documents:
        items.append(model.encode_text(d, "document", args.dim)); names.append(f"doc:{d[:30]}")
        print(f"[doc]   {d!r:50.50}  {_fmt_timings(model.timings)}")
    for p in args.images:
        items.append(model.encode_image(p, args.dim)); names.append(f"image:{Path(p).name}")
        print(f"[image] {p:50.50}  {_fmt_timings(model.timings)}")
    for p in args.audios:
        items.append(model.encode_audio(p, args.dim)); names.append(f"audio:{Path(p).name}")
        print(f"[audio] {p:50.50}  {_fmt_timings(model.timings)}")
    for p in args.videos:
        items.append(model.encode_video(p, dim=args.dim)); names.append(f"video:{Path(p).name}")
        print(f"[video] {p:50.50}  {_fmt_timings(model.timings)}")

    if not Q:
        return
    Q = np.stack(Q)
    if not items:
        print("\nquery x query cosine similarity")
        sims, rows = Q @ Q.T, args.queries
    else:
        print("\nitem x query cosine similarity")
        sims, rows = np.stack(items) @ Q.T, names
    width = max(len(r) for r in rows)
    print(" " * width + "  " + "  ".join(f"q{i}" .rjust(6) for i in range(len(Q))))
    for r, row in zip(rows, sims):
        best = int(np.argmax(row))
        print(r.ljust(width) + "  " + "  ".join(
            (f"{v:6.3f}" + ("*" if j == best else " ")) for j, v in enumerate(row)))
    for i, q in enumerate(args.queries):
        print(f"q{i}: {q}")


if __name__ == "__main__":
    main()
