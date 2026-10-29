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

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_dir import resolve_model_dir  # noqa: E402
from runner import EmbeddingGemma2  # noqa: E402
from utils.runtime import cleanup_npu_after_inference, setup_npu_for_inference  # noqa: E402


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
    ap.add_argument("--top", type=int, default=5, help="Matches listed per query (default: %(default)s)")
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
        # mapped; check that the requested networks fit together.
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
        print(f"[text]  {q!r:40.40}  {_fmt_timings(model.timings)}")
    items, names = [], []
    for d in args.documents:
        items.append(model.encode_text(d, "document", args.dim)); names.append(f"doc:{d[:30]}")
        print(f"[doc]   {d!r:40.40}  {_fmt_timings(model.timings)}")
    for p in args.images:
        items.append(model.encode_image(p, args.dim)); names.append(f"image:{Path(p).name}")
        print(f"[image] {Path(p).name:40.40}  {_fmt_timings(model.timings)}")
    for p in args.audios:
        items.append(model.encode_audio(p, args.dim)); names.append(f"audio:{Path(p).name}")
        print(f"[audio] {Path(p).name:40.40}  {_fmt_timings(model.timings)}")
    for p in args.videos:
        items.append(model.encode_video(p, dim=args.dim)); names.append(f"video:{Path(p).name}")
        print(f"[video] {Path(p).name:40.40}  {_fmt_timings(model.timings)}")

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
    setup_npu_for_inference()
    try:
        main()
    finally:
        cleanup_npu_after_inference()
