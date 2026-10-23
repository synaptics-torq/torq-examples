# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Any-to-any media search with EmbeddingGemma-2.

Text, images, video and audio share one embedding space, so one index answers a query
in words across all of them: "a dog barking" finds the bark recording and the dog
photo, "a turtle swimming in the ocean" finds the moment in a video. An image as the
query finds similar images and video moments.

    # 1. index the example media that setup_demos.py downloaded (or --media-dir <folder>)
    python src/media_search.py index

    # 2. search it: start an interactive prompt (the model loads once), or one query per run
    python src/media_search.py search startup
    python src/media_search.py search --text "a dog barking"
    python src/media_search.py search --image my_photo.jpg

Every image, sound clip and video goes into the index, and so does every second of
each video ("moments"), so a search can point to a time in a video.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp  # noqa: E402
from model_dir import resolve_model_dir  # noqa: E402
from runner import EmbeddingGemma2  # noqa: E402

EVAL = Path(__file__).resolve().parents[2] / "data" / "eval" / "eval_set.json"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".ogv"}
AUDIO_EXT = {".wav"}
# Each modality group is indexed in its own process, which maps only the networks it needs.
GROUPS = {"visual": {"image", "video"}, "audio": {"audio"}}


def fetch_samples(media_dir: Path):
    """The demo library: the images, videos and sounds of the eval set (transformers.js docs)."""
    from huggingface_hub import hf_hub_download

    ev = json.loads(EVAL.read_text())
    files = [it["file"] for key in ("images", "videos", "audios") for it in ev.get(key, [])]
    media_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        if not (media_dir / name).exists():
            hf_hub_download(ev["media_repo"], name, repo_type="dataset", local_dir=media_dir)
    print(f"[fetch] {len(files)} files in {media_dir}", flush=True)


def list_media(media_dir: Path) -> dict[str, list[Path]]:
    files = sorted(p for p in media_dir.rglob("*") if p.is_file())
    return {"image": [p for p in files if p.suffix.lower() in IMAGE_EXT],
            "video": [p for p in files if p.suffix.lower() in VIDEO_EXT],
            "audio": [p for p in files if p.suffix.lower() in AUDIO_EXT]}


def audio_window(model_dir: Path, clips: list) -> int:
    """The smallest audio encoder window (frames of 10 ms) that holds the longest clip."""
    manifest = json.loads((model_dir / "embeddinggemma2_manifest.json").read_text())
    windows = sorted((manifest.get("audio") or {}).get("windows", [1120]))
    longest = max(len(pp.load_audio(str(c))) if not isinstance(c, np.ndarray) else len(c) for c in clips)
    return next((w for w in windows if w * 0.01 >= longest / pp.AUDIO_SAMPLE_RATE), windows[-1])


def open_model(args, modalities: set[str], audio_frames: int | None = None) -> EmbeddingGemma2:
    return EmbeddingGemma2(args.model_dir, backend=args.backend, ort_threads=args.ort_threads,
                           modalities=tuple({"text"} | modalities), audio_frames=audio_frames)


def index_group(args, group: str) -> list[dict]:
    """Embed every file of one modality group. Returns index entries with an `embedding`."""
    media = list_media(Path(args.media_dir))
    entries = []
    if group == "audio":
        if not media["audio"]:
            return entries
        model = open_model(args, {"audio"}, audio_window(Path(args.model_dir), media["audio"]))
        for p in media["audio"]:
            emb = model.encode_audio(p)
            entries.append({"kind": "audio", "file": p.name, "embedding": emb})
            print(f"[audio] {p.name:40.40} {model.timings.get('audio_ms', 0):7.0f} ms audio encoder")
        return entries

    if not (media["image"] or media["video"]):
        return entries
    model = open_model(args, {"image", "video"})
    for p in media["image"]:
        emb = model.encode_image(p)
        entries.append({"kind": "image", "file": p.name, "embedding": emb})
        print(f"[image] {p.name:40.40} {model.timings.get('vision_ms', 0):7.0f} ms vision encoder")
    for p in media["video"]:
        frames, times = pp.sample_video_frames(str(p), fps=args.moment_fps, max_frames=args.max_moments,
                                               return_times=True)
        # Each frame runs through the vision encoder once; its soft tokens give the moment
        # embedding, and a uniform selection of them gives the whole-video embedding.
        feats = [model.vision_tokens(f) for f in frames]
        for t, f in zip(times, feats):
            entries.append({"kind": "moment", "file": p.name, "time": round(float(t), 1),
                            "embedding": model.embed_ids(pp.image_ids(f.shape[0]), f)})
        pick = np.linspace(0, len(feats) - 1, num=min(len(feats), model.max_frames)).round().astype(int)
        clip = [feats[i] for i in pick]
        entries.append({"kind": "video", "file": p.name,
                        "embedding": model.embed_ids(pp.video_ids(len(clip), clip[0].shape[0]), np.concatenate(clip))})
        print(f"[video] {p.name:40.40} {len(frames):3d} moments")
    return entries


def save_entries(entries: list[dict], path: Path):
    np.savez(path, embeddings=np.stack([e.pop("embedding") for e in entries]).astype(np.float32),
             meta=json.dumps(entries))


def load_entries(path: Path) -> tuple[np.ndarray, list[dict]]:
    data = np.load(path)
    return data["embeddings"], json.loads(str(data["meta"]))


def cmd_index(args):
    media_dir = Path(args.media_dir)
    if args.fetch_samples:
        fetch_samples(media_dir)
    index = Path(args.index)
    if args.group:
        # child process: one modality group
        save_entries(index_group(args, args.group), index)
        return
    t0 = time.perf_counter()
    embeddings, meta = [], []
    for group in GROUPS:
        part = index.with_name(f"{index.stem}.{group}.npz")
        cmd = [sys.executable, __file__, "index", "-m", args.model_dir, "--media-dir", args.media_dir,
               "--index", str(part), "--backend", args.backend, "--ort-threads", str(args.ort_threads),
               "--moment-fps", str(args.moment_fps), "--max-moments", str(args.max_moments), "--group", group]
        subprocess.run(cmd, check=True)
        e, m = load_entries(part)
        if len(m):
            embeddings.append(e)
            meta += m
        part.unlink()
    np.savez(index, embeddings=np.concatenate(embeddings), meta=json.dumps(meta))
    kinds = {k: sum(1 for m in meta if m["kind"] == k) for k in ("image", "audio", "video", "moment")}
    print(f"[index] {index}: {kinds} in {time.perf_counter() - t0:.0f} s ({args.backend})")


def _label(m: dict) -> str:
    if m["kind"] == "moment":
        return f"{m['file']} @ {int(m['time']) // 60}:{int(m['time']) % 60:02d}"
    return m["file"]


def print_results(scores: np.ndarray, meta: list[dict], kinds: list[str], top: int, query_file: str = ""):
    for kind, title in (("image", "images"), ("audio", "sounds"), ("video", "videos"), ("moment", "video moments")):
        if kind not in kinds:
            continue
        idx = [i for i, m in enumerate(meta) if m["kind"] == kind and m["file"] != query_file]
        if kind == "moment":
            # best moment per video, so one long video does not fill the list
            best = {}
            for i in idx:
                f = meta[i]["file"]
                if f not in best or scores[i] > scores[best[f]]:
                    best[f] = i
            idx = list(best.values())
        idx = sorted(idx, key=lambda i: -scores[i])[:top]
        if idx:
            print(f"  {title}:")
            for i in idx:
                print(f"    {scores[i]:6.3f}  {_label(meta[i])}")


# Matches across media types are reliable from text; between two media types (a sound
# against photos) the scores are close to noise, so by default an image query returns
# images and video moments (frames are images) and a sound returns sounds.
DEFAULT_KINDS = {"text": "image,audio,video,moment", "image": "image,moment,video", "audio": "audio"}


def interactive(args, embeddings: np.ndarray, meta: list[dict]):
    """Load the text model once, then answer text queries typed at the prompt."""
    t0 = time.perf_counter()
    model = open_model(args, set())
    kinds = (args.kinds or DEFAULT_KINDS["text"]).split(",")
    print(f"[search] {len(meta)} entries from {args.index}, model ready in {time.perf_counter() - t0:.0f} s. "
          "Type a query; 'q', Ctrl-D or Ctrl-C to quit.", flush=True)
    try:
        while True:
            try:
                text = input("\nsearch> ").strip()
            except EOFError:
                print()
                break
            if text in ("q", "quit", "exit"):
                break
            if not text:
                continue
            t = time.perf_counter()
            scores = embeddings @ model.encode_text(text, "query")
            print(f"query: text {text!r}  ({(time.perf_counter() - t) * 1e3:.0f} ms)\n")
            print_results(scores, meta, kinds, args.top)
    except KeyboardInterrupt:
        print()
    finally:
        model.close()
        print("[search] model released", flush=True)


def cmd_search(args):
    if not Path(args.index).exists():
        sys.exit(f"No index at {args.index}. Build one first (indexes the example media):\n"
                 f"  python src/media_search.py index")
    embeddings, meta = load_entries(Path(args.index))
    if args.mode == "startup":
        interactive(args, embeddings, meta)
        return
    if args.text:
        model = open_model(args, set())
        q, what = model.encode_text(args.text, "query"), f"text {args.text!r}"
    elif args.image:
        model = open_model(args, {"image"})
        q, what = model.encode_image(args.image), f"image {Path(args.image).name}"
    else:
        model = open_model(args, {"audio"}, audio_window(Path(args.model_dir), [args.audio]))
        q, what = model.encode_audio(args.audio), f"audio {Path(args.audio).name}"
    ms = sum(v for k, v in model.timings.items() if k.endswith("_ms"))
    print(f"\nquery: {what}  ({ms:.0f} ms to embed)\n")
    kinds = (args.kinds or DEFAULT_KINDS["text" if args.text else "image" if args.image else "audio"]).split(",")
    print_results(embeddings @ q, meta, kinds, args.top, Path(args.image or args.audio or "").name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("index", "search"):
        p = sub.add_parser(name)
        p.add_argument("-m", "--model-dir", help="Directory with the vmfbs/onnx graphs and assets (default: the files setup_demos.py downloaded)")
        p.add_argument("--backend", choices=["torq", "ort"], default="torq")
        p.add_argument("--ort-threads", type=int, default=0)
        p.add_argument("--index", default="media_index.npz", help="Index file (default: %(default)s)")
    ix = sub.choices["index"]
    ix.add_argument("--media-dir", help="Folder of images, videos and .wav sounds (default: the example media "
                                         "setup_demos.py downloaded, <model dir>/examples)")
    ix.add_argument("--fetch-samples", action="store_true", help=argparse.SUPPRESS)
    ix.add_argument("--moment-fps", type=float, default=1.0, help="Video moments per second (default: %(default)s)")
    ix.add_argument("--max-moments", type=int, default=60, help="Moments per video at most (default: %(default)s)")
    ix.add_argument("--group", choices=list(GROUPS), help=argparse.SUPPRESS)
    se = sub.choices["search"]
    se.add_argument("mode", nargs="?", choices=["startup"],
                    help="'startup': load the model once and answer text queries typed at a prompt "
                         "(~0.4 s each)")
    q = se.add_mutually_exclusive_group()
    q.add_argument("--text", help="One text query")
    q.add_argument("--image", help="An image as the query")
    q.add_argument("--audio", help="A sound clip as the query")
    se.add_argument("--top", type=int, default=3, help="Results per kind (default: %(default)s)")
    se.add_argument("--kinds", help="Result kinds to show, from image,audio,video,moment (default: all for a "
                                    "text query, image,moment,video for an image, audio for a sound)")
    args = ap.parse_args()
    if args.cmd == "search" and (args.mode == "startup") == bool(args.text or args.image or args.audio):
        se.error("use `search startup` for the interactive prompt, or one of --text/--image/--audio")
    args.model_dir = resolve_model_dir(ap, args.model_dir)
    if args.cmd == "index" and not args.media_dir:
        args.media_dir = "media" if args.fetch_samples else str(Path(args.model_dir) / "examples")
        if not args.fetch_samples and not Path(args.media_dir).is_dir():
            ap.error(f"no example media in {args.media_dir}; run `python setup_demos.py EmbeddingGemma-2` "
                     "from the torq-examples root, or pass --media-dir")
    cmd_index(args) if args.cmd == "index" else cmd_search(args)


if __name__ == "__main__":
    main()
