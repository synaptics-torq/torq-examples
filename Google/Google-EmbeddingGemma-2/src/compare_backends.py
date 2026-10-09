# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Compare EmbeddingGemma-2 backends on a fixed eval set (accuracy + performance).

Run each backend in its own process (so memory numbers are per backend), then report:

    python src/compare_backends.py run -m <torq_dir> --backend torq -o torq.npz
    python src/compare_backends.py run -m <ort_dir>  --backend ort --ort-threads 2 -o ort.npz
    python src/compare_backends.py report torq.npz ort.npz -o report.md

``run`` embeds every eval item (texts, images, videos), stores the embeddings, the
image soft tokens and per-stage timings, then times ``--repeats`` runs of one item
per path (text S=128, document S=512, image, video). ``report`` compares two runs:
per-item cosine per modality and Matryoshka dimension, retrieval agreement, STS
score agreement, latency and memory.
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp  # noqa: E402
from model_dir import resolve_model_dir  # noqa: E402
from runner import EmbeddingGemma2  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent / "eval" / "eval_set.json"


def _media(media_dir: Path, repo: str, name: str) -> Path:
    p = media_dir / name
    if not p.exists():
        from huggingface_hub import hf_hub_download

        media_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo, name, repo_type="dataset", local_dir=media_dir)
    return p


def _mem() -> dict:
    out = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            k, _, v = line.partition(":")
            if k in ("VmHWM", "VmRSS", "RssAnon", "RssFile"):
                out[k] = int(v.split()[0]) / 1024.0  # MB
    except OSError:
        pass
    return out


def _texts(ev: dict) -> list[tuple[str, str, str | None]]:
    """(id, text, prompt) for every text item."""
    items = []
    for i, r in enumerate(ev["retrieval"]):
        items.append((f"rq{i}", r["query"], "query"))
        items.append((f"rd{i}", r["document"], "document"))
    for i, (a, b) in enumerate(ev["sts"]):
        items.append((f"sa{i}", a, "STS"))
        items.append((f"sb{i}", b, "STS"))
    for i, d in enumerate(ev["long_documents"]):
        items.append((f"ld{i}", d, "document"))
    for i, it in enumerate(ev["images"]):
        items.append((f"iq{i}", it["query"], "query"))
    for i, it in enumerate(ev["videos"]):
        items.append((f"vq{i}", it["query"], "query"))
    for i, it in enumerate(ev.get("audios", [])):
        items.append((f"aq{i}", it["query"], "query"))
    return items


def cmd_run(args):
    ev = json.loads(Path(args.eval).read_text())
    media_dir = Path(args.media_dir)
    cpu0, wall0 = os.times(), time.perf_counter()
    run_mods = set(args.modalities.split(",")) | ({"audio"} if args.audio else set())
    model = EmbeddingGemma2(args.model_dir, backend=args.backend, ort_threads=args.ort_threads,
                            modalities=tuple(run_mods))
    res: dict[str, np.ndarray] = {}
    timing: dict[str, dict] = {}

    mods = set(args.modalities.split(","))
    if args.audio:
        mods.add("audio")
    for tid, text, prompt in _texts(ev):
        if "text" not in mods and not (tid.startswith("aq") and "audio" in mods):
            continue
        res[f"text/{tid}"] = model.encode_text(text, prompt)
        timing[f"text/{tid}"] = dict(model.timings)
    for i, it in enumerate(ev["images"] if "image" in mods else []):
        p = _media(media_dir, ev["media_repo"], it["file"])
        res[f"image/{i}"] = model.encode_image(p)
        timing[f"image/{i}"] = dict(model.timings)
        from PIL import Image

        res[f"vision_tokens/{i}"] = model.vision_tokens(Image.open(p))
    for i, it in enumerate(ev["videos"] if "video" in mods else []):
        p = _media(media_dir, ev["media_repo"], it["file"])
        res[f"video/{i}"] = model.encode_video(p)
        timing[f"video/{i}"] = dict(model.timings)

    for i, it in enumerate(ev.get("audios", []) if "audio" in mods else []):
        p = _media(media_dir, ev["media_repo"], it["file"])
        res[f"audio/{i}"] = model.encode_audio(p)
        timing[f"audio/{i}"] = dict(model.timings)

    # Latency: repeat one item per path after warm-up.
    from PIL import Image

    paths = {}
    if "text" in mods:
        paths["text_s128"] = lambda: model.encode_text(ev["retrieval"][0]["query"], "query")
        paths["text_s512"] = lambda: model.encode_text(ev["long_documents"][0], "document")
    if "image" in mods:
        img = Image.open(_media(media_dir, ev["media_repo"], ev["images"][0]["file"]))
        paths["image"] = lambda: model.encode_image(img)
    if "video" in mods:
        frames = pp.sample_video_frames(str(_media(media_dir, ev["media_repo"], ev["videos"][0]["file"])),
                                        fps=1.0, max_frames=model.max_frames)
        paths["video_7f"] = lambda: model.encode_frames(frames)
    if "audio" in mods and ev.get("audios"):
        wav = pp.load_audio(str(_media(media_dir, ev["media_repo"], ev["audios"][0]["file"])))
        paths["audio_11s"] = lambda: model.encode_audio(wav)
    lat = {}
    for name, fn in paths.items():
        fn()
        ts, stages = [], []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
            stages.append(dict(model.timings))
        lat[name] = {"total_ms": ts, "stages": stages}

    cpu1, wall1 = os.times(), time.perf_counter()
    meta = {
        "backend": args.backend, "ort_threads": args.ort_threads, "model_dir": str(args.model_dir),
        "timing": timing, "latency": lat, "memory_mb": _mem(),
        "cpu_util": ((cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)) / (wall1 - wall0),
        "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    np.savez(args.output, meta=json.dumps(meta), **{k.replace("/", "__"): v for k, v in res.items()})
    print(f"saved {args.output}: {len(res)} arrays, peak RSS {meta['memory_mb'].get('VmHWM', 0):.0f} MB, "
          f"cpu util {meta['cpu_util']:.2f}")


def _load(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    arrs = {k.replace("__", "/"): z[k] for k in z.files if k != "meta"}
    return meta, arrs


def _norm(x, d):
    x = x[..., :d]
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _stats(v):
    v = np.asarray(v, dtype=np.float64)
    return f"{v.mean():.1f} / {np.percentile(v, 50):.1f} / {np.percentile(v, 90):.1f}"


def cmd_report(args):
    (ma, A), (mb, B) = _load(args.a), _load(args.b)
    ev = json.loads(Path(args.eval).read_text())
    la, lb = f"{ma['backend']}", f"{mb['backend']}" + (f" ({mb['ort_threads']}t)" if mb["ort_threads"] else "")
    lines = [f"# EmbeddingGemma-2: {la} vs {lb}", ""]

    # --- per-item cosine
    lines += ["## Embedding agreement (cosine, per item)", "",
              "| Modality | Dim | Mean | Min | Worst 5 |", "|---|---|---|---|---|"]
    for mod in ("text", "image", "video", "audio"):
        keys = sorted(k for k in A if k.startswith(mod + "/") and k in B)
        if not keys:
            continue
        for d in pp.MRL_DIMS:
            c = np.array([float(_norm(A[k], d) @ _norm(B[k], d)) for k in keys])
            worst = ", ".join(f"{x:.4f}" for x in np.sort(c)[:5])
            lines.append(f"| {mod} ({len(keys)}) | {d} | {c.mean():.5f} | {c.min():.5f} | {worst} |")
    vt = [k for k in A if k.startswith("vision_tokens/") and k in B]
    if vt:
        c = np.concatenate([(_norm(A[k], 512) * _norm(B[k], 512)).sum(-1) for k in vt])
        lines += ["", f"Vision soft tokens ({len(vt)} images x 64): token cos mean {c.mean():.5f}, min {c.min():.5f}"]

    # --- retrieval
    def retrieval(qkeys, ikeys, arrs, d):
        Q = np.stack([_norm(arrs[k], d) for k in qkeys]); I = np.stack([_norm(arrs[k], d) for k in ikeys])
        return Q @ I.T

    lines += ["", "## Retrieval (query -> item)", "",
              f"| Task | Dim | top-1 acc {la} | top-1 acc {lb} | top-1 agreement | top-5 overlap | max abs score diff |",
              "|---|---|---|---|---|---|---|"]
    tasks = {
        "text->doc": ([f"text/rq{i}" for i in range(len(ev["retrieval"]))],
                      [f"text/rd{i}" for i in range(len(ev["retrieval"]))]),
        "text->image": ([f"text/iq{i}" for i in range(len(ev["images"]))],
                        [f"image/{i}" for i in range(len(ev["images"]))]),
        "text->video": ([f"text/vq{i}" for i in range(len(ev["videos"]))],
                        [f"video/{i}" for i in range(len(ev["videos"]))]),
    }
    if all(f"audio/{i}" in A and f"audio/{i}" in B for i in range(len(ev.get("audios", [])))) and ev.get("audios"):
        tasks["text->audio"] = ([f"text/aq{i}" for i in range(len(ev["audios"]))],
                                [f"audio/{i}" for i in range(len(ev["audios"]))])
    for name, (qk, ik) in tasks.items():
        if not all(k in A and k in B for k in qk + ik):
            continue
        for d in pp.MRL_DIMS:
            sa, sb = retrieval(qk, ik, A, d), retrieval(qk, ik, B, d)
            ta, tb = sa.argmax(1), sb.argmax(1)
            gt = np.arange(len(qk))
            k = min(5, sa.shape[1])
            top5 = np.mean([len(set(np.argsort(-sa[i])[:k]) & set(np.argsort(-sb[i])[:k])) / k for i in range(len(qk))])
            lines.append(f"| {name} | {d} | {np.mean(ta == gt):.2f} | {np.mean(tb == gt):.2f} | "
                         f"{np.mean(ta == tb):.2f} | {top5:.2f} | {np.abs(sa - sb).max():.4f} |")

    # --- STS
    if not all(f"text/sa{i}" in A and f"text/sa{i}" in B for i in range(len(ev["sts"]))):
        ev = dict(ev, sts=[])
    sa = np.array([float(_norm(A[f"text/sa{i}"], 768) @ _norm(A[f"text/sb{i}"], 768)) for i in range(len(ev["sts"]))])
    sb = np.array([float(_norm(B[f"text/sa{i}"], 768) @ _norm(B[f"text/sb{i}"], 768)) for i in range(len(ev["sts"]))])
    if len(sa) > 1:
        lines += ["", f"STS pairs ({len(sa)}): Spearman {la} vs {lb} = {_spearman(sa, sb):.4f}, "
                  f"max abs score diff {np.abs(sa - sb).max():.4f}"]

    # --- performance
    lines += ["", "## Latency (ms, mean / p50 / p90)", "",
              f"| Path | {la} total | {lb} total | speedup | {la} stages | {lb} stages |", "|---|---|---|---|---|---|"]

    def stage_str(stages):
        keys = [k for k in stages[0] if k.endswith("_ms")]
        return ", ".join(f"{k[:-3]} {np.mean([s.get(k, 0) for s in stages]):.1f}" for k in keys)

    for path in [p for p in ma["latency"] if p in mb["latency"]]:
        ta_, tb_ = ma["latency"][path]["total_ms"], mb["latency"][path]["total_ms"]
        lines.append(f"| {path} | {_stats(ta_)} | {_stats(tb_)} | {np.mean(tb_) / np.mean(ta_):.2f}x | "
                     f"{stage_str(ma['latency'][path]['stages'])} | {stage_str(mb['latency'][path]['stages'])} |")
    lines += ["", "## Memory / CPU", "", f"| | {la} | {lb} |", "|---|---|---|"]
    for k in ("VmHWM", "RssAnon", "RssFile"):
        lines.append(f"| {k} (MB) | {ma['memory_mb'].get(k, 0):.0f} | {mb['memory_mb'].get(k, 0):.0f} |")
    lines.append(f"| CPU utilisation (cores busy) | {ma['cpu_util']:.2f} | {mb['cpu_util']:.2f} |")

    text = "\n".join(lines) + "\n"
    Path(args.output).write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("-m", "--model-dir", help="Directory with the vmfbs/onnx graphs and assets (default: the files setup_demos.py downloaded)")
    r.add_argument("--backend", choices=["torq", "ort"], required=True)
    r.add_argument("--ort-threads", type=int, default=0)
    r.add_argument("--eval", default=str(EVAL))
    r.add_argument("--media-dir", default="eval_media")
    r.add_argument("--repeats", type=int, default=20)
    r.add_argument("--audio", action="store_true", help="Also embed the audio eval items")
    r.add_argument("--modalities", default="text,image,video",
                   help="Comma-separated subset of text,image,video,audio to run (default: %(default)s)")
    r.add_argument("-o", "--output", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("a")
    rp.add_argument("b")
    rp.add_argument("--eval", default=str(EVAL))
    rp.add_argument("-o", "--output", default="report.md")
    args = ap.parse_args()
    if args.cmd == "run":
        args.model_dir = resolve_model_dir(ap, args.model_dir)
    cmd_run(args) if args.cmd == "run" else cmd_report(args)


if __name__ == "__main__":
    main()
