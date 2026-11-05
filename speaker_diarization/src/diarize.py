# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Speaker diarization + transcription demo (RedimNet + Moonshine streaming).

Answers "who spoke when, and what did they say" for a WAV file:

  1. An energy VAD finds the speech regions.
  2. RedimNet (on the NPU) embeds overlapping windows of each region.
  3. The window embeddings are clustered into speakers, and each region is
     split into speaker turns.
  4. Moonshine (on the NPU) transcribes each turn.

RedimNet runs in a helper process (``embed.py``), because its VMFB can need a
different Torq runtime than the Moonshine VMFBs. The NPU serves one process at
a time, so the helper finishes before Moonshine loads.
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "moonshine_streaming" / "src"))

import numpy as np
import soundfile as sf
from tokenizers import Tokenizer

from infer import EnergyVAD, resample  # noqa: E402 (moonshine_streaming/src)
from runner import MoonshineStaticStreamingModel, find_asset  # noqa: E402
from segments import Turn, build_turns, cluster, plan_windows, speech_regions  # noqa: E402
from moonshine_streaming.setup_demo import local_moonshine_streaming_model_dir  # noqa: E402
from utils.inference import add_common_inference_args  # noqa: E402
from utils.log import configure_logging  # noqa: E402
from utils.npu import configure_npu_userspace_frequency  # noqa: E402

logger = logging.getLogger("speaker_diarization")

SAMPLE_RATE = 16000
VAD_CHUNK   = 1280   # 80 ms, the same as the Moonshine chunk
SPEAKER_COLORS = ["\033[36m", "\033[33m", "\033[35m", "\033[32m", "\033[34m", "\033[31m"]


# ── Audio ────────────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Read a WAV file as 16 kHz float32 mono."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return resample(data.astype(np.float32), sr, SAMPLE_RATE)


def _fmt(samples: int) -> str:
    sec = samples / SAMPLE_RATE
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


# ── Speaker embeddings (helper process) ──────────────────────────────────────

def _embed_env(pythonpath: str) -> dict:
    """Environment for the helper process. An empty ``pythonpath`` removes
    PYTHONPATH, so that the helper uses the runtime installed in the venv."""
    env = dict(os.environ)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    else:
        env.pop("PYTHONPATH", None)
    return env


def compute_embeddings(audio: np.ndarray, starts: np.ndarray, win: int,
                       args: argparse.Namespace) -> np.ndarray:
    """Run ``embed.py`` on the windows that start at ``starts`` and return one
    embedding per window."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.save(tmp / "audio.npy", audio)
        np.save(tmp / "starts.npy", starts)
        cmd = [
            args.embed_python, str(Path(__file__).with_name("embed.py")),
            "--vmfb", args.embed_vmfb, "--audio", str(tmp / "audio.npy"),
            "--starts", str(tmp / "starts.npy"), "--out", str(tmp / "emb.npy"),
            "--window", str(win), "--tda", args.tda,
        ]
        subprocess.run(cmd, env=_embed_env(args.embed_pythonpath), check=True,
                       stdout=subprocess.DEVNULL if not args.verbose else None)
        return np.load(tmp / "emb.npy")


def find_turns(audio: np.ndarray, args: argparse.Namespace) -> list[Turn]:
    """VAD, RedimNet embeddings and clustering: the speaker turns of ``audio``."""
    vad     = EnergyVAD(threshold=args.vad_threshold, silence_duration=args.vad_silence)
    regions = speech_regions(audio, vad, VAD_CHUNK, lead_in=SAMPLE_RATE)
    if not regions:
        return []
    win    = int(args.embed_window * SAMPLE_RATE)
    starts = plan_windows(regions, win, int(args.embed_hop * SAMPLE_RATE), len(audio))
    audio  = np.pad(audio, (0, max(win - len(audio), 0)))
    embs   = compute_embeddings(audio, starts, win, args)
    labels = cluster(embs, args.cluster_threshold, args.num_speakers)
    turns  = build_turns(regions, starts, labels, win, VAD_CHUNK, int(args.min_turn * SAMPLE_RATE))
    logger.info("%d speech regions, %d embedding windows, %d speakers",
                len(regions), len(starts), max(t.speaker for t in turns) + 1)
    return turns


# ── Transcription ────────────────────────────────────────────────────────────

def _chunk_at(audio: np.ndarray, pos: int, size: int) -> np.ndarray:
    """``size`` samples from ``pos``; zero-padded outside ``audio``."""
    lo, hi = max(pos, 0), min(pos + size, len(audio))
    out = np.zeros(size, dtype=np.float32)
    if hi > lo:
        out[lo - pos:hi - pos] = audio[lo:hi]
    return out


def _warm_up(model, state, audio: np.ndarray, pos: int):
    """Feed the warm-up chunks that come just before ``pos`` into a fresh
    state. The encoder discards their output, but they fill its conv and
    lookahead buffers, so the first words of the turn are not clipped."""
    for i in range(model.warmup_chunks, 0, -1):
        model.process_audio_chunk(state, _chunk_at(audio, pos - i * model.chunk_len, model.chunk_len))


def _finish(model, state, tokenizer) -> str:
    """Flush the encoder lookahead and decode the buffered audio."""
    model.encode(state, is_final=True)
    return tokenizer.decode(model.decode(state), skip_special_tokens=True).strip()


def _quietest_cut(audio: np.ndarray, start: int, max_chunks: int, chunk: int) -> int:
    """The chunk boundary in the second half of ``max_chunks`` chunks after
    ``start`` with the least energy around it, so a cut falls between words."""
    def energy(k):
        mid = start + k * chunk
        return float(np.mean(audio[mid - chunk // 2:mid + chunk // 2] ** 2))
    return start + chunk * min(range(max_chunks // 2, max_chunks + 1), key=energy)


def _pieces(audio: np.ndarray, turn: Turn, model) -> list[tuple[int, int]]:
    """Split a turn into pieces that fit in the Moonshine cross-KV memory
    (``max_memory_len`` frames, minus the frames of the final flush)."""
    max_chunks = model.max_memory_len // model.feature_stride - model.warmup_chunks
    pieces, pos = [], turn.start
    while turn.end - pos > max_chunks * model.chunk_len:
        cut = _quietest_cut(audio, pos, max_chunks, model.chunk_len)
        pieces.append((pos, cut))
        pos = cut
    pieces.append((pos, turn.end))
    return pieces


def _transcribe_piece(model, state, tokenizer, audio: np.ndarray, start: int, end: int) -> str:
    state.reset()
    _warm_up(model, state, audio, start)
    for pos in range(start, end, model.chunk_len):
        model.process_audio_chunk(state, _chunk_at(audio, pos, model.chunk_len))
    return _finish(model, state, tokenizer)


def transcribe_turn(model, state, tokenizer, audio: np.ndarray, turn: Turn) -> str:
    """Transcribe one turn, in pieces when it is longer than Moonshine can hold."""
    texts = [_transcribe_piece(model, state, tokenizer, audio, start, end)
             for start, end in _pieces(audio, turn, model)]
    return " ".join(t for t in texts if t)


def load_asr(args: argparse.Namespace):
    runtime_flags = [f"--torq_device_allocator={args.tda}"] + (args.runtime_flags or [])
    model = MoonshineStaticStreamingModel(args.model_dir, runtime_flags=runtime_flags,
                                          device_io=args.device_io or args.tda == "dmabuf")
    tokenizer = Tokenizer.from_file(find_asset(model.model_dir, "tokenizer.json"))
    return model, tokenizer


def print_turn(turn: Turn, text: str):
    color = SPEAKER_COLORS[turn.speaker % len(SPEAKER_COLORS)]
    print(f"[{_fmt(turn.start)} - {_fmt(turn.end)}] {color}Speaker {turn.speaker + 1}\033[0m: "
          f"{text if text else '(no words)'}", flush=True)


# ── Entry point ──────────────────────────────────────────────────────────────

def run(args: argparse.Namespace):
    audio = load_audio(args.wav)
    logger.info("Diarizing %s (%.1fs)", args.wav, len(audio) / SAMPLE_RATE)

    t0    = time.perf_counter()
    turns = find_turns(audio, args)
    t1    = time.perf_counter()
    if not turns:
        print("No speech found.")
        return

    model, tokenizer = load_asr(args)
    state = model.create_state()
    t2    = time.perf_counter()
    for turn in turns:
        print_turn(turn, transcribe_turn(model, state, tokenizer, audio, turn))
    t3    = time.perf_counter()

    if args.verbose:
        audio_s = len(audio) / SAMPLE_RATE
        print(f"\n[verbose] diarization {t1 - t0:.1f}s (incl. RedimNet load), "
              f"transcription {t3 - t2:.1f}s, RTF {(t1 - t0 + t3 - t2) / audio_s:.2f}x "
              f"over {audio_s:.1f}s audio", file=sys.stderr)


def main(args: argparse.Namespace):
    if not os.path.isfile(args.wav):
        sys.exit(f"WAV file {args.wav} not found.")
    ok, message = configure_npu_userspace_frequency("max")
    print(f"[NPU] {message}")
    try:
        run(args)
    finally:
        ok, message = configure_npu_userspace_frequency("min")
        print(f"[NPU] {message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker diarization + transcription (RedimNet + Moonshine)")
    parser.add_argument("--wav",            required=True,                  help="WAV file to diarize and transcribe")
    parser.add_argument("--embed-vmfb",     required=True,                  help="RedimNet speaker-embedding VMFB")
    parser.add_argument("-m", "--model-dir", default=None, metavar="DIR",   help="Moonshine streaming model dir (default: the one setup_demos.py downloaded)")
    parser.add_argument("--num-speakers",   type=int,   default=None,       help="Number of speakers, if known (default: found with --cluster-threshold)")
    parser.add_argument("--cluster-threshold", type=float, default=0.4,     help="Cosine similarity above which clusters merge (default: %(default)s)")
    parser.add_argument("--embed-window",   type=float, default=2.0,        help="Embedding window in seconds; must match the RedimNet VMFB (default: %(default)s)")
    parser.add_argument("--embed-hop",      type=float, default=0.5,        help="Step between embedding windows in seconds (default: %(default)s)")
    parser.add_argument("--min-turn",       type=float, default=1.0,        help="Shortest speaker turn in seconds; shorter ones join a neighbour (default: %(default)s)")
    parser.add_argument("--vad-threshold",  type=float, default=0.01,       help="RMS floor for the energy VAD (default: %(default)s)")
    parser.add_argument("--vad-silence",    type=float, default=0.5,        help="Pause in seconds that ends a speech region (default: %(default)s)")
    parser.add_argument("--embed-python",   default=sys.executable,         help="Python for the RedimNet helper process (default: this one)")
    parser.add_argument("--embed-pythonpath", default="",                   help="PYTHONPATH for the helper process; empty uses the venv runtime (default: empty)")
    parser.add_argument("--verbose",        action="store_true",            help="Print helper output and a timing summary")
    add_common_inference_args(parser, with_threads=False, device_io_default=False,
                              device_io_help="Use device-backed inputs and outputs (enabled automatically with --tda dmabuf)")
    args = parser.parse_args()
    configure_logging(args.logging)
    if args.model_dir is None:
        model_dir = local_moonshine_streaming_model_dir()
        if model_dir is None:
            parser.error("no local Moonshine streaming model found; pass -m/--model-dir or run "
                         "`python setup_demos.py moonshine_streaming` from torq-examples root")
        args.model_dir = str(model_dir)
    main(args)
