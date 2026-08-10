# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Piper (VITS) text-to-speech on Torq: writes a .wav and plays it on the speaker.

partA — the text encoder and duration predictor — runs on the CPU under
onnxruntime and yields the exact frame count; the HiFi-GAN vocoder (partB) runs
on the NPU as one of five bf16 NSS-only vmfbs covering 1/2/4/6/8 s windows. The
two overlap, so the CPU encodes the next sentence while the NPU vocodes the
current one and the speaker plays the previous one. Assets download from Hugging
Face (``Synaptics/Piper-TTS``) on first run.
"""

import argparse
import sys
from pathlib import Path

from piper_tts.piper_core.phonemize import Phonemizer
from piper_tts.piper_core.pipeline import PiperTTS
from piper_tts.setup_demo import ensure_piper_models
from utils.npu import enable_npu_clock

SAMPLES = [
    "The morning train was late again. Nobody on the platform seemed surprised.",
    "Rain fell steadily on the harbour road. The ferry would not sail until morning, "
    "and the lamps along the quay came on one by one.",
    "She opened the wooden gate and crossed the wet grass. Below the cliff, the grey "
    "water moved slowly against the rocks.",
    "The bakery on the corner opens at six. By seven the shelves are half empty, and "
    "by nine there is nothing left but rye.",
    "At midnight the lighthouse changed its rhythm. Three short flashes, then a long "
    "pause, exactly as the old keeper had promised.",
]


def report(text, stats, quiet=False):
    """Print one line of timing per utterance."""
    if quiet:
        return
    windows = "+".join(f"{w:.0f}s" for w in stats["windows"]) or "-"
    played = stats["first_sound_s"] is not None
    first = stats["first_sound_s"] if played else (stats["first_audio_s"] or 0.0)
    print(f'  "{text[:64]}{"..." if len(text) > 64 else ""}"')
    print(f"  audio {stats['audio_s']:.2f} s | windows {windows} | "
          f"{'first sound' if played else 'first audio'} {first:.2f} s | "
          f"compute {stats['compute_s']:.2f} s ({stats['rtf']:.2f}x real time)"
          + (f" | saved {stats['wav']}" if stats.get("wav") else ""))


def run_once(tts, phon, text, out_path, play, quiet):
    sentences = phon(text)
    if not sentences:
        print("  nothing to synthesize", file=sys.stderr)
        return False
    on_skip = lambda i, secs: print(f"  !! sentence {i + 1} is {secs:.1f} s of speech, longer than the "
                                    f"{tts.max_seconds:.0f} s window — skipped; try shorter sentences.")
    _, stats = tts.synthesize(sentences, out_path, play=play, on_skip=on_skip)
    report(text, stats, quiet)
    return True


def interactive(tts, phon, out_dir, play, quiet):
    """Menu loop: pick a sample or type your own text; 'q' quits."""
    while True:
        print("\n=== Piper TTS (CPU partA || NPU partB) ===\n   0) type your own text")
        for i, s in enumerate(SAMPLES):
            print(f"  {i + 1:2d}) {s[:66]}...")
        try:
            choice = input(f"Select 0-{len(SAMPLES)}, q to quit: ").strip().lower()
        except EOFError:
            break
        if choice in ("q", "quit", "exit"):
            break
        if not choice.isdigit() or not 0 <= int(choice) <= len(SAMPLES):
            print(f"  invalid: {choice}")
            continue
        k = int(choice)
        if k:
            text, out = SAMPLES[k - 1], Path(out_dir) / f"sample_{k}.wav"
        else:
            try:
                text = input("Your text: ").strip()
            except EOFError:
                break
            if not text:
                print("  empty text")
                continue
            out = Path(out_dir) / "custom.wav"
        print()
        run_once(tts, phon, text, out, play, quiet)
    print("bye")


def main():
    p = argparse.ArgumentParser(description="Piper text-to-speech on Torq (NPU vocoder).")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", help="Text to speak (default: the first built-in sample).")
    src.add_argument("--file", help="Read the text to speak from a file.")
    src.add_argument("--sample", type=int, metavar="N", help=f"Speak built-in sample 1-{len(SAMPLES)}.")
    src.add_argument("--interactive", action="store_true", help="Menu loop: pick a sample or type text.")
    p.add_argument("--list-samples", action="store_true", help="Print the built-in samples and exit.")
    p.add_argument("--output", default=None, help="Output wav (default: tts_out.wav, or out/ when interactive).")
    p.add_argument("--no-play", action="store_true", help="Only write the wav; do not play it.")
    p.add_argument("--audio-device", default=None, help="ALSA device (default: autodetected USB DAC).")
    p.add_argument("--dac-rate", type=int, default=48000, help="Playback rate the DAC accepts (default: %(default)s).")
    p.add_argument("--length-scale", type=float, default=1.0, help="Phoneme duration scale (>1 speaks slower).")
    p.add_argument("--speaker", type=int, default=0, help="Speaker id, 0-903 (default: %(default)s).")
    p.add_argument("--threads", type=int, default=2, help="ORT threads for partA (default: %(default)s).")
    p.add_argument("--model-dir", default=None, help="Asset dir (default: models/Synaptics/Piper-TTS).")
    p.add_argument("--device", default="torq", help="IREE device URI for the vocoder (default: %(default)s).")
    p.add_argument("--no-refresh", action="store_true", help="Skip the Hugging Face update check (offline).")
    p.add_argument("--quiet", action="store_true", help="Suppress per-utterance timing lines.")
    args = p.parse_args()

    if args.list_samples:
        for i, s in enumerate(SAMPLES):
            print(f"{i + 1:2d}) {s}")
        return

    model_dir = ensure_piper_models(args.model_dir, refresh=not args.no_refresh)
    ok, message = enable_npu_clock()
    print(f"[NPU] {message}")

    print("Loading (partA on CPU, partB windows on NPU, espeak resident)...")
    tts = PiperTTS(model_dir, device_uri=args.device, threads=args.threads, length_scale=args.length_scale,
                   speaker=args.speaker, audio_device=args.audio_device, dac_rate=args.dac_rate)
    phon = Phonemizer(model_dir)
    print(f"  partA {tts.load_s['partA']:.1f} s | partB {len(tts.windows)} windows "
          f"({', '.join(f'{w * 256 / 22050:.0f}s' for w in tts.sizes)}) {tts.load_s['partB']:.1f} s | "
          f"speaker {'(silent)' if args.no_play else tts.audio_device}")

    try:
        if args.interactive:
            out_dir = args.output or "out"
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            interactive(tts, phon, out_dir, not args.no_play, args.quiet)
        else:
            text = (Path(args.file).read_text().strip() if args.file
                    else SAMPLES[args.sample - 1] if args.sample else args.text or SAMPLES[0])
            if args.sample and not 1 <= args.sample <= len(SAMPLES):
                p.error(f"--sample must be 1-{len(SAMPLES)}")
            print()
            if not run_once(tts, phon, text, args.output or "tts_out.wav", not args.no_play, args.quiet):
                sys.exit(1)
    finally:
        phon.close(), tts.close()


if __name__ == "__main__":
    main()
