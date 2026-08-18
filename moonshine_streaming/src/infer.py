# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Moonshine streaming WAV file transcription demo (2-Split VMFB).

Transcribes a pre-recorded WAV file, using a self-calibrating energy VAD to
split it into utterances, with committed-prefix incremental decode for a
real-time live preview. See ``runner.py`` for the inference engine.
"""

import argparse
import logging
import os
import queue
import sys
import threading
import time
from collections import deque

import numpy as np

try:
    from tokenizers import Tokenizer
except ImportError:
    print("Error: tokenizers is not installed. Please run:", file=sys.stderr)
    print("  pip install tokenizers", file=sys.stderr)
    sys.exit(1)

from runner import MoonshineStaticStreamingModel, find_asset  # noqa: E402 (sibling import)
from moonshine_streaming.setup_demo import ensure_moonshine_streaming_models
from utils.log import add_logging_args, configure_logging
from utils.npu import configure_npu_userspace_frequency, enable_npu_clock

logger = logging.getLogger("moonshine_streaming")

# Sentinel put on the audio queue to mark end-of-input.
_END_OF_STREAM = object()

# ── VAD ───────────────────────────────────────────────────────────────────────

class _HangoverVAD:
    """
    Shared speech/silence endpointing: given a per-chunk score from a
    subclass, tracks speech_start/speech_end transitions with a fixed
    silence-duration hangover before ending an utterance. 
    """
    def __init__(self, threshold, silence_duration, sample_rate):
        self.threshold                = threshold
        self.silence_duration_samples = int(silence_duration * sample_rate)
        self.sample_rate              = sample_rate
        self.silence_counter          = 0
        self.is_speaking              = False
        self.last_score               = 0.0
        self.silence_remaining_sec    = 0.0

    def _score(self, audio_chunk) -> float:
        raise NotImplementedError

    def process_chunk(self, audio_chunk):
        score = self._score(audio_chunk)
        self.last_score = score

        is_speech = score > self.threshold
        if is_speech:
            self.silence_counter       = 0
            self.silence_remaining_sec = 0.0
            if not self.is_speaking:
                self.is_speaking = True
                return "speech_start"
            return "speech"
        else:
            if self.is_speaking:
                self.silence_counter += len(audio_chunk)
                remaining = max(0, self.silence_duration_samples - self.silence_counter)
                self.silence_remaining_sec = remaining / self.sample_rate
                if self.silence_counter >= self.silence_duration_samples:
                    self.is_speaking           = False
                    self.silence_counter        = 0
                    self.silence_remaining_sec  = 0.0
                    return "speech_end"
                return "speech"
            self.silence_remaining_sec = 0.0
            return "silence"


class EnergyVAD(_HangoverVAD):
    """
    Simple RMS energy-based voice activity detector for streaming.
    Self-calibrating: samples ambient noise during the first 12 chunks (~960 ms).
    """
    def __init__(self, threshold=0.015, silence_duration=2.5, sample_rate=16000,
                 report_calibration=False):
        super().__init__(threshold, silence_duration, sample_rate)
        self.base_threshold     = threshold
        self.report_calibration = report_calibration
        self.ambient_rms        = []
        self.calibrated         = False

    def _score(self, audio_chunk):
        return np.sqrt(np.mean(audio_chunk ** 2)) if len(audio_chunk) > 0 else 0.0

    def process_chunk(self, audio_chunk):
        if not self.calibrated:
            rms = self._score(audio_chunk)
            self.last_score = rms
            self.ambient_rms.append(rms)
            if len(self.ambient_rms) >= 12:
                mean_rms = np.mean(self.ambient_rms)
                std_rms  = np.std(self.ambient_rms)
                self.threshold = max(mean_rms + 4 * std_rms, self.base_threshold)
                if self.report_calibration:
                    print(
                        f"\n[VAD Calibration] Ambient Noise RMS: {mean_rms:.5f} "
                        f"(std: {std_rms:.5f}). Threshold set to: {self.threshold:.5f}",
                        file=sys.stderr,
                    )
                self.calibrated = True
            return "silence"
        return super().process_chunk(audio_chunk)



# ── Terminal renderer ─────────────────────────────────────────────────────────

class TerminalListener:
    """Minimal ANSI redraw: moves the cursor back up over the last drawn block
    and overwrites it in place, so the live preview updates on the same lines
    instead of scrolling. Assumes each line fits on one terminal row, which
    holds for the short status/preview text this demo prints."""
    def __init__(self):
        self.prev_lines = 0
        self._last_live_draw = 0.0

    def draw(self, text):
        if self.prev_lines:
            sys.stdout.write(f"\033[{self.prev_lines}A")
        sys.stdout.write("\r")
        # Paint the new frame over the old one (erasing first is what causes
        # visible flicker at the ~12.5 Hz this is called during speech).
        sys.stdout.write(text)
        sys.stdout.write("\033[J")
        sys.stdout.flush()
        self.prev_lines = text.count("\n")

    def draw_live(self, text, min_interval=0.1):
        """Throttled variant of draw() for the high-frequency live indicator,
        which is otherwise redrawn on every 80 ms audio chunk (~12.5 Hz) even
        when nothing meaningful changed. Skips the redraw if the previous one
        was less than min_interval ago."""
        now = time.monotonic()
        if now - self._last_live_draw < min_interval:
            return
        self._last_live_draw = now
        self.draw(text)

    def complete_line(self):
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.prev_lines = 0


# ── Utilities ─────────────────────────────────────────────────────────────────

def resample(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio
    duration           = len(audio) / orig_sr
    num_target_samples = int(duration * target_sr)
    indices            = np.linspace(0, len(audio) - 1, num_target_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# ── Verbose stats ────────────────────────────────────────────────────────────

class _Stats:
    """Accumulates just enough timing to report RTF and decoder throughput
    for --verbose; attribute mutation (not rebinding) so it needs no
    `nonlocal` from the nested worker closure."""
    def __init__(self):
        self.encode_ms    = 0.0
        self.decode_ms    = 0.0
        self.decode_steps = 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args: argparse.Namespace):
    configure_logging(args.logging)

    model_dir = args.model_dir

    # Refresh the local copy when the upstream HF revision has moved on, so a
    # stale model dir is repaired before the existence check below.
    ensure_moonshine_streaming_models(model_dir, refresh=not args.no_refresh)

    if not os.path.isdir(model_dir):
        logger.error("Model directory %s not found.", model_dir)
        sys.exit(1)

    if args.full_decode:
        logger.info("Decode mode:      full re-decode from BOS (baseline)")
    else:
        logger.info(
            "Decode mode:      incremental committed-prefix "
            "(LocalAgreement-%d, commit-delay %.1fs)",
            args.commit_agreement, args.commit_delay,
        )

    ok, message = enable_npu_clock()
    print(f"[NPU] {message}")
    ok, message = configure_npu_userspace_frequency("max")
    print(f"[NPU] {message}")

    try:
        model     = MoonshineStaticStreamingModel(model_dir,
                                                   runtime_flags=args.runtime_flags)
        tokenizer = Tokenizer.from_file(find_asset(model.model_dir, "tokenizer.json"))
    except Exception as e:
        logger.error("Error initializing models: %s", e)
        sys.exit(1)

    try:
        import soundfile as sf
    except ImportError:
        logger.error("soundfile is required. Install it with: pip install soundfile")
        sys.exit(1)
    if args.mic:
            input_sample_rate = 48000     # AS33980 (SR80) sample rate is 48kHz
            wav_audio = None
            logger.info("Transcribing from AS33980 microphone (live @ 48000 Hz)")
    else:
        if not os.path.isfile(args.wav):
            logger.error("WAV file %s not found.", args.wav)
            sys.exit(1)
        data, input_sample_rate = sf.read(args.wav, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        wav_audio = data.astype(np.float32)
        logger.info(
            "Transcribing WAV file:  %s (%.1fs @ %d Hz)",
            args.wav, len(wav_audio) / input_sample_rate, input_sample_rate,
        )

    audio_queue = queue.Queue()
    running     = True

    logger.info("VAD backend:      energy (self-calibrating, floor %.4f)", args.vad_threshold)
    vad = EnergyVAD(threshold=args.vad_threshold, silence_duration=args.vad_silence,
                        report_calibration=args.verbose)
    terminal = TerminalListener()
    state    = model.create_state()

    stats = _Stats() if args.verbose else None

    def worker():
        tokens              = []
        utterance_count     = 0
        resampled_buffer    = np.array([], dtype=np.float32)
        chunks_since_decode = 0

        # Pre-speech look-behind buffer: rolls over every "silence" chunk so that
        # when speech_start fires, we have a few chunks of real audio (ambient
        # noise + whatever soft onset the VAD hadn't crossed threshold on yet)
        # to replay instead of losing that window to the encoder's warmup
        # discard (see the replay below). Defaults to warmup_chunks so the
        # replay exactly covers the window that would otherwise be thrown away.
        lookback_chunks = args.vad_lookback if args.vad_lookback is not None else model.warmup_chunks
        lookback_buffer = deque(maxlen=max(lookback_chunks, 0))

        def _encode(chunk):
            if stats is not None:
                t0 = time.perf_counter()
            model.process_audio_chunk(state, chunk)
            model.encode(state, is_final=False)
            if stats is not None:
                stats.encode_ms += (time.perf_counter() - t0) * 1000

        def _decode():
            if stats is not None:
                t0 = time.perf_counter()
            if args.full_decode:
                tokens = model.decode(state)
            else:
                tokens = model.decode_incremental(state, args.commit_delay, args.commit_agreement)
            if stats is not None:
                stats.decode_ms    += (time.perf_counter() - t0) * 1000
                stats.decode_steps += state.last_decode_steps
            return tokens

        def _finalize(count):
            """Flush the encoder's remaining lookahead, decode the utterance in
            full, and lock the finalized line in place."""
            model.encode(state, is_final=True)
            text = tokenizer.decode(_decode(), skip_special_tokens=True)
            terminal.draw(f"\033[32m✓\033[0m Utterance #{count}: {text if text else '(empty)'}")
            terminal.complete_line()

        while running:
            try:
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is _END_OF_STREAM:
                # Input exhausted. Finalize whatever utterance is still in flight.
                if state.cross_kv_fill > 0:
                    terminal.draw(f"\033[34m◉\033[0m Utterance #{utterance_count}: processing...")
                    _finalize(utterance_count)
                audio_queue.task_done()
                break

            chunk_16k        = resample(chunk, input_sample_rate, 16000)
            resampled_buffer = np.concatenate([resampled_buffer, chunk_16k])

            chunk_size = model.chunk_len
            while len(resampled_buffer) >= chunk_size:
                audio_chunk_1280 = resampled_buffer[:chunk_size]
                resampled_buffer = resampled_buffer[chunk_size:]

                vad_status = vad.process_chunk(audio_chunk_1280)

                if vad_status == "speech_start":
                    state.reset()
                    tokens = []
                    chunks_since_decode = 0
                    utterance_count += 1
                    terminal.draw(f"\033[32m●\033[0m Utterance #{utterance_count}: [Listening...]")

                    # Replay the buffered pre-speech chunks through the fresh state
                    # before the triggering chunk below. chunk_idx is 0 right after
                    # reset(), so these calls land in the encoder's warmup window
                    # (their cross-KV output is discarded either way, see
                    # process_audio_chunk) — we're just choosing to spend that
                    # discarded window on real pre-onset audio instead of on the
                    # first spoken syllables.
                    for lb_chunk in lookback_buffer:
                        _encode(lb_chunk)
                    lookback_buffer.clear()

                if vad_status in ("speech", "speech_start"):
                    _encode(audio_chunk_1280)
                    chunks_since_decode += 1

                    # Auto-finalize when cross-KV buffer is full
                    if state.cross_kv_fill >= model.max_memory_len:
                        buf_secs = int(model.max_memory_len * 0.020)
                        terminal.draw(
                            f"\033[31m⚠\033[0m Utterance #{utterance_count}:"
                            f" buffer full ({buf_secs}s limit) — finalizing..."
                        )
                        _finalize(utterance_count)
                        state.reset()
                        tokens = []
                        chunks_since_decode = 0
                        utterance_count += 1
                        terminal.draw(f"\033[32m●\033[0m Utterance #{utterance_count}: [Listening...]")
                        continue

                    # Periodic live preview decode
                    if chunks_since_decode >= args.preview_every and state.cross_kv_fill > 0:
                        tokens = _decode()
                        chunks_since_decode = 0

                    text = tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
                    dot = "\033[33m●\033[0m" if vad.silence_remaining_sec > 0 else "\033[32m●\033[0m"
                    indicator = f"{dot} Utterance #{utterance_count}"
                    terminal.draw_live(f"{indicator}\n{text if text else '...'}")

                elif vad_status == "speech_end":
                    terminal.draw(f"\033[34m◉\033[0m Utterance #{utterance_count}: processing...")
                    model.process_audio_chunk(state, audio_chunk_1280)
                    _finalize(utterance_count)
                    chunks_since_decode = 0

                elif vad_status == "silence":
                    lookback_buffer.append(audio_chunk_1280.copy())

            audio_queue.task_done()

    def print_verbose_summary():
        if stats is None:
            return
        audio_s = len(wav_audio) / input_sample_rate
        work_s  = (stats.encode_ms + stats.decode_ms) / 1000
        rtf     = work_s / audio_s if audio_s else 0.0
        tok_s   = stats.decode_steps / (stats.decode_ms / 1000) if stats.decode_ms else 0.0
        print(
            f"\n[verbose] RTF: {rtf:.2f}x realtime  ({work_s:.1f}s work / {audio_s:.1f}s audio)",
            file=sys.stderr,
        )
        print(
            f"[verbose] Decoder: {stats.decode_steps} tokens in {stats.decode_ms / 1000:.1f}s "
            f"→ {tok_s:.1f} tok/s",
            file=sys.stderr,
        )

    def feed_wav_to_queue():
        """Push the WAV file onto audio_queue in fixed-size blocks, then signal
        end-of-stream. A synthetic silence lead-in is prepended so the VAD's
        self-calibration (~1s of ambient noise) has something to sample even
        when the file starts talking immediately."""
        lead_in = np.zeros(int(1.0 * input_sample_rate), dtype=np.float32)
        full    = np.concatenate([lead_in, wav_audio])
        block   = 4096
        pos     = 0
        while pos < len(full) and running:
            end = min(pos + block, len(full))
            audio_queue.put(full[pos:end])
            if args.realtime:
                time.sleep((end - pos) / input_sample_rate)
            pos = end
        audio_queue.put(_END_OF_STREAM)

    def feed_mic_to_queue():
        """AS33980 capture via arecord and push chunks to audio_queue"""
        import subprocess

        capture_rate = 48000
        capture_channels = 2
        block_frames = 4096                      # samples per block
        sample_width = 2                         # S16_LE
        block_bytes = block_frames * sample_width * capture_channels

        cmd = ["arecord", "-D", "plughw:1,0", "-f", "S16_LE",
            "-r", str(capture_rate), "-c", str(capture_channels),
            "-t", "raw", "-q"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        try:
            while running:
                raw = proc.stdout.read(block_bytes)
                if not raw:
                    break
                # int16 stereo → right channel → float32
                audio = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
                right = audio[:, 1].astype(np.float32) / 32768.0
                audio_queue.put(right)
        finally:
            proc.terminate()
            proc.wait()
            audio_queue.put(_END_OF_STREAM)

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    # Hide the terminal cursor while the live transcript is being redrawn in
    # place — draw() writes "\r" (parking the cursor at column 0, right on
    # the leading "●") before overwriting it with new text, so at the redraw
    # rate used here the terminal's own blinking block cursor visibly flashes
    # over that character. Always restored in the finally below.
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()
    try:
        print(f">>> Transcribing {args.wav} ... <<<\n", file=sys.stderr)
        try:
            feed_wav_to_queue()
            worker_thread.join()
        except KeyboardInterrupt:
            print("\n\nInterrupted...", file=sys.stderr)
            running = False
            worker_thread.join(timeout=1.0)
        finally:
            print_verbose_summary()
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        ok, message = configure_npu_userspace_frequency("min")
        print(f"[NPU] {message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Moonshine Static Streaming WAV File Transcription Demo (2-Split VMFB)"
    )
    parser.add_argument("--wav",           type=str,   required=True,          help="WAV file to transcribe")
    parser.add_argument("--realtime",      action="store_true",               help="Pace the feed to match real-time playback speed (default: feed as fast as possible)")
    parser.add_argument("-m", "--model-dir", type=str, required=True, metavar="DIR", help="Path to the flat moonshine-streaming-tiny model dir")
    parser.add_argument("--vad-threshold", type=float, default=0.01,           help="VAD trigger threshold: RMS floor for the energy VAD (default: 0.010)")
    parser.add_argument("--vad-silence",   type=float, default=2.5,            help="Silence gap to split utterances in seconds (default: 2.5)")
    parser.add_argument("--vad-lookback",  type=int,   default=None,           help="Pre-speech chunks to replay into the encoder on speech_start, to avoid clipping word onsets (default: model.warmup_chunks; 0 disables)")
    parser.add_argument("--preview-every", type=int,   default=5,              help="Chunks the decoder waits between live preview decodes (default: 5)")
    parser.add_argument("--commit-agreement", type=int, default=2,             help="LocalAgreement-N: commit a token only if stable across the last N hypotheses (default: 2)")
    parser.add_argument("--commit-delay",  type=float, default=3.0,            help="Only commit tokens at least this many seconds of audio behind the live frontier (default: 3.0)")
    parser.add_argument("--full-decode",   action="store_true",               help="Disable incremental decode; re-decode from BOS each time (baseline behaviour)")
    parser.add_argument("--verbose",       action="store_true",               help="Print a real-time factor (RTF) and decoder tok/s summary on exit")
    parser.add_argument("--no-refresh",    action="store_true", default=False, help="Skip the Hugging Face check for updated models (offline/airgapped runs)")
    add_logging_args(parser)
    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument(
        "--runtime-flags",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="FLAG",
        help=(
            "[Advanced] Extra flags for the Torq runtime (e.g. --torq_hw_type=sim). "
            "Must be specified last; all remaining arguments are forwarded."
        ),
    )
    main(parser.parse_args())
