# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

r"""CPU || NPU text-to-speech pipeline for Piper (VITS) on Torq.

The VITS graph is split where the per-phoneme durations are ceiled and summed,
which is the point the output length becomes exact:

    ids -> [partA: text encoder + duration]  (ORT, CPU) -> z [1,192,F], g [1,512,1]
                     z,g -> [partB: HiFi-GAN vocoder]   (vmfb, NPU) -> audio [F*256]

partA reports the exact frame count F, so the vocoder window is known before it
runs: the smallest of the 1/2/4/6/8 s vmfbs that fits is picked and the latent is
edge-padded up to it. Three threads overlap so the CPU encodes sentence *n+1*
while the NPU vocodes sentence *n* and the speaker plays sentence *n-1*.
"""

import queue
import re
import subprocess
import threading
import time
import wave
from pathlib import Path

import ml_dtypes
import numpy as np
import onnxruntime as ort
from torq.runtime import VMFBInferenceRunner

HOP, SR = 256, 22050                                   # vocoder hop, sample rate
Z_NAME, G_NAME = "/Mul_7_output_0", "/Unsqueeze_output_0"   # partA -> partB seam


def find_audio_device():
    """Pick a USB DAC from ``aplay -l``, falling back to the ALSA default."""
    try:
        out = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return "default"
    m = re.search(r"^card \d+: (\S+) \[.*?\], device (\d+):.*USB", out, re.M)
    return f"plughw:CARD={m.group(1)},DEV={m.group(2)}" if m else "default"


def write_wav(path, audio, rate=SR):
    """Write mono float audio to a 16-bit PCM wav; return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(to_pcm16(audio).tobytes())
    return path


def to_pcm16(audio):
    return (np.clip(audio, -1, 1) * 32767).round().astype(np.int16)


def resample(audio, src_rate, dst_rate):
    if src_rate == dst_rate:
        return audio
    n = int(audio.size * dst_rate / src_rate)
    return np.interp(np.arange(n) / dst_rate, np.arange(audio.size) / src_rate, audio).astype(np.float32)


class PiperTTS:
    """Load partA (CPU) + the partB window vmfbs (NPU) once, synthesize many times."""

    def __init__(self, model_dir, *, device_uri="torq", threads=2, length_scale=1.0,
                 speaker=0, audio_device=None, dac_rate=48000, dac_channels=2):
        d = Path(model_dir)
        self.dac_rate, self.dac_channels = dac_rate, dac_channels
        self.audio_device = audio_device or find_audio_device()
        self.scales = np.array([0.333, length_scale, 0.0], dtype=np.float32)  # noise, length, noise_w
        self.sid = np.array([speaker], dtype=np.int64)
        self.load_s = {}

        t = time.perf_counter()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.partA = ort.InferenceSession(str(d / "onnx" / "partA.onnx"), sess_options=so,
                                          providers=["CPUExecutionProvider"])
        self.a_names = [o.name for o in self.partA.get_outputs()]
        self._encode(np.array([1, 0, 3, 0, 2], dtype=np.int64))          # warm ORT
        self.load_s["partA"] = time.perf_counter() - t

        # One runner per window. The frame width is read from the vmfb signature
        # rather than hardcoded, so a recompiled set of windows just works.
        t = time.perf_counter()
        self.windows = {}
        for p in sorted((d / "vmfb").glob("partB_static_*s.vmfb")):
            r = VMFBInferenceRunner(str(p), device_uri=device_uri)
            self.windows[r.inputs_info[0].shape[2]] = r
        if not self.windows:
            raise FileNotFoundError(f"no partB vmfbs found in {d / 'vmfb'}")
        self.sizes = sorted(self.windows)
        smallest = self.windows[self.sizes[0]]
        smallest.infer([np.zeros(i.shape, dtype=ml_dtypes.bfloat16) for i in smallest.inputs_info])
        self.load_s["partB"] = time.perf_counter() - t

    @property
    def max_seconds(self):
        return self.sizes[-1] * HOP / SR

    def _encode(self, ids):
        """partA: phoneme ids -> (z, g) latents, with F fixed and exact."""
        out = dict(zip(self.a_names, self.partA.run(None, {
            "input": ids[np.newaxis, :], "input_lengths": np.array([ids.size], dtype=np.int64),
            "scales": self.scales, "sid": self.sid})))
        return out[Z_NAME], out[G_NAME]

    def _vocode(self, z, g):
        """partB: pick the smallest window that fits, edge-pad, run on the NPU."""
        F = z.shape[2]
        W = next((s for s in self.sizes if F <= s), None)
        if W is None:
            return None, None
        zp = np.zeros((1, z.shape[1], W), dtype=np.float32)
        zp[:, :, :F], zp[:, :, F:] = z, z[:, :, F - 1:F]      # repeat the last frame
        out = self.windows[W].infer([zp.astype(ml_dtypes.bfloat16), g.astype(ml_dtypes.bfloat16)])
        return np.asarray(out[0]).astype(np.float32).ravel()[:F * HOP], W

    def _play_stream(self, play_q, marks, t0):
        """Stream finished sentences to the speaker as they arrive."""
        player = subprocess.Popen(["aplay", "-q", "-D", self.audio_device, "-t", "raw", "-f", "S16_LE",
                                   "-r", str(self.dac_rate), "-c", str(self.dac_channels), "-"],
                                  stdin=subprocess.PIPE)
        while (item := play_q.get()) is not None:
            marks.setdefault("first_sound", time.perf_counter() - t0)
            pcm = to_pcm16(resample(item, SR, self.dac_rate))
            try:
                player.stdin.write(np.repeat(pcm, self.dac_channels).tobytes()), player.stdin.flush()
            except (BrokenPipeError, OSError):
                break
        try:
            player.stdin.close()
        except OSError:
            pass
        player.wait()

    def synthesize(self, sentences, wav_path=None, *, play=True, on_skip=None):
        """Run the pipeline over per-sentence id arrays; return audio + timings."""
        n = len(sentences)
        audio, used, marks, errors = [None] * n, [None] * n, {}, []
        latents, play_q = queue.Queue(maxsize=2), queue.Queue()
        t0 = time.perf_counter()

        # Both workers always post their sentinel, so a failure in one surfaces
        # as an exception here instead of deadlocking the other on an empty queue.
        def encode_all():
            try:
                for i, ids in enumerate(sentences):
                    latents.put((i, *self._encode(ids)))
            except Exception as e:  # noqa: BLE001 - re-raised on the calling thread
                errors.append(e)
            finally:
                latents.put(None)

        def vocode_all():
            try:
                while (item := latents.get()) is not None:
                    i, z, g = item
                    audio[i], used[i] = self._vocode(z, g)
                    if audio[i] is None:                # longer than the widest window
                        audio[i] = np.zeros(0, dtype=np.float32)
                        on_skip(i, z.shape[2] * HOP / SR) if on_skip else None
                        continue
                    marks.setdefault("first_audio", time.perf_counter() - t0)
                    play_q.put(audio[i]) if play else None
            except Exception as e:  # noqa: BLE001 - re-raised on the calling thread
                errors.append(e)
                while latents.get() is not None:        # unblock a waiting encoder
                    pass
            finally:
                play_q.put(None)

        workers = [threading.Thread(target=f, daemon=True) for f in (encode_all, vocode_all)]
        if play:
            workers.append(threading.Thread(target=self._play_stream, args=(play_q, marks, t0), daemon=True))
        for w in workers:
            w.start()
        workers[0].join(), workers[1].join()
        compute_s = time.perf_counter() - t0     # compute done; the speaker may still be draining
        for w in workers[2:]:
            w.join()
        if errors:
            raise errors[0]

        full = np.concatenate(audio) if n else np.zeros(0, dtype=np.float32)
        stats = {"audio_s": full.size / SR, "compute_s": compute_s,
                 "rtf": (full.size / SR) / compute_s if compute_s else 0.0,
                 "first_audio_s": marks.get("first_audio"), "first_sound_s": marks.get("first_sound"),
                 "windows": [w * HOP / SR for w in used if w]}
        if wav_path:
            stats["wav"] = write_wav(wav_path, full)
        return full, stats

    def close(self):
        self.windows.clear()
