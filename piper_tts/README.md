# Piper TTS Demo

Neural text-to-speech on Torq. Runs **Piper** (a VITS model, voice
`en_US-libritts_r-medium`, 904 speakers, 22.05 kHz) **split across CPU and NPU**:
the text encoder and duration predictor stay on the CPU under onnxruntime, and
the HiFi-GAN vocoder — the expensive 82% — runs on the NPU as bf16 NSS-only
VMFBs. The two halves overlap, so the CPU encodes the next sentence while the
NPU vocodes the current one and the speaker plays the previous one.

Each run **writes a `.wav` file and plays it on the speaker**.

On the SL2619 board this synthesizes **2.2× faster than real time**, against
1.0× for the same model run entirely on the CPU.

## Setup

From the repo root, run:

```sh
cd piper_tts
pip install -r requirements.txt
cd ..
python setup_demos.py piper_tts
```

This verifies the demo's Python dependencies and downloads the assets from
Hugging Face ([`Synaptics/Piper-TTS`](https://huggingface.co/Synaptics/Piper-TTS)).

Downloaded assets are stored at:

```sh
models/Synaptics/Piper-TTS/
├── onnx/partA.onnx                       # text encoder + duration (CPU, onnxruntime)
├── vmfb/partB_static_{1,2,4,6,8}s.vmfb   # HiFi-GAN vocoder (NPU), one per window
├── voice/en_US-libritts_r-medium.onnx.json   # phoneme -> id map + voice config
└── espeak/{phonemizerd, espeak-ng-data/}     # phonemizer daemon + dictionaries
```

The vocoder windows are shipped as five separate VMFBs because the NPU model is
statically shaped; see [How it runs](#how-it-runs). The demo reads each window's
frame width **from the VMFB signature itself**, so recompiling with a different
set of windows needs no code change — just drop the new files in `vmfb/`.

You also need a speaker. The demo autodetects a USB audio card from `aplay -l`
and falls back to the ALSA `default` device; override with `--audio-device`.

## Running

Run the demo from the `piper_tts` directory. The first run downloads the assets
automatically; pass `--no-refresh` for fully offline runs afterwards.

```sh
cd piper_tts

# default: speak the first built-in sample -> tts_out.wav, and play it
python src/infer.py

# your own text
python src/infer.py --text "Hello from the Synaptics board." --output hello.wav

# a file, a built-in sample, or a menu
echo "The bakery on the corner opens at six." > article.txt
python src/infer.py --file article.txt
python src/infer.py --interactive

# write the wav without playing it (e.g. over SSH with no speaker)
python src/infer.py --text "Silent run." --no-play
```

The text is printed in full before synthesis starts, so it is on screen while it
plays; the timing line follows — audio duration, which vocoder windows were used,
the time until the first sound reached the speaker, and the synthesis speed:

```
[NPU] NPU clock enabled
Loading (partA on CPU, partB windows on NPU, espeak resident)...
  partA 6.1 s | partB 5 windows (1s, 2s, 4s, 6s, 8s) 0.8 s | speaker plughw:CARD=C1,DEV=0

  "The morning train was late again. Nobody on the platform seemed surprised."
  audio 3.05 s | windows 2s+2s | first sound 0.91 s | compute 1.41 s (2.16x real time) | saved tts_out.wav
```

Options:

- `--text STR` / `--file PATH` / `--sample N` / `--interactive` — what to speak
  (mutually exclusive; default is sample 1). `--list-samples` prints the samples.
- `--output PATH` — output wav (default `tts_out.wav`). In `--interactive` mode
  this is the *directory* written to instead (default `out/`).
- `--no-play` — write the wav only, don't open the speaker.
- `--audio-device DEV` — ALSA device (default: autodetected USB DAC).
- `--dac-rate HZ` — rate the DAC accepts, 48000 by default; audio is resampled
  from 22.05 kHz to this before playback. The wav on disk is always 22.05 kHz.
- `--speaker N` — speaker id, 0–903 (this voice is multi-speaker).
- `--length-scale F` — phoneme duration scale; `>1` speaks slower. Note it is
  **not proportional** — Piper interleaves a PAD token between phonemes whose
  duration is pinned at one frame by a `Ceil`, so roughly a third of a short
  utterance does not scale. `1.36` gives about +20% on typical text.
- `--threads N` — onnxruntime threads for partA (default 2, the board's core count).
- `--device URI` — IREE device for the vocoder (default `torq`).
- `--model-dir DIR` — asset dir (default `models/Synaptics/Piper-TTS`).
- `--no-refresh` — skip the Hugging Face update check (offline).
- `--quiet` — suppress the per-utterance timing lines.

## How it runs

VITS fixes its output length at the point where the predicted per-phoneme
durations are ceiled and summed. Everything up to that sum is **partA**; the
alignment expansion, flow and vocoder that follow are **partB**. The split is at
`/ReduceSum_output_0` and the interface is two tensors:

```
text --[espeak]--> phoneme ids --> [partA]  (CPU, onnxruntime)  --> z [1,192,F], g [1,512,1]
                                              z,g --> [partB]   (NPU, bf16 vmfb) --> audio [F*256]
```

That cut point is what makes the NPU side tractable. partA holds 85% of the
*nodes* — a swarm of small shape and attention ops — but partB holds 82% of the
*time*, and it is pure convolution, which is what the NPU is good at. It also
means **F, the exact frame count, is known before the vocoder runs**: audio
length is `F × 256` samples, exactly, so the right window can be chosen up
front rather than guessed.

**Static windows.** The NPU model is statically shaped, so the vocoder ships as
five VMFBs covering 1, 2, 4, 6 and 8 seconds of audio. Per sentence the demo
picks the smallest window that fits and edge-pads the latent up to it (repeating
the last frame), then trims the output back to `F × 256` samples. Sentences
longer than the 8 s window are skipped with a warning — split them at a comma.

**Three-stage overlap.** Encoding, vocoding and playback run in separate threads
connected by queues, so all three stay busy:

```
CPU:  [A1] [A2] [A3] ...
NPU:       [B1] [B2] [B3] ...
spk:            [P1] [P2] [P3] ...
```

Because synthesis beats real time, the only real wait is at the front — the
first sentence — which is why time-to-first-sound (~0.9 s) matters more than
total throughput for anything interactive.

**Phonemization** uses espeak-ng through `phonemizerd`, a small resident daemon
that mirrors libpiper's `piper_synthesize_start()` exactly — same espeak call,
same clause-terminator handling — so the phoneme ids match what Piper itself
would produce. It stays loaded, so per-utterance phonemization costs a few
milliseconds instead of a fresh dictionary load. Source:
[`piper_core/phonemizerd.c`](./piper_core/phonemizerd.c).

## Performance

SL2619 board, `en_US-libritts_r-medium`, 2 CPU threads for partA, NPU at full
clock. "compute" excludes speaker drain time.

Sample 1 (two sentences, 3.05 s of audio):

| | compute | speed |
|---|---|---|
| **CPU (partA) ‖ NPU (partB)** | **1.37 s** | **2.22× real time** |
| serial, all-CPU onnxruntime | 3.03 s | 1.01× real time |

**2.2× faster end to end**, and the CPU is left free during vocoding.

The vocoder itself is a steady ~3.9× real time on the NPU at every window size:

| window | frames | NPU time | speed |
|---|---|---|---|
| 1 s | 86 | 254 ms | 3.9× RT |
| 2 s | 172 | 499 ms | 4.0× RT |
| 4 s | 345 | 1059 ms | 3.8× RT |
| 6 s | 517 | 1549 ms | 3.9× RT |
| 8 s | 689 | 2038 ms | 3.9× RT |

Startup, paid once: partA ~6 s (onnxruntime loading a 71 MB graph), all five
VMFBs ~0.8 s, phonemizer ~0.3 s. Resident set with all five windows loaded is
~110 MB — they are kept loaded so no window switch costs a reload.

Typical end-to-end figures: **first sound ~0.9 s** after you press enter,
2.1–2.2× real time sustained.

## Accuracy

The bf16 NPU vocoder against the same partB graph in fp32 on the CPU, same
latents:

| | |
|---|---|
| SNR | **39.6 dB** |
| correlation | **0.99998** |

That is inaudible in practice — the difference is well below the run-to-run
variation of the model itself, which samples noise internally (the same text
synthesized twice is never bit-identical).

## Notes

- **No speaker?** Use `--no-play`; the wav is still written. If autodetection
  picks the wrong card, pass `--audio-device plughw:CARD=<name>,DEV=<n>` — run
  `aplay -l` to list them.
- **Sentence length.** The 8 s window is the ceiling for a single sentence.
  Long sentences split at a comma also *improve* time-to-first-sound, since the
  first unit is shorter.
- **Licensing.** espeak-ng is GPLv3 and `phonemizerd` links it statically; the
  daemon's source ships with this demo at `piper_core/phonemizerd.c` and the
  build command is in its header comment. The Piper voice is MIT; the models are
  redistributed from the [`Synaptics/Piper-TTS`](https://huggingface.co/Synaptics/Piper-TTS)
  repo.
