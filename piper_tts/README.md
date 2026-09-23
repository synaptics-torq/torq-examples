# Piper TTS Demo

Neural text-to-speech on Torq. Runs **Piper** (a VITS model) **split
across CPU and NPU**: the text encoder and duration predictor stay on the CPU
under onnxruntime, and the HiFi-GAN vocoder — the expensive 82% — runs on the NPU
as bf16 NSS-only VMFBs. The two halves overlap, so the CPU encodes the next
sentence while the NPU vocodes the current one and the speaker plays the previous
one.

Three voices ship, selected with `--voice`:

| Voice | Language | Speakers |
|---|---|---|
| `en_US-libritts_r-medium` (default) | English (US) | 904 |
| `en_US-lessac-low` | English (US), 16 kHz | 1 |
| `es_MX-ald-medium` | Spanish (Mexico) | 1 |

They differ in more than weights: the Spanish model is single-speaker, so its
partA takes no speaker id and its vocoder takes the latent alone. The demo reads
both signatures off the models rather than a table, so either shape just runs.

Each run **writes a `.wav` file and plays it on the speaker**.

On the SL2619 board this synthesizes **2.7–3.8× faster than real time** — about
**2× faster** than the same voice run entirely on the CPU.

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
├── en_US-lessac-low/{onnx,vmfb,voice}/       # the same three, per extra voice
├── es_MX-ald-medium/{onnx,vmfb,voice}/
└── espeak/{phonemizerd, espeak-ng-data/}     # phonemizer daemon + dictionaries
```

Only the voice you ask for is downloaded, so running English never fetches the
Spanish vocoder. The espeak dictionaries are shared — one copy covers every
language, so a voice adds no phonemizer assets.

The vocoder windows are shipped as five separate VMFBs because the NPU model is
statically shaped; see [How it runs](#how-it-runs). The demo reads each window's
frame width **from the VMFB signature itself**, so recompiling with a different
set of windows needs no code change — just drop the new files in `vmfb/`.

You also need a speaker. The demo autodetects a USB audio card from `aplay -l`
and falls back to the ALSA `default` device; override with `--audio-device`.

## Running

Run the demo from the `piper_tts` directory. The first run downloads the assets
automatically, and later runs re-download them when the Hugging Face repo has
been updated; pass `--no-refresh` for fully offline runs.

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

# Spanish — each voice carries its own samples, so --interactive and --sample
# give you Spanish ones here
python src/infer.py --voice es_MX-ald-medium --interactive
python src/infer.py --voice es_MX-ald-medium --text "Buenos días. El sistema ya funciona."

# write the wav without playing it (e.g. over SSH with no speaker)
python src/infer.py --text "Silent run." --no-play
```

The text is printed in full before synthesis starts, so it is on screen while it
plays; the timing line follows — audio duration, which vocoder windows were used,
the time until the first sound reached the speaker, and the synthesis speed:

```
[NPU] NPU clock enabled
Loading (partA on CPU, partB windows on NPU, espeak resident)...
  partA 5.9 s | partB 5 windows (1s, 2s, 4s, 6s, 8s) 0.7 s | speaker plughw:CARD=C1,DEV=0

  "The morning train was late again. Nobody on the platform seemed surprised."
  audio 3.05 s | windows 2s+2s | first sound 0.70 s | compute 1.16 s (2.64x real time) | saved tts_out.wav
```

Options:

- `--voice KEY` — which voice to speak with (default `en_US-libritts_r-medium`).
  `--list-voices` prints them.
- `--text STR` / `--file PATH` / `--sample N` / `--interactive` — what to speak
  (mutually exclusive; default is sample 1). `--list-samples` prints the samples
  for the selected voice.
- `--output PATH` — output wav (default `tts_out.wav`). In `--interactive` mode
  this is the *directory* written to instead (default `out/`).
- `--no-play` — write the wav only, don't open the speaker.
- `--audio-device DEV` — ALSA device (default: autodetected USB DAC).
- `--dac-rate HZ` — rate the DAC accepts, 48000 by default; audio is resampled
  from the voice's rate (22.05 kHz, or 16 kHz for `lessac-low`) to this before
  playback. The wav on disk keeps the voice's own rate.
- `--speaker N` — speaker id, for multi-speaker voices only (0–903 on
  `en_US-libritts_r-medium`; `es_MX-ald-medium` has a single speaker).
- `--length-scale F` — phoneme duration scale; `>1` speaks slower. Note it is
  **not proportional** — Piper interleaves a PAD token between phonemes whose
  duration is pinned at one frame by a `Ceil`, so roughly a third of a short
  utterance does not scale. `1.36` gives about +20% on typical text.
- `--threads N` — onnxruntime threads for partA (default 2, the board's core count).
- `--device URI` — IREE device for the vocoder (default `torq`).
- `--model-dir DIR` — asset dir (default `models/Synaptics/Piper-TTS`).
- `--no-refresh` — skip Hugging Face entirely and run against the assets already
  on disk. Anything missing is named in the error instead of being fetched, which
  is also how locally built vocoders are used.
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

`g` is the speaker embedding. A single-speaker voice such as `es_MX-ald-medium`
has none, so its interface is the one tensor `z` and its vocoder is 68 nodes
instead of 73 — the same graph minus the per-block speaker conditioning.

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

SL2619 board, 2 CPU threads for partA, NPU at full clock. A three-sentence sample,
text in to all audio synthesized (phonemization included, model load excluded),
median of 5 runs:

| voice | audio | all-CPU onnxruntime | **CPU (partA) ‖ NPU (partB)** | speedup | first audio (CPU / NPU) |
|---|---|---|---|---|---|
| `en_US-libritts_r-medium` | 7.70 s | 5.62 s (1.37× RT) | **2.82 s (2.73× RT)** | **1.99×** | 0.94 s / **0.77 s** |
| `en_US-lessac-low` | 9.82 s | 5.21 s (1.89× RT) | **2.59 s (3.79× RT)** | **2.01×** | 0.89 s / **0.66 s** |
| `es_MX-ald-medium` | 12.63 s | 8.84 s (1.43× RT) | **4.05 s (3.12× RT)** | **2.18×** | 1.83 s / **1.37 s** |

The CPU is also left free during vocoding. The vocoder alone on the NPU, per window:

| window | libritts / Spanish (22.05 kHz) | lessac-low (16 kHz) |
|---|---|---|
| 1 s | 170 ms (5.9× RT) | 126 ms (7.8× RT) |
| 2 s | 335 ms (6.0× RT) | 249 ms (8.0× RT) |
| 4 s | 688 ms (5.8× RT) | 497 ms (8.0× RT) |
| 6 s | 1032 ms (5.8× RT) | 749 ms (8.0× RT) |
| 8 s | 1200–1356 ms (5.9–6.7× RT) | 1027 ms (7.8× RT) |

The 16 kHz voice has about 27% fewer vocoder frames per second of speech, which
is why it is the fastest.

Startup, paid once: partA ~6 s (onnxruntime loading the graph), all five VMFBs
~0.5–0.7 s, phonemizer ~0.3 s. The windows are kept loaded so no window switch
costs a reload. Forty utterances back to back show no memory growth or slowdown.

## Accuracy

The bf16 NPU vocoder against the same partB graph in fp32 on the CPU, same
latents, every window of every voice:

| | |
|---|---|
| SNR | **37.3–40.7 dB** |
| correlation | **≥ 0.9999** |

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
