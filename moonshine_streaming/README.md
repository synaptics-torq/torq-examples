# Moonshine Streaming Demo

Real-time microphone transcription with Moonshine-tiny English using a 2-split
Torq VMFB model (fused encoder + KV decoder). A self-calibrating energy VAD splits
utterances and a committed-prefix incremental decoder gives a live preview.

> Requires a working microphone and the `sounddevice` package (PortAudio).

## Setup

From the repo root, run:

```sh
python setup_demos.py moonshine_streaming
```

This downloads the default model files to:

```sh
models/Synaptics/moonshine-streaming-tiny-torq/
```

> The HuggingFace repo is not published yet — until then, copy the model files
> into that directory manually (see `PLACEHOLDER.md` there for the exact list).

## Running

Run the demo from the `moonshine_streaming` directory:

```sh
cd moonshine_streaming
python src/infer.py -m ../models/Synaptics/moonshine-streaming-tiny-torq
```

Then start speaking. Press `Ctrl+C` to exit.

The defaults are tuned for the board (`--hw-type astra_machina`, `--device 1`,
`--vad-silence 8`, `--vad-threshold 0.010`, `--preview-every 5`), so the command
above is equivalent to:

```sh
python src/infer.py -m ../models/Synaptics/moonshine-streaming-tiny-torq \
    --hw-type astra_machina --device 1 --vad-silence 8 --vad-threshold 0.010 --preview-every 5
```

List audio input devices and pick a different one:

```sh
python src/infer.py --list-devices
python src/infer.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --device 3
```

Run `python src/infer.py -h` to see all available options (VAD thresholds, decode
mode, hardware type, profiling).

## Running on a board

Target is an aarch64 board running Python 3.12 with the Torq runtime
(`--hw-type astra_machina` is the default).

Prerequisites — PortAudio (for `sounddevice`) and a working microphone:

```sh
sudo apt-get update && sudo apt-get install -y libportaudio2
```

From the repo root:

```sh
# 1. venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Torq runtime wheel (aarch64 / Python 3.12 board build)
pip install https://github.com/synaptics-torq/torq-examples/releases/download/torq-runtime-v2.0-alpha/torq_runtime-2.0.0a1-cp312-cp312-manylinux_2_28_aarch64.whl

# 3. shared deps + this demo's deps
pip install -r requirements.txt
pip install -r moonshine_streaming/requirements.txt

# 4. one-time setup: installs the repo on the Python path (.pth) and verifies the
#    model files (a manifest is present, so no download is attempted)
python setup_demos.py moonshine_streaming

# 5. run
cd moonshine_streaming
python src/infer.py -m ../models/Synaptics/moonshine-streaming-tiny-torq
```

> The model files ship in `models/Synaptics/moonshine-streaming-tiny-torq/` (no
> HuggingFace repo yet). If `models/` is excluded from the copy onto the board,
> transfer that directory manually — see its `PLACEHOLDER.md` for the file list.
> Match the wheel in step 2 to your board's CPU architecture and Python version
> (see the [releases page](https://github.com/synaptics-torq/torq-examples/releases)).

## How it works

This is a **real-time microphone transcriber**: it consumes a live audio stream
chunk-by-chunk, detects when you are speaking, and updates the transcript *as you
talk* — freezing words once it is confident about them. The two ideas that make it
real-time are (a) an encoder that turns streaming audio into an incrementally-grown
cross-attention memory, and (b) a decoder that resumes from a frozen token prefix
with its KV tensors pinned on the NPU, so each preview is cheap.

### Code layout

| File | Role |
|------|------|
| `src/runner.py` | the **engine**: the model, its pre-allocated state, and the thin VMFB session wrapper. No audio, no UI — "give me audio chunks, I return tokens". |
| `src/infer.py` | the **app**: microphone capture, VAD, the worker thread, decode triggering, terminal rendering, the CLI, and the profiler. |
| `setup_demo.py` | downloads/verifies the model files (reuses `utils/`). |

### The model (2-split) and its files

A 2-split Moonshine-tiny: a **fused `encoder`** (audio → cross-attention memory) and
a **`decoder`** (memory → tokens, autoregressively). The model directory holds 7
flat files:

| File | Role |
|------|------|
| `encoder.vmfb`, `decoder.vmfb` | the compiled models that run on the NPU |
| `streaming_config.json` | the streaming knobs (below) |
| `config.json` | model config |
| `adapter_pos_emb.npy` | position-embedding table, looked up host-side per chunk |
| `decoder_token_embeddings.npy` | token-embedding table (the decoder is fed embeddings, not token IDs) |
| `tokenizer.json` | token IDs → text |

A VMFB exposes its inputs *positionally* — it has no argument names. The dict-based
feed interface needs names, so they are **hardcoded** as `ENCODER_INPUT_ORDER` /
`DECODER_INPUT_ORDER` in `runner.py` (input shapes + dtypes come from the runner's
`inputs_info`). This is why the demo needs neither `onnx` nor any sidecar files at
runtime; `_Session` validates the hardcoded arity against `inputs_info` at load and
errors loudly if the model is re-exported with a different number of inputs.

The streaming knobs (`streaming_config.json`) drive everything downstream:

```
chunk_len        = 1280 samples  → 80 ms of audio per chunk @ 16 kHz
feature_stride   = 4 frames/chunk → each active chunk adds 4 cross-KV frames
                                     (80 ms / 4 = 20 ms per frame)
total_lookahead  = 16 frames
warmup_chunks    = 4
max_memory_len   = 400 frames    → 400 × 20 ms = 8 s cross-KV buffer
max_tokens       = 48
layers=6  heads=8  head_dim=40  hidden=320  BOS=1  EOS=2
```

A useful invariant: **`cross_kv_fill` frames × 20 ms = seconds of audio captured**.

### Data flow

```
mic ──callback──> audio_queue ──> worker thread ──> resample to 16 kHz
                                        │
                                        ▼  (slice into fixed 1280-sample chunks)
                                   per-chunk loop:
                                        │
                          ┌─────────────┴──────────────┐
                          ▼                             ▼
                      EnergyVAD                  if speaking:
                   (speech? silence?)       process_audio_chunk()  ← ENCODER
                          │                        │ grows cross-KV
                          │                        ▼
                          │                  every N chunks OR on speech-end:
                          │                  decode_incremental()  ← DECODER
                          │                        │ produces tokens
                          ▼                        ▼
                   TerminalListener  <──────  tokenizer.decode()
```

Two threads: the **audio callback** only does `audio_queue.put(chunk)` (it must
return fast or audio drops); the **worker thread** does resample → VAD → encode →
decode → render. This decoupling keeps capture glitch-free while inference runs.

### Step by step

**1. Capture.** `sounddevice` delivers `blocksize=4096` samples at the device's
native rate. The callback just queues a copy.

**2. Normalize to fixed chunks.** The worker resamples each block to 16 kHz
(linear interpolation), accumulates into a buffer, and slices off **exactly 1280
samples (80 ms)** at a time. So the whole pipeline sees uniform chunks regardless of
the mic's native rate/blocksize; leftover samples carry to the next block.

**3. Pre-allocated state** (`MoonshineStaticStreamingState`, allocated once, reused
across utterances):

| Buffer | Shape | Meaning |
|--------|-------|---------|
| `conv1_buffer` / `conv2_buffer` | `(1,320,4)` / `(1,640,4)` | rolling conv left-context |
| `features_buffer` | `(1,16,320)` | rolling feature/lookahead window |
| `enc_bufs` | list of `(1,16,320)` | encoder internal layer state |
| `k_cross` / `v_cross` | `(6,1,8,400,40)` | **the cross-attention memory** |
| `k_self` / `v_self` | `(6,1,8,48,40)` | decoder self-attention KV |
| `pos_offset` | `[0]` | running index into the position-embedding table |

Nothing is reallocated during streaming. `reset()` (on `speech_start`) clears
`cross_kv_fill`, `chunk_idx`, `committed_tokens`, `recent_hyps`.

**4. VAD** (`EnergyVAD`): a self-calibrating RMS energy detector. It samples the
room for the first ~12 chunks (~1 s) and sets `threshold = max(mean + 4·std,
--vad-threshold)`, then per chunk emits `speech_start` / `speech` / `speech_end`
(after `--vad-silence` s of quiet) / `silence`. (The `[VAD Calibration]` line only
prints with `--profile`.)

**5. Encoder step** (`process_audio_chunk`, runs on every speech chunk). Builds a
feed dict keyed by the encoder's input names (audio, the three rolling buffers, a
position-embedding slice looked up from `pos_offset`, and `buf_*`), runs the VMFB,
and unpacks outputs as
`[k_cross_new, v_cross_new, conv1_out, conv2_out, features_out, *enc_buf_outs]`. The
rolling buffers are always updated. Then:

- **Warmup** (`chunk_idx < 4`): discard the new cross-KV and encoder-buffer updates —
  not enough context yet.
- **Active** (`chunk_idx ≥ 4`): append the 4 new cross-KV frames at `cross_kv_fill`,
  advance `cross_kv_fill += 4` and `pos_offset += 4`.

So each active 80 ms chunk grows the memory by 4 frames (80 ms). `encode(is_final=
False)` is a no-op — all the work lives in `process_audio_chunk`.

**6. Finalize / flush** (`encode(is_final=True)`): because the frontend has 16
frames of lookahead, the last audio hasn't fully propagated when you stop. Finalize
feeds **4 zero chunks** through the same path so the tail of your sentence is pushed
out of the lookahead pipeline (otherwise the last word or two is lost).

**7. When the decoder runs.** Three triggers in the worker:
1. **Live preview** — `chunks_since_decode ≥ --preview-every` (≈ every 5 × 80 ms =
   400 ms while speaking).
2. **Buffer full** — `cross_kv_fill ≥ 400` (8 s): force-finalize, start a new utterance.
3. **Speech end** — VAD said so: flush + final decode + commit the line.

**8. Incremental decode** (`decode_incremental`). First it computes
`max_tokens = min(ceil(seconds × 6.5), 48)` (speech is ~4–6.5 tok/s) and a
`cross_attn_bias` that masks the empty tail of the 400-frame memory (`-1e9` beyond
`cross_kv_fill`) so the decoder only attends to real audio. Then it **resumes from
the committed prefix** instead of from BOS:

```
C = len(committed_tokens)
start at position C with current_token = committed[-1]   (or BOS at position 0 if C==0)
loop: feed embedding(current_token) + bias + position + self/cross-KV
      → logits → argmax → next_token; stop on EOS(2) or max_tokens
```

A preview that already committed 12 tokens and finds 4 new ones runs ~5 decoder
passes, not 16 — that is the O(tail) speedup vs. O(T²) re-decode-from-BOS.

**9. KV residency** (the per-token cost saver). The decoder session is created with
`device_outputs=True`, and:
- **P1** (`_upload_cross_kv`): cross-KV is constant for a whole decode call → uploaded
  once as device buffers and reused every token (instead of re-uploading 6×8×400×40
  values per token).
- **P2** (`_ensure_self_kv_device`): self-KV stays resident — each step's self-KV
  *output* handle becomes the next step's *input*, no host round-trip. Only the
  (tiny) logits are copied back, for the argmax. `_Session.run_raw` returns the raw
  `DeviceArray`s that make this possible.

**10. Commit decision** (what gets frozen on screen). After decoding `result_tokens`
(length `T`), a token is committed only when **both** gates pass:

```
la_len  = longest prefix shared by the last `--commit-agreement` (2) hypotheses
          → "the model has stopped changing its mind"
delay_frames = --commit-delay (3 s) / 0.020 = 150 frames
age_len = int(T × max(0, (cross_kv_fill − 150) / cross_kv_fill))
          → tokens at least 3 s of audio behind the live frontier
commit_len = max(C, min(la_len, age_len))     # monotonic — never un-commit
```

*Worked example:* `cross_kv_fill = 250` (5 s), `T = 20`, previously `C = 5`. Then
`age_len = int(20 × (250−150)/250) = 8`; if the last two hypotheses agree on 12
tokens, `commit_len = max(5, min(12, 8)) = 8` → freeze 8 tokens. **LocalAgreement**
stops you committing a word still being revised; **commit-delay** stops you
committing a word so recent that more audio could still change it; `max(C, …)`
guarantees on-screen committed text never rewrites itself. `--full-decode` bypasses
all of this (re-decode from BOS every time: correct, but O(T²) and it flickers).

**11. Render & lifecycle.** `TerminalListener` overwrites the current line(s) in
place (ANSI) with the live transcript plus volume/buffer bars. On `speech_end` (or
buffer-full) the final line is locked with `complete_line()`, the utterance counter
ticks, `state.reset()` clears the memory + committed prefix, and it returns to
"Listening…".

## Notes

- `--full-decode` restores the baseline re-decode-from-BOS behaviour (instead of
  the default committed-prefix incremental decode).
- `--profile` records per-chunk worker timing and prints a real-time keep-up
  summary on exit (and shows the VAD calibration line).
- `--hw-type` selects the Torq hardware target (default `astra_machina`; use `sim`
  for a software simulation).
- `--preview-every`, `--commit-agreement`, `--commit-delay` tune the live-preview
  cadence and how eagerly tokens are frozen; `--vad-threshold` / `--vad-silence`
  tune speech detection and utterance splitting.
