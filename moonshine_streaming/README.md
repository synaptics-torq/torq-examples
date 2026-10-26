# Moonshine Streaming Demo

WAV file transcription with Moonshine-tiny English using a Torq VMFB model split into a fused encoder and a KV decoder. A self-calibrating energy VAD splits the file into utterances and a committed-prefix incremental decoder gives a live preview as each utterance is transcribed.

## Setup

From the repo root, run:

```sh
python setup_demos.py moonshine_streaming
```

This downloads the default model files from our HuggingFace repo to:

```sh
models/Synaptics/moonshine-streaming-tiny-torq/
```

By default the demo downloads the model version matching this repo's `VERSION` file. The [repo README](../README.md) describes the versioning scheme. `setup_demo.py` also supports `--model-version` to pin a specific release tag and `--no-update` to skip model tracking entirely.

## Transcribing a WAV file

```sh
python src/infer.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --wav /path/to/sample.wav
```

This reads the file with `soundfile`, mixes stereo to mono, resamples to 16 kHz, and feeds the result through the VAD/encode/decode pipeline. It exits automatically once the file is fully transcribed. A silent 1 s lead-in is synthesized ahead of the file so the VAD has something to calibrate against, since a recording often starts talking immediately.

> [!NOTE]
> The demo defaults to the DMA/dmabuf allocator and automatically uses device-backed I/O. Use `--tda cpu` to run with host NumPy inputs, or `--device-io` to explicitly enable device-backed I/O with another allocator.

## Options

- `--wav <file>` specifies the required wave file to transcribe
- `--realtime` paces the feed to match playback speed.
- `--full-decode` restores the baseline re-decode-from-BOS behaviour in place of the default committed-prefix incremental decode
- `--preview-every`, `--commit-agreement`, `--commit-delay` tune the live-preview cadence and how eagerly tokens are frozen
-  `--vad-threshold` / `--vad-silence` / `--vad-lookback` tune speech detection and utterance splitting.
- `--no-refresh` skips the Hugging Face check for updated models
- `--verbose` prints a real-time factor summary on exit
- `--runtime-flags` forwards flags straight to the Torq runtime and must come last, since every remaining argument is passed through

### Notes on Options

By default the file is fed as fast as possible for quick batch testing. Pass `--realtime` to pace the feed to match the file's real playback speed instead, so you can watch the live preview update the way it would from a live stream:

```sh
python src/infer.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --wav /path/to/sample.wav --realtime
```

The board defaults are `--vad-silence 2.5`, `--vad-threshold 0.010` and `--preview-every 5`, so the command above is equivalent to:

```sh
python src/infer.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --wav /path/to/sample.wav \
    --vad-silence 2.5 --vad-threshold 0.010 --preview-every 5
```

> [!TIP]
> Tune the VAD parameters for your audio. The defaults are a starting point and may need adjustment. Transcription quality depends far more on where utterances get split than on anything else in the pipeline.
>
> `--vad-silence` matters most. It sets how long a pause must last before the current utterance is closed and flushed. Too low and the VAD cuts mid-sentence on natural pauses, so the decoder loses the context that makes the rest of the sentence accurate; too high and utterances run together and the transcript lags. If your output is fragmented, raise it; if it feels sluggish or sentences merge, lower it.
>
> Also worth adjusting:
>
> - `--vad-threshold`: raise it for a noisy recording so background noise does not trigger speech, lower it for a quiet/soft recording.
> - `--vad-lookback`: raise it if word onsets are clipped at the start of utterances.

Run `python src/infer.py -h` to see all available options, including VAD thresholds, decode mode, runtime flags and verbose stats.

## How it works

This demo consumes a WAV file chunk-by-chunk, detects when speech is present, and updates the transcript incrementally as it's processed, freezing words once it is confident about them. Two ideas make this cheap:

- **A growing memory instead of a growing input.** The encoder is fed one small audio chunk at a time and folds each chunk into a cross-attention memory, rather than re-encoding the whole utterance so far. The self-calibrating `EnergyVAD` decides when an utterance starts and ends, so this memory is reset between utterances instead of growing forever.
- **Resuming instead of restarting.** Live previews don't re-decode the utterance from scratch. Once the model has stopped revising a token *and* enough new audio has arrived behind it, that token is "committed" and its decoder state is reused as-is on the next preview. Only the still-changing tail is regenerated. This makes each preview cost O(new tokens) instead of O(all tokens so far), and it means committed text on screen never rewrites itself. Passing `--full-decode` disables this and re-decodes from scratch every time, which is simpler but slower and visibly flickers as it revises earlier words.

The decoder runs periodically during speech to update the live preview and performs a final decode when the VAD detects the end of an utterance. It also finalizes when cross-attention memory fills, limiting the size of long utterances, and at end-of-file to flush any remaining utterance.

### Code layout

| File | Role |
|------|------|
| `src/runner.py` | the **engine**: the model, its pre-allocated state, and the inference logic. |
| `src/infer.py` | the **app**: WAV file feeding, VAD, the worker thread, decode triggering, terminal rendering, and the CLI. |
| `setup_demo.py` | downloads/verifies the model files using `utils/`. |

The Moonshine-tiny export has two parts: a fused `encoder.vmfb` that maps audio to cross-attention memory and a `decoder.vmfb` that generates tokens autoregressively from that memory. Streaming settings such as chunk size, lookahead and buffer capacity live in `streaming_config.json` alongside the VMFBs in the model directory.

## References

This demo's streaming architecture and portions of its code are adapted from [moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine), using its MIT-licensed C++ streaming VAD/STT implementation in [moonshine-ai/moonshine's `micro`](https://github.com/moonshine-ai/moonshine/tree/main/micro) directory.

Only the MIT-licensed code and English-language models from that repo apply here. Moonshine-ai's non-English models ship under a separate, more restrictive Community License with revenue caps and registration requirements. This demo does not use those models.
