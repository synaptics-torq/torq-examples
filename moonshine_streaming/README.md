# Moonshine Streaming Demo

WAV file transcription with Moonshine-tiny English using a 2-split Torq VMFB
model (fused encoder + KV decoder). A self-calibrating energy VAD splits the
file into utterances and a committed-prefix incremental decoder gives a live
preview as each utterance is transcribed.

## Setup

From the repo root, run:

```sh
python setup_demos.py moonshine_streaming
```

This downloads the default model files from our HuggingFace repo to:

```sh
models/Synaptics/moonshine-streaming-tiny-torq/
```

## Transcribing a WAV file

```sh
python src/infer.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --wav /path/to/sample.wav
```

This reads the file with `soundfile` (mixed to mono if stereo, resampled to
16 kHz), feeds it through the VAD/encode/decode pipeline, and exits
automatically once the file is fully transcribed. A silent 1 s lead-in is
synthesized ahead of the file so the VAD has something to calibrate against,
since a recording often starts talking immediately.


## Options
- `--wav <file>` (required) wave file to transcribe
- `--realtime` paces the feed to match playback speed.
- `--full-decode` restores the baseline re-decode-from-BOS behaviour (instead of
  the default committed-prefix incremental decode)
- `--preview-every`, `--commit-agreement`, `--commit-delay` tune the live-preview
  cadence and how eagerly tokens are frozen 
-  `--vad-threshold` / `--vad-silence` / `--vad-lookback` tune speech detection and utterance splitting.
- `--no-refresh` skips the Hugging Face check for updated models
- `--verbose` prints a real-time factor (RTF) and decoder tok/s summary on exit
- `--runtime-flags` forwards flags straight to the Torq runtime and must come
last, since every remaining argument is passed through

### Notes on Options
By default the file is fed as fast as possible (for quick batch testing). Pass
`--realtime` to pace the feed to match the file's real playback speed instead,
so you can watch the live preview update the way it would from a live stream:

```sh
python src/infer.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --wav /path/to/sample.wav --realtime
```

The defaults are tuned for the board (`--vad-silence 2.5`, `--vad-threshold 0.010`, 
`--preview-every 5`), so the command above is equivalent to:

```sh
python src/infer.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --wav /path/to/sample.wav \
    --vad-silence 2.5 --vad-threshold 0.010 --preview-every 5
```

> [!TIP]
> **Tune the VAD parameters for your own audio — the defaults are a starting
> point, not a good fit for every recording.** Transcription quality depends
> far more on where utterances get split than on anything else in the
> pipeline.
>
> `--vad-silence` matters most. It sets how long a pause must last before the
> current utterance is closed and flushed. Too low and the VAD cuts mid-sentence
> on natural pauses, so the decoder loses the context that makes the rest of the
> sentence accurate; too high and utterances run together and the transcript
> lags. If your output is fragmented, raise it; if it feels sluggish or
> sentences merge, lower it.
>
> Also worth adjusting:
> - `--vad-threshold` — raise it for a noisy recording so background noise does
>   not trigger speech, lower it for a quiet/soft recording.
> - `--vad-lookback` — raise it if word onsets are clipped at the start of
>   utterances.

Run `python src/infer.py -h` to see all available options (VAD thresholds, decode
mode, runtime flags, verbose stats).

## How it works

This demo consumes a WAV file chunk-by-chunk, detects when speech is present,
and updates the transcript incrementally as it's processed — freezing words
once it is confident about them. Two ideas make this cheap:

- **A growing memory instead of a growing input.** The encoder is fed one
  small audio chunk at a time and folds each chunk into a cross-attention
  memory, rather than re-encoding the whole utterance so far. A self-calibrating
  energy VAD (`EnergyVAD`) decides when an utterance starts and ends, so this
  memory is reset between utterances instead of growing forever.
- **Resuming instead of restarting.** Live previews don't re-decode the
  utterance from scratch. Once the model has stopped revising a token *and*
  enough new audio has arrived behind it, that token is "committed" and its
  decoder state is reused as-is on the next preview — only the still-changing
  tail is regenerated. This makes each preview cost O(new tokens) instead of
  O(all tokens so far), and it means committed text on screen never rewrites
  itself. Passing `--full-decode` disables this and re-decodes from scratch
  every time, which is simpler but slower and visibly flickers as it revises
  earlier words.

The decoder runs on a few different triggers: periodically while an utterance
is still being spoken (the live preview), when the VAD detects the end of an
utterance (final decode), when the cross-attention memory fills up (forced
finalize, so a long utterance doesn't grow unbounded), and at end-of-file (to
flush whatever utterance was still in progress).

### Code layout

| File | Role |
|------|------|
| `src/runner.py` | the **engine**: the model, its pre-allocated state, and the inference logic. |
| `src/infer.py` | the **app**: WAV file feeding, VAD, the worker thread, decode triggering, terminal rendering, and the CLI. |
| `setup_demo.py` | downloads/verifies the model files (reuses `utils/`). |

The model is a 2-split Moonshine-tiny export: a fused `encoder.vmfb` (audio →
cross-attention memory) and a `decoder.vmfb` (memory → tokens, autoregressively).
Its streaming-specific tunables (chunk size, lookahead, buffer capacity, etc.)
live in `streaming_config.json` alongside the VMFBs in the model directory.

## References

This demo's streaming architecture and portions of its code are adapted from
[moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine), used
under the MIT License — specifically the C++ streaming VAD/STT implementation
in [moonshine-ai/moonshine's `micro`](https://github.com/moonshine-ai/moonshine/tree/main/micro)
directory.

Note: only the MIT-licensed code and English-language models from that repo
apply here — moonshine-ai's non-English models ship under a separate, more
restrictive Community License (revenue-capped, registration required) that
this demo does not use.


