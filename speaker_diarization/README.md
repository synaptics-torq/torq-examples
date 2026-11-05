# Speaker Diarization Demo

This demo finds who spoke when in a WAV file, and what each speaker said.
It uses two models on the NPU:

- **RedimNet** (`Redimnet2_b0-vox2-lm`, bf16, 2 s input) makes a 192-value speaker embedding for each audio window.
- **Moonshine streaming tiny** transcribes each speaker turn (see [moonshine_streaming](../moonshine_streaming/README.md)).

## How it works

1. An energy VAD finds the speech regions.
2. RedimNet embeds 2 s windows over each region, with a 0.5 s step.
3. Average-linkage clustering on cosine similarity groups the windows into speakers.
4. Each window votes for its speaker on the 80 ms chunks that it covers. Runs of one speaker become turns.
5. Moonshine transcribes each turn. A turn that is longer than the Moonshine memory (about 4.6 s) is cut at its quietest point.

RedimNet runs in a helper process (`src/embed.py`). The RedimNet VMFB and the
Moonshine VMFBs can need different Torq runtime versions, and one Python process
can load only one runtime. The NPU serves one process at a time, so the helper
finishes before Moonshine loads.

## Setup

1. Download the Moonshine model:

   ```sh
   python setup_demos.py moonshine_streaming
   ```

2. Get a RedimNet VMFB for the 2 s input (`1x32000xbf16` in, `1x192xbf16` out, function `main_graph`).

## Running

```sh
cd speaker_diarization
python src/diarize.py --wav /path/to/conversation.wav --embed-vmfb /path/to/Redimnet2_b0-vox2-lm_bf16_sl2610.vmfb --verbose
```

If Moonshine needs a different runtime than the venv runtime, set `PYTHONPATH` for the main process.
The helper process removes `PYTHONPATH` and uses the venv runtime:

```sh
PYTHONPATH=/path/to/torq-runtime-2.3.0 python src/diarize.py --wav conversation.wav --embed-vmfb redimnet.vmfb
```

Example output:

```
[00:00.8 - 00:02.7] Speaker 1: And so my fellow America!
[00:06.6 - 00:11.5] Speaker 2: Mr. Quilter is the apostle of the middle classes. And we are glad to welcome his gospel.
[00:13.0 - 00:19.3] Speaker 3: I have a dream That one day. Yes!
```

## Options

- `--num-speakers N` sets the number of speakers. Without it, `--cluster-threshold` (default 0.4) sets when clusters merge. Increase it to find more speakers.
- `--embed-hop` sets the step between embedding windows (default 0.5 s).
- `--min-turn` sets the shortest turn (default 1.0 s). A shorter turn joins a neighbour.
- `--vad-threshold` and `--vad-silence` tune the speech detection. `--vad-silence` (default 0.5 s) is the pause that ends a speech region.
- `--embed-python` and `--embed-pythonpath` select the Python and runtime for the helper process.
- `--tda cpu|dmabuf` selects the device allocator for both models (default dmabuf).
- `--verbose` prints the helper output and a timing summary.

## Limits

- The demo processes a complete file. It does not stream.
- Speaker changes are found with a resolution of about 1 s, because each embedding window is 2 s long.
- Overlapping speech gets one speaker only.
