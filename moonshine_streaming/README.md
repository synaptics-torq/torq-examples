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

## Notes

- `--full-decode` restores the baseline re-decode-from-BOS behaviour (instead of
  the default committed-prefix incremental decode).
- `--profile` records per-chunk worker timing and prints a real-time keep-up
  summary on exit.
- `--hw-type` selects the Torq hardware target (default `astra_machina`; use `sim`
  for a software simulation).
