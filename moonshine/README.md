# Moonshine Demo

WAV file transcription with Moonshine-tiny English using Torq VMFB models.

## Setup

From the repo root, run:

```sh
python setup_demos.py moonshine
```

This downloads the default model files to:

```sh
models/Synaptics/moonshine-tiny-bf16-torq/
```

## Running

Run the demo from the `moonshine` directory and pass one or more WAV files:

```sh
cd moonshine
python src/infer.py -m ../models/Synaptics/moonshine-tiny-bf16-torq path/to/audio.wav
```

> [!NOTE]
> The demo defaults to the DMA/dmabuf allocator. Use `--tda cpu` to run with the CPU allocator, or `--device-io` to experiment with explicit device-backed encoder I/O.

Run `python src/infer.py -h` to see all available inference options.
