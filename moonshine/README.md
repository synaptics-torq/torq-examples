# Moonshine Demo

WAV file transcription with Moonshine-tiny English using Torq VMFB models.

## Setup

See repo [README.md](../README.md) for installing the virtual environment and base dependencies.

Enter the demo directory. Install its dependencies. Jump back to the repo root.

```sh
cd moonshine
pip install -r requirements.txt
cd ..
```

From the repo root, run:

```sh
python setup_demos.py moonshine
```

This downloads the default model files to `models/Synaptics/moonshine-tiny-bf16-torq/`

By default the demo downloads the current `latest` model revision from the HF repo. To pin a specific release tag instead, use:

```sh
cd moonshine
python setup_demo.py --model-version v2.1.0
```

The `--model-version` flag accepts a Hugging Face revision or tag name, such as `latest` or a release like `v2.1.0`.


## Running

Run the demo from the `moonshine` directory and pass one or more WAV files:

```sh
cd moonshine
python src/infer.py -m ../models/Synaptics/moonshine-tiny-bf16-torq path/to/audio.wav
```

> [!NOTE]
> The demo defaults to the DMA/dmabuf allocator. Use `--tda cpu` to run with the CPU allocator, or `--device-io` to experiment with explicit device-backed encoder I/O.

Run `python src/infer.py -h` to see all available inference options.
