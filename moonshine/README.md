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

By default the demo downloads the model version matching this repo's `VERSION` file (see the
[repo README](../README.md) for the versioning scheme). To pin a specific release tag instead, use:

```sh
cd moonshine
python setup_demo.py --model-version v2.2.0
```

A pinned version is kept in sync with its own tag but never upgraded. `--no-update` skips model tracking entirely (no `.manifest.json`), at your own risk.


## Running

Run the demo from the `moonshine` directory and pass one or more WAV files:

```sh
cd moonshine
python src/infer.py -m ../models/Synaptics/moonshine-tiny-bf16-torq path/to/audio.wav
```

> [!NOTE]
> The demo defaults to the DMA/dmabuf allocator and automatically uses device-backed I/O. Use `--tda cpu` to run with host NumPy inputs, or `--device-io` to explicitly enable device-backed I/O with another allocator.

Run `python src/infer.py -h` to see all available inference options.

## Validation

`src/validate.py` transcribes 20 utterances of LibriSpeech test-clean ([openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr)) and scores each transcript against its reference with the word error rate:

```sh
cd moonshine
python src/validate.py
```

The rows are downloaded from Hugging Face on the first run and cached under `$MODELS/datasets/` (copy that folder to a board without network access). The data is not part of this repository: check the dataset card for its license and terms of use. `--dataset`, `--config`, `--split` and `--samples` select other data (the defaults are the `validation` list in `info.json`).

The script prints PASS/FAIL against the demo's maximum WER and exits 1 on FAIL; `--metrics-out` records the same metrics as `src/infer.py`. Run `python src/validate.py -h` to see all available validation options.
