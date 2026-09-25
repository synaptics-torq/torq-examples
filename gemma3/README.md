# Gemma 3 Demo

Interactive chat with Gemma 3 270M using Torq VMFB models.

## Setup

See repo [README.md](../README.md) for installing the virtual environment and base dependencies.

Enter the demo directory. Install its dependencies. Jump back to the repo root.

```sh
cd gemma3
pip install -r requirements.txt
cd ..
```

From the repo root, run:

```sh
python setup_demos.py gemma3
```

This downloads the default instruct model files to: `models/Synaptics/gemma-3-270m-it-torq/`

By default the demo downloads the model version matching this repo's `VERSION` file (see the
[repo README](../README.md) for the versioning scheme). To pin a specific release tag instead, use:

```sh
cd gemma3
python setup_demo.py --model-version v2.1.0
```

A pinned version is kept in sync with its own tag but never upgraded. `--no-update` skips model tracking entirely (no `.manifest.json`), at your own risk.

By default setup does **not** download the optional batched prefill model. To also fetch and track `transformer_prefill.vmfb`, pass `--with-prefill`:

```sh
cd gemma3
python setup_demo.py --with-prefill
```

`transformer_prefill.vmfb` is an extra model, so it uses more memory, and its prompt-chunk size is fixed to 64 tokens. It is a no-op with a warning for repos that have no prefill model. Re-running setup without the flag removes it from tracking (the local file is kept but no longer checked or refreshed).

## Running

Run the demo from the `gemma3` directory:

```sh
cd gemma3
python src/infer.py --instruct-model
```

`-m`/`--model` defaults to the model downloaded during setup. Pass `-m` to use a different (e.g. a custom HF repo) model instead.

> [!NOTE]
> The demo defaults to the DMA/dmabuf allocator with device I/O enabled. Use `--tda cpu` to run with the CPU allocator, or `--no-device-io` to pass user inputs as NumPy arrays.

> [!TIP]
> When the model directory contains a `transformer_prefill.vmfb` (a fixed-size batched prefill model exported alongside `transformer.vmfb`), the demo picks it up automatically: complete prompt chunks run through the prefill model and only the final prompt unit uses the LM head, so time-to-first-token drops sharply.
> Any prompt remainder and all generated tokens still use `transformer.vmfb`, and both paths share the same KV cache. Use `--prefill-model PATH` to point at a specific prefill model, or `--no-prefill-model` to force single-token prompt prefill.

Type `exit` or `quit` to stop the chat session. While an answer is being generated, press <kbd>Ctrl</kbd> + <kbd>C</kbd> or <kbd>Ctrl</kbd> + <kbd>D</kbd> to interrupt it and return to the prompt.

Run `python src/infer.py -h` to see all available inference options.

## Validation

Gemma 3 includes a validation script for text translation datasets. For example:

```sh
cd gemma3
python src/validate.py --instruct-model --max-samples 10
```

Run `python src/validate.py -h` to see all available validation options.
