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

This downloads the default model, `instruct-int4-gptq`, to: `models/Synaptics/gemma-3-270-GPTQ-int4-it-torq/`

The demo has three models:

| Name | Hugging Face repo | Model |
|---|---|---|
| `instruct-int4-gptq` (default) | `Synaptics/gemma-3-270-GPTQ-int4-it-torq` | Instruct, INT4 weights (GPTQ, block size 32), separate LM head (`transformer.vmfb`, `lm_head.vmfb.trim`), no batched prefill model |
| `instruct` | `Synaptics/gemma-3-270m-it-torq` | Instruct, with a batched prefill model |
| `default` | `Synaptics/gemma-3-270m-torq` | Base (pre-trained), with a batched prefill model |

To set up another model, give its name to the demo's setup:

```sh
cd gemma3
python setup_demo.py instruct
```

By default the demo downloads the model version matching this repo's `VERSION` file (see the
[repo README](../README.md) for the versioning scheme). To pin a specific release tag instead, use:

```sh
cd gemma3
python setup_demo.py --model-version v2.3.0
```

A pinned version is kept in sync with its own tag but never upgraded. `--no-update` skips model tracking entirely (no `.manifest.json`), at your own risk.

By default setup does **not** download the optional batched prefill model. To also fetch and track `transformer_prefill.vmfb`, pass `--with-batch-prefill`:

```sh
cd gemma3
python setup_demo.py --with-batch-prefill
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
> Any prompt remainder and all generated tokens still use `transformer.vmfb`, and both paths share the same KV cache. Use `--batch-prefill-model PATH` to point at a specific prefill model, or `--no-batch-prefill-model` to force single-token prompt prefill.

Type `exit` or `quit` to stop the chat session. While an answer is being generated, press <kbd>Ctrl</kbd> + <kbd>C</kbd> or <kbd>Ctrl</kbd> + <kbd>D</kbd> to interrupt it and return to the prompt.

Run `python src/infer.py -h` to see all available inference options.

## Validation

Gemma 3 includes a validation script for text translation datasets. For example:

```sh
cd gemma3
python src/validate.py --instruct-model --max-samples 10
```

Run `python src/validate.py -h` to see all available validation options.

## Build the model

The instruct model on Hugging Face (`Synaptics/gemma-3-270-GPTQ-int4-it-torq`)
comes from one [torq-tools](https://github.com/synaptics-torq/torq-tools) command
(see its `src/torq/models/gemma3/README.md`):

```sh
torq-export-model gemma3 --instruct-model --trim-vocab
```

The command quantizes the weights to INT4 with the GPTQ checkpoint of the model
repository and compiles with the stream-schedule prefetch. It writes
`transformer.vmfb`, `transformer_prefill.vmfb`, `lm_head.vmfb.trim` and the host
files to `models/google/gemma-3-270m-it/export/trim/split_lm_head/quantized/static/compiled/`.
Copy that directory to the board and pass its `transformer.vmfb` with `-m` to
use your own build. The demo picks up `transformer_prefill.vmfb` (see the tip
in [Running](#running)).
