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

Batched prompt prefill comes from the model itself. Model tags publish one of two layouts:

* **Native single-file build:** a lone `transformer.vmfb` with the prefill graphs compiled in, sharing one weight load. Nothing extra to download.
* **Legacy split build:** `transformer.vmfb` (decode) plus a separate fixed-token `transformer_prefill.vmfb`. That extra model uses more memory, so setup downloads it only with `--with-batch-prefill`:

```sh
cd gemma3
python setup_demo.py --with-batch-prefill
```

Re-running setup without the flag removes a downloaded `transformer_prefill.vmfb` from tracking (the local file is kept but no longer checked or refreshed).

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
> Batched prompt prefill is picked up automatically when available, in this order:
>
> * **Native prefill family:** a `transformer.vmfb` compiled with the native batch prefill feature contains multiple entries in one image that share a single weight load. The demo chunks the prompt with the largest entry that fits (a 21-token prompt runs as `main_16`, `main_4`, `main`), all but the final unit skip the LM head.
> * **Sibling prefill model:** when the main model has no native entries, a `transformer_prefill.vmfb` next to it (a fixed-size batched prefill model exported alongside `transformer.vmfb`) is picked up automatically.
>
> Any prompt remainder and all generated tokens still use `transformer.vmfb`, and the
> chunked and decode paths share the same KV/conv cache. Use `--batch-prefill-model PATH`
> to force a specific sibling prefill model, or `--no-batch-prefill-model` to force
> single-token prompt prefill.

Type `exit` or `quit` to stop the chat session. While an answer is being generated, press <kbd>Ctrl</kbd> + <kbd>C</kbd> or <kbd>Ctrl</kbd> + <kbd>D</kbd> to interrupt it and return to the prompt.

When the KV cache fills, the sliding window keeps the system prompt and re-prefills up to `--kv-cache-window N` recent tokens at their new positions (default: 2). It retains fewer history tokens when needed to fit the next prompt or generation step. Re-prefilling adds work at rollover, but keeps the cached keys' position encodings correct. With `--no-kv-cache-window`, generation stops at capacity and a later prompt starts a fresh conversation when it no longer fits.

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
