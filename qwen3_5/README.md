# Qwen3.5 Demo

Interactive chat with [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (text only) on
the Torq NPU, in non-thinking mode (the assistant turn starts with an empty `<think></think>`
block). The weights are INT4.

Qwen3.5 mixes 18 Gated DeltaNet (linear attention) layers with 6 gated-attention layers. The
DeltaNet layers keep a fixed-size state for the whole context, so the KV-cache window option has
no effect: when the 256-token context is full, the answer stops.

## Setup

See repo [README.md](../README.md) for installing the virtual environment and base dependencies.

Then enter the demo directory, install its dependencies, and jump back to the repo root:

```sh
cd qwen3_5
pip install -r requirements.txt
cd ..
```

From the repo root, run:

```sh
python setup_demos.py qwen3_5
```

This downloads the model files to `models/Synaptics/Qwen3.5-0.8B-int4-torq/`:
`transformer.vmfb`, `lm_head.vmfb`, `token_embeddings.npy`, `token_id_lut.npy`,
`config.json` and `tokenizer.json`.

By default the demo downloads the model version matching this repo's `VERSION` file (see the
[repo README](../README.md) for the versioning scheme). To pin a specific release tag instead, use:

```sh
cd qwen3_5
python setup_demo.py --model-version v2.3.0
```

## Running

Run the demo from the `qwen3_5` directory:

```sh
cd qwen3_5
python src/infer.py --instruct-model
```

`-m`/`--model` defaults to the downloaded `transformer.vmfb`. The sibling
`lm_head.vmfb` is picked up automatically. The context window is fixed at 256 tokens
(system prompt + conversation + answer).

> [!NOTE]
> The demo defaults to the DMA/dmabuf allocator with device I/O enabled. Use `--tda cpu` to run with the CPU allocator, or `--no-device-io` to pass user inputs as NumPy arrays.

Type `exit` or `quit` to stop the chat session. Run `python src/infer.py -h` to see all
available inference options.

## Build the model

The model files on Hugging Face come from one
[torq-tools](https://github.com/synaptics-torq/torq-tools) command (see its
`src/torq/models/qwen3_5/README.md`; it needs transformers 5.x in a separate virtualenv):

```sh
torq-export-model qwen3_5
```

It writes `transformer.vmfb`, `lm_head.vmfb` and the host files to
`models/Qwen/Qwen3.5-0.8B/export/torq/`. Pass that `transformer.vmfb` with `-m` to use
your own build.
