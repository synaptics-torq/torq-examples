# Qwen2.5 Demo

Interactive chat with [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
on the Torq NPU. The weights are INT4, calibrated with GPTQ.

## Setup

See repo [README.md](../README.md) for installing the virtual environment and base dependencies.

Then enter the demo directory, install its dependencies, and jump back to the repo root:

```sh
cd qwen2_5
pip install -r requirements.txt
cd ..
```

From the repo root, run:

```sh
python setup_demos.py qwen2_5
```

This downloads the model files to `models/Synaptics/Qwen2.5-0.5B-Instruct-GPTQ-int4-torq/`:
`transformer.vmfb`, `lm_head.vmfb.trim`, `token_embeddings.npy`, `token_id_lut.npy`,
`config.json` and `tokenizer.json`.

By default the demo downloads the model version matching this repo's `VERSION` file (see the
[repo README](../README.md) for the versioning scheme). To pin a specific release tag instead, use:

```sh
cd qwen2_5
python setup_demo.py --model-version v2.3.0
```

## Running

Run the demo from the `qwen2_5` directory:

```sh
cd qwen2_5
python src/infer.py --instruct-model
```

`-m`/`--model` defaults to the downloaded `transformer.vmfb`. The sibling
`lm_head.vmfb.trim` is picked up automatically. The context window is fixed at 256 tokens
(system prompt + conversation + answer).

> [!NOTE]
> The demo defaults to the DMA/dmabuf allocator with device I/O enabled. Use `--tda cpu` to run with the CPU allocator, or `--no-device-io` to pass user inputs as NumPy arrays.

Type `exit` or `quit` to stop the chat session. Run `python src/infer.py -h` to see all
available inference options.

## Build the model

The model files on Hugging Face come from one
[torq-tools](https://github.com/synaptics-torq/torq-tools) command (see its
`src/torq/models/qwen/README.md`):

```sh
torq-export-model qwen -s qwen2.5-0.5b-instruct
```

It writes `transformer.vmfb`, `lm_head.vmfb.trim` and the host files to
`models/<hf-repo>/export/torq/converted/static/`. Pass that `transformer.vmfb`
with `-m` to use your own build.
