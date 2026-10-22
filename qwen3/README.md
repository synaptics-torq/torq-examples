# Qwen3 Demo

Interactive chat with [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on the
Torq NPU, in thinking mode: the model reasons in a `<think>` block before it
answers. The weights are INT4, calibrated with GPTQ.

## Setup

See repo [README.md](../README.md) for installing the virtual environment and base dependencies.

Then enter the demo directory, install its dependencies, and jump back to the repo root:

```sh
cd qwen3
pip install -r requirements.txt
cd ..
```

From the repo root, run:

```sh
python setup_demos.py qwen3
```

This downloads the model files to `models/Synaptics/Qwen3-0.6B-GPTQ-int4-torq/`:
`transformer.vmfb`, `lm_head.vmfb.trim`, `token_embeddings.npy`, `token_id_lut.npy`,
`config.json` and `tokenizer.json`.

By default the demo downloads the model version matching this repo's `VERSION` file (see the
[repo README](../README.md) for the versioning scheme). To pin a specific release tag instead, use:

```sh
cd qwen3
python setup_demo.py --model-version v2.3.0
```

## Running

Run the demo from the `qwen3` directory:

```sh
cd qwen3
python src/infer.py --instruct-model
```

`-m`/`--model` defaults to the downloaded `transformer.vmfb`. The sibling
`lm_head.vmfb.trim` is picked up automatically. The context window is fixed at 256 tokens
(system prompt + conversation + answer).

### Thinking mode

By default the model thinks before it answers. While it thinks, the demo shows
`[thinking...]`, and then it prints only the answer. Thinking gives better
answers on reasoning, word problems and code, but it takes time: a short
question usually needs 100 to 300 thinking tokens, which is about 20 to 50 s
before the answer starts.

| Option | Effect |
|---|---|
| `--show-thinking` | Print the `<think>` block before the answer |
| `--no-thinking` | Answer at once, without reasoning (faster, less accurate) |

The 256-token context also holds the thinking. When the context is full before
the model finishes thinking, the demo stops and prints a note instead of an
answer. Use `--no-thinking` for long questions.

> [!NOTE]
> The demo defaults to the DMA/dmabuf allocator with device I/O enabled. Use `--tda cpu` to run with the CPU allocator, or `--no-device-io` to pass user inputs as NumPy arrays.

Type `exit` or `quit` to stop the chat session. Run `python src/infer.py -h` to see all
available inference options.

## Build the model

The model files on Hugging Face come from one
[torq-tools](https://github.com/synaptics-torq/torq-tools) command (see its
`src/torq/models/qwen/README.md`):

```sh
torq-export-model qwen -s qwen3-0.6b
```

It writes `transformer.vmfb`, `lm_head.vmfb.trim` and the host files to
`models/<hf-repo>/export/torq/converted/static/`. Pass that `transformer.vmfb`
with `-m` to use your own build.
