# Qwen3 Demo (experimental)

Interactive chat with [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on the
Torq NPU, in non-thinking mode (the assistant turn starts with an empty
`<think></think>` block). The weights are INT4, calibrated with GPTQ.

> [!WARNING]
> Experimental. The model was compiled with a development Torq compiler and has not
> been validated on hardware yet. Its INT4 accuracy is below the Gemma 3 demo's
> (see the [model card](https://huggingface.co/Synaptics/Qwen3-0.6B-GPTQ-int4-torq)).
> The output vocabulary is trimmed to Latin text, punctuation and digits.

## Setup

See repo [README.md](../README.md) for installing the virtual environment and base dependencies.

The model repo `Synaptics/Qwen3-0.6B-GPTQ-int4-torq` is **private**
for now. Log in once with a Hugging Face account that can read the Synaptics org:

```sh
huggingface-cli login     # huggingface-hub 0.31 (requirements.txt); `hf auth login` on newer versions
```

Alternatively, export `HF_TOKEN=<token>` in the shell that runs setup.

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
`transformer.vmfb`, `lm_head.vmfb`, `token_embeddings.npy`, `token_id_lut.npy`,
`config.json` and `tokenizer.json`.

The repo has no release tags yet, so setup tracks its `main` branch. You can pin a
specific commit or tag with `--model-version` instead:

```sh
cd qwen3
python setup_demo.py --model-version <commit-or-tag>
```

## Running

Run the demo from the `qwen3` directory:

```sh
cd qwen3
python src/infer.py --instruct-model
```

`-m`/`--model` defaults to the downloaded `transformer.vmfb`. The sibling
`lm_head.vmfb` is picked up automatically. The context window is fixed at 256 tokens
(system prompt + conversation + answer).

> [!NOTE]
> The demo defaults to the DMA/dmabuf allocator with device I/O enabled. Use `--tda cpu` to run with the CPU allocator, or `--no-device-io` to pass user inputs as NumPy arrays.

Type `exit` or `quit` to stop the chat session. Run `python src/infer.py -h` to see all
available inference options.
