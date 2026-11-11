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

## Validation

`src/validate.py` runs the model with greedy decoding on 20 GSM8K grade-school math problems ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)); an answer passes when its last number is the labelled result:

```sh
cd qwen2_5
python src/validate.py --instruct-model
```

The rows are downloaded from Hugging Face on the first run and cached under `$MODELS/datasets/` (copy that folder to a board without network access). The data is not part of this repository: check the dataset card for its license and terms of use. `--dataset`, `--config`, `--split` and `--samples` select other data (the defaults are the `validation` list in `info.json`).

The script prints each item's result and the overall accuracy; `--metrics-out` records it as the `accuracy` metric next to the performance metrics of `src/infer.py`. There is no pass mark: the score is there so you can weigh accuracy against speed (e.g. between model variants). Run `python src/validate.py -h` to see all available validation options.

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
