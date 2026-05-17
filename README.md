# Torq Examples

Simple examples demonstrating inference and profiling with [Torq](https://synaptics-torq.github.io/torq-compiler/v/latest/), using pre-compiled VMFB model binaries.

## Available Demos

| Demo | Description |
|------|-------------|
| [gemma3](gemma3/) | Interactive chat with Gemma 3 270M |
| [moonshine](moonshine/) | WAV file transcription with Moonshine-tiny (EN) |

## Setup

Requires Python 3. Use a virtual environment and install requirements:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install https://github.com/synaptics-torq/torq-examples/releases/download/torq-runtime-v2.0-alpha/torq_runtime-2.0.0a1-cp312-cp312-manylinux_2_28_aarch64.whl
pip install -r requirements.txt
```

Run the setup script to install the repo on your Python path and download model assets:

```sh
# Set up a specific demo
python setup_demos.py gemma3

# Or set up everything
python setup_demos.py --all
```

Individual demos also have their own `setup.py` for customizing setup, but `setup_demos.py` must be run at least once first.

Downloaded models are stored in `./models/` by default. Override with the `$MODELS` environment variable. Setup writes a small `.manifest.json` next to each downloaded model, so re-running setup reuses complete downloads and repairs incomplete model directories.

> [!TIP]
> Some models may require a HuggingFace access token. Set `HF_TOKEN` in your environment before running setup:
> ```sh
> export HF_TOKEN=hf_...
> ```
> or
> ```sh
> HF_TOKEN=hf_... python setup_demos.py
> ```

## Running a Demo

Each demo lives in its own directory. To run a demo, `cd` into its directory and install any demo-specific dependencies first:

```sh
cd gemma3
pip install -r requirements.txt  # if present
```

Then run the demo scripts from inside the demo directory. For example, Gemma 3 interactive chat:

```sh
python src/infer.py -m ../models/Synaptics/gemma-3-270m-it-torq/model.vmfb.trim --instruct-model
```

If the downloaded repo only contains `model.vmfb`, use that path instead.

Run `python src/infer.py -h` to see all available inference options.

## Profiling

`profile.py` at the repo root is a model-agnostic profiling tool. Point it at any VMFB:

```sh
python profile.py models/Synaptics/gemma-3-270m-it-torq/model.vmfb -r 5
```

## Validation

Some demos include built-in validation scripts. For example, Gemma 3 can be validated on a text translation dataset:

```sh
cd gemma3
python src/validate.py -m ../models/Synaptics/gemma-3-270m-it-torq/model.vmfb.trim --instruct-model --max-samples 10
```

Run the validation script with `-h` to see all available options.
