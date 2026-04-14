# Torq Examples

Simple examples demonstrating inference and profiling with [Torq](https://github.com/synaptics-synap/torq), using pre-compiled VMFB model binaries.

## Available Demos

| Demo | Description |
|------|-------------|
| [gemma3](gemma3/) | Interactive chat with Gemma 3 270M |

## Setup

Requires Python 3. Use a virtual environment and install requirements:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install torq_runtime-*.whl
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

Downloaded models are stored in `./models/` by default. Override with the `$MODELS` environment variable.

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
python src/infer.py -m ../models/Synaptics/gemma-3-270m-it/model.vmfb --instruct-model
```

Run `python src/infer.py -h` to see all available inference options.

## Profiling

`profile.py` at the repo root is a model-agnostic profiling tool. Point it at any VMFB:

```sh
python profile.py models/Synaptics/gemma-3-270m-it/model.vmfb -r 5
```
