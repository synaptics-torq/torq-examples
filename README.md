# Torq Examples

Simple examples demonstrating inference and profiling with [Torq](https://synaptics-torq.github.io/torq-compiler/v/latest/), using pre-compiled VMFB model binaries.

## Available Demos

| Demo | Description |
|------|-------------|
| [gemma3](gemma3/) | Interactive chat with Gemma 3 270M |
| [moonshine](moonshine/) | WAV file transcription with Moonshine-tiny (EN) |
| [object_detection](object_detection/) | YOLOv8n image and video object detection |

## Setup

Requires Python 3. Use a virtual environment and install requirements:

```sh
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install https://github.com/synaptics-torq/torq-examples/releases/download/torq-runtime-v2.0-alpha/torq_runtime-2.0.0a1-cp312-cp312-manylinux_2_28_aarch64.whl
pip install -r requirements.txt
```

Additionally, install any demo-specific dependencies:

```sh
# Example: Gemma 3
cd gemma3
pip install -r requirements.txt  # if present
cd ..

# Example: Object Detection
cd object_detection
pip install -r requirements.txt
cd ..
```

Run the setup script to install the repo on your Python path and download model assets:

```sh
# Set up a specific demo
python setup_demos.py gemma3

# Or set up everything
python setup_demos.py --all
```

Individual demos also have their own `setup_demo.py` for customizing setup, but the top-level `setup_demos.py` must be run at least once first.

Downloaded models are stored in `./models/` by default. Override with the `$MODELS` environment variable. Setup writes a small `.manifest.json` next to each downloaded model, so re-running setup reuses complete downloads and repairs incomplete model directories.

The manifest also records the Hugging Face revision the files came from. When a model repo is updated upstream, re-running setup detects the change and automatically refreshes the local copy — there's no need to manually delete the model directory. The demos apply the same check when they start, so inference refreshes stale models even if setup wasn't re-run. If Hugging Face is unreachable (e.g. offline), the existing local files are used and a warning is logged that they may be out of date. To skip the update check entirely (for fast or airgapped runs), pass `--no-refresh` to a demo's `infer.py`.

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

Each demo lives in its own directory. To run a demo, `cd` into its directory and run the demo scripts from inside the demo directory. For example, Gemma 3 interactive chat:

```sh
python src/infer.py -m ../models/Synaptics/gemma-3-270m-it-torq/model.vmfb.trim --instruct-model
```

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
