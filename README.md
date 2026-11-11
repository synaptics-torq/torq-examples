# Torq Examples

Simple examples demonstrating inference and profiling with [Torq](https://synaptics-torq.github.io/torq-compiler/v/latest/), using pre-compiled VMFB model binaries.

## Available Demos

| Demo | Task category | Description |
|------|---------------|-------------|
| [gemma3](gemma3/) | text_generation | Interactive chat with Gemma 3 270M |
| [functiongemma](functiongemma/) | text_generation | Offline function calling with FunctionGemma 270M (BF16 recommended; precision comparison included) |
| [qwen2_5](qwen2_5/) | text_generation | Interactive chat with Qwen2.5 0.5B Instruct |
| [qwen3](qwen3/) | text_generation | Interactive chat with Qwen3 0.6B (thinking mode) |
| [LiquidAI-LFM2.5](lfm2_5/) | text_generation | Interactive text chat with LiquidAI LFM2.5 (230M / 350M) |
| [moonshine](moonshine/) | automatic_speech_recognition | WAV file transcription with Moonshine-tiny (EN) |
| [moonshine_streaming](moonshine_streaming/) | automatic_speech_recognition | Streaming WAV transcription with Moonshine-tiny (EN) |
| [object_detection](object_detection/) | object_detection | YOLOv8n, YOLOv26n and YOLOv26s image and video object detection |
| [pose_estimation](pose_estimation/) | keypoint_detection | YOLOv8n pose estimation |
| [face_id](face_id/) | face_recognition | Face detection on an image file |
| [LiquidAI-LFM2-VL-450M](lfm2_vl/) | image_text_to_text | Image Q&A / VLM with LiquidAI LFM2-VL-450M (INT8) |
| [EmbeddingGemma-2](embeddinggemma2/) | feature_extraction | Multimodal embeddings (text, image, video, audio) with Google EmbeddingGemma-2, and search over a media folder |

> [!NOTE]
> Renamed demo folders: `LiquidAI/LiquidAI-LFM2.5` is now `lfm2_5`, `LiquidAI/LiquidAI-LFM2-VL-450M` is now `lfm2_vl`, `Face_ID` is now `face_id`, and `EmbeddingGemma-2` is now `embeddinggemma2`. `setup_demos.py` and `run_demo.py` still accept the old names for one release, with a deprecation warning (the `EmbeddingGemma-2` demo name itself is unchanged; only its folder moved).


## Setup

Requires Python 3. 

1. Use a virtual environment and install requirements. 

    From the repo root:

    ```sh
    python3 -m venv .venv --system-site-packages

    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. Additionally, install any demo-specific dependencies:

    Enter the demo directory. Install its requirements. Jump back to the repo root.

    ```sh
    cd <demo app directory>
    pip install -r requirements.txt  # if present
    cd ..
    ```

    Examples: 
    ```sh
    # Example: Gemma 3
    cd gemma3
    pip install -r requirements.txt
    cd ..

    # Example: Object Detection
    cd object_detection
    pip install -r requirements.txt
    cd ..
    ```

3. Run the setup script to install the repo on your Python path and download model assets:


    From the repo root: 
    ```sh
    # Set up a specific demo
    python setup_demos.py gemma3

    # Or set up everything
    python setup_demos.py --all
    ```

Individual demos also have their own `setup_demo.py` for customizing setup, but the top-level `setup_demos.py` must be run at least once first.

Downloaded models are stored in `./models/` by default. Override with the `$MODELS` environment variable. Setup writes a small `.manifest.json` next to each downloaded model, so re-running setup reuses complete downloads and repairs incomplete model directories.

### Model versions

This repo's version is recorded in the `VERSION` file at the root and tracks the torq-compiler release the demos are tested against. Built-in model repos on Hugging Face carry matching tags, so a checkout of examples `2.2.0` downloads and maintains the models at tag `v2.2.0`. Setup **fails loudly** if that version does not exist in a model repo.

- Re-running setup or starting a demo re-checks the tracked tag and refreshes the local copy if the tag moved upstream or local files went missing/corrupt. When the tag moved, only the files whose content actually changed are re-downloaded (the manifest records per-file hashes); a copy whose manifest predates per-file tracking is re-downloaded in full once and then upgrades its manifest.
- A model is **never** upgraded to a newer version on its own: it stays on the version it was set up with until you explicitly change it. There is no `latest` tracking.
- `--model-version vX.Y.Z` (per-demo `setup_demo.py` or `setup_demos.py`) pins a specific model version. A pinned version is kept in sync with its own tag but never upgraded.
- `name:version` in a model argument pins a single model, e.g. `python setup_demo.py default:v2.0.0 custom/gemma3`.
- Custom (non-Synaptics) repos have no default version: `python setup_demo.py custom/gemma3` downloads the repo's latest (HEAD) revision, untracked.
- `--no-update` downloads without writing a `.manifest.json`; the model is then excluded from the tracking system entirely (no checks, no refreshes), at your own risk.

### Offline / airgapped runs

Version checks use a short timeout (default 5s; override with `TORQ_HF_CHECK_TIMEOUT`) and fall back to the local files with a warning when Hugging Face is unreachable, so the demos run fully offline once their models are present. To skip the check entirely (for fast or airgapped runs), pass `--no-refresh` to a demo's `infer.py`.

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

Each demo lives in its own folder. From the repo root, `run_demo.py` runs a demo by its `info.json` name (or its folder) and passes the remaining arguments through unchanged:

```sh
python run_demo.py gemma3 --instruct-model
python run_demo.py object_detection:infer_video --video /path/to/video.mp4   # another script of the demo
```

The part after the `:` names the script to run (default `infer`). You can also run the demo's script directly — the setup step above puts the repo root on the Python path, so this works from the repo root or from inside the demo folder:

```sh
python gemma3/src/infer.py --instruct-model
```

The `-m`/`--model` (or `--model-dir`) flag is optional in every demo: it defaults to the model the demo's setup downloaded, and fails with a pointer to `setup_demos.py` when no local model is present.

Every inference script accepts `--metrics-out FILE` to write the run's performance metrics as JSON to FILE at exit (the format is documented in [CONTRIBUTING.md](CONTRIBUTING.md)).

Run `python run_demo.py <demo> -h` to see all available inference options.

## Profiling

`profile.py` at the repo root is a model-agnostic profiling tool. Point it at any VMFB:

```sh
python profile.py models/Synaptics/gemma-3-270-GPTQ-int4-it-torq/transformer.vmfb -r 5
```

## Validation

Every demo has a `src/validate.py` that scores the model on public Hugging Face datasets (the `validation` list in the demo's `info.json`), downloaded on the first run and cached under `$MODELS/datasets/`; the repository ships no validation data:

```sh
python run_demo.py gemma3:validate --instruct-model
```

The script prints one line per item, then the overall score (accuracy, or word error rate for speech), which `--metrics-out` records next to the performance metrics, together with the datasets and sample counts it was measured on. Validation never fails a model: the score is there so you can weigh accuracy against speed, for example when choosing between model variants. It exits non-zero only on errors.

Run the validation script with `-h` to see all available options.