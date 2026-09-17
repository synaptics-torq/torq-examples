#!/bin/bash
set -euo pipefail

if [ -z "${DUT_REPO_DIR:-}" ]; then
    echo "[ERROR] DUT_REPO_DIR is required and was not set" >&2
    exit 1
fi
repo_root="${DUT_REPO_DIR}"
echo "[INFO] repo_root: ${repo_root}"
cd "${repo_root}"

if [ -f ".venv/bin/activate" ]; then
    . .venv/bin/activate
fi

model_dir="${MODEL_DIR:-$repo_root/models/Synaptics/moonshine-streaming-tiny-torq}"
DUT_WAV="${HELLO_WAV:-/home/torq-examples/hello_world_16k.wav}"
wav_path=""

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model-dir)
            shift
            if [ $# -gt 0 ]; then
                model_dir="$1"
            fi
            ;;
        --model-dir=*)
            model_dir="${1#*=}"
            ;;
        --wav)
            shift
            if [ $# -gt 0 ]; then
                wav_path="$1"
            fi
            ;;
        --wav=*)
            wav_path="${1#*=}"
            ;;
        *)
            wav_path="$1"
            ;;
    esac
    shift
done

if [ -z "$wav_path" ]; then
    wav_path="$DUT_WAV"
fi

for required_file in encoder.vmfb decoder.vmfb decoder_token_embeddings.npy tokenizer.json; do
    if [ ! -f "$model_dir/$required_file" ]; then
        echo "[ERROR] Moonshine streaming model assets missing in $model_dir. Required file: $required_file" >&2
        echo "[ERROR] Run: python3 setup_demos.py moonshine_streaming" >&2
        exit 3
    fi
done

if [ ! -f "$wav_path" ]; then
    echo "[ERROR] WAV input not found for Moonshine streaming: $wav_path" >&2
    exit 2
fi

cd "$repo_root/moonshine_streaming"
python3 src/infer.py -m "$model_dir" --wav "$wav_path"
