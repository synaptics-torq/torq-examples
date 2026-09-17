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

model_dir="${MODEL_DIR:-$repo_root/models/Synaptics/moonshine-tiny-bf16-torq}"
DUT_WAV="${HELLO_WAV:-/home/torq-examples/hello_world_16k.wav}"
MOONSHINE_TDA="${MOONSHINE_TDA:-dmabuf}"
wav_inputs=()

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
        --tda)
            shift
            if [ $# -gt 0 ]; then
                MOONSHINE_TDA="$1"
            fi
            ;;
        --tda=*)
            MOONSHINE_TDA="${1#*=}"
            ;;
        --)
            shift
            wav_inputs+=("$@")
            break
            ;;
        *)
            wav_inputs+=("$1")
            ;;
    esac
    shift
done

if [ ${#wav_inputs[@]} -eq 0 ]; then
    wav_inputs+=("$DUT_WAV")
fi

for required_file in encoder.vmfb decoder.vmfb decoder_token_embeddings.npy tokenizer.json; do
    if [ ! -f "$model_dir/$required_file" ]; then
        echo "[ERROR] Moonshine model assets missing in $model_dir. Required file: $required_file" >&2
        echo "[ERROR] Run: python3 setup_demos.py moonshine" >&2
        exit 3
    fi
done

if [ ! -f "${wav_inputs[0]}" ]; then
    echo "[ERROR] WAV input not found for Moonshine: ${wav_inputs[0]}" >&2
    exit 2
fi

cd "$repo_root/moonshine"

# Default path: keep the original behavior and let the runtime use the normal dmabuf allocator.
python3 src/infer.py -m "$model_dir" "${wav_inputs[@]}"
#
# The CPU workaround is kept only as a debugging / compatibility fallback.
# It is useful when a DUT hits the known XRAM write issue, but it is noticeably slower.
# Example:
#   MOONSHINE_TDA=cpu python3 src/infer.py -m "$model_dir" --tda cpu "${wav_inputs[@]}"
#   python3 src/infer.py -m "$model_dir" --tda cpu "${wav_inputs[@]}"
#
# echo "[INFO] Moonshine TDA mode: ${MOONSHINE_TDA} (CPU fallback for XRAM write issue; slower runtime)"
# python3 src/infer.py -m "$model_dir" --tda "$MOONSHINE_TDA" "${wav_inputs[@]}"
