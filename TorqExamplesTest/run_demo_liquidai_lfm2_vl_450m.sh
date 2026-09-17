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

model_root="${MODEL_ROOT:-$repo_root/models/Synaptics/LiquidAI-LFM2-VL-450M}"
model_path="${MODEL_PATH:-$model_root/decoder_nolm.vmfb}"
lm_head="${LM_HEAD:-$model_root/lm_head.vmfb}"
vision_model="${VISION_MODEL:-$model_root/vision_encoder_256.vmfb}"
image_decoder="${IMAGE_DECODER:-$model_root/decoder_image_2part_}"
image_path="${IMAGE_PATH:-$model_root/cats-and-dogs-256.jpg}"
prompt="${PROMPT:-What is in this image?}"

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model)
            shift
            if [ $# -gt 0 ]; then
                model_path="$1"
            fi
            ;;
        --model=*)
            model_path="${1#*=}"
            ;;
        --lm-head)
            shift
            if [ $# -gt 0 ]; then
                lm_head="$1"
            fi
            ;;
        --lm-head=*)
            lm_head="${1#*=}"
            ;;
        --vision)
            shift
            if [ $# -gt 0 ]; then
                vision_model="$1"
            fi
            ;;
        --vision=*)
            vision_model="${1#*=}"
            ;;
        --image-decoder)
            shift
            if [ $# -gt 0 ]; then
                image_decoder="$1"
            fi
            ;;
        --image-decoder=*)
            image_decoder="${1#*=}"
            ;;
        --image)
            shift
            if [ $# -gt 0 ]; then
                image_path="$1"
            fi
            ;;
        --image=*)
            image_path="${1#*=}"
            ;;
        --prompt)
            shift
            if [ $# -gt 0 ]; then
                prompt="$1"
            fi
            ;;
        --prompt=*)
            prompt="${1#*=}"
            ;;
        *)
            prompt="$1"
            ;;
    esac
    shift
done

if [ ! -f "$model_path" ]; then
    echo "[ERROR] LiquidAI-LFM2-VL-450M decoder model not found: $model_path" >&2
    exit 2
fi

if [ -n "$lm_head" ] && [ ! -f "$lm_head" ]; then
    echo "[ERROR] LM head not found: $lm_head" >&2
    exit 2
fi

if [ ! -f "$vision_model" ]; then
    echo "[ERROR] Vision model not found: $vision_model" >&2
    exit 2
fi

if [ ! -f "$image_path" ]; then
    echo "[ERROR] Input image not found: $image_path" >&2
    exit 2
fi

cd "$repo_root/LiquidAI/LiquidAI-LFM2-VL-450M"
cmd=(python3 src/infer.py -m "$model_path" --vision "$vision_model" --image "$image_path" --prompt "$prompt")
if [ -n "$lm_head" ]; then
    cmd+=(--lm-head "$lm_head")
fi
if [ -n "$image_decoder" ]; then
    cmd+=(--image-decoder "$image_decoder")
fi

echo "[IMAGE] $image_path"
echo "[PROMPT] $prompt"
"${cmd[@]}"
