#!/bin/bash
# script usage:
#cd /home/torq-examples
#. .venv/bin/activate
#bash ./run_demo_gemma3.sh --instruct-model --prompt "Hello there"
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

model_path="${MODEL_PATH:-$repo_root/models/Synaptics/gemma-3-270m-it-torq/transformer.vmfb}"
lm_head="${LM_HEAD:-}"
prompt="${PROMPT:-Hello there}"
instruct_model=0

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
        --instruct-model)
            instruct_model=1
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
        --)
            shift
            if [ $# -gt 0 ]; then
                prompt="$1"
            fi
            break
            ;;
        *)
            prompt="$1"
            ;;
    esac
    shift
done

if [ ! -f "$model_path" ]; then
    for candidate in \
        "$repo_root/models/Synaptics/gemma-3-270m-it-torq/transformer.vmfb" \
        "$repo_root/models/Synaptics/gemma-3-270m-it-torq/model.vmfb.trim" \
        "$repo_root/models/Synaptics/gemma-3-270m-it-torq/model.vmfb"
    do
        if [ -f "$candidate" ]; then
            model_path="$candidate"
            break
        fi
    done
fi

if [ ! -f "$model_path" ]; then
    echo "[ERROR] Gemma3 model not found. Checked: $model_path" >&2
    exit 2
fi

if [ -n "$lm_head" ] && [ ! -f "$lm_head" ]; then
    echo "[ERROR] LM head not found: $lm_head" >&2
    exit 2
fi

cd "$repo_root/gemma3"
cmd=(python3 src/infer.py -m "$model_path")
if [ "$instruct_model" -eq 1 ]; then
    cmd+=(--instruct-model)
fi
if [ -n "$lm_head" ]; then
    cmd+=(--lm-head "$lm_head")
fi

echo "[PROMPT] $prompt"
printf '%s\n' "$prompt" | "${cmd[@]}"
