#!/bin/bash
set -euo pipefail
set -x
PS4='[CMD] '

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

demo_name="${1:-${DEMO_NAME:-moonshine}}"

install_requirements_file() {
    local requirements_file="$1"
    echo "[SETUP] installing requirements from ${requirements_file}"
    python3 -m pip install -r "${requirements_file}"
}

install_demo_requirements() {
    local target_demo="$1"
    case "$target_demo" in
        gemma3)
            echo "[SETUP] ${target_demo}: installing requirements from ${repo_root}/gemma3/requirements.txt"
            install_requirements_file "$repo_root/gemma3/requirements.txt"
            ;;
        LiquidAI-LFM2.5-230M)
            echo "[SETUP] ${target_demo}: installing requirements from ${repo_root}/LiquidAI/LiquidAI-LFM2.5-230M/requirements.txt"
            install_requirements_file "$repo_root/LiquidAI/LiquidAI-LFM2.5-230M/requirements.txt"
            ;;
        moonshine)
            echo "[SETUP] ${target_demo}: installing requirements from ${repo_root}/moonshine/requirements.txt"
            install_requirements_file "$repo_root/moonshine/requirements.txt"
            ;;
        moonshine_streaming)
            echo "[SETUP] ${target_demo}: installing requirements from ${repo_root}/moonshine_streaming/requirements.txt"
            install_requirements_file "$repo_root/moonshine_streaming/requirements.txt"
            ;;
        LiquidAI-LFM2-VL-450M)
            echo "[SETUP] ${target_demo}: installing requirements from ${repo_root}/LiquidAI/LiquidAI-LFM2-VL-450M/requirements.txt"
            install_requirements_file "$repo_root/LiquidAI/LiquidAI-LFM2-VL-450M/requirements.txt"
            ;;
        object_detection)
            echo "[SETUP] ${target_demo}: installing requirements from ${repo_root}/object_detection/requirements.txt"
            install_requirements_file "$repo_root/object_detection/requirements.txt"
            ;;
        pose_estimation)
            echo "[SETUP] ${target_demo}: installing requirements from ${repo_root}/pose_estimation/requirements.txt"
            install_requirements_file "$repo_root/pose_estimation/requirements.txt"
            ;;
        *)
            echo "[WARN] ${target_demo}: no per-demo requirements file; skip specific install" >&2
            ;;
    esac
}

install_base_requirements() {
    local base_requirements_file="${repo_root}/requirements.txt"
    if [ -f "${base_requirements_file}" ]; then
        echo "[SETUP] installing shared DUT requirements from ${base_requirements_file} using default pip index"
        python3 -m pip install -r "${base_requirements_file}"
    else
        echo "[WARN] shared DUT requirements file not found: ${base_requirements_file}" >&2
    fi
}

setup_one_demo_with_retry() {
    local target_demo="$1"
    local attempt

    echo "[SETUP] target demo: ${target_demo}"
    for attempt in 1 2 3; do
        echo "[SETUP] ${target_demo}: attempt ${attempt}/3"

        install_base_requirements
        install_demo_requirements "$target_demo"
        echo "[SETUP] ${target_demo}: running setup_demos.py ${target_demo}"
        if python3 setup_demos.py "$target_demo"; then
            echo "[SETUP] ${target_demo}: setup succeeded on attempt ${attempt}"
            return 0
        fi

        echo "[WARN] ${target_demo}: setup failed on attempt ${attempt}; retrying next attempt" >&2
    done

    echo "[ERROR] ${target_demo}: setup failed after 3 attempts" >&2
    return 1
}

setup_one_demo_with_retry "$demo_name"
