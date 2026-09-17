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

python3 object_detection/src/infer.py "$@"
