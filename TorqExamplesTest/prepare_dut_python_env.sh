#!/bin/bash
set -euo pipefail

ANDROID_SERIAL="${ANDROID_SERIAL:-}"
DUT_REPO_DIR="${DUT_REPO_DIR:-/home/torq-examples}"
TORQ_EXAMPLES_REPO_URL="${TORQ_EXAMPLES_REPO_URL:-https://github.com/synaptics-torq/torq-examples.git}"
TORQ_EXAMPLES_BRANCH="${TORQ_EXAMPLES_BRANCH:-main}"
TORQ_RUNTIME_WHEEL_URL="${TORQ_RUNTIME_WHEEL_URL:-https://github.com/synaptics-torq/torq-compiler/releases/download/v2.1.0/torq_runtime-2.1.0-cp312-cp312-manylinux_2_28_aarch64.whl}"

if [[ -z "${ANDROID_SERIAL}" ]]; then
    echo "[ERROR] ANDROID_SERIAL is not set. Please export the DUT serial number before running this script." >&2
    exit 1
fi

if ! command -v adb >/dev/null 2>&1; then
    echo "[ERROR] adb is not in PATH" >&2
    exit 1
fi

ADB_CMD=(adb -s "${ANDROID_SERIAL}")

check_dut_connection() {
    echo "[INFO] Checking DUT ADB connectivity"
    if ! "${ADB_CMD[@]}" shell "echo adb_ok" >/dev/null 2>&1; then
        echo "[ERROR] DUT is not reachable through ADB: ${ANDROID_SERIAL:-default}" >&2
        return 1
    fi
    echo "[INFO] DUT ADB connectivity check passed"
}

prepare_dut_repo() {
    echo "[INFO] Prepare DUT repo at ${DUT_REPO_DIR}"
    local repo_exists
    repo_exists="$(${ADB_CMD[@]} shell "if [ -d '${DUT_REPO_DIR}/.git' ]; then echo yes; else echo no; fi" 2>/dev/null)"
    if [[ "${repo_exists}" == "yes" ]]; then
        echo "[INFO] Repo already exists on DUT, skip re-clone"
        return 0
    fi

    echo "[INFO] Cloning torq-examples repo onto DUT"
    "${ADB_CMD[@]}" shell "mkdir -p /home && cd /home && if [ -d torq-examples ]; then rm -rf torq-examples; fi && git clone -b ${TORQ_EXAMPLES_BRANCH} ${TORQ_EXAMPLES_REPO_URL} torq-examples"
}

ensure_venv() {
    echo "[INFO] Creating Python venv on DUT under ${DUT_REPO_DIR}/.venv"
    "${ADB_CMD[@]}" shell "cd '${DUT_REPO_DIR}' && python3 -m venv --system-site-packages .venv"

    echo "[INFO] Installing Torq runtime required by torq-examples"
    "${ADB_CMD[@]}" shell "cd '${DUT_REPO_DIR}' && . .venv/bin/activate && python -m pip install '${TORQ_RUNTIME_WHEEL_URL}'"

    echo "[INFO] Activating and validating the DUT Python venv"
    "${ADB_CMD[@]}" shell "cd '${DUT_REPO_DIR}' && . .venv/bin/activate && python - <<'PY'
import torq.runtime
print('VENV_READY')
print('Python executable:', __import__('sys').executable)
print('TORQ_RUNTIME_READY')
PY"

    echo "[INFO] Python virtual environment is ready on DUT. Stopping here by design."
}

main() {
    echo "[INFO] Host-side DUT venv preparation only"
    echo "[INFO] DUT repo path: ${DUT_REPO_DIR}"

    check_dut_connection || return 1
    prepare_dut_repo || return 1
    ensure_venv || return 1

    return 0
}

main "$@"
exit $?
