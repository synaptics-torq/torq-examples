#!/bin/bash
SQA_GIT_SERVER="${SQA_GIT_SERVER:-http://sha2uvp-gog01.synaptics.com:4080}"
export SQA_GIT_SERVER

TORQ_EXAMPLES_REPO_URL="${TORQ_EXAMPLES_REPO_URL:-https://github.com/synaptics-torq/torq-examples.git}"
TORQ_EXAMPLES_REF="${TORQ_EXAMPLES_REF:-}"
# TORQ_EXAMPLES_BRANCH is removed; script requires TORQ_EXAMPLES_REF as input.
TORQ_RUNTIME_WHEEL_URL="${TORQ_RUNTIME_WHEEL_URL:-https://github.com/synaptics-torq/torq-compiler/releases/download/v2.1.0/torq_runtime-2.1.0-cp312-cp312-manylinux_2_28_aarch64.whl}"
# /home/torq-examples on DUT
DUT_REPO_DIR="/home/torq-examples"
DUT_LOG_DIR="${DUT_REPO_DIR}/log"
BUILD_PROFILE="${BUILD_PROFILE:-sl2619_scarthgap}"

# /home/torq-examples/TorqExamplesTest on DUT
SUITE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_MAIN="test_demo_board.py"
DEFAULT_DEMO_NAME="${DEMO_NAME:-}"
RUN_FULL_DEMO_SUITE="${RUN_FULL_DEMO_SUITE:-1}"
export DEMO_NAME="${DEMO_NAME:-${DEFAULT_DEMO_NAME}}"
export DEMO_NAMES="${DEMO_NAMES:-}"
export RUN_FULL_DEMO_SUITE
export BUILD_PROFILE
export TORQ_EXAMPLES_REF

if [[ -z "${DUT_REPO_DIR}" ]]; then
    echo "[ERROR] DUT_REPO_DIR is not set. Please export the DUT repo path before running this script." >&2
    exit 1
fi

export DUT_REPO_DIR SUITE_SCRIPT_DIR 

# Allow passing the torq-examples commit SHA as the first positional argument
if [[ -n "$1" && -z "${TORQ_EXAMPLES_REF}" ]]; then
    TORQ_EXAMPLES_REF="$1"
    export TORQ_EXAMPLES_REF
fi

# TORQ_EXAMPLES_REF is required
if [[ -z "${TORQ_EXAMPLES_REF}" ]]; then
    echo "[ERROR] TORQ_EXAMPLES_REF is required (pass as env or first arg)" >&2
    exit 2
fi

# Supported demos are hardcoded here (do not parse README)
SUPPORTED_DEMOS_RAW="gemma3,LiquidAI-LFM2.5-230M,moonshine,moonshine_streaming,LiquidAI-LFM2-VL-450M,object_detection,pose_estimation"
IFS=',' read -r -a SUPPORTED_DEMOS_ARR <<< "${SUPPORTED_DEMOS_RAW}"

# Print help/usage
print_help() {
        cat <<EOF
Usage: ${0##*/} <TORQ_EXAMPLES_REF> [DEMO_NAME[,DEMO_NAME...]]

Host-side runner for torq-examples demo validation.

Parameters:
    TORQ_EXAMPLES_REF   Commit SHA (40 chars) to checkout on DUT (required)
    DEMO_NAME           Comma-separated list of demos to run (optional). If omitted, all demos run.

Supported demos:
    ${SUPPORTED_DEMOS_RAW}

Examples:
    run all supported demos:
    ${0##*/} 0123456789abcdef0123456789abcdef01234567
    run assigned demos:
    ${0##*/} 0123456789abcdef0123456789abcdef01234567 object_detection,pose_estimation

EOF
}

# If user asked for help anywhere in args, print and exit
for a in "$@"; do
        if [[ "$a" == "--help" || "$a" == "-h" ]]; then
                print_help
                exit 0
        fi
done

# If user did not provide a second argument, run all demos by default.
if [[ -z "$2" ]]; then
    echo "[INFO] user not assign demo name, run all demo."
    DEMO_NAMES="${SUPPORTED_DEMOS_RAW}"
    DEMO_NAME=$(echo "${SUPPORTED_DEMOS_RAW}" | cut -d',' -f1)
    export DEMO_NAMES
    export DEMO_NAME
else
    # User provided a comma-separated demo list; validate entries against the hardcoded list
    REQUESTED_DEMOS_RAW="$2"
    IFS=',' read -r -a REQUESTED_ARR <<< "${REQUESTED_DEMOS_RAW}"
    VALIDATED=()
    for raw in "${REQUESTED_ARR[@]}"; do
        dm=$(echo "${raw}" | sed -e 's/^\s*//' -e 's/\s*$//')
        if [[ -z "${dm}" ]]; then
            continue
        fi
        ok=0
        for s in "${SUPPORTED_DEMOS_ARR[@]}"; do
            if [[ "${s}" == "${dm}" ]]; then
                ok=1
                break
            fi
        done
        if [[ "${ok}" -ne 1 ]]; then
            cat >&2 <<EOF
[ERROR] Unsupported demo name in list: ${dm}

Usage: ${0##*/} <TORQ_EXAMPLES_REF> [DEMO_NAME[,DEMO_NAME...]]

Parameters:
  TORQ_EXAMPLES_REF   Commit SHA (40 chars) to checkout on DUT (required)
  DEMO_NAME           Comma-separated list of demos to run (optional). Supported demos:
    ${SUPPORTED_DEMOS_RAW}

Example:
  ${0##*/} 0123456789abcdef0123456789abcdef01234567 object_detection,pose_estimation

EOF
            exit 3
        fi
        VALIDATED+=("${dm}")
    done
    if [[ ${#VALIDATED[@]} -gt 0 ]]; then
        # Export DEMO_NAMES as comma-separated list and set DEMO_NAME to first demo for compatibility
        DEMO_NAMES=$(IFS=','; echo "${VALIDATED[*]}")
        DEMO_NAME="${VALIDATED[0]}"
        export DEMO_NAMES
        export DEMO_NAME
    fi
fi


resolve_dut_build_version() {
    echo "[INFO] Reading DUT build info from /etc/buildinfo"
    local build_info
    if ! build_info=$(cat /etc/buildinfo 2>/dev/null); then
        echo "[ERROR] Failed to read /etc/buildinfo from DUT" >&2
        return 1
    fi

    local build_version
    build_version="$(printf '%s\n' "${build_info}" | grep -E '^[[:space:]]*SYNA_SDK_REVISION[[:space:]]*=' | head -n 1 | sed -E 's/^[[:space:]]*SYNA_SDK_REVISION[[:space:]]*=[[:space:]]*//; s/[[:space:]]*$//')"
    if [[ -z "${build_version}" ]]; then
        echo "[ERROR] SYNA_SDK_REVISION not found in /etc/buildinfo" >&2
        printf '%s\n' "${build_info}" >&2
        return 1
    fi

    local version=""
    if [[ "${build_version}" =~ ^[0-9]{12}_([0-9]{12})$ ]]; then
        echo "[INFO] Valid full version format: '${build_version}'"
        version="${BASH_REMATCH[1]}"
        echo "[INFO] Extract actual build version: '${version}'"
        build_version="${version}"
    elif [[ "${build_version}" =~ [0-9]{12} ]]; then
        echo "[INFO] CI image short version format: '${build_version}'"
    else
        echo "[ERROR] Invalid full version format: '${build_version}'" >&2
    fi

    export SYNA_SDK_REVISION="${build_version}"
    echo "[INFO] SYNA_SDK_REVISION: ${SYNA_SDK_REVISION}"
}

validate_torq_examples_ref() {
    if [[ -z "${TORQ_EXAMPLES_REF}" ]]; then
        echo "[ERROR] TORQ_EXAMPLES_REF is not set" >&2
        return 1
    fi

    if [[ ! "${TORQ_EXAMPLES_REF}" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "[ERROR] TORQ_EXAMPLES_REF must be a 40-character commit SHA" >&2
        return 1
    fi
}

clean_suite_work_space(){
    echo "[INFO] Cleaning suite work space"
    echo "[INFO] Cleaning HTML and XML files in ${SUITE_SCRIPT_DIR} left from previous runs"

    rm -f "${SUITE_SCRIPT_DIR}/"*.html
    rm -f "${SUITE_SCRIPT_DIR}/"*.xml
}

prepare_dut_repo() {
    echo "[INFO] Prepare DUT repo at ${DUT_REPO_DIR}"

    validate_torq_examples_ref || return 1

    echo "[INFO] Checking whether git is available on DUT"
    if ! command -v git >/dev/null 2>&1; then
        echo "[WARN] git is not found on DUT, trying to install it"
        if ! bash -lc "
            if command -v apt-get >/dev/null 2>&1; then
                cat > /etc/apt/sources.list <<'EOF'
deb [trusted=yes] http://10.46.130.134/deb_2.0/sl2619/all ./
deb [trusted=yes] http://10.46.130.134/deb_2.0/sl2619/cortexa55 ./
deb [trusted=yes] http://10.46.130.134/deb_2.0/sl2619/sl2619 ./
EOF
                apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && apt-get install git
                exit \$?
            fi
            echo '[ERROR] apt-get is not available on DUT' >&2
            exit 1
        "; then
            echo "[ERROR] Failed to install git on DUT" >&2
            return 1
        fi

        if ! command -v git >/dev/null 2>&1; then
            echo "[ERROR] git is still unavailable on DUT after installation attempt" >&2
            return 1
        fi
    fi
    echo "[INFO] git is available on DUT"

    local repo_exists
    repo_exists="$(if [ -d "${DUT_REPO_DIR}/.git" ]; then echo yes; else echo no; fi)"

    if [[ "${repo_exists}" != "yes" ]]; then
        echo "[INFO] Cloning torq-examples on DUT"
        if ! bash -lc "mkdir -p /home && rm -rf '${DUT_REPO_DIR}' && git clone '${TORQ_EXAMPLES_REPO_URL}' '${DUT_REPO_DIR}'"; then
            echo "[ERROR] Failed to clone torq-examples on DUT" >&2
            return 1
        fi
    else
        echo "[INFO] torq-examples already exists on DUT; updating remote"
        if ! bash -lc "cd '${DUT_REPO_DIR}' && git remote set-url origin '${TORQ_EXAMPLES_REPO_URL}'"; then
            echo "[ERROR] Failed to configure torq-examples remote on DUT" >&2
            return 1
        fi
    fi

    echo "[INFO] Checking out torq-examples commit: ${TORQ_EXAMPLES_REF}"
    if ! bash -lc "cd '${DUT_REPO_DIR}' && git fetch --depth=1 origin '${TORQ_EXAMPLES_REF}' && git checkout --detach FETCH_HEAD"; then
        echo "[ERROR] Failed to checkout torq-examples commit ${TORQ_EXAMPLES_REF}" >&2
        return 1
    fi

    local actual_ref
    actual_ref="$(bash -lc "cd '${DUT_REPO_DIR}' && git rev-parse HEAD" 2>/dev/null | tr -d '\r' | tail -n 1)"
    echo "[INFO] Actual torq-examples commit on DUT: ${actual_ref}"

    if [[ -n "${TORQ_EXAMPLES_REF}" && "${actual_ref}" != "${TORQ_EXAMPLES_REF}" ]]; then
        echo "[ERROR] DUT checkout does not match requested torq-examples commit" >&2
        echo "[ERROR] Requested: ${TORQ_EXAMPLES_REF}" >&2
        echo "[ERROR] Actual:    ${actual_ref}" >&2
        return 1
    fi
}

check_dut_python_env() {
    echo "[INFO] Checking DUT Python virtual environment"
    local env_ok
    env_ok="$(bash -lc "
        cd '${DUT_REPO_DIR}' && \
        if [ -f '.venv/bin/python' ] && [ -f 'requirements.txt' ]; then \
            . .venv/bin/activate && \
            python - <<'PY'
import importlib.util
missing = []
for name in ['pip', 'pytest', 'torq.runtime']:
    try:
        found = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError):
        found = None
    if found is None:
        missing.append(name)
if missing:
    print('venv_missing_modules=' + ','.join(missing))
    raise SystemExit(1)
print('venv_ok')
PY
        else
            echo 'venv_missing'
            exit 1
        fi
    " 2>/dev/null)"

    if [[ "${env_ok}" == "venv_ok" ]]; then
        echo "[INFO] DUT Python venv is already valid; skipping reinstall"
        return 0
    fi



    return 0
}

resolve_demo_queue() {
    local demo_queue
    if [[ -n "${DEMO_NAMES:-}" ]]; then
        demo_queue="${DEMO_NAMES}"
    elif [[ -n "${DEMO_NAME:-}" ]]; then
        demo_queue="${DEMO_NAME}"
    else
        local demo_queue_raw
        if demo_queue_raw="$(python3 "${SUITE_SCRIPT_DIR}/read_demo_names.py" 2>&1)"; then
            demo_queue="${demo_queue_raw}"
        else
            echo "[WARN] Could not resolve demo queue from host-side config.ini: ${demo_queue_raw}" >&2
            demo_queue="[]"
        fi
    fi
    printf '%s\n' "${demo_queue}"
}

setup_dut_python_env() {
    echo "[INFO] Create or refresh DUT venv and install dependencies for pytest execution"
    local dut_python_version
    dut_python_version="$(bash -lc "python3 -c \"import sys; print('.'.join(map(str, sys.version_info[:3])))\"" 2>/dev/null | tr -d '\r' | tail -n 1)"
    if [[ -z "${dut_python_version}" ]]; then
        echo "[ERROR] Failed to detect DUT python3 version" >&2
        return 1
    fi
    echo "[INFO] DUT python3 version: ${dut_python_version}"
    if ! bash -lc "python3 -c \"import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)\"" >/dev/null 2>&1; then
        echo "[ERROR] DUT python3 version must be >= 3.12, current: ${dut_python_version}" >&2
        return 1
    fi

    if ! bash -lc "python3 -c \"import venv\"" >/dev/null 2>&1; then
        echo "[WARN] DUT python3 venv module is missing, installing python3-venv"
        if ! bash -lc "apt-get install -y python3-venv"; then
            echo "[ERROR] Failed to install python3-venv on DUT" >&2
            return 1
        fi
    fi

    echo "[INFO] Some build version may missing python3-terminal. This can lead tty fail for some demo."
    if ! bash -lc "apt-get install -y python3-terminal"; then
        echo "[ERROR] Failed to install python3-terminal on DUT" >&2
        return 1
    fi


    echo "[WARN] Prevent pip is missing on DUT, installing python3-pip"
    if ! bash -lc "apt-get install -y python3-pip"; then
        echo "[WARNING] Failed to install pip on DUT" >&2
    fi

    echo "[INFO] Consolidating DUT logs under ${DUT_REPO_DIR}/log"
    bash -lc "mkdir -p '${DUT_REPO_DIR}/log'" || {
        echo "[ERROR] Failed to create DUT log folder: ${DUT_REPO_DIR}/log" >&2
        return 1
    }

    echo "[INFO] Installing pytest 7.1.0 into DUT local python environment"
    bash -lc "python3 -m pip install 'pytest==7.1.0'" || {
        echo "[ERROR] Failed to install pytest 7.1.0 in DUT local environment" >&2
        return 1
    }

    if ! bash -lc "cd '${DUT_REPO_DIR}' && python3 -m venv --system-site-packages .venv"; then
        echo "[WARN] ensurepip failed, installing python3-venv and python3-pip before retry"
        if ! bash -lc "if command -v apt-get >/dev/null 2>&1; then apt-get install -y python3-venv python3-pip; else exit 1; fi && cd '${DUT_REPO_DIR}' && rm -rf .venv && python3 -m venv --without-pip --system-site-packages .venv"; then
            echo "[ERROR] Failed to create DUT Python virtual environment with system pip" >&2
            return 1
        fi
    fi

    if ! bash -lc "cd '${DUT_REPO_DIR}' && .venv/bin/python -m pip --version" >/dev/null 2>&1; then
        echo "[ERROR] pip is unavailable in DUT Python virtual environment" >&2
        return 1
    fi

    echo "[INFO] install setuptools"
    bash -lc "cd '${DUT_REPO_DIR}' && . .venv/bin/activate && python -m pip install setuptools wheel 2>&1 | tee '${DUT_REPO_DIR}/log/dut_pip_upgrade.log'"
    echo "[INFO] install ${TORQ_RUNTIME_WHEEL_URL}"
    bash -lc "cd '${DUT_REPO_DIR}' && . .venv/bin/activate && python -m pip install '${TORQ_RUNTIME_WHEEL_URL}' 2>&1 | tee '${DUT_REPO_DIR}/log/dut_torq_runtime_install.log'"
    echo "[INFO] install pytest in venv"
    bash -lc "cd '${DUT_REPO_DIR}' && . .venv/bin/activate && python -m pip install pytest -r requirements.txt 2>&1 | tee '${DUT_REPO_DIR}/log/dut_requirements_install.log'"
    echo "[INFO] install torq.runtime in venv"
    bash -lc "cd '${DUT_REPO_DIR}' && . .venv/bin/activate && python -c 'import requests; import torq.runtime; print(\"DUT_PYTHON_DEPENDENCIES_READY\")'"
}

run_dut_pytest() {
    echo "[INFO] Running host-side pytest suite for demo board validation"
    local demo_queue
    if [[ -n "${DEMO_NAMES:-}" ]]; then
        demo_queue="${DEMO_NAMES}"
    elif [[ -n "${DEMO_NAME:-}" ]]; then
        demo_queue="${DEMO_NAME}"
    else
        local demo_queue_raw
        if demo_queue_raw="$(python3 "${SUITE_SCRIPT_DIR}/read_demo_names.py" 2>&1)"; then
            demo_queue="${demo_queue_raw}"
        else
            echo "[WARN] Could not resolve demo queue from suite-side config.ini: ${demo_queue_raw}" >&2
            demo_queue="[]"
        fi
    fi
    export DEMO_NAMES="${demo_queue}"
    echo "[INFO] Demo queue: ${demo_queue}"
    if [[ ! -f "${TEST_MAIN}" ]]; then
        echo "[ERROR] Host pytest harness not found: ${TEST_MAIN}. Verify the repo checkout and working directory." >&2
        return 1
    fi

    export PYTHONUNBUFFERED=1

    local build_version="${SYNA_SDK_REVISION:-unknown}"

    local junit_xml_path="TEST-TorqExamplesTest.xml"
    echo "[INFO] JUnit XML report path: ${junit_xml_path}"

    local pytest_status=0
    python3 -m pytest "${TEST_MAIN}" -q --junitxml="${junit_xml_path}" || pytest_status=$?
    if [[ -f "${junit_xml_path}" ]]; then
        python3 "${SUITE_SCRIPT_DIR}/normalize_junit_report.py" "${junit_xml_path}" || \
            echo "[WARN] Failed to normalize JUnit XML report naming" >&2
    fi
    return ${pytest_status}
}

collect_dut_logs() {
    echo "[INFO] Collecting DUT device logs and host pytest report"
    local junit_report_path="TEST-TorqExamplesTest.xml"
    if [[ -f "${junit_report_path}" ]]; then
        echo "[INFO] Host JUnit report saved at ${junit_report_path}"
    fi
    dmesg | tail -n 200 > "${SUITE_SCRIPT_DIR}/dut_dmesg.log" 2>&1 || true
    echo "[INFO] DUT test artifacts saved to ${SUITE_SCRIPT_DIR}"
}

generate_profile_summary() {
    echo "[INFO] Generating profiling summary from DUT logs"
    if command -v python3 >/dev/null 2>&1; then
        if python3 "${SUITE_SCRIPT_DIR}/collect_profile_metrics.py" --logs-dir "${DUT_LOG_DIR}" --out "profile_summary.html"; then
            echo "[INFO] Profile summary generated: profile_summary.html"
        else
            echo "[WARN] Failed to generate profile summary" >&2
        fi
    else
        echo "[WARN] python3 not found; skipping profile summary generation" >&2
    fi
}

generate_test_summary() {
    echo "[INFO] Generating combined test + profile summary"
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[WARN] python3 not found; skipping combined summary generation" >&2
        return 0
    fi
    local junit="TEST-TorqExamplesTest.xml"
    local profile="profile_summary.html"
    local out="test_and_profile_summary.html"
    if python3 "${SUITE_SCRIPT_DIR}/generate_test_summary.py" --junit "${junit}" --profile "${profile}" --out "${out}"; then
        echo "[INFO] Combined summary generated: ${out}"
    else
        echo "[WARN] Failed to generate combined summary" >&2
    fi
}

main() {
    echo "[INFO] torq-examples pytest validation"
    echo "[INFO] Repo: ${TORQ_EXAMPLES_REPO_URL}"
    echo "[INFO] torq-examples commit: ${TORQ_EXAMPLES_REF}" 
    echo "[INFO] DUT Repo path: ${DUT_REPO_DIR}"
    echo "[INFO] Suite Script DIR: ${SUITE_SCRIPT_DIR}"
    clean_suite_work_space || return 1
    resolve_dut_build_version || return 1
    prepare_dut_repo || return 1
    setup_dut_python_env || return 1

    # Run pytest but always continue to collect logs and generate profile summary.
    local pytest_status=0
    run_dut_pytest || pytest_status=$?

    collect_dut_logs || true
    generate_profile_summary || true
    echo "[INFO] Generate summary with commit:${TORQ_EXAMPLES_REF}, DUT version:${SYNA_SDK_REVISION}"
    generate_test_summary || true 
    # Exit with pytest status so callers see the test result, but ensure post-processing ran.
    if [[ "${pytest_status}" -ne 0 ]]; then
        return "${pytest_status}"
    fi
}

main "$@"
exit $?
