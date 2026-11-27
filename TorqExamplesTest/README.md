# TorqExamplesTest

Host-driven test suite for the torq-examples demo validation flow.

## Architecture (reconstruction, branch `newHostMethod`)

Preflight and two stages, all orchestrated from the CI host:

0. **DUT preflight** (`conftest.py`, session-scoped `dut_ready`) — power-cycles
   the DUT and checks ADB before any host container or model preparation.
   It waits 15 seconds before polling ADB every 5 seconds for up to another
   30 seconds. Failure blocks both host preparation and DUT deployment and
   requests manual environment checks. This runs once per pytest session,
   including when selecting only host or only DUT pipeline tests; helper
   unit tests do not trigger it. The fixed Netbooter sequence is in
   `dut_power_cycle.py`.
1. **Host-side model preparation** (`host_converter.py`) — runs
   `setup_demos.py` (or per-model `setup_demo.py` for multi-model demos) inside a Docker container
   (`conversion/Dockerfile`, ubuntu 24.04 base) to provide an isolated,
   consistent Python environment for model downloads and validation.
   Model artifacts land in `<examples checkout>/models/`.
2. **DUT deploy + inference + profiling** (`test_dut_deploy.py` and
   `dut_deploy.py`) — after the shared preflight and host preparation,
   the module fixture locks `sl2619npu01` without another power cycle.
   It then wipes and rebuilds the DUT workspace (no stale-file pollution), pushes the
   models, installs the aarch64 runtime wheel, and either swaps the NPU
   kernel module or keeps the already-loaded board module when no KO artifact
   was supplied.
   Missing models or per-model upload failures do not block other model
   uploads. KO/runtime installation continues after a partial model-push
   failure, and the deployment test reports the failure afterwards so the
   remaining models can still be tested. Connectivity, workspace, KO, and
   runtime installation failures still stop deployment immediately.
   It then runs the per-demo inference and profiling matrices (repo-root
   `profile.py --repeat 3`, seven metrics). The fixed Netbooter sequence is
   in `dut_power_cycle.py`.

The runtime wheel and optional KO under test come from a test-build run's
artifacts (`TORQ_DUT_RUNTIME_WHEEL`, `TORQ_DUT_KO`). Their SHA-256 fingerprints
are recorded in `prepare_metadata.json`, the HTML/JSON reports, and the
workflow artifact metadata.
The reported Torq Runtime Version is collected from the DUT after the test
wheel is successfully installed, not from the previously installed runtime.
Failure to collect or save this information fails the runtime installation stage.

## Test entries

- `test_host_conversion.py` — container environment + per-demo host setup.
  Multi-model demos prepare each model repository independently using the
  demo's `setup_demo.py` CLI and save separate
  `<demo>_<model>_host_setup.log` files. One model's failure does not block
  preparation of another model in the same demo. Single-model demos keep
  using `setup_demos.py`. Each case checks that its model directory exists
  and is non-empty; setup errors are printed immediately.
- `test_dut_deploy.py` — `test_dut_deploy` (deploy, one case) +
  `test_dut_infer[demo]` + `test_dut_profile[demo]` (12 demos each;
  demos without an inference command builder are skipped with a reason,
  profiling applies to all)
- EmbeddingGemma-2 inference runs the text-query session by default. Runtime
  errors (including XRAM allocation failures) are reported as failures, not skips.

## Infrastructure

- `demo_registry.py`: AST discovery of demos/models from `setup_demos.py`
  (never executed; also the authoritative source for app dirs and model
  repo IDs — `dut_deploy.py` prefers it over its built-in spec table)
- `conftest.py`: repository-local pytest plugin (artifact cleanup, console
  tee, JUnit + combined HTML generation, failure semantics)
- `generate_test_summary.py`: merges JUnit results with structured
  profiling records into one HTML report. Results are grouped into Host Setup,
  DUT Deployment & Inference, and DUT Profiling tables. Only Host Setup includes
  the Hugging Face revision column. Profiling retains per-file metrics and
  shows failed cases even when no profiling data was produced.

## Run and report generation

### Structured ADB command results

The bundled `torq.lab.pipeline.remote.ADBCommandRunner` provides
`run_cmd_result(command, timeout=...)`, returning a
`subprocess.CompletedProcess[str]` with `returncode`, `stdout`, and `stderr`.
Unlike `run_cmd()`, it captures stdout/stderr separately and returns non-zero
ADB/shell exit codes instead of raising for them. Timeouts and process-launch
failures raise `RemoteCommandError`.

Omitting `timeout` or passing `None` uses the runner's instance timeout;
a numeric value overrides only that invocation. String scripts preserve
quoting. For both ADB and SSH, `run_cmd()` accepts either a complete shell
script string (unchanged) or an argv list of raw, unquoted arguments (serialized
with `shlex.join()`). Use strings for shell syntax such as pipes, redirections,
variable assignments, loops, and `time` blocks; do not wrap a whole script in
a list or pre-quote individual argv elements. `run_cmd_result()` uses the same
contract on ADB. Return values, exit-code handling, stream capture, and timeout
behavior remain unchanged.

```python
runner.run_cmd(["printf", "%s", "Hello world"])
runner.run_cmd("printf '%s\\n' 'Hello world' | cat")
```

The deployment driver's `_shell()` uses this API and maps the exit code and
separate streams into `PhaseResult`, retaining its per-call timeouts and
command logs. Non-zero exits remain phase results; runner initialization or
execution exceptions are reported as `DutDeployError`. The ADB runner is reused
for the acquired board and closed when it is released.

Model directories, kernel modules, runtime wheels, and generated WAV files are
uploaded through the same ADB runner's `copy_files(..., board_dst=True)`.
On ADB, `copy_files()` also accepts a per-call `timeout` and `check=False` to
return a `CompletedProcess[str]` with separate streams and a non-raising exit
code, which `_push()` maps into `PhaseResult`. Without `check=False`, its
existing return value (`None`) and raise-on-nonzero behavior are unchanged.
Model uploads retain their 10800-second timeout; other uploads use 3600 seconds.
For binary payloads that must be streamed to a remote shell command's stdin,
use `run_cmd_result_with_input(command, input_data, timeout=...)`; for example,
the source archive is sent to `tar -xf -` this way without creating a temporary
archive file.

### Manual setup on a host computer

The full suite requires a Linux/POSIX host with access to the lab DUT and its
power controller. Native Windows Python/Git Bash is not sufficient: host model
preparation uses POSIX UID/GID, and board locking uses `fcntl`.

**Before running:** reserve the DUT and ensure no other job is using it. The
suite power-cycles the board, replaces its runtime, optionally replaces its NPU
module, and deletes/rebuilds the configured DUT workspace. It also cleans
previous report artifacts at the configured output locations. Never point
`TORQ_DUT_REMOTE_ROOT` at a shared directory or a system directory.

#### Host and DUT prerequisites

On the host:

- Python 3.12 with pip and venv; install `pytest` and `requests` into the host
  environment. `requests` is imported by the bundled board-control module.
- Working Docker CLI and daemon, with permission for the current user to build
  and run containers. The suite builds its Ubuntu 24.04 / Python 3.12 image
  automatically and installs the examples/demo dependencies inside it.
- `bash`, `tar`, `adb`, and the lab `netbooter` command available. Git is needed
  to obtain/update the source checkouts.
- Network access from the host/container to the Docker image registry, Ubuntu
  package repositories, Python package index, and Hugging Face. Configure
  corporate proxy and CA trust if required. The container mounts the host CA
  material and forwards the proxy/HF variables listed below.
- Writable source/output directories and enough disk space for downloaded
  models, the Docker image, and reports.

On the DUT:

- The DUT must be connected to the host through ADB. Run `adb devices` on the
  host and confirm that the intended DUT is listed with status `device`.
  The first column is its ADB serial/device identifier, not a required hardware
  name; it can be any identifier reported by ADB. The example below uses
  `sl2619npu01`, the current lab serial. The current suite configuration still
  checks that specific serial, as explained under "Current lab configuration".
- ADB shell permissions to create the workspace and load/unload the NPU kernel
  module.
- Python 3 with venv/pip and `tar`. The suite creates the DUT venv and installs
  root and per-demo requirements; the DUT therefore also needs package-index
  access and appropriate proxy/CA trust.
- Firmware compatible with the selected KO/runtime. The runtime wheel must
  match the DUT's aarch64 architecture and Python ABI, not the host's.
  The runtime and downloaded models must also use compatible VMFB formats.

No compiler wheel is needed on the host: both modes download pre-compiled
models. Plain pytest does **not** download the runtime wheel or KO, resolve
release artifacts, or perform the workflow's firmware burn stage. Obtain the
artifacts and prepare compatible DUT firmware before starting.

#### Current lab configuration

[dut_power_cycle.py](dut_power_cycle.py) currently fixes these values:

| Setting | Current value |
| --- | --- |
| Environment script sourced by power-control/ADB checks | `/home/jenkins/cases/env_config.sh` |
| Netbooter IP | `10.70.58.211` |
| Netbooter outlet | `2` |
| DUT ADB serial | `sl2619npu01` |

These power-control values are not environment-variable overrides. A different
computer can run the suite only if it has the required environment script,
tools, and access to this lab bench. Setting `TORQ_DUT_SERIAL` to another board
does not retarget the full pytest flow; its preflight and fixture reject a
serial different from the fixed power-cycle target. Adapting to another bench
requires changing that configuration consistently before running.

Check ADB connectivity before starting:

```text
$ adb devices
List of devices attached
sl2619npu01    device
```

Use the serial actually shown for your DUT; `sl2619npu01` is only the lab
example, not an ADB naming requirement. An empty list, `offline`, or
`unauthorized` is not a successful connection: resolve connectivity,
permissions, or authorization first. If multiple devices are listed, verify
the intended DUT with `adb -s <your-adb-serial> shell echo ok`.
For a different serial, the current fixed suite target must also be adapted;
setting the environment variable alone is not yet sufficient.

The suite explicitly uses the local ADB board-control backend; Jenkins API
credentials are not required. The lock file is initialized from connected ADB
devices if absent. Its parent directory must exist and be writable. Concurrent
users of the same DUT must use the same control file, and must reserve the
bench separately because the preflight power cycle occurs before board locking.

#### Environment variables

| Variable | Requirement/default | Purpose |
| --- | --- | --- |
| `TORQ_TEST_MODE` | Recommended; default `release` | `release` or `dev` model policy. |
| `TORQ_MODEL_VERSION` | Required in `dev`; recommended in `release` | HF revision such as `v2.3.0`; `main` is also supported in dev. Release mode without a pin uses the examples `VERSION`. |
| `TORQ_DUT_RUNTIME_WHEEL` | Required | Absolute host path to the aarch64 runtime wheel. |
| `TORQ_DUT_KO` | Required and non-empty | Absolute host path to `syna_npu.ko`, or `on_board_default` to retain the already-loaded module. |
| `TORQ_DUT_SERIAL` | Recommended: `sl2619npu01` | Pin the board through the shared ADB lock; the full pytest flow requires this fixed serial. |
| `TORQ_DUT_REMOTE_ROOT` | Optional: `/home/torq-examples` | Dedicated DUT workspace; deleted and rebuilt during deployment. |
| `TORQ_BOARDS_CONTROL` | Optional: `board_control.json` beside `dut_deploy.py` | Shared board-lock file; use the same absolute path across runs/users of the bench. |
| `TORQ_HOST_IMAGE` | Optional: `torq-examples-host-setup:local` | Tag for the automatically rebuilt Docker image. |
| `HF_TOKEN` | When needed | Token for gated/private model access; do not commit it. |
| `HF_ENDPOINT` | Optional | Alternative Hugging Face endpoint. |
| `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` (or lowercase forms) | Network-dependent | Forwarded to the host-setup container; exclude lab endpoints from proxying where appropriate. |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD` | Recommended: `1` | Prevent unrelated installed pytest plugins from affecting the run. |

GitHub workflow inputs such as `rel`, `source_run_id`, `examples_ref`, and
`suite_branch` are not pytest inputs. For a manual run, choose the source
checkouts and set model/artifact paths explicitly.

#### Example: manual release run

Place `TorqExamplesTest/` directly inside the **torq-examples checkout under
test**, alongside `setup_demos.py` and `requirements.txt`. This ensures that
the suite uses the intended examples source rather than another nearby
checkout. Use this layout for manual experiments:

```text
torq-examples/
    setup_demos.py
    requirements.txt
    VERSION
    TorqExamplesTest/
        pytest.ini
        conftest.py
        test_host_conversion.py
        test_dut_deploy.py
```

Run the following in **Bash on the Linux host**, from the root of that
torq-examples checkout. Replace the checkout and artifact placeholders with
real absolute paths:

```bash
cd /absolute/path/to/torq-examples

# Load the lab tool environment into the current shell, not only a subprocess.
source /home/jenkins/cases/env_config.sh

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install pytest requests

export TORQ_TEST_MODE=release
export TORQ_MODEL_VERSION=v2.3.0
export TORQ_DUT_RUNTIME_WHEEL=/absolute/path/to/runtime.whl
export TORQ_DUT_KO=/absolute/path/to/syna_npu.ko
# Alternatively, keep the loaded board module:
# export TORQ_DUT_KO=on_board_default

export TORQ_DUT_SERIAL=sl2619npu01
export TORQ_DUT_REMOTE_ROOT=/home/torq-examples
# For multiple host users, use an agreed shared writable directory instead.
mkdir -p "$HOME/.cache/torq-examples-test"
export TORQ_BOARDS_CONTROL="$HOME/.cache/torq-examples-test/board_control.json"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

python --version
docker version
command -v bash tar adb netbooter
adb devices
test -f "$TORQ_DUT_RUNTIME_WHEEL"
if [ "$TORQ_DUT_KO" != on_board_default ]; then
    test -f "$TORQ_DUT_KO"
fi

# Continue only after the checks succeed and the target appears as "device".
python -m pytest TorqExamplesTest
```

For dev testing, set `TORQ_TEST_MODE=dev`, choose an explicit
`TORQ_MODEL_VERSION`, and point to the intended dev runtime/KO artifacts.
Changing the mode alone does not select or download a different runtime.

If startup fails, check the environment-script path, Netbooter reachability,
ADB authorization/permissions, Docker access, proxy/CA settings, and artifact
paths first. Missing HF revisions must be resolved by selecting an available,
compatible revision; the suite does not silently fall back to another version.

### Running and inspecting reports

Run the complete suite from the root of the torq-examples checkout containing
`TorqExamplesTest/` (fixture dependencies and
conftest enforce DUT preflight -> host preparation -> DUT deploy/infer/profile,
regardless of collection order):

	python3.12 -m pytest TorqExamplesTest

The default test collection includes only the host preparation and DUT
deployment/inference/profiling modules, not helper unit tests in `unit_tests/`.
The shared session preflight performs one DUT
power-off/power-on sequence before the pipeline starts; later stages and
teardown do not repeat it.

All helper unit tests live in `unit_tests/` and mock hardware interactions.
Test files use `test_*.py` without an additional `_unit` suffix; production
and utility modules use descriptive snake_case names without the `test_` prefix.
Run them explicitly from the examples root, with separate artifact paths to
avoid cleaning previous pipeline results:

```bash
unit_results="$(mktemp -d)"
python -m pytest TorqExamplesTest/unit_tests \
  --junitxml="$unit_results/results.xml" \
  --torq-profile-dir="$unit_results/logs" \
  --torq-summary-html="$unit_results/summary.html"
```

Standalone debugging without pytest (engine drivers, no DUT needed for
host stage, no suite layout needed):

	TORQ_TEST_MODE=dev TORQ_MODEL_VERSION=v2.3.0 \
	  python3 TorqExamplesTest/host_converter.py <examples_root> [demo ...]
	TORQ_DUT_RUNTIME_WHEEL=... TORQ_DUT_KO=... \
	  python3 TorqExamplesTest/dut_deploy.py <examples_root> [demo ...]

Every invocation produces:

- `TEST-TorqExamplesTest.xml`: final pytest JUnit report
- `test_and_profile_summary.html`: combined test + profiling report
  (run info, per-case results with model VERSION / HF revision, TTFT /
  tok/s for LLM demos, seven profiling metrics)
- `torq-examples/logs/`: per-phase logs and `*_infer.json` / `*_profile.json` records

Runtime inference measurements are stored as numeric values in each
`logs/<demo>_infer.json` record's `metrics`: `ms`, `ttft_ms`, `tok_s`, `rtf`,
and, for per-frame detection/pose/face inference, `inference_ms` and `fps`.
Only values actually printed by the demo are recorded; missing measurements
remain absent. FPS is derived as `1000 / inference_ms` (not end-to-end video
throughput). Multiple measurements within one command are averaged; FPS uses
the mean frame inference time. The HTML inference table includes derived FPS
and lists FPS separately by VMFB when a test covers multiple model files;
other metrics from multiple commands in the same row use arithmetic means.
JSON export retains each command's measurements without display rounding.
Reports prefer inference JSON; old runs without these fields retain log parsing.
Per-VMFB profiling data stays separate in `*_profile.json`.
The HTML profiling table shows Infer and Profile results without a duplicate
testcase-result column. When the profiling failure reports no available VMFB,
both result cells show `NO model file`. Other failures retain their result
labels; this presentation change does not change pytest or JSON statuses.

To regenerate only the HTML from existing artifacts:

	python TorqExamplesTest/generate_test_summary.py --junit TEST-TorqExamplesTest.xml --profile-dir torq-examples/logs --out test_and_profile_summary.html

### CI run isolation

`test-torq-examples.yml` creates a fresh temporary directory for each run
and retry. Release downloads/extraction, the burn venv, burn output, and
collected test results use that directory. Wheel/ko paths in the workspace
are published only after the current downloads pass validation.
All checkouts explicitly enable cleaning; before pytest, the workflow
removes only the checkout's known model/output directories, pytest cache,
and report files. Failed or skipped test stages cannot upload old workspace
reports: uploads read only this run's result directory.

The workflow clears inherited artifact/version paths, Python path overrides,
pytest options/extra plugins, and disables third-party pytest plugin autoload.
It rebuilds the host image from the checked-out Dockerfile and pins the DUT
serial/workspace.
Proxy/CA/HF credentials, board/INI configuration, and the shared TFTP
firmware cache are preserved. Inherited `IMAGE_LOCAL` overrides are cleared
so firmware selection uses the configured TFTP root and target build/profile.
No global Docker/cache pruning or deletion of other runs is performed.

## Model modes

- The `test-torq-examples.yml` workflow currently supports release mode only.
  Trigger it with a public compiler release tag (`v2.3.0` by default):

      gh workflow run .github/workflows/test-torq-examples.yml --repo synaptics-torq/torq-compiler-dev --ref wip/torq_examples-ci-cd -f rel=v2.3.0

  `rel` is a workflow input, passed using `-f rel=...` (not `--rel`).
  Repository dispatch callers may also provide `client_payload.rel`; push
  and dispatch runs without a release input use `v2.3.0`.
  The workflow verifies the public tag and release, downloads
  `astra-release.tar.gz` and the matching CPython 3.12 aarch64 runtime wheel,
  and extracts `lib/syna_npu.ko`. It sets `TORQ_TEST_MODE=release`,
  and exports the absolute DUT artifact paths for subsequent steps. Missing
  tags, releases, or artifacts fail the workflow before pytest starts.
  Before checking out/preparing the examples, the workflow maps the release
  tag to its major/minor public compiler branch (`v2.3.0` -> `v2.3`) and reads
  `scripts/astra_build_version.txt` from that branch (not from the host
  checkout). Missing or invalid version files stop the workflow.
  It then runs the same `python -m torq.lab burn` flow as
  `test-image-burn.yml`, including its isolated environment, board INI,
  firmware download, PDU control, version-based flash/skip decision, and
  platform-info/diagnostic artifact uploads. Only successful completion
  (flash or skip) permits the demo tests to start.
  The manual inputs `profile`, `image_url`, `board`, `section`, `force_flash`,
  and `use_pdu_ac` retain the burn workflow defaults: all configured boards,
  profile `sl2619_oobe_scarthgap`, no forced flashing, and PDU AC enabled.
  The two workflows share concurrency group `syna-uboot-burn-astra-sh` to
  serialize firmware preparation and the subsequent example tests.
- `TORQ_TEST_MODE=release` (default): demos consume pre-compiled models
  published on HuggingFace, version-anchored to the checked-out torq-examples
  `VERSION` file (e.g. 2.3.0 -> v2.3.0). Setting `TORQ_MODEL_VERSION` pins an
  explicit release instead (the daily release workflow passes the `rel` tag).
  No compiler is installed; only the DUT runtime wheel and kernel module
  are under test.
- `TORQ_TEST_MODE=dev`: also downloads pre-compiled models from HuggingFace,
  pinned by `TORQ_MODEL_VERSION` (required; the DEV workflow takes it from the
  `models_version` input or the torq-examples `VERSION` file). No compiler
  wheel or model compilation is required. The runtime wheel and optional
  `astra-sl-release` KO are identified in the reports by SHA-256.

## Important boundary

This layer is intentionally not a validation suite for the helper
implementation itself. The object under test is the real demo stack
(runtime × ko × models × examples code) on the DUT, not the Python
orchestration used to drive it.
