# Torq Examples Manual Runbook

This runbook is for manual DUT-side validation of the Torq demos. It is intended for engineers who need to reproduce the same flow as the CI pipeline without relying on the Jenkins wrapper alone.

## 1. Prerequisites

- A DUT is connected over ADB.
- `ANDROID_SERIAL` is set to the device serial number.
- `DUT_REPO_DIR` is set to `/home/torq-examples` on the DUT.
- `git`, `python3`, and `adb` are available on the target environment.

Example:

```bash
export ANDROID_SERIAL=<device-serial>
export DUT_REPO_DIR=/home/torq-examples
export HOST_LOG_FOLDER=./log
```

---

## 2. One-time DUT repo setup

Run the following from the host machine (or a shell that can reach the DUT through `adb`):

```bash
adb -s "$ANDROID_SERIAL" shell
mkdir -p /home
cd /home
if [ -d torq-examples ]; then rm -rf torq-examples; fi
git clone -b main https://github.com/synaptics-torq/torq-examples.git torq-examples
```

Then initialize the Python environment on the DUT:

```bash
adb -s "$ANDROID_SERIAL" shell "cd /home/torq-examples && python3 -m venv --system-site-packages .venv && . .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel && python -m pip install https://github.com/synaptics-torq/torq-compiler/releases/download/v2.1.0/torq_runtime-2.1.0-cp312-cp312-manylinux_2_28_aarch64.whl && python -m pip install -r requirements.txt && python -c 'import torq.runtime; print(\"TORQ_RUNTIME_READY\")'"
```

The Torq runtime is required by every inference entrypoint. The Jenkins wrapper performs the same installation and refuses to reuse a venv when `torq.runtime` cannot be imported. All DUT-side automation logs, including the runtime installation output, are stored under `/home/torq-examples/log/`.

If you only want to run the Jenkins wrapper, you can also use:

```bash
bash JenkinsRun.sh
```

The wrapper writes logs in the host log folder and names the JUnit XML by DUT build version and profile:

```text
./log/
├── main.log
├── suite.log
├── execute.log
├── adb.log
├── adb_devices.log
├── dut_dmesg.log
├── device_dut_pip_upgrade.log
├── device_dut_torq_runtime_install.log
├── device_dut_requirements_install.log
├── device_run_demo_setup_gemma3.log
├── device_run_demo_gemma3_....log
├── 202608252006-sl2619_scarthgap-TorqExamplesTest.Report.xml
└── ...
```

All DUT-side logs are written under `/home/torq-examples/log/`. At test teardown, each DUT log file is pulled directly into the Host `./log/` folder with a `device_` filename prefix. This avoids the previous `./log/dut_log/log/` nesting and prevents DUT logs from overwriting Host-generated logs. After every DUT log file has been pulled successfully, `/home/torq-examples/log/` is deleted from the DUT to prevent stale logs from contaminating later runs. If any pull fails, the DUT directory is preserved for recovery and debugging.

The JUnit report name follows the current pattern:

```text
${SYNA_SDK_REVISION}-${BUILD_PROFILE}-TorqExamplesTest.Report.xml
```

with `BUILD_PROFILE` defaulting to `sl2619_scarthgap`.

---

## 3. Common setup for each demo

From the DUT repo root:

```bash
adb -s "$ANDROID_SERIAL" shell
cd /home/torq-examples
. .venv/bin/activate
```

Then run the one-time setup for a specific demo:

```bash
python setup_demos.py <demo_name>
```

Examples:

```bash
pip install -r /home/torq-examples/gemma3/requirements.txt
python setup_demos.py gemma3
python setup_demos.py moonshine
python setup_demos.py object_detection
python setup_demos.py pose_estimation
python setup_demos.py LiquidAI-LFM2.5-230M
python setup_demos.py LiquidAI-LFM2-VL-450M
python setup_demos.py moonshine_streaming
```

The repo-level setup downloads model assets into `models/Synaptics/...` and is required before manual inference.

---

## 3.5 Automated test coverage in this repo

The DUT-side automation now defaults to a full demo suite run unless a narrower target is explicitly requested.

Default execution policy:

- `RUN_FULL_DEMO_SUITE` defaults to `1`
- `DEMO_NAME` is optional and only used for a single-demo override
- `DEMO_NAMES` can be used to set an explicit comma-separated list
- If a setup step fails for one demo, the XML report still records the failed setup case and marks remaining demo cases as skipped so the overall report remains complete

Total automated coverage count (default full-suite mode):

- `test_demo_setup_on_dut`: parameterized across all available demo names
- `test_object_detection_inference_on_dut`: 1 inference test
- `test_pose_estimation_inference_on_dut`: 1 inference test
- `test_demo_profiling_on_dut`: parameterized across successfully prepared demo names (records non-blocking NPU performance metrics)
- Total: full-suite execution across the current demo manifest

Covered demo setup checks:

1. `gemma3`
2. `LiquidAI-LFM2.5-230M`
3. `moonshine`
4. `moonshine_streaming`
5. `LiquidAI-LFM2-VL-450M`
6. `object_detection`
7. `pose_estimation`

Covered runtime inference checks:

1. `object_detection` model inference
2. `pose_estimation` model inference

Covered automated profiling checks:

- Parameterized profiling executes on successfully set up models, run via `run_demo_profile.sh` with a default repetition of 3.
- Performance logs & metrics (e.g., median, mean latencies) are automatically parsed from the stdout and recorded under `logs/`.
- Metric profiling is non-blocking: any profiling failures trigger a warning/skip logic but do not cause the entire integration pipeline to fail or report a red build.

Important distinction:

- The automation defaults to a full demo suite run and validates setup for the full demo manifest on the DUT.
- It also executes runtime inference checks for deterministic demos with explicit, repeatable entrypoints, notably the object-detection and pose-estimation models.
- For Gemma3 / Moonshine / LiquidAI and similar interactive demos, the manual flow below remains the intended one-by-one validation path.
- When a setup phase fails, the remaining demo entries are not silently lost: they remain in the JUnit XML as skipped tests so the report shows the full intended execution plan.

This means the default automation run is a full demo validation sweep, not a single smoke-only target.


## 3.6 Profiling summary generation

The automated profiling tests (`test_demo_profiling_on_dut`) produce DUT-side profiling logs which are pulled to the host `./log/` folder during teardown. After all demos have completed and all DUT logs are pulled, run the host-side parser to generate a single Excel summary (example: `book1.xlsx` / `profile_summary.xlsx`).

Recommended position in CI: Jenkins `Post-processing` stage (after `Run Pytest` and `Pull DUT logs`).

Quick usage (run on the host where logs were pulled):

```bash
python3 TorqExamplesTest/collect_profile_metrics.py --logs-dir ./log --out ./profile_summary.html
# then archive ./log/profile_summary.html as a build artifact
```

Notes:
- The parser looks for lines emitted by the profiling helper marked with `[METRICS] --- <demo> Profile Summary ---` and collects the block until the matching end marker. It extracts numeric values (mean/median/latency/fps) when present and writes a table with one row per demo-log.
- The parser uses only the Python standard library, so it does not depend on `pandas` or any NumPy-linked binary packages.


## 4. Demo-by-demo manual run steps

### 4.1 Gemma 3

Current repo builds typically download the newer model set:

- `transformer.vmfb`
- `lm_head.vmfb.trim` (or `lm_head.vmfb`)
- tokenizer/config files

Use the current command below:

```bash
adb -s "$ANDROID_SERIAL" shell "cd /home/torq-examples/gemma3 && . /home/torq-examples/.venv/bin/activate && python src/infer.py -m ../models/Synaptics/gemma-3-270m-it-torq/transformer.vmfb --instruct-model"
```

If the repo still contains a legacy single-file model instead, use:

```bash
python src/infer.py -m ../models/Synaptics/gemma-3-270m-it-torq/model.vmfb --instruct-model
```

If the LM head is not auto-discovered, specify it explicitly:

```bash
python src/infer.py -m ../models/Synaptics/gemma-3-270m-it-torq/transformer.vmfb --lm-head ../models/Synaptics/gemma-3-270m-it-torq/lm_head.vmfb.trim --instruct-model
```

---

### 4.2 Moonshine

Current DUT behavior: the Moonshine wrapper defaults to the CPU allocator workaround because the dmabuf allocator can trigger a `failed to writeXram()` runtime failure on some DUTs. If you run the infer command directly, add `--tda cpu` explicitly.

```bash
adb -s "$ANDROID_SERIAL" shell "cd /home/torq-examples/moonshine && . /home/torq-examples/.venv/bin/activate && python src/infer.py -m ../models/Synaptics/moonshine-tiny-bf16-torq --tda cpu /path/to/audio.wav"
```

For a local sample file, replace `/path/to/audio.wav` with the actual audio path. The wrapper script also exposes this as `MOONSHINE_TDA=cpu` or a `--tda` argument override.

---

### 4.3 Moonshine Streaming

```bash
adb -s "$ANDROID_SERIAL" shell "cd /home/torq-examples/moonshine_streaming && . /home/torq-examples/.venv/bin/activate && python src/infer.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --wav /path/to/sample.wav"
```

Optional realtime mode:

```bash
python src/infer.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --wav /path/to/sample.wav --realtime
```

---

### 4.4 Object Detection

```bash
adb -s "$ANDROID_SERIAL" shell "cd /home/torq-examples/object_detection && . /home/torq-examples/.venv/bin/activate && python src/infer.py --model ../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb --image ../models/Synaptics/yolov8-od-nano-320-int8-torq/samples/dog_bike_car.jpg --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json --device torq --device-io"
```

Optional annotated display output:

```bash
python src/infer.py --model ../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb --image ../models/Synaptics/yolov8-od-nano-320-int8-torq/samples/dog_bike_car.jpg --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json --device torq --device-io --save-image --display
```

---

### 4.5 Pose Estimation

```bash
adb -s "$ANDROID_SERIAL" shell "cd /home/torq-examples/pose_estimation && . /home/torq-examples/.venv/bin/activate && python src/infer.py --model ../models/Synaptics/yolov8-pose-nano-320-int8-torq/yolo_pose.vmfb --image ../models/Synaptics/yolov8-pose-nano-320-int8-torq/samples/pose.jpg --device torq --device-io"
```

Optional annotated image save:

```bash
python src/infer.py --model ../models/Synaptics/yolov8-pose-nano-320-int8-torq/yolo_pose.vmfb --image ../models/Synaptics/yolov8-pose-nano-320-int8-torq/samples/pose.jpg --device torq --device-io --save-image --display
```

---

### 4.6 LiquidAI LFM2.5-230M

```bash
adb -s "$ANDROID_SERIAL" shell "cd /home/torq-examples/LiquidAI/LiquidAI-LFM2.5-230M && . /home/torq-examples/.venv/bin/activate && python src/infer.py -m ../../models/Synaptics/LiquidAI-LFM2.5-230M/body.vmfb --lm-head ../../models/Synaptics/LiquidAI-LFM2.5-230M/lm_head.vmfb --instruct-model"
```

This is the interactive chat flow for the 230M model. Use `quit` or `exit` to stop the session.

---

### 4.7 LiquidAI LFM2-VL-450M

```bash
adb -s "$ANDROID_SERIAL" shell "cd /home/torq-examples/LiquidAI/LiquidAI-LFM2-VL-450M && . /home/torq-examples/.venv/bin/activate && MODELS=../../models/Synaptics/LiquidAI-LFM2-VL-450M && python src/infer.py -m \$MODELS/decoder_nolm.vmfb --lm-head \$MODELS/lm_head.vmfb --vision \$MODELS/vision_encoder_256.vmfb --image-decoder \$MODELS/decoder_image_2part_ --image \$MODELS/cats-and-dogs-256.jpg"
```

This runs the visual-language prompt loop for the provided sample image.

---

## 5. Useful troubleshooting checks

### Check the DUT repo exists

```bash
adb -s "$ANDROID_SERIAL" shell "ls -ld /home/torq-examples"
```

### Check ADB connection

```bash
adb devices
adb -s "$ANDROID_SERIAL" shell "echo adb_ok"
```

### Re-run setup for a specific demo

```bash
adb -s "$ANDROID_SERIAL" shell "cd /home/torq-examples && . .venv/bin/activate && python setup_demos.py <demo_name>"
```

### Collect logs after a run

```bash
adb -s "$ANDROID_SERIAL" shell "dmesg | tail -n 200" > ./log/device/dut_dmesg.txt
```

---

## 6. Recommended manual validation order

If you are testing manually for the first time, run in this order:

1. `object_detection`
2. `pose_estimation`
3. `gemma3`
4. `moonshine`
5. `moonshine_streaming`
6. `LiquidAI-LFM2.5-230M`
7. `LiquidAI-LFM2-VL-450M`

This order is simple to debug and matches the typical model download + inference setup flow.

For automated runs, the default behavior is full-suite execution: the harness prints the full demo queue before starting, and then runs the full manifest unless `DEMO_NAME`, `DEMO_NAMES`, or `RUN_FULL_DEMO_SUITE=0` narrows the scope.
