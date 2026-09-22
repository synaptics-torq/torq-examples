# TorqExamplesTest Runbook

Git clone `torq-examples` with branch `wip/ci_cd_test`, which includes the test suite `TorqExamplesTest`, to `/home` on the DUT.

Move `TorqExamplesTest/` to also under `/home`. Keep `torq-examples/` and `TorqExamplesTest/` at the same level under `/home`.

## Layout

```text
/home/
├── torq-examples/
└── TorqExamplesTest/
    ├── RunSuite.sh
    ├── test_demo_board.py
    ├── prepare_dut_python_env.sh
    ├── run_demo_setup.sh
    ├── run_demo_gemma3.sh
    ├── run_demo_liquidai_lfm2_5_230m.sh
    ├── run_demo_liquidai_lfm2_vl_450m.sh
    ├── run_demo_moonshine.sh
    ├── run_demo_moonshine_streaming.sh
    ├── run_demo_object_detection.sh
    ├── run_demo_pose_estimation.sh
    ├── run_demo_profile.sh
    ├── collect_profile_metrics.py
    ├── generate_test_summary.py
    ├── hello_world_16k.wav
    ├── normalize_junit_report.py
    ├── read_demo_names.py
    ├── README.md
    └── requirements-host.txt
```

## Usage

`TorqExamplesTest` can test based on target commit and target demo. For example:

`https://github.com/synaptics-torq/torq-examples/commit/50b7cdb526eddd226042bbd111da4b4085903ccd`

Run the command below to test a target commit with a target demo:

```bash
bash RunSuite.sh 50b7cdb526eddd226042bbd111da4b4085903ccd LiquidAI-LFM2-VL-450M
```

If you run the command without a demo name, for example:

```bash
bash RunSuite.sh 50b7cdb526eddd226042bbd111da4b4085903ccd
```

The test suite will run all 7 supported demos and their profiling:

1. `gemma3`
2. `LiquidAI-LFM2.5-230M`
3. `moonshine`
4. `moonshine_streaming`
5. `LiquidAI-LFM2-VL-450M`
6. `object_detection`
7. `pose_estimation`

## Log File Location

After test run related log files are generated under the target repo directory `/home/torq-examples/`.

```text
/home/
└── torq-examples/
    └── log/
        ├── execute.log
        ├── main.log
        ├── run_demo_profile_home_torq-examples_models_Synaptics_LiquidAI-LFM2-VL-450M_decod.log
        ├── run_demo_liquidai_lfm2_vl_450m_--model_home_torq-examples_models_Synaptics_Liqui.log
        └── suite.log
```

Three result files are generated under the test suite directory `/home/TorqExamplesTest/`:

```text
/home/
└── TorqExamplesTest/
    ├── profile_summary.html
    ├── test_and_profile_summary.html
    └── TEST-TorqExamplesTest.xml
```

## File Guide

Main entry points:

- `RunSuite.sh`
    Main entry for the whole test suite. It checks out the target `torq-examples` commit, prepares the DUT Python environment, selects one demo or the full demo list, runs `pytest`, and generates the final XML and HTML summaries.

- `test_demo_board.py`
    The pytest scheduler for the suite. It loads the demo queue, runs per-demo setup, dispatches each demo inference script, runs model profiling, and controls skip/fail behavior in the test report.

Environment and setup helpers:

- `prepare_dut_python_env.sh`
    Older ADB-based environment preparation helper. It checks DUT connectivity, clones `torq-examples`, creates the DUT virtual environment, and installs the Torq runtime wheel.

- `run_demo_setup.sh`
    Shared per-demo setup wrapper. It installs base requirements, installs demo-specific requirements when needed, and runs `python3 setup_demos.py <demo>` with retry logic.

Per-demo runtime wrappers:

- `run_demo_gemma3.sh`
    Runs the Gemma3 demo with a target model path and optional LM head, then feeds a single prompt into the interactive infer script for automated validation.

    Actual command pattern:

    ```bash
    cd /home/torq-examples/gemma3
    printf '%s\n' "$PROMPT" | python3 src/infer.py -m "$MODEL_PATH" [--instruct-model] [--lm-head "$LM_HEAD"]
    ```

- `run_demo_liquidai_lfm2_5_230m.sh`
    Runs the LiquidAI LFM2.5-230M text model with body VMFB, LM head, and optional prompt for automated single-turn validation.

    Actual command pattern:

    ```bash
    cd /home/torq-examples/LiquidAI/LiquidAI-LFM2.5-230M
    printf '%s\n' "$PROMPT" | python3 src/infer.py -m "$MODEL_PATH" [--instruct-model] [--lm-head "$LM_HEAD"]
    ```

- `run_demo_liquidai_lfm2_vl_450m.sh`
    Runs the LiquidAI LFM2-VL-450M vision-language demo with decoder, LM head, vision encoder, image decoder, image input, and prompt.

    Actual command pattern:

    ```bash
    cd /home/torq-examples/LiquidAI/LiquidAI-LFM2-VL-450M
    python3 src/infer.py -m "$MODEL_PATH" --vision "$VISION_MODEL" --image "$IMAGE_PATH" --prompt "$PROMPT" [--lm-head "$LM_HEAD"] [--image-decoder "$IMAGE_DECODER"]
    ```

- `run_demo_moonshine.sh`
    Runs the Moonshine speech demo against a WAV input, using the prepared demo assets under `models/Synaptics/moonshine-tiny-bf16-torq`.

    Actual command pattern:

    ```bash
    cd /home/torq-examples/moonshine
    python3 src/infer.py -m "$MODEL_DIR" "$WAV_PATH"
    ```

- `run_demo_moonshine_streaming.sh`
    Runs the Moonshine streaming speech demo against a WAV input, using the prepared streaming model assets.

    Actual command pattern:

    ```bash
    cd /home/torq-examples/moonshine_streaming
    python3 src/infer.py -m "$MODEL_DIR" --wav "$WAV_PATH"
    ```

- `run_demo_object_detection.sh`
    Thin wrapper for `object_detection/src/infer.py`. The pytest suite passes the resolved model, image, label, and device arguments through this script.

    Actual command pattern:

    ```bash
    cd /home/torq-examples
    python3 object_detection/src/infer.py "$@"
    ```

- `run_demo_pose_estimation.sh`
    Thin wrapper for `pose_estimation/src/infer.py`. The pytest suite passes the resolved model, image, and device arguments through this script.

    Actual command pattern:

    ```bash
    cd /home/torq-examples
    python3 pose_estimation/src/infer.py "$@"
    ```

- `run_demo_profile.sh`
    Wrapper around `profile.py` used by the profiling test stage to collect model performance numbers for prepared VMFB models.

    Actual command pattern:

    ```bash
    cd /home/torq-examples
    python3 profile.py "$@"
    ```

Reporting and queue helpers:

- `collect_profile_metrics.py`
    Parses profiling log files and generates `profile_summary.html`.

- `generate_test_summary.py`
    Combines the JUnit XML report and profiling HTML into one `test_and_profile_summary.html` page.

- `normalize_junit_report.py`
    Rewrites pytest-generated JUnit case names into a more stable and readable test naming format.

- `read_demo_names.py`
    Resolves the effective demo list from defaults plus `config.ini`, and prints the enabled demos as a comma-separated list.

Static assets and support files:

- `hello_world_16k.wav`
    Default sample WAV file used by the Moonshine and Moonshine Streaming tests.

- `requirements-host.txt`
    Host-side Python dependency list for running the test harness and summary tools.

- `README.md`
    Quick-start runbook for this test suite.
