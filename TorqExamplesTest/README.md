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
