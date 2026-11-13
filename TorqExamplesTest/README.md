# TorqExamplesTest

Host-driven test suite for the torq-examples demo validation flow.

## Architecture (reconstruction, branch `newHostMethod`)

Two stages, all orchestrated from the CI host:

1. **Host-side model preparation** (`host_converter.py`) — runs
   `setup_demos.py` for every demo inside a Docker container
   (`conversion/Dockerfile`, ubuntu 24.04 base, required because the
   compiler wheel is `manylinux_2_39` and the CI host may be older).
   Model artifacts land in `<examples checkout>/models/`.
2. **DUT deploy + inference + profiling** (`dut_deploy.py`) — wipes and
   rebuilds the DUT workspace (no stale-file pollution), pushes the models,
   swaps the NPU kernel module, force-installs the aarch64 runtime wheel,
   then runs the per-demo inference matrix and the per-demo profiling
   matrix (repo-root `profile.py --repeat 3`, seven metrics).

The compiler/runtime/ko under test come from a test-build run's artifacts
(passed via `TORQ_HOST_COMPILER_WHEEL`, `TORQ_DUT_RUNTIME_WHEEL`,
`TORQ_DUT_KO`) so the tested stack is version-pinned per run.

## Test entries

- `test_host_conversion.py` — container environment + per-demo host setup
- `test_dut_deploy.py` — `test_dut_deploy` (deploy, one case) +
  `test_dut_infer[demo]` + `test_dut_profile[demo]` (12 demos each;
  demos without an inference command builder are skipped with a reason,
  profiling applies to all)

## Infrastructure

- `demo_registry.py`: AST discovery of demos/models from `setup_demos.py`
  (never executed; also the authoritative source for app dirs and model
  repo IDs — `dut_deploy.py` prefers it over its built-in spec table)
- `conftest.py`: repository-local pytest plugin (artifact cleanup, console
  tee, JUnit + combined HTML generation, failure semantics)
- `generate_test_summary.py`: merges JUnit results with structured
  profiling records into one HTML report

## Run and report generation

Run the complete suite from the workspace root (conftest enforces
host-before-DUT ordering regardless of collection order):

	python3.12 -m pytest TorqExamplesTest

Standalone debugging without pytest (engine drivers, no DUT needed for
host stage, no suite layout needed):

	python3 TorqExamplesTest/host_converter.py <examples_root> [demo ...]
	TORQ_DUT_RUNTIME_WHEEL=... TORQ_DUT_KO=... \
	  python3 TorqExamplesTest/dut_deploy.py <examples_root> [demo ...]

Every invocation produces:

- `TEST-TorqExamplesTest.xml`: final pytest JUnit report
- `test_and_profile_summary.html`: combined test + profiling report
  (run info, per-case results with model VERSION / HF revision, TTFT /
  tok/s for LLM demos, seven profiling metrics)
- `torq-examples/logs/`: per-phase logs and `*_profile.json` records

To regenerate only the HTML from existing artifacts:

	python TorqExamplesTest/generate_test_summary.py --junit TEST-TorqExamplesTest.xml --profile-dir torq-examples/logs --out test_and_profile_summary.html

## Model modes

- `TORQ_MODEL_MODE=release` (default): demos consume pre-compiled models
  published on HuggingFace (version-anchored per release, e.g. v2.3.0).
  No compiler is required on the host; only the DUT runtime wheel and
  kernel module must match the release under test.
- `TORQ_MODEL_MODE=dev`: build models from source with the PR compiler so
  a not-yet-released compiler can be validated against examples. Not
  implemented yet - the suite fails explicitly until it lands.

## Important boundary

This layer is intentionally not a validation suite for the helper
implementation itself. The object under test is the real demo stack
(compiler × runtime × ko × models × examples code) on the DUT, not the
Python orchestration used to drive it.
