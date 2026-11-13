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
   `setup_demos.py` for every demo inside a Docker container
   (`conversion/Dockerfile`, ubuntu 24.04 base, required because the
   compiler wheel is `manylinux_2_39` and the CI host may be older).
   Model artifacts land in `<examples checkout>/models/`.
2. **DUT deploy + inference + profiling** (`test_dut_deploy.py` and
   `dut_deploy.py`) — after the shared preflight and host preparation,
   the module fixture locks `sl2619npu01` without another power cycle.
   It then wipes and rebuilds the DUT workspace (no stale-file pollution), pushes the
   models, swaps the NPU kernel module, force-installs the aarch64 runtime
   wheel, and runs the per-demo inference and profiling matrices (repo-root
   `profile.py --repeat 3`, seven metrics). The fixed Netbooter sequence is
   in `dut_power_cycle.py`.

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

Run the complete suite from the workspace root (fixture dependencies and
conftest enforce DUT preflight -> host preparation -> DUT deploy/infer/profile,
regardless of collection order):

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
It resets container reuse/keep and DUT skip flags, rebuilds the host image
from the checked-out Dockerfile, and pins the DUT serial/workspace.
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
  and extracts `lib/syna_npu.ko`. It sets `TORQ_MODEL_MODE=release`,
  `TORQ_HOST_COMPILER_VERSION` without the leading `v`,
  `TORQ_HOST_COMPILER_WHEEL` to an empty string, and the absolute DUT artifact
  paths for subsequent steps. Missing tags, releases, or artifacts fail the
  workflow before pytest starts.
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
