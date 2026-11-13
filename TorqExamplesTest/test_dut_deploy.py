"""DUT deploy + inference matrix tests (host-driven reconstruction, step 2).

test_dut_deploy:        single case deploying the whole stack to the DUT
                        (workspace wipe/rebuild, model push, ko swap, ARM wheel)
test_dut_infer[<case>]:  per-model-dir inference cases (13 total, matching the
                        per-model-dir setup cases); known-debt demos (no command
                        builder yet) are skipped with an explicit reason
test_dut_profile[<case>]: per-model-dir profiling cases (repo-root profile.py,
                        --repeat 3, seven metrics) - applies to all demos

Requires on the host:
    the DUT attached to adb; the board is acquired through
    torq.utils.boards (shared control-file locking, like the legacy
    executor). TORQ_DUT_SERIAL pins a specific board (still locked);
    TORQ_BOARDS_CONTROL overrides the control-file path.
    TORQ_DUT_RUNTIME_WHEEL  aarch64 torq-runtime wheel path
    TORQ_DUT_KO             syna_npu.ko path
    models prepared on the host by step 1 (host_converter.py)
"""

from __future__ import annotations

import os

import pytest

from TorqExamplesTest.demo_registry import REPO_ROOT, demo_model_cases
from TorqExamplesTest.dut_deploy import DEMO_SPECS, DutDeployError, DutDeployRunner

# Per-model-dir (repo) granularity, identical to the setup cases: 13 entries.
# A multi-repo demo (object_detection) therefore yields two infer/profile cases
# instead of one, so a failure localises to the repo while the report shows
# every vmfb underneath it.
MODEL_CASES = demo_model_cases()


@pytest.fixture(scope="module")
def dut_deploy() -> DutDeployRunner:
    missing = [
        name
        for name, value in (
            ("TORQ_DUT_RUNTIME_WHEEL", os.environ.get("TORQ_DUT_RUNTIME_WHEEL", "")),
            ("TORQ_DUT_KO", os.environ.get("TORQ_DUT_KO", "")),
        )
        if not value.strip()
    ]
    if missing:
        pytest.fail(
            "DUT deploy inputs missing (set before pytest; the action flow "
            f"provides them from the run's artifacts): {', '.join(missing)}"
        )
    runner = DutDeployRunner(REPO_ROOT)
    try:
        runner.acquire_board()
    except DutDeployError as exc:
        pytest.fail(f"DUT not available: {exc}")
    try:
        yield runner
    finally:
        runner.release_board()


def test_dut_deploy(dut_deploy: DutDeployRunner) -> None:
    """Deploy the host-prepared stack to the DUT (wipe/rebuild + push + ko + wheel)."""
    phases = (
        ("connectivity", dut_deploy.phase_connectivity),
        ("workspace prep", dut_deploy.phase_prepare_workspace),
        ("model push", dut_deploy.phase_push_models),
        ("ko swap", dut_deploy.phase_ko_swap),
        ("ARM wheel install", dut_deploy.phase_wheel_install),
    )
    for name, fn in phases:
        try:
            result = fn()
        except DutDeployError as exc:
            pytest.fail(f"phase '{name}' aborted: {exc}")
        result.phase = name
        dut_deploy._write_log(result)
        if not result.passed:
            pytest.fail(
                f"phase '{name}' failed rc={result.returncode}\n"
                f"stdout tail:\n{result.stdout[-2000:]}\n"
                f"stderr tail:\n{result.stderr[-2000:]}\n"
                f"(DUT left in current state; inspect via adb -s {dut_deploy.serial} shell)"
            )


@pytest.mark.parametrize("model_case", MODEL_CASES, ids=lambda c: c.case_id)
def test_dut_infer(dut_deploy: DutDeployRunner, model_case) -> None:
    """Run inference for one model dir (repo) on the DUT.

    Granularity matches the per-model-dir setup cases (13 total). Known-debt
    demos (no command builder yet) are skipped with an explicit reason. Every
    vmfb inside a runnable model dir is executed (no short-circuit); the case
    is marked failed if any vmfb fails so it localises to the repo, while the
    per-model-file outcome is captured in {demo}_infer.json for the report.
    """
    spec = DEMO_SPECS[model_case.demo_name]
    if spec["builder"] is None:
        # Record the repo-level "skipped" marker in {demo}_infer.json (no DUT
        # work) so the report shows SKIPPED, not NOT RUN, for the debt vmfbs;
        # then skip the junit case itself.
        dut_deploy.phase_infer(model_case.demo_name, model_case.repo_id)
        pytest.skip(f"known debt: {spec.get('debt', 'no command builder')}")
    try:
        result = dut_deploy.phase_infer(model_case.demo_name, model_case.repo_id)
    except DutDeployError as exc:
        pytest.fail(f"infer {model_case.case_id} aborted: {exc}")
    dut_deploy._write_log(result)
    if not result.passed:
        pytest.fail(
            f"infer {model_case.case_id} failed rc={result.returncode}\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )


@pytest.mark.parametrize("model_case", MODEL_CASES, ids=lambda c: c.case_id)
def test_dut_profile(dut_deploy: DutDeployRunner, model_case) -> None:
    """Profile the VMFBs of one model dir (repo) via profile.py --repeat 3.

    Granularity matches the per-model-dir setup cases (13 total); applies to
    all demos including known debt. Every vmfb in the model dir is profiled
    (no short-circuit) and the per-model-file status is recorded in
    {demo}_profile.json.
    """
    try:
        result = dut_deploy.phase_profile(model_case.demo_name, model_case.repo_id)
    except DutDeployError as exc:
        pytest.fail(f"profile {model_case.case_id} aborted: {exc}")
    dut_deploy._write_log(result)
    if not result.passed:
        pytest.fail(
            f"profile {model_case.case_id} failed rc={result.returncode}\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )
