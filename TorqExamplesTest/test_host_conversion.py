"""
Host-side demo model preparation tests.

Demo model preparation (setup_demos.py: fetch + convert/validate) now runs
on the CI host inside a container (see host_converter.py) instead of on the
DUT board. Each case verifies that one demo's model artifacts exist in the
host checkout, ready to be pushed to the board for inference.

TorqExamplesTest has two model modes, selected by TORQ_MODEL_MODE:
    release (default): download and validate models already published on
        Hugging Face and pinned to a release (e.g. moonshine v2.3.0).
        No host-side model compilation is required.
    dev: build models from source using the PR compiler, as specified by
        IOTMMSQA-18868 ("build all models from their sources").

Model preparation runs on the CI host inside a Docker container (see
host_converter.py), producing artifacts for subsequent DUT deployment
and inference. Docker is required on the host.
"""

from __future__ import annotations

import pytest

from TorqExamplesTest.demo_registry import REPO_ROOT, demo_model_cases, demo_registry
from TorqExamplesTest.host_converter import HostSetupError, HostSetupRunner

REGISTRY = demo_registry()
MODEL_CASES = demo_model_cases()

_converted_demos: set[str] = set()
_failed_demos: set[str] = set()


@pytest.fixture(scope="session")
def host_setup(dut_ready: None) -> HostSetupRunner:
    runner = HostSetupRunner(examples_root=REPO_ROOT, log_dir=REPO_ROOT / "logs")
    if not runner.docker_available():
        pytest.fail("docker is not available on the host; host-side setup cannot run")
    try:
        runner.ensure_image()
        runner.start()
        yield runner
    finally:
        runner.stop()


def test_host_setup_environment(host_setup: HostSetupRunner) -> None:
    """Host environment ready: venv + requirements (release) or container
    venv + verified compiler (dev)."""
    result = host_setup.setup_environment()
    if host_setup.model_mode == "dev":
        assert "torq.compiler import OK" in result.stdout, (
            "compiler wheel did not import inside the container "
            "(check glibc/GLIBCXX compatibility of the base image)"
        )
    else:
        assert "compiler: not installed (release mode)" in result.stdout
    assert "host environment ready" in result.stdout


@pytest.mark.parametrize("model_case", MODEL_CASES, ids=lambda case: case.case_id)
def test_demo_setup_on_host(host_setup: HostSetupRunner, model_case) -> None:
    """Prepare demo models for TORQ_MODEL_MODE: release downloads or dev builds."""
    demo = model_case.demo_name
    print(
        f"\n[host_setup] Starting {model_case.case_id}: "
        f"mode={host_setup.model_mode}, repo_id={model_case.repo_id}",
        flush=True,
    )
    if demo in _converted_demos:
        print(
            f"[host_setup] {model_case.case_id}: reusing earlier demo preparation; "
            "checking this model directory",
            flush=True,
        )
    elif demo in _failed_demos:
        pytest.fail(f"demo '{demo}' host setup already failed in an earlier case "
                    f"(see its log: logs/{demo}_host_setup.log)")
    else:
        spec = REGISTRY[demo]
        try:
            result = host_setup.setup_demo(demo, spec.app_dir)
        except HostSetupError:
            _failed_demos.add(demo)
            raise
        if not result.passed:
            _failed_demos.add(demo)
        assert result.passed, (
            f"host setup for demo '{demo}' failed rc={result.returncode}\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )
        _converted_demos.add(demo)

    model_dir = host_setup.examples_root / "models" / model_case.repo_id
    assert model_dir.is_dir() and any(model_dir.rglob("*")), (
        f"[model check] {model_case.case_id}: missing or empty model dir: {model_dir}"
    )
    print(
        f"[host_setup] Completed {model_case.case_id}: "
        f"model directory check passed for {model_dir}",
        flush=True,
    )
