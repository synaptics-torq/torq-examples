"""
Host-side demo model preparation tests.

Demo model preparation (setup_demos.py: fetch + convert/validate) now runs
on the CI host inside a container (see host_converter.py) instead of on the
DUT board. Each case verifies that one demo's model artifacts exist in the
host checkout, ready to be pushed to the board for inference.

TorqExamplesTest has two model modes, selected by TORQ_MODEL_MODE:
    release (default): download and validate models already published on
        Hugging Face using the normal release version resolution.
    dev: download pre-compiled models from Hugging Face pinned by
        TORQ_MODEL_VERSION. No compiler or model compilation is involved.

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
    """Host environment is ready without installing a compiler in either mode."""
    result = host_setup.setup_environment()
    if host_setup.model_mode == "dev":
        assert "compiler: not installed (dev mode: HF model download)" in result.stdout, (
            "dev model downloads must not depend on a compiler installation"
        )
    else:
        assert "compiler: not installed (release mode)" in result.stdout
    assert "host environment ready" in result.stdout


@pytest.mark.parametrize("model_case", MODEL_CASES, ids=lambda case: case.case_id)
def test_demo_setup_on_host(host_setup: HostSetupRunner, model_case) -> None:
    """Prepare release models, pinning DEV downloads to TORQ_MODEL_VERSION."""
    demo = model_case.demo_name
    print(
        f"\n[host_setup] Starting {model_case.case_id}: "
        f"mode={host_setup.model_mode}, repo_id={model_case.repo_id}",
        flush=True,
    )
    spec = REGISTRY[demo]
    try:
        if len(spec.setup_model_repositories) > 1:
            result = host_setup.setup_demo(
                demo, spec.app_dir, model_name=model_case.model_name
            )
        else:
            result = host_setup.setup_demo(demo, spec.app_dir)
    except HostSetupError as exc:
        print(f"[host_setup] FAILED {model_case.case_id}: {exc}", flush=True)
        raise
    if not result.passed:
        message = (
            f"host setup for model '{model_case.case_id}' failed rc={result.returncode}\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )
        print(f"[host_setup] FAILED {model_case.case_id}: {message}", flush=True)
        raise AssertionError(message)

    model_dir = host_setup.examples_root / "models" / model_case.repo_id
    assert model_dir.is_dir() and any(model_dir.rglob("*")), (
        f"[model check] {model_case.case_id}: missing or empty model dir: {model_dir}"
    )
    print(
        f"[host_setup] Completed {model_case.case_id}: "
        f"model directory check passed for {model_dir}",
        flush=True,
    )
