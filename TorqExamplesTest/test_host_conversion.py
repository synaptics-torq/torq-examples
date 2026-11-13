"""Host-side demo setup tests (host-driven reconstruction, step 1).

Demo model preparation (setup_demos.py: fetch + convert/validate) now runs
on the CI host inside a container (see host_converter.py) instead of on the
DUT board. Each case verifies that one demo's model artifacts exist in the
host checkout, ready to be pushed to the board for inference (step 2).

The legacy DUT-side conversion cases stay in place for this step; they are
replaced by the push-to-board stage in step 2.

Requires docker on the host. Compiler wheel selection via environment:
    TORQ_HOST_COMPILER_WHEEL=/path/torq_compiler-*.whl   (CI: PR-built wheel)
    TORQ_HOST_COMPILER_VERSION=2.3.0                     (dev: published release)
"""

from __future__ import annotations

import pytest

from TorqExamplesTest.demo_registry import REPO_ROOT, demo_model_cases, demo_registry
from TorqExamplesTest.host_converter import HostSetupRunner

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
    """Prepare one demo's models on the host and validate the artifacts."""
    demo = model_case.demo_name
    if demo in _converted_demos:
        print(f"[host_setup] {demo}: artifacts already prepared by an earlier case")
        return
    if demo in _failed_demos:
        pytest.fail(f"demo '{demo}' host setup already failed in an earlier case "
                    f"(see its log: logs/{demo}_host_setup.log)")

    spec = REGISTRY[demo]
    result = host_setup.setup_demo(demo, spec.app_dir, model_case.repo_id)
    assert result.passed, (
        f"host setup for demo '{demo}' failed rc={result.returncode}\n"
        f"stdout tail:\n{result.stdout[-2000:]}\n"
        f"stderr tail:\n{result.stderr[-2000:]}"
    )
    _converted_demos.add(demo)
