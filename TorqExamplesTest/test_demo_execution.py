"""Per-model execution tests for the Python-only refactor flow.

Cases are generated from each demo's dynamically discovered model repositories.
They intentionally do not rely on global skip flags or shell wrappers.
"""

from __future__ import annotations

import json

import pytest

from TorqExamplesTest.demo_executor import DemoExecutor
from TorqExamplesTest.demo_registry import demo_model_cases


MODEL_CASES = demo_model_cases()


@pytest.fixture(scope="session")
def executor():
    """Keep one Torq-managed DUT locked for the complete pytest session."""
    with DemoExecutor() as suite_executor:
        yield suite_executor


def test_prepare(executor, record_property):
    """Push the repository and prepare the shared DUT Python environment once."""
    result = executor.prepare_dut()
    if result.model_dirs:
        record_property(
            "model_dirs",
            json.dumps(result.model_dirs, ensure_ascii=False),
        )
    for name, value in result.metadata.items():
        record_property(name, value)
    if not result.passed:
        pytest.fail(
            f"DUT preparation failed with rc={result.returncode}\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}",
            pytrace=False,
        )


@pytest.mark.parametrize("model_case", MODEL_CASES, ids=lambda case: case.case_id)
def test_convert_demo(executor, model_case, record_property):
    result = executor.convert_demo_on_dut(
        model_case.demo_name,
        model_path=model_case.model_path,
    )
    if result.model_dirs:
        record_property(
            "model_dirs",
            json.dumps(result.model_dirs, ensure_ascii=False),
        )
    if result.model_versions:
        record_property(
            "model_versions",
            json.dumps(result.model_versions, ensure_ascii=False),
        )
    if not result.passed:
        pytest.fail(
            f"Demo conversion '{model_case.case_id}' failed with rc={result.returncode}\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}",
            pytrace=False,
        )


@pytest.mark.parametrize("model_case", MODEL_CASES, ids=lambda case: case.case_id)
def test_run_demo(executor, model_case):
    result = executor.run_demo(
        model_case.demo_name,
        model_path=model_case.model_path,
    )
    if not result.passed:
        pytest.fail(
            f"Demo '{model_case.case_id}' failed with rc={result.returncode}\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}",
            pytrace=False,
        )


@pytest.mark.parametrize("model_case", MODEL_CASES, ids=lambda case: case.case_id)
def test_profile_demo(executor, model_case, record_property):
    """Profile one dynamically selected demo model repository over ADB."""
    result = executor.profile_demo(
        model_case.demo_name,
        model_path=model_case.model_path,
    )
    if result.profile_records:
        record_property(
            "profile_metrics",
            json.dumps(result.profile_records, ensure_ascii=False),
        )
    if result.log_path:
        record_property("profile_log", result.log_path)
    if not result.passed:
        pytest.fail(
            f"Demo profiling '{model_case.case_id}' failed with rc={result.returncode}\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}",
            pytrace=False,
        )
