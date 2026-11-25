import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from TorqExamplesTest import dut_power_cycle
from TorqExamplesTest.dut_power_cycle import adb_device_is_online


def _skip_sleep(seconds: int) -> None:
    assert seconds > 0


def test_adb_device_is_online_requires_expected_serial_and_device_state() -> None:
    assert adb_device_is_online(
        "List of devices attached\nsl2619npu01\tdevice\n"
    )
    assert not adb_device_is_online(
        "List of devices attached\nsl2619npu01\toffline\n"
    )
    assert not adb_device_is_online(
        "List of devices attached\nanother-device\tdevice\n"
    )


def test_power_cycle_runs_sequence_and_checks_adb(monkeypatch, capsys) -> None:
    calls: list[tuple[list[str], object]] = []
    adb_attempts = 0

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal adb_attempts
        calls.append((command, kwargs.get("timeout")))
        if "adb devices" in command[2]:
            adb_attempts += 1
            state = "device" if adb_attempts == 3 else "offline"
            stdout = f"List of devices attached\nsl2619npu01\t{state}\n"
        else:
            stdout = "Netbooter command completed\n"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(dut_power_cycle.subprocess, "run", fake_run)
    monkeypatch.setattr(dut_power_cycle.time, "sleep", _skip_sleep)

    dut_power_cycle.power_cycle_dut()

    scripts = [call[0][2] for call in calls]
    assert len(scripts) == 5
    assert "-c OFF" in scripts[0]
    assert "-c ON" in scripts[1]
    assert all("adb devices" in script for script in scripts[2:])
    for call in calls:
        expected_timeout = (
            dut_power_cycle.ADB_CHECK_TIMEOUT_SECONDS
            if "adb devices" in call[0][2]
            else dut_power_cycle.COMMAND_TIMEOUT_SECONDS
        )
        assert call[1] == expected_timeout
    output = capsys.readouterr().out
    assert "Power-off wait: 1/5s elapsed." in output
    assert "Power-off wait: 5/5s elapsed." in output
    assert "Initial boot wait: 15/15s elapsed." in output
    assert output.count("ADB discovery wait: 5/5s elapsed.") == 3
    assert "checking ADB devices at 30s (attempt 3/6)" in output
    assert "ADB device sl2619npu01 is online after 30s." in output


def test_power_cycle_fails_if_device_stays_offline_for_45_seconds(
    monkeypatch,
    capsys,
) -> None:
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        assert isinstance(kwargs.get("timeout"), int)
        stdout = (
            "List of devices attached\nsl2619npu01\toffline\n"
            if "adb devices" in command[2]
            else ""
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(dut_power_cycle.subprocess, "run", fake_run)
    monkeypatch.setattr(dut_power_cycle.time, "sleep", _skip_sleep)

    try:
        dut_power_cycle.power_cycle_dut()
    except dut_power_cycle.DutPowerCycleError as exc:
        assert "within 45s after power-on" in str(exc)
        assert "Need manual intervention to check the environment!!!!" in str(exc)
    else:
        raise AssertionError("power_cycle_dut should fail after 45 seconds")
    assert calls == 2 + dut_power_cycle.ADB_POLL_ATTEMPTS
    assert "Need manual intervention to check the environment!!!!" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("selection", "power_cycle_fails"),
    [("", False), ("", True), ("host", False), ("dut", False)],
)
def test_pipeline_checks_dut_before_host_setup(
    tmp_path, selection: str, power_cycle_fails: bool
) -> None:
    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(
            f"""
            import os
            import pytest
            from TorqExamplesTest import conftest as suite
            from TorqExamplesTest import test_host_conversion as host
            from TorqExamplesTest import test_dut_deploy as dut

            dut_ready = suite.dut_ready
            host_setup = host.host_setup
            dut_deploy = dut.dut_deploy

            class HostRunner:
                def __init__(self, **kwargs):
                    print("EVENT host_init")
                def docker_available(self):
                    return True
                def ensure_image(self):
                    print("EVENT host_image")
                def start(self):
                    print("EVENT host_start")
                def stop(self):
                    print("EVENT host_stop")

            class DutRunner:
                serial = suite.ADB_SERIAL
                def __init__(self, root):
                    print("EVENT dut_init")
                def acquire_board(self):
                    print("EVENT acquire_board")
                def release_board(self):
                    print("EVENT release_board")

            def power_cycle():
                assert os.environ["TORQ_DUT_SERIAL"] == suite.ADB_SERIAL
                print("EVENT power_cycle")
                if {power_cycle_fails!r}:
                    raise suite.DutPowerCycleError("preflight failed")
                print("EVENT adb_ready")

            @pytest.fixture(scope="session", autouse=True)
            def mock_hardware():
                with pytest.MonkeyPatch.context() as patch:
                    patch.setenv("TORQ_DUT_SERIAL", suite.ADB_SERIAL)
                    patch.setenv("TORQ_DUT_RUNTIME_WHEEL", "runtime.whl")
                    patch.setenv("TORQ_DUT_KO", "syna_npu.ko")
                    patch.setattr(suite, "power_cycle_dut", power_cycle)
                    patch.setattr(host, "HostSetupRunner", HostRunner)
                    patch.setattr(dut, "DutDeployRunner", DutRunner)
                    yield
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "test_pipeline.py").write_text(
        textwrap.dedent(
            """
            def test_host_first(host_setup):
                print("EVENT host_test_first")

            def test_host_second(host_setup):
                print("EVENT host_test_second")

            def test_dut(dut_deploy):
                print("EVENT dut_test")
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    workspace = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (workspace, env.get("PYTHONPATH", "")) if path
    )
    command = [sys.executable, "-m", "pytest", "-q", "-s", str(tmp_path)]
    if selection:
        command.extend(["-k", selection])
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    assert output.count("EVENT power_cycle") == 1, output
    if power_cycle_fails:
        assert result.returncode == pytest.ExitCode.TESTS_FAILED, output
        assert "Host-side DUT power cycle failed: preflight failed" in output
        assert "EVENT host_" not in output
        assert "EVENT dut_" not in output
        assert "EVENT acquire_board" not in output
    else:
        assert result.returncode == pytest.ExitCode.OK, output
        if selection != "dut":
            assert output.index("EVENT adb_ready") < output.index("EVENT host_init")
            assert output.count("EVENT host_start") == 1
        if selection != "host":
            assert output.index("EVENT adb_ready") < output.index("EVENT acquire_board")
        if not selection:
            assert output.index("EVENT host_test_second") < output.index("EVENT acquire_board")
