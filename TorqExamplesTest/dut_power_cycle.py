"""Power-cycle the DUT from the host and verify that it returns on ADB."""

from __future__ import annotations

import shlex
import subprocess
import sys
import time


ENV_CONFIG = "/home/jenkins/cases/env_config.sh"
NETBOOTER_IP = "10.70.58.211"
NETBOOTER_OUTLET = "2"
ADB_SERIAL = "sl2619npu01"
COMMAND_TIMEOUT_SECONDS = 180
POWER_OFF_WAIT_SECONDS = 5
POWER_ON_INITIAL_WAIT_SECONDS = 15
POWER_ON_TOTAL_WAIT_SECONDS = 45
ADB_POLL_INTERVAL_SECONDS = 5
ADB_CHECK_TIMEOUT_SECONDS = 5
ADB_POLL_ATTEMPTS = (
    (POWER_ON_TOTAL_WAIT_SECONDS - POWER_ON_INITIAL_WAIT_SECONDS)
    // ADB_POLL_INTERVAL_SECONDS
)


class DutPowerCycleError(RuntimeError):
    """Raised when the host cannot power-cycle or rediscover the DUT."""


def _run_shell_step(
    step: str,
    command: str,
    *,
    timeout_seconds: int = COMMAND_TIMEOUT_SECONDS,
    allow_failure: bool = False,
) -> str | None:
    """Run one host command and print its progress and output to the test log."""
    print(f"[dut_power_cycle] Starting: {step}", flush=True)
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise DutPowerCycleError("bash was not found on the host") from exc
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        if stdout:
            print(stdout, end="", flush=True)
        if stderr:
            print(stderr, file=sys.stderr, end="", flush=True)
        if allow_failure:
            print(
                f"[dut_power_cycle] {step} timed out; will retry.",
                flush=True,
            )
            return None
        raise DutPowerCycleError(
            f"{step} timed out after {timeout_seconds}s"
        ) from exc

    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="", flush=True)
    if result.returncode != 0:
        if allow_failure:
            print(
                f"[dut_power_cycle] {step} exited with code "
                f"{result.returncode}; will retry.",
                flush=True,
            )
            return None
        raise DutPowerCycleError(
            f"{step} failed with exit code {result.returncode}"
        )
    print(f"[dut_power_cycle] Completed: {step}", flush=True)
    return result.stdout


def _wait_with_progress(label: str, seconds: int, interval_seconds: int) -> None:
    """Wait for *seconds*, printing elapsed time at each interval."""
    elapsed = 0
    print(f"[dut_power_cycle] {label}: waiting {seconds}s.", flush=True)
    while elapsed < seconds:
        interval = min(interval_seconds, seconds - elapsed)
        time.sleep(interval)
        elapsed += interval
        print(
            f"[dut_power_cycle] {label}: {elapsed}/{seconds}s elapsed.",
            flush=True,
        )


def adb_device_is_online(output: str, serial: str = ADB_SERIAL) -> bool:
    """Return whether ``adb devices`` reports *serial* in the ``device`` state."""
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == serial and fields[1] == "device":
            return True
    return False


def power_cycle_dut() -> None:
    """Power the configured DUT off/on once and require it to reappear on ADB."""
    env_config = shlex.quote(ENV_CONFIG)
    netbooter = (
        f"netbooter -s {shlex.quote(NETBOOTER_IP)} "
        f"-o {shlex.quote(NETBOOTER_OUTLET)}"
    )
    print(
        f"[dut_power_cycle] Target: {NETBOOTER_IP} outlet {NETBOOTER_OUTLET}; "
        f"ADB serial: {ADB_SERIAL}.",
        flush=True,
    )
    _run_shell_step(
        "powering DUT OFF",
        f"set -e; source {env_config}; {netbooter} -c OFF",
    )
    _wait_with_progress("Power-off wait", POWER_OFF_WAIT_SECONDS, 1)
    _run_shell_step(
        "powering DUT ON",
        f"set -e; source {env_config}; {netbooter} -c ON",
    )
    _wait_with_progress(
        "Initial boot wait",
        POWER_ON_INITIAL_WAIT_SECONDS,
        ADB_POLL_INTERVAL_SECONDS,
    )
    adb_command = f"set -e; source {env_config}; adb devices"
    for attempt in range(ADB_POLL_ATTEMPTS):
        _wait_with_progress(
            "ADB discovery wait",
            ADB_POLL_INTERVAL_SECONDS,
            ADB_POLL_INTERVAL_SECONDS,
        )
        elapsed = POWER_ON_INITIAL_WAIT_SECONDS + (attempt + 1) * ADB_POLL_INTERVAL_SECONDS
        adb_output = _run_shell_step(
            f"checking ADB devices at {elapsed}s "
            f"(attempt {attempt + 1}/{ADB_POLL_ATTEMPTS})",
            adb_command,
            timeout_seconds=ADB_CHECK_TIMEOUT_SECONDS,
            allow_failure=True,
        )
        if adb_output is not None and adb_device_is_online(adb_output):
            print(
                f"[dut_power_cycle] ADB device {ADB_SERIAL} is online "
                f"after {elapsed}s.",
                flush=True,
            )
            return

        print(
            f"[dut_power_cycle] ADB device {ADB_SERIAL} is not ready at "
            f"{elapsed}s.",
            flush=True,
        )
        if attempt + 1 < ADB_POLL_ATTEMPTS:
            _wait_with_progress(
                "ADB retry wait",
                ADB_POLL_INTERVAL_SECONDS,
                ADB_POLL_INTERVAL_SECONDS,
            )

    message = (
        f"Could not detect ADB device {ADB_SERIAL} within "
        f"{POWER_ON_TOTAL_WAIT_SECONDS}s after power-on; "
        "manual intervention is required to check the environment. "
        "需要人工介入检查环境。"
    )
    print(f"[dut_power_cycle] {message}", flush=True)
    raise DutPowerCycleError(message)


if __name__ == "__main__":
    power_cycle_dut()
