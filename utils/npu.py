# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

from pathlib import Path
import subprocess


_NPU_DEVFREQ_PATH = Path("/sys/class/devfreq/f7600000.synpu")
_NPU_GOVERNOR_PATH = _NPU_DEVFREQ_PATH / "governor"
_NPU_SET_FREQ_PATH = _NPU_DEVFREQ_PATH / "userspace" / "set_freq"
_NPU_MAX_FREQ_PATH = _NPU_DEVFREQ_PATH / "max_freq"
_NPU_MIN_FREQ_PATH = _NPU_DEVFREQ_PATH / "min_freq"
_NPU_USERSPACE_GOVERNOR = "userspace"


def configure_npu_userspace_frequency(level: str = "max") -> tuple[bool, str]:
    if level not in {"max", "min"}:
        return False, "NPU userspace frequency level must be 'max' or 'min'"

    if not _NPU_GOVERNOR_PATH.exists():
        return False, "NPU devfreq controls not available"

    try:
        current_governor = _NPU_GOVERNOR_PATH.read_text().strip()
        if current_governor != _NPU_USERSPACE_GOVERNOR:
            _NPU_GOVERNOR_PATH.write_text(_NPU_USERSPACE_GOVERNOR)

        if not _NPU_SET_FREQ_PATH.exists():
            return False, "NPU userspace frequency control not available"

        freq_bound_path = _NPU_MAX_FREQ_PATH if level == "max" else _NPU_MIN_FREQ_PATH
        if not freq_bound_path.exists():
            return False, f"NPU {level}_freq control not available"

        target_freq_hz = freq_bound_path.read_text().strip()
        if not target_freq_hz:
            return False, f"NPU {level}_freq is empty"

        _NPU_SET_FREQ_PATH.write_text(target_freq_hz)
    except Exception as exc:
        return False, f"NPU userspace {level} frequency setup failed: {exc}"

    return True, f"NPU userspace {level} frequency applied"


def enable_npu_clock() -> tuple[bool, str]:
    try:
        subprocess.run(
            ["devmem", "0xf7e104b0", "32", "0x216"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return False, f"NPU clock enable failed: {exc}"

    return True, "NPU clock enabled"
