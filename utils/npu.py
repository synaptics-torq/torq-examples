# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import subprocess


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
