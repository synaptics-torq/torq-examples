# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations


def build_runtime_flags(tda, extra_runtime_flags=None):
    return [f"--torq_device_allocator={tda}"] + (extra_runtime_flags or [])