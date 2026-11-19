# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared audio helpers for the speech demos."""

import numpy as np


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Linearly resample mono *audio* from *orig_sr* to *target_sr* Hz."""
    if orig_sr == target_sr:
        return audio
    num_target_samples = int(len(audio) / orig_sr * target_sr)
    indices = np.linspace(0, len(audio) - 1, num_target_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
