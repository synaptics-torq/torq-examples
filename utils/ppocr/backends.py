# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Detection/recognition backends for the PP-OCR demo.

Each backend exposes a single ``run(x)`` method taking an ``[N,3,H,W]`` float32
batch and returning float32 predictions, so the pipeline is agnostic to whether
a stage executes on the Torq NPU or on ONNX Runtime.

The NPU backends wrap :class:`~utils.inference.SimpleVMFBInferenceRunner`. The
vmfbs are compiled for bf16 IO and for one static input shape, so the backend
casts to bf16 and feeds one sample at a time.
"""

from __future__ import annotations

import numpy as np

from utils.inference import SimpleVMFBInferenceRunner

# PP-OCR vmfbs keep the exported name from the PaddlePaddle graph rather than
# the usual "main"; discovered per-file so a renamed export still works.
_RESERVED_FUNCTIONS = ("__init", "__reset")


def entry_function(vmfb_path) -> str:
    """Return the exported function name of a vmfb (no device needed)."""
    import iree.runtime as rt

    module = rt.VmModule.mmap(rt.VmInstance(), str(vmfb_path))
    names = [n for n in module.function_names if n not in _RESERVED_FUNCTIONS]
    if not names:
        raise ValueError(f"No exported function found in '{vmfb_path}'")
    return names[0]


def _to_float32(out) -> np.ndarray:
    """Normalize a runner output to float32, handling bf16 returned as void."""
    import ml_dtypes

    arr = np.asarray(out)
    if arr.dtype.kind == "V":  # bf16 arrives as an opaque 2-byte dtype
        arr = arr.view(ml_dtypes.bfloat16)
    return arr.astype(np.float32)


class NPUBackend:
    """A bf16 Torq vmfb driven through the shared inference runner."""

    def __init__(self, vmfb_path, *, device_uri="torq", runtime_flags=None, device_io=False):
        import ml_dtypes

        self._bf16 = ml_dtypes.bfloat16
        self.runner = SimpleVMFBInferenceRunner(
            vmfb_path,
            device_uri=device_uri,
            function=entry_function(vmfb_path),
            runtime_flags=runtime_flags,
            device_io=device_io,
        )
        self.num_classes = None  # recognizers learn this from the first output

    @property
    def infer_time_ms(self):
        return self.runner.infer_time_ms

    def run(self, batch: np.ndarray) -> np.ndarray:
        """Run ``[N,3,H,W]`` float32 through the vmfb one sample at a time."""
        outs = [
            _to_float32(self.runner.infer(batch[i:i + 1].astype(self._bf16)))
            for i in range(batch.shape[0])
        ]
        out = np.concatenate(outs, 0)
        self.num_classes = out.shape[-1]
        return out


class ORTBackend:
    """An fp32 ONNX model on ONNX Runtime — the CPU reference for comparisons."""

    def __init__(self, onnx_path):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_outputs()[0].shape
        self.num_classes = shape[-1] if isinstance(shape[-1], int) else None

    def run(self, batch: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: batch.astype(np.float32)})[0]
