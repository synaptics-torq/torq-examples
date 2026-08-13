# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Detection/recognition backends for the PP-OCR demo.

Each backend exposes ``run(x)`` taking an ``[N,3,H,W]`` float32 batch and returning float32
predictions, so the pipeline is agnostic to whether a stage runs on the NPU or ONNX Runtime.
"""

from __future__ import annotations

import numpy as np

from utils.inference import SimpleVMFBInferenceRunner

_RESERVED_FUNCTIONS = ("__init", "__reset")


def entry_function(vmfb_path) -> str:
    """Exported function name of a vmfb; PP-OCR keeps the PaddlePaddle graph name, not "main"."""
    import iree.runtime as rt

    module = rt.VmModule.mmap(rt.VmInstance(), str(vmfb_path))
    names = [n for n in module.function_names if n not in _RESERVED_FUNCTIONS]
    if not names:
        raise ValueError(f"No exported function found in '{vmfb_path}'")
    return names[0]


def _to_float32(out) -> np.ndarray:
    """Runner output to float32; bf16 arrives as an opaque 2-byte dtype."""
    import ml_dtypes

    arr = np.asarray(out)
    if arr.dtype.kind == "V":
        arr = arr.view(ml_dtypes.bfloat16)
    return arr.astype(np.float32)


class NPUBackend:
    """A bf16 Torq vmfb driven through the shared inference runner."""

    def __init__(self, vmfb_path, *, device_uri="torq", runtime_flags=None, device_io=False):
        import ml_dtypes

        self._bf16 = ml_dtypes.bfloat16
        fn = entry_function(vmfb_path)
        self.runner = SimpleVMFBInferenceRunner(vmfb_path, device_uri=device_uri, function=fn, runtime_flags=runtime_flags, device_io=device_io)
        self.num_classes = None  # recognizers learn this from the first output

    @property
    def infer_time_ms(self):
        return self.runner.infer_time_ms

    def run(self, batch: np.ndarray) -> np.ndarray:
        """The vmfbs are compiled for one static shape, so feed one sample at a time."""
        outs = [_to_float32(self.runner.infer(batch[i:i + 1].astype(self._bf16))) for i in range(batch.shape[0])]
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
