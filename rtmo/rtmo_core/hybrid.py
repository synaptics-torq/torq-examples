r"""Hybrid 3-vmfb runner for the RTMO demo.

Chains the three NSS-only vmfbs — int8 conv backbone, bf16 AIFI transformer, int8
detection head — behind the same ``.infer([input]) -> outputs`` /
``.infer_time_ms`` interface as ``torq.runtime.VMFBInferenceRunner``, so rtmo.py
treats it like any other quantized model.

Dataflow (all requant/transpose happens host-side between the vmfbs, exactly as
it would on-device):

    image(int8) -> [backbone] -> P3,P4,P5 (int8)
                                   P5 -{dequant}-> bf16 -> [transformer] -> P5' (bf16, NCHW)
    P3,P4 -{requant to head scales}-\                          |
                                     P5' -{NHWC, requant}------ +-> [head] -> 8 heads (int8)
"""

import time

import numpy as np
import ml_dtypes

from torq.runtime import VMFBInferenceRunner


class HybridRunner:
    def __init__(self, backbone_vmfb, transformer_vmfb, head_vmfb, seams,
                 *, device_uri="torq", function="main", torq_hw_type=None):
        # torq_hw_type is only accepted by newer (host) VMFBInferenceRunner; the
        # board build omits it, so only pass it through when explicitly set.
        common = dict(device_uri=device_uri, function=function)
        if torq_hw_type is not None:
            common["torq_hw_type"] = torq_hw_type
        self._bb = VMFBInferenceRunner(backbone_vmfb, **common)
        self._tf = VMFBInferenceRunner(transformer_vmfb, **common)
        self._hd = VMFBInferenceRunner(head_vmfb, **common)
        self._s = seams
        self.infer_time_ms = 0.0
        # per-part timings for reporting
        self.part_ms = {}
        # Seam fusion: the P3/P4 skip connections go int8(backbone) -> int8(head).
        # When the backbone-output scale/zp equal the head-input scale/zp (they do,
        # to ~1e-8 — same PTQ calibration), the int8 values are already correct for
        # the head, so the dequant->requant round-trip is a no-op. Pass through and
        # skip it (saves ~5 ms/frame of host numpy on the two largest skip tensors).
        self._p3_passthrough = self._scales_match(seams["bb_p3"], seams["hd_p3"])
        self._p4_passthrough = self._scales_match(seams["bb_p4"], seams["hd_p4"])

    @staticmethod
    def _scales_match(a, b, rtol=1e-4):
        return abs(a[0] - b[0]) <= rtol * abs(b[0]) and a[1] == b[1]

    @staticmethod
    def _q(x, sz, dt=np.int8):
        scale, zp = sz
        info = np.iinfo(dt)
        return np.clip(np.rint(x / scale) + zp, info.min, info.max).astype(dt)

    @staticmethod
    def _dq(x, sz):
        scale, zp = sz
        return (x.astype(np.float32) - zp) * scale

    @staticmethod
    def _by_shape(arrs, shape):
        return next(a for a in arrs if tuple(np.asarray(a).shape) == shape)

    def infer(self, inputs):
        s = self._s
        x = inputs[0]  # int8 NHWC image
        t0 = time.perf_counter()

        bb_out = self._bb.infer([x])
        t1 = time.perf_counter()
        p3 = self._by_shape(bb_out, s["p3_shape"])
        p4 = self._by_shape(bb_out, s["p4_shape"])
        p5 = self._by_shape(bb_out, s["p5_shape"])

        # backbone P5 -> bf16 -> transformer
        p5_bf16 = self._dq(p5, s["bb_p5"]).astype(ml_dtypes.bfloat16)
        p5t = self._tf.infer([p5_bf16])[0]                    # bf16 NCHW (1,256,10,10)
        t2 = time.perf_counter()
        p5t = np.transpose(np.asarray(p5t).astype(np.float32), (0, 2, 3, 1))  # -> NHWC

        # P5 always requants (transformer changed it); P3/P4 pass through when
        # backbone and head scales match (fused seam).
        p5_h = self._q(p5t, s["hd_p5"])
        p3_h = p3 if self._p3_passthrough else self._q(self._dq(p3, s["bb_p3"]), s["hd_p3"])
        p4_h = p4 if self._p4_passthrough else self._q(self._dq(p4, s["bb_p4"]), s["hd_p4"])

        hd_out = self._hd.infer([p3_h, p4_h, p5_h])           # order: P3, P4, P5
        t3 = time.perf_counter()

        self.part_ms = {"backbone": (t1 - t0) * 1e3, "transformer": (t2 - t1) * 1e3,
                        "head": (t3 - t2) * 1e3}
        self.infer_time_ms = (t3 - t0) * 1e3
        return hd_out
