# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared Torq runtime utilities for all inference scripts.

``RuntimeConfig`` carries the runtime options for one session (built once
from the CLI args), ``Model`` is the only place a
``torq.runtime.VMFBInferenceRunner`` is constructed, and ``npu_clock()``
holds the NPU at its max frequency for the duration of inference.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
from dataclasses import dataclass

import ml_dtypes
import numpy as np

from utils.log import add_logging_args
from utils.metrics import add_metrics_args

try:
    from torq.runtime import VMFBInferenceRunner
except Exception:  # torq.runtime needs the NPU runtime; keep this module importable on the host
    VMFBInferenceRunner = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime options for one inference session.

    Built once from the CLI args (see ``add_common_inference_args``) and
    passed to every model load in the session, so one run uses one allocator
    and one I/O mode.
    """

    device: str = "torq"
    tda: str = "dmabuf"
    device_io: bool = True
    threads: int | None = None
    extra_flags: tuple[str, ...] = ()

    def __post_init__(self):
        # The dmabuf allocator is incompatible with host NumPy I/O (known
        # runtime bug); it only works with device arrays. Do not remove this
        # forcing.
        if self.tda == "dmabuf" and not self.device_io:
            logger.warning("--tda dmabuf requires device I/O; forcing --device-io on")
            object.__setattr__(self, "device_io", True)

    @classmethod
    def from_args(cls, args) -> "RuntimeConfig":
        return cls(
            device=getattr(args, "device", "torq"),
            tda=getattr(args, "tda", "dmabuf"),
            device_io=getattr(args, "device_io", True),
            threads=getattr(args, "threads", None),
            extra_flags=tuple(getattr(args, "runtime_flags", None) or ()),
        )

    @property
    def flags(self) -> list[str]:
        return [f"--torq_device_allocator={self.tda}", *self.extra_flags]


def to_host(x):
    """Copy a device-backed value to the host; host values pass through unchanged."""
    return x.to_host() if hasattr(x, "to_host") else x


def load_lut(path) -> np.ndarray:
    """Load a token LUT (embeddings or token-id remap) from a ``.npy`` file,
    memory-mapped on the host.

    Un-typed bfloat16 exports store raw 2-byte rows (dtype ``V2``); they are
    viewed as ``ml_dtypes.bfloat16`` so lookups stay zero-copy.
    """
    arr = np.load(path, mmap_mode="r")
    if arr.dtype == np.dtype("V2"):
        arr = arr.view(ml_dtypes.bfloat16)
    return arr


class Model:
    """A loaded .vmfb, wrapped by composition around a ``VMFBInferenceRunner``.

    *runtime* (a ``RuntimeConfig``) sets the allocator, device, I/O mode and
    threads for the underlying runner. With device I/O on, :meth:`infer`
    stages host inputs on the device and returns the raw outputs;
    :meth:`run` copies them home.
    """

    def __init__(self, path, runtime: RuntimeConfig, *, function="main", device_outputs=False):
        if VMFBInferenceRunner is None:
            raise RuntimeError(
                "torq.runtime is not importable in this environment; "
                "models load on the board (tests stub the runtime)"
            )
        self._runtime = runtime
        self.runner = VMFBInferenceRunner(
            path,
            function=function,
            device_uri=runtime.device,
            n_threads=runtime.threads,
            runtime_flags=runtime.flags,
            device_outputs=device_outputs or runtime.device_io,
        )

    # Read-only pass-throughs for what the scripts already read off the runner.
    @property
    def path(self):
        return self.runner.model_path

    @property
    def inputs_info(self):
        return self.runner.inputs_info

    @property
    def outputs_info(self):
        return self.runner.outputs_info

    @property
    def device(self):
        return self.runner.device

    @property
    def infer_time_ms(self):
        return self.runner.infer_time_ms

    def allocate_device_array(self, array):
        return self.runner.allocate_device_array(array)

    def to_device(self, array):
        """Stage a host array on the device when device I/O is on.

        Device arrays (and everything in host I/O mode) pass through
        unchanged, so pre-staged inputs are never uploaded twice.
        """
        if self._runtime.device_io and isinstance(array, np.ndarray):
            return self.runner.allocate_device_array(array)
        return array

    def infer(self, inputs: list) -> list:
        """One pass over *inputs*; returns the raw outputs."""
        return self.runner.infer([self.to_device(x) for x in inputs])

    def run(self, *inputs) -> list:
        """``infer`` plus ``to_host`` on every output — the common case."""
        return [to_host(out) for out in self.infer(list(inputs))]


def run_profile_if_requested(args, runtime: RuntimeConfig):
    """Run the profiler and return True if --profile was set (caller should return after this)."""
    if not args.profile:
        return False
    from torq.runtime import profile_vmfb_resources

    results = profile_vmfb_resources(
        args.model, device=runtime.device, n_iters=1, n_threads=None,
        function="main", runtime_flags=runtime.flags,
        device_io=runtime.device_io, do_warmup=True,
    )
    print(results.summary())
    return True


@contextlib.contextmanager
def npu_clock():
    """Hold the NPU at its max userspace frequency while the block runs,
    restoring the minimum when done (including on error)."""
    from utils.npu import configure_npu_userspace_frequency

    ok, message = configure_npu_userspace_frequency("max")
    print(f"[NPU] {message}")
    try:
        yield
    finally:
        ok, message = configure_npu_userspace_frequency("min")
        print(f"[NPU] {message}")


# ── Shared inference CLI ─────────────────────────────────────────────────────


_NO_REFRESH_HELP: str = (
    "Skip the Hugging Face check for updated models (offline/airgapped runs)"
)
_RUNTIME_FLAGS_HELP: str = (
    "[Advanced] Extra flags for the Torq runtime. "
    "Must be specified last; all remaining arguments are forwarded."
)


def add_common_inference_args(
    parser: argparse.ArgumentParser,
    *,
    with_threads: bool = True,
    with_no_refresh: bool = True,
    with_allocator: bool = True,
    tda_default: str = "dmabuf",
    device_io_default: bool = True,
    device_io_help: str | None = None,
) -> None:
    """Flags shared by the demos' inference scripts: the runtime group a
    ``RuntimeConfig`` is built from, plus the logging and metrics args.

    Adds ``-j/--threads`` (unless *with_threads*), ``--no-refresh`` (unless
    *with_no_refresh*), a "runtime" group with ``--tda`` (default
    *tda_default*), ``--device-io`` (default *device_io_default*, unless
    *with_allocator*) and ``--runtime-flags``, the logging args, and
    ``--metrics-out``.
    """
    if with_threads:
        parser.add_argument(
            "-j", "--threads", type=int,
            help="Number of cores to use for CPU execution (default: all)",
        )
    if with_no_refresh:
        parser.add_argument(
            "--no-refresh", action="store_true", default=False,
            help=_NO_REFRESH_HELP,
        )
    runtime_group = parser.add_argument_group("runtime")
    if with_allocator:
        runtime_group.add_argument(
            "--tda",
            type=str,
            choices=["cpu", "dmabuf"],
            default=tda_default,
            help="Allocator backing Torq device buffers (default: %(default)s)",
        )
        runtime_group.add_argument(
            "--device-io",
            action=argparse.BooleanOptionalAction,
            default=device_io_default,
            help=device_io_help or (
                "Preallocate inputs and keep cache outputs as device arrays "
                f"(default: {'enabled' if device_io_default else 'disabled'})"
            ),
        )
    runtime_group.add_argument(
        "--runtime-flags",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="FLAG",
        help=_RUNTIME_FLAGS_HELP,
    )
    add_logging_args(parser)
    add_metrics_args(parser)
