# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Per-run performance metrics and the JSON format defined in CONTRIBUTING.md.

A :class:`MetricsRun` around a demo's inference block records named
measurements (aggregated as means) plus system counters sampled by
:class:`SystemSampler` (CPU/NPU load, peak memory). At exit it writes
``{"metrics": [...], "measurements": {...}}`` to the ``--metrics-out`` file,
if one was given, even when the block is interrupted.
"""

import argparse
import json
import os
import resource
import tempfile
from time import perf_counter_ns

__all__ = [
    "METRICS",
    "SystemSampler",
    "MetricsRun",
    "add_metrics_args",
]

PROC_STAT_PATH = "/proc/stat"
NPU_INFERENCE_TIME_PATH = "/sys/class/misc/torq/statistics/torq_inference_time"

# Metric name -> (description, unit).
METRICS: dict[str, tuple[str, str]] = {
    "time_to_first_token": ("Time taken to generate the first token", "ns"),
    "tokens_per_second": ("Generated answer tokens divided by the time taken to generate them", "hz"),
    "latency": ("Latency", "ns"),
    "frames_per_second": ("Frames per second", "hz"),
    "real_time_factor": ("Real-time factor (RTF*100; 100 = real-time speed)", "percent"),
    "cpu_load": ("CPU load during demo execution", "percent"),
    "npu_load": ("NPU load during demo execution", "percent"),
    "peak_memory": ("Peak memory usage", "bytes"),
}


def _metric_info(name: str) -> tuple[str, str]:
    """The (description, unit) of *name*.

    ``latency_<suffix>`` names (per-modality latency) reuse latency's unit
    with a description of ``Latency (<suffix>)``.

    Raises:
        ValueError: if *name* is not a known metric.
    """
    if name in METRICS:
        return METRICS[name]
    if name.startswith("latency_"):
        return f"Latency ({name[len('latency_'):]})", METRICS["latency"][1]
    raise ValueError(f"Unknown metric name: {name!r}")


def _clamp_percent(value: float) -> float:
    return min(100.0, max(0.0, value))


def _read_cpu_jiffies(path: str) -> tuple[int, int]:
    """The (idle, total) jiffies of the aggregate ``cpu`` line of /proc/stat."""
    with open(path) as f:
        for line in f:
            if line.startswith("cpu "):
                nums = [int(x) for x in line.split()[1:]]
                # idle and iowait are the 4th and 5th numbers.
                return nums[3] + nums[4], sum(nums)
    raise ValueError(f"No aggregate 'cpu' line in {path}")


def _read_npu_time_us(path: str) -> int | None:
    """The cumulative NPU active time (us, sl26xx), or None if unavailable."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


class SystemSampler:
    """Samples system counters (CPU load, NPU load, peak memory) over a run."""

    def start(self) -> None:
        self._t0 = perf_counter_ns()
        self._cpu0 = _read_cpu_jiffies(PROC_STAT_PATH)
        self._npu0 = _read_npu_time_us(NPU_INFERENCE_TIME_PATH)

    def stop(self) -> dict[str, float]:
        wall_ns = perf_counter_ns() - self._t0

        idle, total = _read_cpu_jiffies(PROC_STAT_PATH)
        d_idle = idle - self._cpu0[0]
        d_total = total - self._cpu0[1]
        cpu_load = (
            _clamp_percent(100.0 * (1.0 - d_idle / d_total)) if d_total > 0 else 0.0
        )

        values: dict[str, float] = {"cpu_load": cpu_load}
        npu_time = _read_npu_time_us(NPU_INFERENCE_TIME_PATH)
        if self._npu0 is not None and npu_time is not None and wall_ns > 0:
            # NPU active us over the wall ns -> duty cycle percent.
            values["npu_load"] = _clamp_percent(100.0 * (npu_time - self._npu0) * 1000 / wall_ns)

        values["peak_memory"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return values


class MetricsRun:
    """Collects the measurements of a run and writes the metrics JSON at exit.

    Args:
        path: Output file (CONTRIBUTING.md format), or None to only collect.
    """

    def __init__(self, path: str | None = None):
        self._path = path
        self._sampler = SystemSampler()
        self._values: dict[str, list[float]] = {}

    def __enter__(self) -> "MetricsRun":
        self._sampler.start()
        return self

    def add(self, name: str, value: float) -> None:
        """Record one measurement of *name*; repeated names are averaged."""
        _metric_info(name)
        self._values.setdefault(name, []).append(value)

    def __exit__(self, exc_type, exc, tb) -> bool:
        measurements = {name: sum(vals) / len(vals) for name, vals in self._values.items()}
        measurements.update(self._sampler.stop())
        if self._path is not None:
            self._write(measurements)
        return False

    def _write(self, measurements: dict[str, float]) -> None:
        entries = []
        for name in measurements:
            description, unit = _metric_info(name)
            entries.append({"name": name, "description": description, "unit": unit})
        doc = {"metrics": entries, "measurements": measurements}

        path = os.fspath(self._path)
        fd, tmp = tempfile.mkstemp(prefix=".metrics-", suffix=".tmp", dir=os.path.dirname(path) or ".")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(doc, f, indent=2)
                f.write("\n")
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def add_metrics_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metrics-out",
        metavar="FILE",
        default=None,
        help="Write performance metrics as JSON to FILE at exit (CONTRIBUTING.md format)",
    )
