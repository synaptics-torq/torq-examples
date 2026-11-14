#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""
Power monitor API for hwmon sysfs channels.

Usage:
    mon = PowerMonitor("/sys/class/hwmon/hwmon2")
    print(mon.channels)              # list of discovered label names
    r = mon.read("VCORE")            # PowerReading(label, voltage_mv, current_ma, power_mw)
    all = mon.read_all()             # {label: PowerReading, ...}
    mon.close()                      # or: use as context manager

    # Continuous background sampling with named sessions:
    sampler = PowerSampler(mon, interval_s=0.1, csv_file="power.csv")
    sampler.start("encode")          # session name required
    # ... run encoder ...
    result = sampler.stop()          # returns SessionResult
    print(result.energy_mj)          # {channel: mJ}
    print(result.total_energy_mj)    # sum across all channels

    # Energy per inference, with idle (baseline) power subtracted:
    sampler.measure_baseline(5.0)    # idle power, before inference starts
    for prompt in prompts:
        with sampler.measure("translate") as m:
            out = model(prompt)
            m.inference_time_s = out.latency_s   # optional, else wall-clock
    sampler.print_summary()          # per-session table + totals

Units: power is mW, energy is mJ (mW x s == mJ).
"""

from __future__ import annotations

import contextlib
import csv
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

_logger = logging.getLogger(__name__)


DEFAULT_HWMON_PATH = Path("/sys/class/hwmon/hwmon2")

_CHANNEL_INDICES = (1, 2, 3)


@dataclass(frozen=True)
class PowerReading:
    label: str
    voltage_mv: float
    current_ma: float

    @property
    def power_mw(self) -> float:
        return self.voltage_mv * self.current_ma / 1000


@dataclass(frozen=True)
class SessionResult:
    """Results from a completed power measurement session."""

    session: str
    duration_s: float
    n_samples: int
    energy_mj: dict[str, float]  # {channel: millijoules}, gross, over the whole session
    avg_power_mw: dict[str, float] = field(default_factory=dict)  # mean of samples
    baseline_mw: dict[str, float] = field(default_factory=dict)  # idle power, if measured
    inference_time_s: float | None = None  # caller-supplied; defaults to duration_s

    @property
    def total_energy_mj(self) -> float:
        """Total energy summed across all measured channels, in millijoules."""
        return sum(self.energy_mj.values())

    @property
    def total_avg_power_mw(self) -> float:
        return sum(self.avg_power_mw.values())

    @property
    def has_baseline(self) -> bool:
        return bool(self.baseline_mw)

    @property
    def active_time_s(self) -> float:
        """Time the energy is attributed to: inference time if given, else wall-clock."""
        return self.inference_time_s if self.inference_time_s is not None else self.duration_s

    @property
    def net_power_mw(self) -> dict[str, float]:
        """Average power above baseline per channel (clamped at 0). Equals
        avg_power_mw when no baseline was measured."""
        return {
            ch: max(0.0, p - self.baseline_mw.get(ch, 0.0))
            for ch, p in self.avg_power_mw.items()
        }

    @property
    def total_net_power_mw(self) -> float:
        return sum(self.net_power_mw.values())

    @property
    def net_energy_mj(self) -> float:
        """Energy attributable to the inference: (avg - baseline) power x inference time."""
        return self.total_net_power_mw * self.active_time_s


class _Measurement:
    """Handle yielded by PowerSampler.measure(); set inference_time_s inside the block."""

    def __init__(self) -> None:
        self.inference_time_s: float | None = None
        self.result: SessionResult | None = None


class _Channel:
    __slots__ = ("label", "_volt_fh", "_curr_fh")

    def __init__(self, label: str, volt_fh: IO | None, curr_fh: IO | None):
        self.label = label
        self._volt_fh = volt_fh
        self._curr_fh = curr_fh

    def read(self, n_avg: int = 3) -> PowerReading:
        v_sum = 0.0
        c_sum = 0.0
        for _ in range(n_avg):
            if self._volt_fh is not None:
                self._volt_fh.seek(0)
                v_sum += float(self._volt_fh.read())
            if self._curr_fh is not None:
                self._curr_fh.seek(0)
                c_sum += float(self._curr_fh.read())
        return PowerReading(label=self.label, voltage_mv=v_sum / n_avg, current_ma=c_sum / n_avg)

    def close(self):
        if self._volt_fh is not None:
            self._volt_fh.close()
        if self._curr_fh is not None:
            self._curr_fh.close()


class PowerMonitor:
    """
    Reads voltage and current from an hwmon sysfs directory.

    Discovers available channels on init by reading in*_label files and
    keeps file descriptors open for fast repeated reads.

    If the hwmon directory does not exist (e.g. running on a board without the
    power monitor), construction still succeeds: a single warning is logged,
    ``available`` is False and there are no channels, so callers can carry on
    without power measurements.
    """

    def __init__(self, path: str | Path = DEFAULT_HWMON_PATH):
        self._path = Path(path)
        self._channels: dict[str, _Channel] = {}
        self._available = self._path.is_dir()
        if not self._available:
            _logger.warning(
                "[PWR] hwmon path %s not found — power measurement disabled", self._path
            )
            return
        self._init_channels()

    def _open_if_exists(self, p: Path) -> IO | None:
        try:
            return open(p, "r")
        except OSError:
            return None

    def _init_channels(self):
        for idx in _CHANNEL_INDICES:
            label_path = self._path / f"in{idx}_label"
            if not label_path.exists():
                continue
            try:
                label = label_path.read_text().strip()
            except OSError:
                continue
            volt_fh = self._open_if_exists(self._path / f"in{idx}_input")
            curr_fh = self._open_if_exists(self._path / f"curr{idx}_input")
            self._channels[label] = _Channel(label, volt_fh, curr_fh)

    @property
    def available(self) -> bool:
        """True if the hwmon path exists and at least one channel was found."""
        return self._available and bool(self._channels)

    @property
    def channels(self) -> list[str]:
        """Labels of all discovered measurement channels."""
        return list(self._channels.keys())

    def read(self, label: str) -> PowerReading:
        """Return a PowerReading for the given label. Raises KeyError if not found."""
        return self._channels[label].read()

    def read_all(self) -> dict[str, PowerReading]:
        """Read all channels. Returns {label: PowerReading}."""
        return {label: ch.read() for label, ch in self._channels.items()}

    def close(self):
        for ch in self._channels.values():
            ch.close()
        self._channels.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class PowerSampler:
    """
    Samples PowerMonitor channels at a fixed interval in a daemon thread.

    Each start()/stop() pair is a named session. Energy consumed during the
    session is accumulated from power × sample interval and returned by stop().

    If csv_file is provided, each sample is written as one row with columns:
        session, timestamp, <ch>_mv, <ch>_ma, <ch>_mw, ...
    one column group per channel, all channels on the same row. The file is
    opened on the first start() call and kept open across sessions; close()
    flushes and closes it.

    Log lines are emitted at INFO level with a [PWR] prefix so they interleave
    with application logs in a single chronological record.

    Energy per inference: call measure_baseline() once while the system is idle
    (before inference starts), then wrap each inference in start()/stop() or
    ``with measure(...)``. Each result's net_energy_mj is
    (avg power - baseline power) x inference time. Results accumulate and
    print_summary() reports them at the end of the run. If the monitor is not
    available, all of this is a silent no-op (stop() returns None, the summary
    is empty), so callers need no special handling.
    """

    def __init__(
        self,
        monitor: PowerMonitor,
        *,
        interval_s: float = 0.1,
        label: str = "",
        csv_file: str | Path | None = None,
        log_samples: bool = True,
    ):
        self._monitor = monitor
        self._interval = interval_s
        self._label = label
        self._log_samples = log_samples
        self._csv_path = Path(csv_file) if csv_file else None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._session: str = ""
        self._start_time: float = 0.0
        self._energy_acc: dict[str, float] = {}  # channel -> accumulated mW·s
        self._n_samples: int = 0
        self._csv_fh = None
        self._csv_writer = None
        self._csv_channels: list[str] = []
        self._power_sum: dict[str, float] = {}  # channel -> sum of sampled mW
        self._is_baseline = False
        self._baseline_mw: dict[str, float] = {}
        self._results: list[SessionResult] = []

    @property
    def baseline_mw(self) -> dict[str, float]:
        """Idle power per channel from measure_baseline() ({} if not measured)."""
        return dict(self._baseline_mw)

    @property
    def results(self) -> list[SessionResult]:
        """Completed (non-baseline) sessions, in order."""
        return list(self._results)

    def clear_results(self) -> None:
        self._results.clear()

    def start(self, session: str) -> None:
        """Begin a named power measurement session."""
        if not session:
            raise ValueError("session name must be non-empty")
        if self._thread is not None and self._thread.is_alive():
            return
        if not self._monitor.available:
            # Already warned when the monitor was created; stay a silent no-op.
            return
        channels = [self._label] if self._label else self._monitor.channels
        if self._label and self._label not in self._monitor.channels:
            _logger.warning("[PWR] Label '%s' not found — power sampling disabled", self._label)
            return
        if not channels:
            _logger.warning("[PWR] No channels found — power sampling disabled")
            return
        self._session = session
        self._energy_acc = {ch: 0.0 for ch in channels}
        self._power_sum = {ch: 0.0 for ch in channels}
        self._n_samples = 0
        self._start_time = time.perf_counter()
        self._open_csv(channels)
        _logger.info(
            "[PWR] Session '%s': sampling %s every %.0fms",
            session, channels, self._interval * 1000,
        )
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pwr-sampler", daemon=True)
        self._thread.start()

    def stop(self, inference_time_s: float | None = None) -> SessionResult | None:
        """End the current session and return its results.

        inference_time_s: time the inference itself took (e.g. reported by the
        model). Energy is computed over this instead of the session wall-clock
        time. Non-baseline results are also kept for print_summary().
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if not self._session:
            return None
        if self._n_samples == 0:
            # Session shorter than one sampling interval: take one sample now so
            # the average power is defined.
            self._sample_once()
        duration = time.perf_counter() - self._start_time
        n = self._n_samples
        energy_mj = {ch: mws / 1000.0 for ch, mws in self._energy_acc.items()}
        avg_power = {ch: (total / n if n else 0.0) for ch, total in self._power_sum.items()}
        is_baseline = self._is_baseline
        result = SessionResult(
            session=self._session,
            duration_s=duration,
            n_samples=n,
            energy_mj=energy_mj,
            avg_power_mw=avg_power,
            baseline_mw={} if is_baseline else dict(self._baseline_mw),
            inference_time_s=None if is_baseline else inference_time_s,
        )
        if is_baseline:
            _logger.info(
                "[PWR] Session '%s' complete: %.2fs  %d samples  avg=%.0fmW  per-channel=%s",
                self._session, duration, n, result.total_avg_power_mw,
                {k: f"{v:.0f}mW" for k, v in avg_power.items()},
            )
        else:
            _logger.info(
                "[PWR] Session '%s' complete: %.2fs  %d samples  total=%.3fmJ  per-channel=%s",
                self._session, duration, n, result.total_energy_mj,
                {k: f"{v:.3f}mJ" for k, v in energy_mj.items()},
            )
            if result.has_baseline:
                _logger.info(
                    "[PWR] Session '%s': avg=%.0fmW  net=%.0fmW  inference=%.3fs  net energy=%.3fmJ",
                    result.session, result.total_avg_power_mw, result.total_net_power_mw,
                    result.active_time_s, result.net_energy_mj,
                )
            self._results.append(result)
        self._session = ""
        self._is_baseline = False
        return result

    def measure_baseline(self, duration_s: float = 5.0, name: str = "baseline") -> dict[str, float] | None:
        """Measure idle power for duration_s seconds (blocking) and store it as
        the baseline subtracted from later sessions.

        Call once before inferencing starts, while the system is idle. Sessions
        measured before this call have no baseline. Returns {channel: mW}, or
        None if power measurement is unavailable.
        """
        if self._thread is not None and self._thread.is_alive():
            _logger.warning("[PWR] Cannot measure baseline while session '%s' is running", self._session)
            return None
        self.start(name)
        if self._thread is None:  # monitor unavailable / no channels
            return None
        self._is_baseline = True
        try:
            time.sleep(duration_s)
        finally:
            result = self.stop()
        if result is None:
            return None
        self._baseline_mw = dict(result.avg_power_mw)
        return self.baseline_mw

    @contextlib.contextmanager
    def measure(self, session: str):
        """Context manager for one inference: start(), yield a handle, stop().

        Set ``handle.inference_time_s`` inside the block to use the model's own
        timing for the energy calculation; ``handle.result`` is set on exit
        (None if power measurement is unavailable).
        """
        handle = _Measurement()
        self.start(session)
        try:
            yield handle
        finally:
            handle.result = self.stop(handle.inference_time_s)

    def format_summary(self, per_inference: bool = False) -> str:
        """Table of all measured sessions, grouped by session name."""
        if not self._results:
            return "[PWR] No power measurements recorded."
        lines = ["[PWR] Power summary"]
        if not self._baseline_mw:
            lines.append("  No baseline measured: net power/energy equal gross.")

        groups: dict[str, list[SessionResult]] = {}
        for r in self._results:
            groups.setdefault(r.session, []).append(r)

        hdr = f"  {'session':<16}{'n':>4}{'time ms':>10}{'avg mW':>9}{'base mW':>9}{'net mW':>9}{'mJ/inf':>10}"
        lines += [hdr, "  " + "-" * (len(hdr) - 2)]

        def mean(vals):
            return sum(vals) / len(vals)

        def base_col(mw):
            return "-" if mw is None else f"{mw:.0f}"

        for name, rs in groups.items():
            lines.append(
                f"  {name:<16}{len(rs):>4}"
                f"{mean([r.active_time_s for r in rs]) * 1000:>10.1f}"
                f"{mean([r.total_avg_power_mw for r in rs]):>9.0f}"
                f"{base_col(mean([sum(r.baseline_mw.values()) for r in rs]) if rs[0].has_baseline else None):>9}"
                f"{mean([r.total_net_power_mw for r in rs]):>9.0f}"
                f"{mean([r.net_energy_mj for r in rs]):>10.2f}"
            )
            if per_inference:
                for i, r in enumerate(rs, 1):
                    lines.append(
                        f"    #{i:<13}{'':>4}"
                        f"{r.active_time_s * 1000:>10.1f}"
                        f"{r.total_avg_power_mw:>9.0f}"
                        f"{base_col(sum(r.baseline_mw.values()) if r.has_baseline else None):>9}"
                        f"{r.total_net_power_mw:>9.0f}"
                        f"{r.net_energy_mj:>10.2f}"
                        f"  ({r.n_samples} samples)"
                    )
        if len(groups) > 1:
            lines.append(
                f"  {'ALL':<16}{len(self._results):>4}{'':>10}{'':>9}{'':>9}{'':>9}"
                f"{mean([r.net_energy_mj for r in self._results]):>10.2f}"
            )
        few = [r for r in self._results if r.n_samples < 3]
        if few:
            lines.append(
                f"  Note: {len(few)} inference(s) had fewer than 3 power samples; "
                "averages are coarse. Use a smaller interval_s for short inferences."
            )
        lines.append("  mJ/inf = (avg - base) power x inference time; all channels summed.")
        return "\n".join(lines)

    def print_summary(self, per_inference: bool = False) -> None:
        print(self.format_summary(per_inference=per_inference))

    def close(self):
        """Stop sampling and close the CSV file."""
        self.stop()
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None
            self._csv_writer = None
            self._csv_channels = []

    def _open_csv(self, channels: list[str]) -> None:
        if self._csv_path is None or self._csv_fh is not None:
            return
        is_new = not self._csv_path.exists()
        self._csv_fh = open(self._csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_fh)
        self._csv_channels = channels
        if is_new:
            header = ["session", "timestamp"]
            for ch in channels:
                header += [f"{ch}_mv", f"{ch}_ma", f"{ch}_mw"]
            self._csv_writer.writerow(header)
            self._csv_fh.flush()

    def _run(self):
        while not self._stop.wait(self._interval):
            self._sample_once()

    def _sample_once(self) -> None:
        try:
            if self._label:
                readings = [self._monitor.read(self._label)]
            else:
                readings = list(self._monitor.read_all().values())
        except (OSError, ValueError) as e:
            # A failed or malformed read skips this sample; sampling continues.
            _logger.warning("[PWR] Read error: %s", e)
            return
        if not readings:
            return

        ts = time.strftime("%H:%M:%S")
        by_label = {r.label: r for r in readings}

        for r in readings:
            self._energy_acc[r.label] = (
                self._energy_acc.get(r.label, 0.0) + r.power_mw * self._interval
            )
            self._power_sum[r.label] = self._power_sum.get(r.label, 0.0) + r.power_mw

        if self._log_samples:
            parts = "  ".join(
                f"{r.label}: {r.voltage_mv:.0f}mV {r.current_ma:.0f}mA {r.power_mw:.0f}mW"
                for r in readings
            )
            _logger.info("[PWR] %s  [%s]  %s", ts, self._session, parts)

        if self._csv_writer is not None:
            row: list = [self._session, ts]
            for ch in self._csv_channels:
                r = by_label.get(ch)
                if r is not None:
                    row += [f"{r.voltage_mv:.1f}", f"{r.current_ma:.1f}", f"{r.power_mw:.1f}"]
                else:
                    row += ["", "", ""]
            self._csv_writer.writerow(row)
            self._csv_fh.flush()

        self._n_samples += 1


# ---------------------------------------------------------------------------
# Test main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="Read power monitor values from hwmon sysfs.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_HWMON_PATH),
        help=f"hwmon directory (default: {DEFAULT_HWMON_PATH})",
    )
    parser.add_argument("--label", help="Read a single label instead of all channels.")
    parser.add_argument(
        "--bench",
        type=int,
        metavar="N",
        help="Benchmark: read all channels N times and report average latency.",
    )
    parser.add_argument(
        "--sample",
        type=float,
        metavar="SECS",
        help="Run PowerSampler for SECS seconds at 100ms interval and print log output.",
    )
    parser.add_argument("--csv", metavar="FILE", help="Write samples to a CSV file.")
    args = parser.parse_args()

    mon = PowerMonitor(args.path)
    print(f"hwmon path : {args.path}")
    print(f"Channels   : {mon.channels}\n")

    if not mon.available:
        print("Power monitor not available — check the path and label files.")

    elif args.sample:
        sampler = PowerSampler(mon, interval_s=0.1, csv_file=args.csv)
        sampler.start("test")
        time.sleep(args.sample)
        result = sampler.stop()
        sampler.close()
        if result:
            print(f"\nSession     : {result.session}")
            print(f"Duration    : {result.duration_s:.2f}s")
            print(f"Samples     : {result.n_samples}")
            print(f"Energy (mJ) : {result.energy_mj}")
            print(f"Total (mJ)  : {result.total_energy_mj:.3f}")

    elif args.bench:
        mon.read_all()  # warm up
        t0 = time.perf_counter_ns()
        for _ in range(args.bench):
            mon.read_all()
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        print(f"{elapsed_ms / args.bench * 1000:.1f} µs avg/call")

    elif args.label:
        try:
            r = mon.read(args.label)
            print(f"{r.label}:")
            print(f"  voltage  : {r.voltage_mv:.1f} mV")
            print(f"  current  : {r.current_ma:.1f} mA")
            print(f"  power    : {r.power_mw:.1f} mW")
        except KeyError:
            print(f"Label '{args.label}' not found. Available: {mon.channels}")

    else:
        readings = mon.read_all()
        col = max(len(l) for l in readings)
        print(f"{'Label':<{col}}  {'Voltage (mV)':>12}  {'Current (mA)':>12}  {'Power (mW)':>10}")
        print("-" * (col + 40))
        for r in readings.values():
            print(f"{r.label:<{col}}  {r.voltage_mv:>12.1f}  {r.current_ma:>12.1f}  {r.power_mw:>10.1f}")
