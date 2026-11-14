# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
import math
from pathlib import Path
from torq.runtime import profile_vmfb_resources


def profile_power(args, runtime_flags, sampler, avg_inference_ms):
    """Measure power and energy per inference for the model in ``args.model``.

    ``sampler`` is a PowerSampler whose idle baseline was already measured.
    Runs a second profiling pass (so the regular profile is not disturbed by the
    power sampling thread) and records power only while the timed iterations
    run, excluding model loading and warm-up. Net power is the average power
    during inference minus the baseline; energy per inference is net power x
    inference time.
    """
    from torq.runtime.vmfb import VMFBInferenceRunner

    # A fast model finishes -r iterations in milliseconds, fewer than one power
    # sampling interval. Run enough iterations to fill --power-min-seconds.
    n_iters = max(args.repeat, math.ceil(args.power_min_seconds * 1000 / avg_inference_ms))
    n_warmup = 0 if args.no_warmup else 1
    state = {"calls": 0, "infer_ms": 0.0}
    original_infer = VMFBInferenceRunner.infer

    def infer_with_power(self, *a, **kw):
        # Start sampling just before the first timed iteration, stop right after the last.
        if state["calls"] == n_warmup:
            sampler.start("inference")
        out = original_infer(self, *a, **kw)
        if state["calls"] >= n_warmup:
            state["infer_ms"] += self.infer_time_ms
        state["calls"] += 1
        if state["calls"] == n_warmup + n_iters:
            sampler.stop(inference_time_s=state["infer_ms"] / n_iters / 1000)
        return out

    VMFBInferenceRunner.infer = infer_with_power
    try:
        profile_vmfb_resources(
            args.model,
            device=args.iree_device,
            n_iters=n_iters,
            n_threads=args.threads,
            function=args.function,
            runtime_flags=runtime_flags,
            device_io=args.device_io,
            do_warmup=not args.no_warmup,
        )
    finally:
        VMFBInferenceRunner.infer = original_infer
        sampler.close()  # also ends the session if profiling raised

    if not sampler.results:
        print("\nPower: no measurements were recorded")
        return
    r = sampler.results[0]
    print(f"\nPower profile ({n_iters} iterations over {r.duration_s:.1f} s, {r.n_samples} samples):")
    print(f"  Baseline power:         {sum(r.baseline_mw.values()):.0f} mW")
    print(f"  Avg power:              {r.total_avg_power_mw:.0f} mW")
    print(f"  Net power:              {r.total_net_power_mw:.0f} mW")
    print(f"  Avg inference time:     {r.active_time_s * 1000:.3f} ms")
    print(f"  Energy per inference:   {r.net_energy_mj:.3f} mJ")
    if r.n_samples < 3:
        print(
            f"  Warning: only {r.n_samples} power sample(s) were taken; the inference window "
            f"({r.duration_s * 1000:.0f} ms) is too short for a reliable average.\n"
            f"           Increase --power-min-seconds or reduce --power-interval."
        )
    print(f"  Per channel (mW):       {'avg':>6}{'base':>7}{'net':>7}")
    for ch, avg in r.avg_power_mw.items():
        print(f"    {ch:<20}{avg:>6.0f}{r.baseline_mw.get(ch, 0.0):>7.0f}{r.net_power_mw[ch]:>7.0f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=Path,
        help="Path to VMFB model"
    )
    parser.add_argument(
        "-d", "--iree-device",
        type=str,
        choices=["local-task", "torq"],
        default="torq",
        help="IREE backend device: %(choices)s, default: %(default)s"
    )
    parser.add_argument(
        "--function",
        type=str,
        default="main",
        help="VMFB function name (default: %(default)s)"
    )
    parser.add_argument(
        "-r", "--repeat",
        type=int,
        default=5,
        help="Number of iterations to profile for (default: %(default)s)"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        default=False,
        help="Skip model warm-up before profiling"
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)"
    )
    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument(
        "--tda", type=str, choices=["cpu", "dmabuf"], default="dmabuf",
        help="Allocator backing Torq device buffers (default: %(default)s)",
    )
    runtime_group.add_argument(
        "--device-io",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preallocate inputs and keep cache outputs as device arrays (default: enabled)",
    )
    runtime_group.add_argument(
        "--runtime-flags",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="FLAG",
        help=(
            "[Advanced] Extra flags for the Torq runtime. "
            "Must be specified last; all remaining arguments are forwarded."
        ),
    )
    power_group = parser.add_argument_group("power")
    power_group.add_argument(
        "--power",
        action="store_true",
        help=(
            "Also measure power and energy per inference from the board's hwmon power "
            "monitor. Measures idle baseline power first, so keep the system idle at start."
        ),
    )
    power_group.add_argument(
        "--power-baseline-seconds",
        type=float,
        default=5.0,
        help="Idle baseline measurement time (default: %(default)s)",
    )
    power_group.add_argument(
        "--power-min-seconds",
        type=float,
        default=2.0,
        help=(
            "Minimum time the power pass keeps running inferences, so fast models "
            "still yield enough power samples (default: %(default)s)"
        ),
    )
    power_group.add_argument(
        "--power-interval",
        type=float,
        default=0.1,
        help="Power sampling interval in seconds (default: %(default)s)",
    )
    power_group.add_argument(
        "--hwmon-path",
        type=Path,
        default=None,
        help="hwmon directory of the power monitor (default: the board's standard path)",
    )
    power_group.add_argument(
        "--power-csv",
        type=Path,
        default=None,
        help="Write raw power samples to this CSV file",
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

    runtime_flags = [f"--torq_device_allocator={args.tda}"] + (args.runtime_flags or [])

    sampler = None
    if args.power:
        from utils import pwr_meas

        monitor = pwr_meas.PowerMonitor(args.hwmon_path or pwr_meas.DEFAULT_HWMON_PATH)
        if monitor.available:
            sampler = pwr_meas.PowerSampler(
                monitor, interval_s=args.power_interval, csv_file=args.power_csv, log_samples=False
            )
            # Measured first, before the model is loaded, while the system is idle.
            print(f"Measuring idle power for {args.power_baseline_seconds:.0f} s, keep the system idle...")
            sampler.measure_baseline(args.power_baseline_seconds)
        else:
            print("Power monitor not available; continuing without power measurements.")

    results = profile_vmfb_resources(
        args.model,
        device=args.iree_device,
        n_iters=args.repeat,
        n_threads=args.threads,
        function=args.function,
        runtime_flags=runtime_flags,
        device_io=args.device_io,
        do_warmup=not args.no_warmup,
    )
    print(results.summary())

    if sampler is not None:
        profile_power(args, runtime_flags, sampler, results.avg_inference_time_ms)
