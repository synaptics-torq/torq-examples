# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared Torq runtime utilities for all inference scripts."""

from __future__ import annotations


def build_runtime_flags(tda, extra_runtime_flags=None):
    return [f"--torq_device_allocator={tda}"] + (extra_runtime_flags or [])


def run_profile_if_requested(args, runtime_flags):
    """Run the profiler and return True if --profile was set (caller should return after this)."""
    if not args.profile:
        return False
    from torq.runtime import profile_vmfb_resources
    results = profile_vmfb_resources(
        args.model, device=args.device, n_iters=1, n_threads=None,
        function="main", runtime_flags=runtime_flags,
        device_io=args.device_io, do_warmup=True,
    )
    print(results.summary())
    return True


def setup_npu_for_inference():
    """Enable NPU clock and set max frequency for inference."""
    from utils.npu import configure_npu_userspace_frequency, enable_npu_clock

    ok, message = enable_npu_clock()
    print(f"[NPU] {message}")
    ok, message = configure_npu_userspace_frequency("max")
    print(f"[NPU] {message}")


def cleanup_npu_after_inference():
    """Restore NPU to min frequency after inference completes."""
    from utils.npu import configure_npu_userspace_frequency

    ok, message = configure_npu_userspace_frequency("min")
    print(f"[NPU] {message}")


def setup_npu_and_runner(args, runtime_flags):
    """Setup NPU clocks, Wayland env, optional camera controls, and create the inference runner.

    Works for both image inference (no camera args) and video inference (with camera args).
    """
    import os
    import sys
    from utils.inference import SimpleVMFBInferenceRunner

    setup_npu_for_inference()

    if getattr(args, "display", False):
        os.environ.setdefault("XDG_RUNTIME_DIR", "/var/run/user/0")
        os.environ.setdefault("WAYLAND_DISPLAY", "wayland-1")

    if getattr(args, "camera_device", None):
        from utils.video import configure_camera, resolve_camera_device
        try:
            args.camera_device = resolve_camera_device(args.camera_device)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        ctrl_device = args.camera_control_device or args.camera_device
        configure_camera(ctrl_device, {
            "brightness": args.brightness,
            "contrast": args.contrast,
            "saturation": args.saturation,
            "sharpness": args.sharpness,
            "gain": args.gain,
            "exposure_auto": args.exposure_auto,
            "exposure_absolute": args.exposure_absolute,
        })

    return SimpleVMFBInferenceRunner(
        args.model,
        device_uri=args.device,
        runtime_flags=runtime_flags,
        device_io=args.device_io,
    )
