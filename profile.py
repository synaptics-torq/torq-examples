# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from torq.runtime import profile_vmfb_resources


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
        "--tda", type=str, choices=["cpu", "dmabuf"], default="cpu",
        help="Allocator backing Torq device buffers (default: %(default)s)",
    )
    runtime_group.add_argument(
        "--device-io", action="store_true",
        help="Preallocate inputs and keep outputs as device arrays",
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
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

    runtime_flags = [f"--torq_device_allocator={args.tda}"] + (args.runtime_flags or [])

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
