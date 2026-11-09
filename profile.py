# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from pathlib import Path
from torq.runtime import profile_vmfb_resources


if __name__ == "__main__":
    import argparse

    from utils.log import configure_logging
    from utils.runtime import RuntimeConfig, add_common_inference_args

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
    add_common_inference_args(parser, with_no_refresh=False)
    args = parser.parse_args()
    configure_logging(args.logging)

    runtime = RuntimeConfig.from_args(args)

    results = profile_vmfb_resources(
        args.model,
        device=args.iree_device,
        n_iters=args.repeat,
        n_threads=args.threads,
        function=args.function,
        runtime_flags=runtime.flags,
        device_io=runtime.device_io,
        do_warmup=not args.no_warmup,
    )
    print(results.summary())
