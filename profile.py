# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging
from pathlib import Path
from torq.runtime import profile_vmfb_inference_time


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
        "-j", "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)"
    )
    args = parser.parse_args()
    logger = logging.getLogger().setLevel(logging.DEBUG)
    t_avg = profile_vmfb_inference_time(
        args.model,
        device=args.iree_device,
        n_iters=args.repeat,
        n_threads=args.threads,
        function=args.function
    )
    print(f"Avg infer time for {args.model} ({args.repeat} iters): {t_avg:.3f} ms")
