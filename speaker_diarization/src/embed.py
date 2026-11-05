# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""RedimNet speaker-embedding helper process.

``diarize.py`` starts this script as a separate process, because the RedimNet
VMFB and the Moonshine VMFBs can need different Torq runtime versions, and one
Python process can load only one runtime. The NPU also serves only one process
at a time, so this process runs to completion before Moonshine loads.

Inputs (``--audio``): 16 kHz float32 mono audio as ``.npy``.
Inputs (``--starts``): int64 window start offsets in samples as ``.npy``.
Output (``--out``): float32 embeddings, one row per window, as ``.npy``.
"""

import argparse
import sys
import time

import numpy as np

from torq.runtime import VMFBInferenceRunner


def load_runner(vmfb: str, tda: str) -> VMFBInferenceRunner:
    """Open the RedimNet VMFB. With the dmabuf allocator, the inputs must be
    device arrays, so the outputs are kept on device too."""
    return VMFBInferenceRunner(
        vmfb,
        function="main_graph",
        device_uri="torq://",
        runtime_flags=[f"--torq_device_allocator={tda}"],
        device_outputs=(tda == "dmabuf"),
    )


def embed_window(runner: VMFBInferenceRunner, window: np.ndarray, device_io: bool) -> np.ndarray:
    """Run one window through RedimNet and return its embedding as float32."""
    x = window.reshape(1, -1).astype(runner.inputs_info[0].dtype)
    if device_io:
        x = runner.allocate_device_array(x)
    out = runner.infer([x])[0]
    if hasattr(out, "to_host"):
        out = out.to_host()
    return np.asarray(out, dtype=np.float32).ravel()


def main(args: argparse.Namespace):
    audio  = np.load(args.audio)
    starts = np.load(args.starts)
    runner = load_runner(args.vmfb, args.tda)

    win = runner.inputs_info[0].shape[1]
    if win != args.window:
        sys.exit(f"RedimNet VMFB takes {win} samples, but diarize.py planned {args.window}")

    t0   = time.perf_counter()
    embs = np.stack([embed_window(runner, audio[s:s + win], args.tda == "dmabuf") for s in starts])
    elapsed_ms = (time.perf_counter() - t0) * 1000
    np.save(args.out, embs)
    print(f"[embed] {len(starts)} windows in {elapsed_ms:.0f} ms", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RedimNet speaker embeddings on the Torq NPU")
    parser.add_argument("--vmfb",   required=True,           help="RedimNet VMFB (input 1xN bf16, output 1x192)")
    parser.add_argument("--audio",  required=True,           help="16 kHz float32 audio (.npy)")
    parser.add_argument("--starts", required=True,           help="Window start offsets in samples (.npy)")
    parser.add_argument("--out",    required=True,           help="Output embeddings (.npy)")
    parser.add_argument("--window", type=int, required=True, help="Window length in samples planned by the caller")
    parser.add_argument("--tda",    choices=["cpu", "dmabuf"], default="dmabuf",
                        help="Allocator backing Torq device buffers (default: %(default)s)")
    main(parser.parse_args())
