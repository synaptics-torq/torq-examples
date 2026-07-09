# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Diagnostic: load each split, run a single forward pass, unload,
print per-split timing and memory.  Each split runs in its own
subprocess so the OS actually reclaims memory between splits.

Usage:
    PYTHONPATH=src python -u src/split_diag.py \\
        -d /home/root/torq-examples/models/Synaptics/LFM2.5-350M-torq/split \\
        --prompt "What is the capital of France?"
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np
from tokenizers import Tokenizer

logger = logging.getLogger("split_diag")


def _rss_kb() -> int:
    """Current process VmRSS in KB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def _free_kb() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


# ---------- worker: invoked as a subprocess for one split ----------
def _worker_main() -> None:
    """Read a single pickled dict from stdin, run inference, write a
    pickled result to stdout, then exit.  Lives entirely in this
    subprocess so the OS reclaims memory on return."""
    payload = pickle.load(sys.stdin.buffer)
    vmfb = payload["vmfb"]
    inputs = payload["inputs"]
    save_kv = payload.get("save_kv", False)
    restore_kv = payload.get("restore_kv")

    from utils.inference import ManagedSelfAttnCacheRunner

    runner = ManagedSelfAttnCacheRunner(vmfb, cache_start_idx=1)
    if restore_kv is not None:
        runner.restore_kv_state(restore_kv)

    t0 = time.perf_counter_ns()
    outs = runner.infer(inputs)
    t1 = time.perf_counter_ns()

    # Bring the first output (hidden_out or logits) back to host.
    first = outs[0]
    first_host = first.to_host().copy() if hasattr(first, "to_host") else np.asarray(first).copy()

    result = {
        "out0": first_host,
        "infer_ns": t1 - t0,
        "rss_kb": _rss_kb(),
    }
    if save_kv:
        result["kv"] = runner.save_kv_state()
    pickle.dump(result, sys.stdout.buffer)
    sys.stdout.buffer.flush()


def _run_split_in_subprocess(
    vmfb_path: str,
    inputs: list[np.ndarray],
    restore_kv: list[np.ndarray] | None = None,
    save_kv: bool = False,
) -> dict:
    payload = {"vmfb": vmfb_path, "inputs": inputs, "save_kv": save_kv, "restore_kv": restore_kv}
    py = sys.executable
    proc = subprocess.run(
        [py, "-u", __file__, "--_worker"],
        input=pickle.dumps(payload),
        capture_output=True,
        env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode(errors="replace"))
        raise RuntimeError(f"Worker for {vmfb_path} failed (exit={proc.returncode})")
    return pickle.loads(proc.stdout)


# ---------- master: orchestrate one forward pass ----------
def _build_inputs(roles: list[str], hidden: np.ndarray, pos_buf: np.ndarray,
                  mask_buf: np.ndarray) -> list[np.ndarray]:
    out = []
    for r in roles:
        if r == "hidden_in":
            out.append(hidden)
        elif r == "position_ids":
            out.append(pos_buf)
        elif r == "attention_mask":
            out.append(mask_buf)
        else:
            raise ValueError(f"unknown role {r!r}")
    return out


def _peek_roles(vmfb_path: str) -> tuple[list[str], list[str]]:
    """Run a tiny subprocess to inspect a vmfb's input/output info and
    derive the role list (hidden_in / position_ids / attention_mask)."""
    code = (
        "import sys, pickle\n"
        "from utils.inference import ManagedSelfAttnCacheRunner\n"
        f"r = ManagedSelfAttnCacheRunner({vmfb_path!r}, cache_start_idx=1)\n"
        "in_info = r.inputs_info\n"
        "n_cache = len(r.outputs_info) - 1\n"
        "non_cache = in_info[: len(in_info) - n_cache]\n"
        "roles = []\n"
        "dtypes = []\n"
        "for t in non_cache:\n"
        "    s = list(t.shape)\n"
        "    if len(s) == 3:\n"
        "        roles.append('hidden_in')\n"
        "    elif len(s) == 2 and s[1] == 1:\n"
        "        roles.append('position_ids')\n"
        "    elif len(s) == 2 and s[1] > 1:\n"
        "        roles.append('attention_mask')\n"
        "    else:\n"
        "        roles.append('?')\n"
        "    dtypes.append(str(t.dtype))\n"
        "pickle.dump({'roles': roles, 'dtypes': dtypes}, sys.stdout.buffer)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True,
        env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode(errors="replace"))
        raise RuntimeError(f"peek failed for {vmfb_path}")
    info = pickle.loads(proc.stdout)
    return info["roles"], info["dtypes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-d", "--models-dir", type=str, required=False)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--instruct-model", action="store_true", default=False)
    args = parser.parse_args()

    if args._worker:
        _worker_main()
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mdir = Path(args.models_dir)
    paths = [str(mdir / f"split_{i}.vmfb") for i in range(4)] + [str(mdir / "head.vmfb")]
    names = [f"split_{i}" for i in range(4)] + ["head"]

    # Load shared metadata.
    with open(mdir / "config.json") as f:
        cfg = json.load(f)
    tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
    emb = np.load(mdir / "token_embeddings.npy", mmap_mode="r")
    if emb.dtype == np.dtype("V2"):
        emb = emb.view(ml_dtypes.bfloat16)

    print(f"Free before any load: {_free_kb()/1024:.1f} MB")
    print(f"This-proc RSS:       {_rss_kb()/1024:.1f} MB\n")

    # Peek roles per split.
    all_roles = [_peek_roles(p)[0] for p in paths]
    for n, r in zip(names, all_roles):
        print(f"  {n} roles: {r}")
    print()

    # Build initial hidden from one token from the prompt.
    ids = tok.encode(args.prompt).ids
    token_id = ids[0]
    print(f"Token: {token_id} ({tok.decode([token_id])!r}); embedding from npy")
    hidden_dim = emb.shape[-1]
    hidden = np.array(emb[token_id], dtype=ml_dtypes.bfloat16).reshape(1, 1, hidden_dim)

    # Shared mask + pos buffers (matching dtype of each split's input).
    max_seq = 256  # from static export
    pos = np.zeros((1, 1), dtype=np.int32)
    mask = np.ones((1, max_seq), dtype=np.int32)

    # No persisted KV between splits for this single-pass test (each split
    # starts with zero cache supplied by the runner).
    times_ns = []
    for i, (path, roles, name) in enumerate(zip(paths, all_roles, names)):
        inputs = _build_inputs(roles, hidden, pos, mask)
        free_before = _free_kb()
        rss_before = _rss_kb()
        t0 = time.perf_counter_ns()
        result = _run_split_in_subprocess(path, inputs)
        t_wall = time.perf_counter_ns() - t0
        free_after = _free_kb()
        rss_after = _rss_kb()
        infer_ms = result["infer_ns"] / 1e6
        wall_ms = t_wall / 1e6
        load_unload_ms = wall_ms - infer_ms
        times_ns.append(result["infer_ns"])
        print(
            f"[{name}] vmfb infer={infer_ms:8.1f} ms  "
            f"subprocess wall={wall_ms:8.1f} ms  "
            f"(load+unload={load_unload_ms:7.1f} ms)  "
            f"worker_peak_RSS={result['rss_kb']/1024:6.1f} MB  "
            f"master_RSS_before/after={rss_before/1024:.1f}/{rss_after/1024:.1f} MB  "
            f"board_free_before/after={free_before/1024:.1f}/{free_after/1024:.1f} MB"
        )
        hidden = result["out0"]

    print(f"\nFinal output shape: {hidden.shape} dtype={hidden.dtype}")
    if hidden.shape[-1] == cfg["vocab_size"]:
        # Sample greedily.
        flat = hidden.reshape(-1, hidden.shape[-1]).astype(np.float32)
        tok_id = int(np.argmax(flat[-1]))
        print(f"Greedy next token: {tok_id} -> {tok.decode([tok_id])!r}")
    print(f"Total chip-side inference: {sum(times_ns)/1e6:.1f} ms across {len(times_ns)} splits")


if __name__ == "__main__":
    main()
