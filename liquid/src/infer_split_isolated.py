# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""End-to-end LFM2.5 inference using subprocess-isolated splits.

Each call into one of the 5 vmfbs (split_0..split_3 + head) is run in a
fresh Python subprocess, so the OS reclaims the IREE runtime's leaked
nanobind objects between calls.  Host-side keeps the KV / conv caches
for each split and pickles them in/out per step.

Slow but memory-safe: master process stays ~100 MB; each subprocess
peaks at ~320 MB then exits.  Per-token cost ≈ 5 × 1.5 s subprocess
startup + ~0.5 s chip-side ≈ 8 s.

Usage:
    PYTHONPATH=src python -u src/infer_split_isolated.py \\
        -d /home/root/torq-examples/models/Synaptics/LFM2.5-350M-torq/split \\
        --instruct-model \\
        -p "What is the capital of France?"
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


logger = logging.getLogger("isolated")


def _rss_kb() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


# ---------- worker: invoked as a subprocess for one split ----------
def _worker_main() -> None:
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
    infer_ns = time.perf_counter_ns() - t0

    first = outs[0]
    first_host = first.to_host().copy() if hasattr(first, "to_host") else np.asarray(first).copy()
    result = {"out0": first_host, "infer_ns": infer_ns}
    if save_kv:
        result["kv"] = runner.save_kv_state()
    pickle.dump(result, sys.stdout.buffer)
    sys.stdout.buffer.flush()


def _run_split_subproc(vmfb: str, inputs: list[np.ndarray],
                       restore_kv: list | None, save_kv: bool) -> dict:
    payload = {"vmfb": vmfb, "inputs": inputs, "save_kv": save_kv, "restore_kv": restore_kv}
    proc = subprocess.run(
        [sys.executable, "-u", __file__, "--_worker"],
        input=pickle.dumps(payload),
        capture_output=True,
        env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode(errors="replace"))
        raise RuntimeError(f"Worker for {vmfb} failed (exit={proc.returncode})")
    return pickle.loads(proc.stdout)


# ---------- master: orchestrates the chained inference ----------
def _peek_roles(vmfb_path: str) -> list[str]:
    code = (
        "import sys, pickle\n"
        "from utils.inference import ManagedSelfAttnCacheRunner\n"
        f"r = ManagedSelfAttnCacheRunner({vmfb_path!r}, cache_start_idx=1)\n"
        "in_info = r.inputs_info\n"
        "n_cache = len(r.outputs_info) - 1\n"
        "non_cache = in_info[: len(in_info) - n_cache]\n"
        "roles = []\n"
        "for t in non_cache:\n"
        "    s = list(t.shape)\n"
        "    if len(s) == 3: roles.append('hidden_in')\n"
        "    elif len(s) == 2 and s[1] == 1: roles.append('position_ids')\n"
        "    elif len(s) == 2 and s[1] > 1: roles.append('attention_mask')\n"
        "    else: roles.append('?')\n"
        "pickle.dump(roles, sys.stdout.buffer)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True,
        env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode(errors="replace"))
        raise RuntimeError(f"peek failed for {vmfb_path}")
    return pickle.loads(proc.stdout)


SYS_PROMPT = (
    "You are a helpful AI assistant. "
    "Answer in 1-2 sentences. No lists, no bullet points, no repetition."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-d", "--models-dir", type=str)
    parser.add_argument("-p", "--prompt", type=str, default="Hello")
    parser.add_argument("--instruct-model", action="store_true", default=False)
    parser.add_argument("--max-gen-tokens", type=int, default=32)
    args = parser.parse_args()

    if args._worker:
        _worker_main()
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mdir = Path(args.models_dir)

    paths = [str(mdir / f"split_{i}.vmfb") for i in range(2)]
    head_path = str(mdir / "head.vmfb")
    n_layer_splits = len(paths)

    # Metadata
    with open(mdir / "config.json") as f:
        cfg = json.load(f)
    tokenizer = Tokenizer.from_file(str(mdir / "tokenizer.json"))
    bos_id = cfg["bos_token_id"]
    eos_id = cfg["eos_token_id"]
    vocab_size = cfg["vocab_size"]
    emb = np.load(mdir / "token_embeddings.npy", mmap_mode="r")
    if emb.dtype == np.dtype("V2"):
        emb = emb.view(ml_dtypes.bfloat16)

    logger.info("Discovering split roles...")
    split_roles = [_peek_roles(p) for p in paths]
    head_roles = _peek_roles(head_path)
    for i, r in enumerate(split_roles):
        logger.info("  split_%d roles=%s", i, r)
    logger.info("  head    roles=%s", head_roles)

    max_seq = 256
    pos_buf = np.zeros((1, 1), dtype=np.int32)
    mask_buf = np.ones((1, max_seq), dtype=np.int32)
    emb_dim = emb.shape[-1]

    # Persistent host-side KV/conv state per layer split.
    kv_states: list[list[np.ndarray] | None] = [None] * n_layer_splits

    def llm_step(token_id: int, seq_pos: int, sample: bool = True) -> int | None:
        # CPU embedding lookup.
        hidden = np.asarray(emb[token_id], dtype=ml_dtypes.bfloat16).reshape(1, 1, emb_dim).copy()
        pos_buf[0, 0] = seq_pos

        for s_idx, (path, roles) in enumerate(zip(paths, split_roles)):
            inputs = []
            for r in roles:
                if r == "hidden_in":
                    inputs.append(hidden)
                elif r == "position_ids":
                    inputs.append(pos_buf)
                elif r == "attention_mask":
                    inputs.append(mask_buf)
            result = _run_split_subproc(
                path, inputs, restore_kv=kv_states[s_idx], save_kv=True,
            )
            hidden = result["out0"]
            kv_states[s_idx] = result["kv"]

        # Head split — no caches.
        head_inputs = [hidden if r == "hidden_in" else None for r in head_roles]
        head_inputs = [h for h in head_inputs if h is not None]
        result = _run_split_subproc(head_path, head_inputs, restore_kv=None, save_kv=False)
        logits = result["out0"][0, -1].astype(np.float32)
        if not sample:
            return None
        return int(np.argmax(logits))

    # ---- Tokenize prompt ----
    def tokenize_chat(text: str, role: str) -> list[int]:
        if role == "assistant":
            return tokenizer.encode("<|im_start|>assistant\n").ids
        ids = tokenizer.encode(f"<|im_start|>{role}\n{text}<|im_end|>\n").ids
        # strip auto-prepended BOS, we add it once
        if ids and ids[0] == bos_id:
            ids = ids[1:]
        return ids

    if args.instruct_model:
        sys_tokens = [bos_id] + tokenize_chat(SYS_PROMPT, "system")
        user_tokens = tokenize_chat(args.prompt, "user") + tokenize_chat("", "assistant")
    else:
        sys_tokens = []
        user_tokens = tokenizer.encode(args.prompt).ids

    logger.info("system tokens: %d, user+assistant prefix tokens: %d",
                len(sys_tokens), len(user_tokens))

    # ---- Prefill (system + user/assistant prefix) ----
    pos = 0
    t_start = time.perf_counter_ns()
    for tok in sys_tokens + user_tokens[:-1]:
        llm_step(tok, pos, sample=False)
        pos += 1
        logger.info("prefill pos=%d", pos)

    # Get the first generated token
    next_tok = llm_step(user_tokens[-1], pos)
    pos += 1
    t_first_token = time.perf_counter_ns() - t_start

    gen = [next_tok]
    logger.info("FIRST TOKEN id=%d (%r)", next_tok, tokenizer.decode([next_tok]))

    # ---- Decode loop ----
    while True:
        if next_tok == eos_id:
            logger.info("hit EOS")
            break
        if len(gen) >= args.max_gen_tokens:
            logger.info("hit max_gen_tokens")
            break
        next_tok = llm_step(next_tok, pos)
        pos += 1
        gen.append(next_tok)
        logger.info("gen pos=%d id=%d (%r)", pos, next_tok, tokenizer.decode([next_tok]))

    total_ns = time.perf_counter_ns() - t_start
    answer = tokenizer.decode(gen)
    print("\n=== ANSWER ===")
    print(answer)
    print()
    print(f"Time to first token: {t_first_token/1e9:.2f} s")
    print(f"Total wall time:     {total_ns/1e9:.2f} s")
    print(f"Tokens generated:    {len(gen)}")
    if len(gen) > 1:
        dec_ms = (total_ns - t_first_token) / 1e6
        print(f"Decode rate:         {(len(gen)-1) / (dec_ms/1000):.2f} tok/s")


if __name__ == "__main__":
    main()
