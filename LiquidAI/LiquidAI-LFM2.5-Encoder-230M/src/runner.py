# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""LFM2.5-Encoder-230M runner: bidirectional encoder body on Torq.

The vmfb holds the encoder *body* with a static sequence length S:

    token_embedding [1, S, 1024] bf16   (host embedding LUT lookup)
    attention_mask  [1, S]       bf16   (1.0 = real token, 0.0 = padding)
      -> hidden     [1, S, 1024] bf16   (final-norm output)

The masked-LM head is tied to the token embeddings, so MLM logits at any
position are `hidden[pos] @ token_embeddings.T`, computed on the host only
for the candidate tokens a task needs (fill-mask top-k over the full vocab
is a single [1024] x [1024, 65536] matmul; routing scores only a handful of
label tokens).

Two backends:
  * ``.vmfb``  — Torq NPU via ``torq.runtime.VMFBInferenceRunner``.
  * ``.onnx``  — onnxruntime CPU (fp32 export), for host-side development.
"""

import json
import logging
import time
from pathlib import Path

import ml_dtypes
import numpy as np

logger = logging.getLogger("LiquidEncoder.runner")

_DEFAULT_ROUTES: dict[str, str] = {
    # route -> one in-context example (few-shot anchors; see route())
    "Coding": "How do I reverse a linked list in Java?",
    "Sales": "I want to buy 50 licenses, can I get a volume deal?",
    "Creative writing": "Compose a haiku about autumn leaves.",
    "General knowledge": "What year did the French Revolution begin?",
}


class LiquidEncoderStatic:
    def __init__(self, model_path: str, seq_len: int | None = None):
        model_path = Path(model_path)
        self.model_dir = model_path.parent
        self._load_assets()
        self.seq_len = seq_len or self.manifest.get("seq_lens", [256])[0]
        self.infer_time_ms = 0.0

        if model_path.suffix == ".vmfb":
            from torq.runtime import VMFBInferenceRunner

            try:
                self.runner = VMFBInferenceRunner(str(model_path))
            except ValueError:
                # older exports name the entrypoint after torch's "main_graph"
                self.runner = VMFBInferenceRunner(
                    str(model_path), function="main_graph")
            self._backend = "torq"
        elif model_path.suffix == ".onnx":
            import onnxruntime as ort

            self.sess = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"])
            self._backend = "ort"
        else:
            raise ValueError(f"unsupported model type: {model_path}")
        logger.info("backend=%s seq_len=%d", self._backend, self.seq_len)

    def _load_assets(self):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(
            str(self.model_dir / "tokenizer.json"))
        # bf16 [vocab, hidden]; np.load returns raw V2 for bf16 files
        self.embeddings = np.load(self.model_dir / "token_embeddings.npy")
        if self.embeddings.dtype == np.dtype("V2"):
            self.embeddings = self.embeddings.view(ml_dtypes.bfloat16)
        manifest_path = self.model_dir / "encoder_manifest.json"
        self.manifest = (json.loads(manifest_path.read_text())
                         if manifest_path.exists() else {})
        self.mask_token_id = self.manifest.get("mask_token_id", 16)
        self.pad_token_id = self.manifest.get("pad_token_id", 0)
        self.bos_token_id = 1

    # ---------------- core inference ----------------

    def tokenize(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if not ids or ids[0] != self.bos_token_id:
            ids = [self.bos_token_id] + ids
        return ids

    def encode_ids(self, ids: list[int]) -> np.ndarray:
        """Run the encoder body; returns fp32 hidden states [n, hidden]."""
        n = len(ids)
        if n > self.seq_len:
            raise ValueError(
                f"sequence of {n} tokens exceeds static seq_len {self.seq_len}")
        padded = np.full(self.seq_len, self.pad_token_id, dtype=np.int64)
        padded[:n] = ids
        embeds = self.embeddings[padded][None]  # [1, S, H] bf16
        mask = np.zeros((1, self.seq_len), dtype=np.float32)
        mask[0, :n] = 1.0

        t0 = time.perf_counter_ns()
        if self._backend == "torq":
            outputs = self.runner.infer([
                embeds.astype(ml_dtypes.bfloat16),
                mask.astype(ml_dtypes.bfloat16),
            ])
            hidden = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            if hasattr(hidden, "to_host"):
                hidden = hidden.to_host()
        else:
            hidden = self.sess.run(None, {
                "token_embedding": embeds.astype(np.float32),
                "attention_mask": mask,
            })[0]
        self.infer_time_ms = (time.perf_counter_ns() - t0) / 1e6
        return np.asarray(hidden)[0, :n].astype(np.float32)

    def mlm_logits(self, ids: list[int], pos: int,
                   cand_ids: list[int] | None = None) -> np.ndarray:
        """MLM logits at position `pos` (optionally only candidate tokens)."""
        hidden = self.encode_ids(ids)
        h = hidden[pos]
        E = self.embeddings if cand_ids is None else self.embeddings[cand_ids]
        return h @ E.astype(np.float32).T

    # ---------------- tasks ----------------

    def fill_mask(self, text: str, topk: int = 5) -> list[tuple[str, float]]:
        ids = self.tokenize(text)
        if self.mask_token_id not in ids:
            raise ValueError("input has no mask token")
        pos = ids.index(self.mask_token_id)
        logits = self.mlm_logits(ids, pos)
        top = np.argsort(logits)[::-1][:topk]
        return [(self.tokenizer.decode([int(t)]).strip(), float(logits[t]))
                for t in top]

    def _first_token(self, text: str) -> int:
        return self.tokenizer.encode(text, add_special_tokens=False).ids[0]

    def route(self, text: str,
              routes: dict[str, str] | list[str] | None = None,
              ) -> list[dict[str, float | str]]:
        """Few-shot MLM routing: label the mask with one of the route names.

        `routes` maps route name -> one example request (the in-context
        anchor). A plain list uses the route names without anchors (weaker).
        Returns [{route, score}] sorted best-first; scores are softmaxed over
        the routes' first-token logits.
        """
        if routes is None:
            routes = _DEFAULT_ROUTES
        if isinstance(routes, list):
            routes = {r: None for r in routes}

        parts = ["Route each request to one category: "
                 + ", ".join(routes) + "."]
        for name, example in routes.items():
            if example:
                parts.append(f"Request: {example}\nCategory: {name}")
        mask_tok = "<|mask|>"
        parts.append(f"Request: {text}\nCategory: {mask_tok}")
        template = "\n\n".join(parts)

        ids = self.tokenize(template)
        pos = ids.index(self.mask_token_id)
        logits = self.mlm_logits(ids, pos)
        scores = {}
        for name in routes:
            cands = [self._first_token(v) for v in (name, " " + name)]
            scores[name] = max(float(logits[c]) for c in cands)
        arr = np.array(list(scores.values()))
        probs = np.exp(arr - arr.max())
        probs /= probs.sum()
        return sorted(
            ({"route": n, "score": float(p)}
             for n, p in zip(scores, probs)),
            key=lambda d: d["score"], reverse=True)
