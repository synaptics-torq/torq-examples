# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Inference runner for LFM2.5 deployed as 9 small VMFBs (8 layer splits
+ 1 head split) instead of one big VMFB.  Each split is a self-contained
VMFB that takes a ``hidden_in`` (+ its share of KV / conv caches) and
returns a ``hidden_out`` (+ updated caches) — except the final ``head``
split, which returns ``logits``.

The chained runner threads the hidden state through the 8 layer splits
in order and finally through the head split, sampling one token per
``llm_step`` like the monolithic runner.
"""

import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

import ml_dtypes
import numpy as np
from tokenizers import Tokenizer
from utils.inference import ManagedSelfAttnCacheRunner

DEFAULT_SYS_PROMPT: Final[str] = (
    "You are a helpful AI assistant. "
    "Answer in 1-2 sentences. No lists, no bullet points, no repetition."
)

StopCheck = Callable[[], bool]


class InferenceInterrupted(Exception):
    """Raised when interactive inference is cancelled by the user."""


def _raise_if_stopped(should_stop: StopCheck | None) -> None:
    if should_stop is not None and should_stop():
        raise InferenceInterrupted


class _SplitRunner:
    """Wraps one ``ManagedSelfAttnCacheRunner`` plus a description of
    which inputs are non-cache (and need to be supplied per call) vs.
    cache (managed by the runner).

    The ONNX subgraph stores inputs in this order (per the splitter's
    ``io_sort_key``):
        0:   hidden_in (or ``token_embedding`` for split 0)
        1+:  position_ids / attention_mask (if any layer in the split is
             an attention block / shared-conv consumer)
        last: cache tensors (past_conv.X, past_key_values.X.key_value)
    Output order:
        0:   hidden_out (or ``logits`` for the head split)
        1+:  cache tensors (present_conv.X, present.X.key_value)
    """

    # Marker strings for the three kinds of non-cache inputs the runner
    # has to populate.  Roles are derived from inputs_info SHAPES, since
    # the runtime TensorInfo doesn't carry input names.
    ROLE_HIDDEN = "hidden_in"
    ROLE_POS = "position_ids"
    ROLE_MASK = "attention_mask"

    def __init__(self, vmfb_path: str, *, n_threads: int | None = None,
                 runtime_flags: list[str] | None = None):
        # cache_start_idx=1 because output[0] is always hidden_out/logits.
        self._model = ManagedSelfAttnCacheRunner(
            vmfb_path, cache_start_idx=1,
            n_threads=n_threads, runtime_flags=runtime_flags,
        )
        in_info = self._model.inputs_info
        n_cache_in = len(self._model.outputs_info) - 1  # one logits/hidden_out
        self._n_non_cache_in = len(in_info) - n_cache_in
        # Identify roles from shapes (the runtime drops names).
        # - hidden_in: rank 3 with last dim == hidden_size (1024)
        # - position_ids: rank 2 with shape [1, 1]
        # - attention_mask: rank 2 with shape [1, max_seq_len] (max_seq_len > 1)
        self._input_roles: list[str] = []
        self._input_dtypes = []
        for t in in_info[: self._n_non_cache_in]:
            shape = list(t.shape)
            if len(shape) == 3:
                self._input_roles.append(self.ROLE_HIDDEN)
            elif len(shape) == 2 and shape[1] == 1:
                self._input_roles.append(self.ROLE_POS)
            elif len(shape) == 2 and shape[1] > 1:
                self._input_roles.append(self.ROLE_MASK)
            else:
                raise ValueError(
                    f"Cannot classify non-cache input with shape {shape} "
                    f"in vmfb '{vmfb_path}'"
                )
            self._input_dtypes.append(np.dtype(t.dtype))

    @property
    def input_roles(self) -> list[str]:
        return self._input_roles

    @property
    def input_names(self) -> list[str]:  # back-compat alias
        return self._input_roles

    def reset_kv(self) -> None:
        self._model.reset_kv()

    def save_kv_state(self):
        return self._model.save_kv_state()

    def restore_kv_state(self, state) -> None:
        self._model.restore_kv_state(state)

    def shift_kv(self, keep_last_n: int, protect_first_n: int = 0) -> None:
        self._model.shift_kv(keep_last_n, protect_first_n=protect_first_n)

    def infer(self, inputs: list[np.ndarray]) -> list:
        return self._model.infer(inputs)

    @property
    def infer_time_ms(self) -> float:
        return self._model.infer_time_ms


class LiquidSplitStatic:
    """Drop-in chained replacement for ``LiquidStatic`` that uses 9 split
    VMFBs.  Tokenization, sampling, ChatML template, and KV-cache reset
    are identical; the only difference is that each ``llm_step`` runs
    the 8 layer splits and the head split sequentially."""

    def __init__(
        self,
        models_dir: str | os.PathLike,
        max_seq_len: int | None = None,
        max_prompt_tokens: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = False,
        *,
        cache_keep_n: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 64,
        runtime_flags: list[str] | None = None,
        sys_prompt: str | None = None,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._models_dir = Path(models_dir)

        # Load all 4 layer splits + 1 head split once at startup.
        # IREE's Python runtime leaks per-context resources on every
        # `del`, so repeated lazy load+unload is not viable.  Keeping
        # the runners alive lets the cumulative memory stabilize.
        self._n_layer_splits = 4
        self._splits: list[_SplitRunner] = []
        for i in range(self._n_layer_splits):
            path = self._models_dir / f"split_{i}.vmfb"
            self._splits.append(_SplitRunner(
                str(path), n_threads=n_threads, runtime_flags=runtime_flags,
            ))
            self._logger.info("Loaded split %d: %s (roles=%s)",
                              i, path.name, self._splits[-1].input_roles)
        self._head = _SplitRunner(
            str(self._models_dir / "head.vmfb"),
            n_threads=n_threads, runtime_flags=runtime_flags,
        )
        self._logger.info("Loaded head (roles=%s)", self._head.input_roles)
        # Mirrors for the all-loaded path:
        self._split_roles = [s.input_roles for s in self._splits]
        self._split_dtypes = [s._input_dtypes for s in self._splits]
        self._head_roles = self._head.input_roles
        self._head_dtypes = self._head._input_dtypes

        # Read config / tokenizer from the models dir (shared across splits).
        with open(self._models_dir / "config.json") as f:
            cfg = json.load(f)
        self._n_layers: int = cfg["num_hidden_layers"]
        self._bos_token_id: int = cfg["bos_token_id"]
        self._eos_token_id: int = cfg["eos_token_id"]
        self._pad_token_id: int = cfg.get("pad_token_id") or 0
        self._instruct_model = instruct_model
        self._tokenizer = Tokenizer.from_file(str(self._models_dir / "tokenizer.json"))
        self._nl_token_id: int = self._tokenizer.encode("\n").ids[-1]
        self._double_nl_token_id: int = self._tokenizer.encode("\n\n").ids[-1]
        self._bos_token: str = self._tokenizer.decode(
            [self._bos_token_id], skip_special_tokens=False
        )
        self._eos_token: str = self._tokenizer.decode(
            [self._eos_token_id], skip_special_tokens=False
        )

        self._max_prompt_tokens = max_prompt_tokens
        self._max_user_tokens: int | None = None
        self._sys_prompt = (sys_prompt or DEFAULT_SYS_PROMPT) if instruct_model else None
        self._cache_keep_n = cache_keep_n
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k

        # Embedding LUT (CPU-side lookup, same as monolithic runner).
        self._token_embeddings = self._load_embeddings()
        if self._token_embeddings is None:
            raise FileNotFoundError(
                f"token_embeddings.npy not found in {self._models_dir}"
            )

        # max_seq_len: derive from any split that has an attention_mask
        # input (rank 2), else fall back to the head's hidden_in shape.
        self._max_seq_len = max_seq_len or self._derive_max_seq_len()
        if self._max_seq_len is None:
            raise ValueError("Could not derive max_seq_len from any split")

        # Pre-allocate reusable buffers per split (only the non-cache ones).
        emb_dim = self._token_embeddings.shape[-1]
        self._buffers: list[dict[str, np.ndarray]] = []
        all_roles = self._split_roles + [self._head_roles]
        all_dtypes = self._split_dtypes + [self._head_dtypes]
        for roles, dtypes in zip(all_roles, all_dtypes):
            bufs: dict[str, np.ndarray] = {}
            for role, dt in zip(roles, dtypes):
                if role == _SplitRunner.ROLE_HIDDEN:
                    bufs[role] = np.zeros((1, 1, emb_dim), dtype=dt)
                elif role == _SplitRunner.ROLE_POS:
                    bufs[role] = np.zeros((1, 1), dtype=dt)
                elif role == _SplitRunner.ROLE_MASK:
                    bufs[role] = np.ones((1, self._max_seq_len), dtype=dt)
            self._buffers.append(bufs)

        self._warmup_len = self._warmup() if instruct_model else 0
        if self._warmup_len > 0:
            self._reset_cache_state = [s.save_kv_state() for s in self._splits]
        else:
            self._reset_cache_state = []

        self._n_tokens_gen: int = 0
        self._last_infer_ns: int = 0
        self._time_to_first_token_ns: int = 0
        self._start_time_ns: int = 0

        self._logger.info("Loaded %d split runners + head", len(self._splits))

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    @property
    def last_infer_time(self) -> float:
        return self._last_infer_ns / 1e6

    @property
    def time_to_first_token(self) -> float:
        return self._time_to_first_token_ns / 1e6

    @property
    def generated_tokens(self) -> int:
        return self._n_tokens_gen

    @property
    def is_instruct_model(self) -> bool:
        return self._instruct_model

    def _load_embeddings(self) -> np.ndarray | None:
        paths = list(self._models_dir.glob("token_embeddings.npy"))
        if not paths:
            return None
        arr = np.load(paths[0], mmap_mode="r")
        if arr.dtype == np.dtype("V2"):
            arr = arr.view(ml_dtypes.bfloat16)
        return arr

    def _derive_max_seq_len(self) -> int | None:
        s = self._splits[0]
        for role, info in zip(s.input_roles, s._model.inputs_info[: s._n_non_cache_in]):
            if role == _SplitRunner.ROLE_MASK and isinstance(info.shape[1], int):
                return int(info.shape[1])
        for info in s._model.inputs_info:
            if len(info.shape) == 4 and isinstance(info.shape[2], int):
                return int(info.shape[2])
        return None

    def _reset_cache(self):
        if self._reset_cache_state:
            for s, st in zip(self._splits, self._reset_cache_state):
                s.restore_kv_state(st)
        else:
            for s in self._splits:
                s.reset_kv()

    def tokenize(self, text: str, role: str | None = None) -> list[int]:
        if not self._instruct_model or role is None:
            return self._tokenizer.encode(text).ids
        if role == "assistant":
            ids = self._tokenizer.encode("<|im_start|>assistant\n").ids
        else:
            ids = self._tokenizer.encode(
                "<|im_start|>" + role + "\n" + text + "<|im_end|>\n"
            ).ids
        if ids and ids[0] == self._bos_token_id:
            ids = ids[1:]
        return ids

    def llm_step(self, token: int, seq_pos: int, *, sample: bool = True) -> int:
        # CPU embedding lookup into split 0's hidden_in buffer.
        bufs0 = self._buffers[0]
        bufs0[_SplitRunner.ROLE_HIDDEN][0, 0, :] = self._token_embeddings[token]
        # position_ids: update wherever present.
        for s_idx, roles in enumerate(self._split_roles + [self._head_roles]):
            if _SplitRunner.ROLE_POS in roles:
                self._buffers[s_idx][_SplitRunner.ROLE_POS][0, 0] = seq_pos

        hidden = None
        for s_idx, s in enumerate(self._splits):
            bufs = self._buffers[s_idx]
            inputs = []
            for role in s.input_roles:
                if role == _SplitRunner.ROLE_HIDDEN and s_idx > 0:
                    inputs.append(hidden)
                else:
                    inputs.append(bufs[role])
            outs = s.infer(inputs)
            hidden = outs[0]  # device array passed straight to next split

        # Head split: hidden -> logits.
        head_bufs = self._buffers[-1]
        head_inputs = []
        for role in self._head.input_roles:
            if role == _SplitRunner.ROLE_HIDDEN:
                head_inputs.append(hidden)
            else:
                head_inputs.append(head_bufs[role])
        logits_dev = self._head.infer(head_inputs)[0]

        if not sample:
            return 0
        return self._sample(logits_dev.to_host()[0, -1])

    def _sample(self, logits: np.ndarray) -> int:
        logits = logits.astype(np.float32, copy=False)
        if self._temperature <= 0:
            return int(logits.argmax())
        k = min(self._top_k, logits.shape[-1])
        top_k_idx = np.argpartition(logits, -k)[-k:]
        x = logits[top_k_idx]
        x /= self._temperature
        x -= x.max()
        np.exp(x, out=x)
        x /= x.sum()
        order = np.argsort(x)[::-1]
        cdf = np.cumsum(x[order])
        cut = int(np.searchsorted(cdf, self._top_p)) + 1
        keep = order[:cut]
        p = x[keep]
        p /= p.sum()
        return int(np.random.choice(top_k_idx[keep], p=p))

    def _prefill(self, tokens: list[int], start: int = 0,
                 should_stop: StopCheck | None = None) -> tuple[int, int]:
        pos = start
        for tok in tokens[:-1]:
            _raise_if_stopped(should_stop)
            self.llm_step(tok, pos, sample=False)
            pos += 1
        if tokens:
            tok = self.llm_step(tokens[-1], pos)
        else:
            tok = 0
        pos += 1
        return tok, pos

    def _stop(self, token: int, gen: list[int]) -> bool:
        if token == self._eos_token_id:
            return True
        if not self._instruct_model and len(gen) > 2:
            if token == self._double_nl_token_id:
                return True
            return all(t == self._nl_token_id for t in gen[-2:])
        return False

    def _warmup(self) -> int:
        if not self._instruct_model:
            return 0
        self._logger.info("Warm-up started...")
        sys_tokens = [self._bos_token_id] + self.tokenize(self._sys_prompt, "system")
        if isinstance(self._max_prompt_tokens, int):
            sys_tokens = sys_tokens[: self._max_prompt_tokens]
            self._max_user_tokens = max(0, self._max_prompt_tokens - len(sys_tokens))
        n = len(sys_tokens)
        self._prefill(sys_tokens)
        self._logger.info(
            "Warm-up complete: system prompt consumed %d tokens, remaining capacity %d",
            n, self._max_seq_len - n,
        )
        return n

    def reset(self) -> None:
        self._reset_cache()

    def run(self, user_input: str, should_stop: StopCheck | None = None) -> str:
        return "".join(self.run_stream(user_input, should_stop=should_stop))

    def run_stream(self, user_input: str, should_stop: StopCheck | None = None):
        self._reset_cache()
        self._n_tokens_gen = 0
        self._last_infer_ns = 0
        self._time_to_first_token_ns = 0

        tokens = self.tokenize(user_input, "user")
        if self._instruct_model:
            tokens += self.tokenize("", "assistant")
        limit = self._max_user_tokens if self._max_user_tokens is not None else self._max_prompt_tokens
        if isinstance(limit, int):
            if len(tokens) > limit:
                tokens = tokens[:limit]
            elif len(tokens) < limit:
                tokens += [self._pad_token_id] * (limit - len(tokens))

        gen: list[int] = []
        self._start_time_ns = time.perf_counter_ns()
        yield_ns = 0
        try:
            next_tok, pos = self._prefill(tokens, start=self._warmup_len, should_stop=should_stop)
            self._time_to_first_token_ns = time.perf_counter_ns() - self._start_time_ns
            prev_text = self._tokenizer.decode([next_tok])
            yield_start = time.perf_counter_ns()
            yield prev_text
            yield_ns += time.perf_counter_ns() - yield_start
            gen = [next_tok]
            while not self._stop(next_tok, gen):
                _raise_if_stopped(should_stop)
                if pos >= self._max_seq_len:
                    if self._cache_keep_n is not None:
                        for s in self._splits:
                            s.shift_kv(self._cache_keep_n, protect_first_n=self._warmup_len)
                        pos = self._warmup_len + self._cache_keep_n
                    else:
                        self._logger.warning("Max generation tokens reached")
                        break
                next_tok = self.llm_step(next_tok, pos)
                gen.append(next_tok)
                pos += 1
                full = self._tokenizer.decode(gen)
                chunk = full[len(prev_text):]
                yield_start = time.perf_counter_ns()
                yield chunk
                yield_ns += time.perf_counter_ns() - yield_start
                prev_text = full
        finally:
            self._n_tokens_gen = max(0, len(gen) - 1)
            self._last_infer_ns = (time.perf_counter_ns() - self._start_time_ns - yield_ns)
