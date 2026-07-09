# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

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
from torq.runtime import VMFBInferenceRunner

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


class LiquidStatic:
    """LFM2.5 inference runner for static-shape Torq VMFB models.

    LFM2.5 is a hybrid model — each of the 16 layers is either a conv
    block (sliding ``past_conv.N`` state of shape ``[1, 1024, 3]``) or a
    standard attention block (combined ``past_key_values.X.key_value``
    of shape ``[1, 16, 256, 64]``).  ``ManagedSelfAttnCacheRunner`` is
    agnostic to what each cached tensor represents — it just zero-inits
    each per-layer cache from the model's input shape metadata and
    shuttles each output back to its matching input.

    The model takes three non-cache inputs in order:
        1. ``token_embedding``  : the embedded vector ``[1, 1, 1024]``
           (LFM2.5 uses extracted token embeddings — a CPU-side LUT
           lookup; the VMFB does not contain the 65 K embedding table).
        2. ``position_ids``     : ``[1, 1]`` int32, the absolute decode
           position of the current token.
        3. ``attention_mask``   : ``[1, max_seq_len]`` int32 ones (the
           static model takes the full mask).
    Followed by the 16 per-layer cache inputs in graph-input order.
    """

    __slots__ = (
        "_logger", "_model", "_model_dir", "_tokenizer",
        "_max_prompt_tokens", "_max_seq_len", "_max_user_tokens",
        "_sys_prompt", "_temperature", "_top_p", "_top_k",
        "_n_layers", "_n_kv_heads", "_head_dim",
        "_instruct_model",
        "_bos_token_id", "_eos_token_id", "_pad_token_id",
        "_nl_token_id", "_double_nl_token_id",
        "_bos_token", "_eos_token",
        "_warmup_len", "_warmup_snapshot",
        "_token_embeddings", "_pos_buf", "_emb_buf", "_attn_mask",
        "_cache_keep_n", "_lmhead", "_cache_specs", "_caches", "_non_cache_count",
        "_n_tokens_gen", "_last_infer_ns",
        "_time_to_first_token_ns", "_start_time_ns",
    )

    def __init__(
        self,
        model_path: str | os.PathLike,
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
        lm_head_path: str | os.PathLike | None = None,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)

        # Raw runner with manual conv/KV cache threading. ManagedSelfAttnCacheRunner
        # mishandles this hybrid model's mixed conv+KV cache (degenerate output);
        # threading the caches by hand is bit-exact. With ``lm_head_path`` the
        # decoder is split body (hidden) + standalone lm_head, so prefill skips the
        # [1024,65536] lm_head (lower TTFT, no decode-throughput cost).
        self._model = VMFBInferenceRunner(
            model_path, function="main", device_uri="torq",
            n_threads=n_threads, runtime_flags=runtime_flags,
            load_method="preload", load_model_to_mem=True, device_outputs=True,
        )
        self._lmhead = None
        if lm_head_path is not None:
            # Preload (not mmap): mmap faulting during a dispatch blows the NPU's
            # 5 s per-job timeout. Body + lm_head fit on a freshly-booted NPU.
            self._lmhead = VMFBInferenceRunner(
                lm_head_path, function="main", device_uri="torq",
                n_threads=n_threads, runtime_flags=runtime_flags,
                device_outputs=True,
            )
            self._logger.info("Loaded standalone lm_head '%s'", str(lm_head_path))

        model_seq_len = self._query_model_seq_len()
        if max_seq_len is not None and model_seq_len is not None:
            if max_seq_len != model_seq_len:
                self._logger.warning(
                    "max_seq_len=%d does not match model KV cache dim=%d; using %d",
                    max_seq_len, model_seq_len, model_seq_len,
                )
            max_seq_len = model_seq_len
        elif max_seq_len is None and model_seq_len is not None:
            max_seq_len = model_seq_len
            self._logger.debug("Derived max_seq_len=%d from model metadata", max_seq_len)
        elif max_seq_len is None:
            raise ValueError(
                "Cannot determine max_seq_len: model has no reflection metadata. "
                "Pass max_seq_len explicitly."
            )

        self._model_dir = Path(model_path).parent
        with open(self._model_dir / "config.json") as f:
            cfg = json.load(f)
        self._n_layers: int = cfg["num_hidden_layers"]
        self._n_kv_heads: int = cfg["num_key_value_heads"]
        self._head_dim: int = int(cfg.get("head_dim") or (cfg["hidden_size"] // cfg["num_attention_heads"]))
        self._bos_token_id: int = cfg["bos_token_id"]
        self._eos_token_id: int = cfg["eos_token_id"]
        self._pad_token_id: int = cfg.get("pad_token_id") or 0
        self._instruct_model = instruct_model
        self._tokenizer = Tokenizer.from_file(str(self._model_dir / "tokenizer.json"))
        self._nl_token_id: int = self._tokenizer.encode("\n").ids[-1]
        self._double_nl_token_id: int = self._tokenizer.encode("\n\n").ids[-1]
        self._bos_token: str = self._tokenizer.decode(
            [self._bos_token_id], skip_special_tokens=False
        )
        self._eos_token: str = self._tokenizer.decode(
            [self._eos_token_id], skip_special_tokens=False
        )

        self._max_prompt_tokens = max_prompt_tokens
        self._max_seq_len = max_seq_len
        self._max_user_tokens: int | None = None
        if instruct_model:
            self._sys_prompt = sys_prompt or DEFAULT_SYS_PROMPT
        else:
            self._sys_prompt = None
        self._cache_keep_n = cache_keep_n
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k

        self._token_embeddings = self._load_embeddings()
        if self._token_embeddings is None:
            raise FileNotFoundError(
                f"token_embeddings.npy not found in {self._model_dir}; "
                "LFM2.5 export uses --extract-embeddings, this file is required."
            )

        # Match the dtypes that the compiled VMFB declares for the
        # non-cache inputs (--torq-convert-io-dtype produces int32 / bf16
        # I/O), and pre-allocate the small reusable buffers.
        in_info = self._model.inputs_info
        self._pos_buf = np.zeros((1, 1), dtype=np.dtype(in_info[1].dtype))
        self._emb_buf = np.zeros(
            (1, 1, self._token_embeddings.shape[-1]),
            dtype=np.dtype(in_info[0].dtype),
        )
        # input 0 = token_embedding, 1 = position_ids; input 2 may be a 2D
        # attention_mask (if the graph kept it). Everything after that is a
        # per-layer cache (conv [1,1024,3] or KV [1,16,256,64]) threaded by hand.
        if len(in_info) >= 3 and len(in_info[2].shape) == 2:
            self._attn_mask = np.ones(
                (1, self._max_seq_len), dtype=np.dtype(in_info[2].dtype)
            )
            self._non_cache_count = 3
        else:
            self._attn_mask = None
            self._non_cache_count = 2
        self._cache_specs = [
            (tuple(t.shape), np.dtype(t.dtype)) for t in in_info[self._non_cache_count:]
        ]
        self._caches: list = []
        self._reset_caches()

        # Warm up the system prompt, then snapshot the caches so each turn can be
        # reset to the post-warmup state (replaces save_kv_state/restore_kv_state).
        self._warmup_len = self._warmup() if instruct_model else 0
        if self._warmup_len > 0:
            self._warmup_snapshot = [self._cache_to_host(c) for c in self._caches]
        else:
            self._warmup_snapshot = None

        self._n_tokens_gen: int = 0
        self._last_infer_ns: int = 0
        self._time_to_first_token_ns: int = 0
        self._start_time_ns: int = 0

        self._logger.info("Loaded model '%s'", str(model_path))

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
        paths = list(self._model_dir.glob("token_embeddings.npy"))
        if not paths:
            return None
        arr = np.load(paths[0], mmap_mode="r")
        if arr.dtype == np.dtype("V2"):
            arr = arr.view(ml_dtypes.bfloat16)
        return arr

    def _query_model_seq_len(self) -> int | None:
        """Extract max sequence length: prefer attention_mask shape if
        present, else fall back to the KV-cache shape (seq_len axis 2 of
        a ``past_key_values.*`` input)."""
        info = self._model.inputs_info
        if info is None:
            return None
        # Try attention_mask (2D): shape[1] is seq_len.
        if len(info) >= 3 and len(info[2].shape) == 2 and isinstance(info[2].shape[1], int):
            return info[2].shape[1]
        # Fall back: first KV cache (rank 4) -> shape[2] is seq_len.
        for t in info:
            if len(t.shape) == 4 and isinstance(t.shape[2], int):
                return t.shape[2]
        return None

    def _reset_caches(self) -> None:
        """Zero-init the per-layer caches as on-device arrays."""
        self._caches = [
            self._model.allocate_device_array(np.zeros(shape, dtype))
            for shape, dtype in self._cache_specs
        ]

    @staticmethod
    def _cache_to_host(arr) -> np.ndarray:
        a = np.ascontiguousarray(np.asarray(arr.to_host()))
        return a.view(ml_dtypes.bfloat16) if a.dtype.kind == "V" else a

    def _reset_cache(self):
        """Reset to the post-warmup snapshot (instruct) or to zeros."""
        if self._warmup_snapshot is not None:
            self._caches = [
                self._model.allocate_device_array(c) for c in self._warmup_snapshot
            ]
        else:
            self._reset_caches()

    def tokenize(self, text: str, role: str | None = None) -> list[int]:
        if not self._instruct_model or role is None:
            return self._tokenizer.encode(text).ids
        # LFM2.5 ChatML format: <|im_start|>role\ntext<|im_end|>\n
        # BOS (<|startoftext|>) is added once at warmup start; strip
        # any auto-prepended BOS here.
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
        # CPU-side embedding lookup populates the first input buffer.
        self._emb_buf[0, 0, :] = self._token_embeddings[token]
        self._pos_buf[0, 0] = seq_pos

        non_cache = [self._emb_buf, self._pos_buf]
        if self._attn_mask is not None:
            non_cache.append(self._attn_mask)
        out = self._model.infer([*non_cache, *self._caches])
        # out[0] = logits (full decoder) or hidden_out (body); out[1:] = present
        # caches -> become past for the next step.
        self._caches = list(out[1:])

        if not sample:
            return 0
        if self._lmhead is not None:
            # Body -> hidden; lm_head -> logits. Independently compiled, so
            # normalize the [1,1,1024] (2 KB) hidden through the host.
            hidden = self._cache_to_host(out[0])
            logits = self._cache_to_host(self._lmhead.infer([hidden])[0])
        else:
            logits = self._cache_to_host(out[0])
        token_id = self._sample(logits[0, -1])
        self._logger.debug(
            "Token ID: %d, Token: %r",
            token_id, self._tokenizer.decode([token_id], skip_special_tokens=False),
        )
        return token_id

    def _sample(self, logits: np.ndarray) -> int:
        st = time.perf_counter_ns()
        logits = logits.astype(np.float32, copy=False)

        if self._temperature <= 0:
            token_id = int(logits.argmax())
            self._logger.debug("Sampling time: %.3f ms", (time.perf_counter_ns() - st) / 1e6)
            return token_id

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
        token_id = int(np.random.choice(top_k_idx[keep], p=p))
        self._logger.debug("Sampling time: %.3f ms", (time.perf_counter_ns() - st) / 1e6)
        return token_id

    def _prefill(
        self,
        tokens: list[int],
        start: int = 0,
        should_stop: StopCheck | None = None,
    ) -> tuple[int, int]:
        pos = start
        for tok_id in tokens[:-1]:
            _raise_if_stopped(should_stop)
            self.llm_step(tok_id, pos, sample=False)
            pos += 1
            _raise_if_stopped(should_stop)
        if tokens:
            _raise_if_stopped(should_stop)
            tok = self.llm_step(tokens[-1], pos)
            _raise_if_stopped(should_stop)
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
        # LFM2.5 format: <|startoftext|><|im_start|>system\n{sys_prompt}<|im_end|>\n
        sys_tokens = [self._bos_token_id] + self.tokenize(self._sys_prompt, "system")
        if isinstance(self._max_prompt_tokens, int):
            sys_tokens = sys_tokens[: self._max_prompt_tokens]
            self._max_user_tokens = max(0, self._max_prompt_tokens - len(sys_tokens))
        n = len(sys_tokens)
        self._prefill(sys_tokens)
        self._logger.info(
            "Warm-up complete: system prompt consumed %d tokens, remaining capacity is %d tokens",
            n, self._max_seq_len - n
        )
        return n

    def reset(self) -> None:
        """Reset the model to its post-warmup state."""
        self._reset_cache()

    def prefill_tokens(
        self,
        tokens: list[int],
    ) -> tuple[int, int]:
        return self._prefill(
            tokens,
            start=self._warmup_len,
            should_stop=None,
        )

    def run(
        self,
        user_input: str,
        should_stop: StopCheck | None = None,
    ) -> str:
        return "".join(self.run_stream(user_input, should_stop=should_stop))

    def run_stream(
        self,
        user_input: str,
        should_stop: StopCheck | None = None,
    ):
        """Yield decoded text chunks as tokens are generated."""

        self._reset_cache()
        self._n_tokens_gen = 0
        self._last_infer_ns = 0
        self._time_to_first_token_ns = 0

        tokens = self.tokenize(user_input, "user")
        if self._instruct_model:
            tokens += self.tokenize("", "assistant")
        # Truncate / pad to max user length
        limit = (
            self._max_user_tokens
            if self._max_user_tokens is not None
            else self._max_prompt_tokens
        )
        if isinstance(limit, int):
            if len(tokens) > limit:
                tokens = tokens[:limit]
            elif len(tokens) < limit:
                tokens += [self._pad_token_id] * (limit - len(tokens))

        gen: list[int] = []
        self._start_time_ns = time.perf_counter_ns()
        yield_ns = 0
        try:
            next_tok, pos = self._prefill(
                tokens,
                start=self._warmup_len,
                should_stop=should_stop,
            )
            self._time_to_first_token_ns = time.perf_counter_ns() - self._start_time_ns

            prev_text = self._tokenizer.decode([next_tok])
            yield_start_ns = time.perf_counter_ns()
            yield prev_text
            yield_ns += time.perf_counter_ns() - yield_start_ns

            gen = [next_tok]
            while not self._stop(next_tok, gen):
                _raise_if_stopped(should_stop)
                if pos >= self._max_seq_len:
                    # Manual-cache runner does not implement the sliding-window
                    # shift; stop at the KV cache limit.
                    self._logger.warning("Max sequence length (%d) reached", self._max_seq_len)
                    break
                next_tok = self.llm_step(next_tok, pos)
                _raise_if_stopped(should_stop)
                gen.append(next_tok)
                pos += 1
                full_text = self._tokenizer.decode(gen)
                chunk = full_text[len(prev_text):]
                yield_start_ns = time.perf_counter_ns()
                yield chunk
                yield_ns += time.perf_counter_ns() - yield_start_ns
                prev_text = full_text
        finally:
            self._n_tokens_gen = max(0, len(gen) - 1)
            self._last_infer_ns = (
                time.perf_counter_ns() - self._start_time_ns - yield_ns
            )


def format_answer(
    answer: str,
    infer_time: float,
    ttft: float,
    stats: list[str] | None = None,
    agent_name: str = "Agent",
) -> str:
    GREEN: Final[str] = "\033[32m"
    RESET: Final[str] = "\033[0m"
    metrics = [f"{infer_time:.3f} ms", f"TTFT: {ttft:.3f} ms"]
    metrics.extend(str(s) for s in (stats or []))
    return GREEN + f"{agent_name}: {answer}" + RESET + f" ({', '.join(metrics)})"
