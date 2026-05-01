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
from utils.cache_runner import ManagedSelfAttnCacheRunner

DEFAULT_SYS_PROMPT: Final[str] = (
    "You are a helpful AI assistant named Gemma. "
    "Answer in 1-2 sentences. No lists, no bullet points, no repetition."
)

StopCheck = Callable[[], bool]


class InferenceInterrupted(Exception):
    """Raised when interactive inference is cancelled by the user."""


def _raise_if_stopped(should_stop: StopCheck | None) -> None:
    if should_stop is not None and should_stop():
        raise InferenceInterrupted


class Gemma3Static:

    __slots__ = (
        "_logger", "_model", "_model_dir", "_tokenizer",
        "_max_prompt_tokens", "_max_seq_len", "_max_user_tokens",
        "_sys_prompt", "_temperature", "_top_p", "_top_k",
        "_n_layers", "_n_kv_heads", "_head_dim",
        "_instruct_model",
        "_bos_token_id", "_eos_token_id", "_pad_token_id",
        "_nl_token_id", "_double_nl_token_id",
        "_bos_token", "_eos_token", "_end_of_turn_id",
        "_reset_cache_state", "_warmup_len",
        "_token_embeddings", "_pos_buf", "_emb_buf",
        "_cache_keep_n",
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
    ):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._model = ManagedSelfAttnCacheRunner(model_path, n_threads=n_threads, runtime_flags=runtime_flags)

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
        self._model_dir = Path(self._model.model_path).parent
        with open(self._model_dir / "config.json") as f:
            cfg = json.load(f)
        self._n_layers: int = cfg["num_hidden_layers"]
        self._n_kv_heads: int = cfg["num_key_value_heads"]
        self._head_dim: int = cfg["head_dim"]
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
        self._end_of_turn_id: int = self._tokenizer.token_to_id("<end_of_turn>")

        self._max_prompt_tokens = max_prompt_tokens
        self._max_seq_len = max_seq_len
        self._max_user_tokens: int | None = None
        self._sys_prompt = DEFAULT_SYS_PROMPT if instruct_model else None
        self._cache_keep_n = cache_keep_n
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k

        self._token_embeddings = self._load_embeddings()
        self._pos_buf = np.zeros((1, 1), dtype=np.int32)
        if self._token_embeddings is not None:
            self._emb_buf = np.zeros(
                (1, 1, self._token_embeddings.shape[-1]),
                dtype=self._token_embeddings.dtype,
            )
        else:
            self._emb_buf = None

        self._warmup_len = self._warmup() if instruct_model else 0
        # snapshot for reset (only meaningful for instruct after warmup)
        if self._warmup_len > 0:
            self._reset_cache_state = self._model.save_kv_state()
        else:
            self._reset_cache_state = []

        self._n_tokens_gen: int = 0
        self._last_infer_ns: int = 0
        self._time_to_first_token_ns: int = 0
        self._start_time_ns: int = 0

        self._logger.info("Loaded model '%s'", str(model_path))

    @property
    def last_infer_time(self) -> float:
        return self._last_infer_ns / 1e6

    @property
    def time_to_first_token(self) -> float:
        return self._time_to_first_token_ns / 1e6

    @property
    def generated_tokens(self) -> int:
        return self._n_tokens_gen

    def _load_embeddings(self) -> np.ndarray | None:
        paths = list(self._model_dir.glob("token_embeddings.npy"))
        if not paths:
            return None
        arr = np.load(paths[0], mmap_mode="r")
        if arr.dtype == np.dtype("V2"):
            arr = arr.view(ml_dtypes.bfloat16)
        return arr

    def _query_model_seq_len(self) -> int | None:
        """Extract max sequence length from the KV cache input shape."""
        info = self._model.inputs_info
        if info is None or len(info) < 3:
            return None
        # KV cache inputs start at index 2, shape: (1, 2*n_kv_heads, seq_len, head_dim)
        kv_shape = info[2].shape
        if len(kv_shape) >= 3 and isinstance(kv_shape[2], int):
            return kv_shape[2]
        return None

    def _reset_cache(self):
        if self._reset_cache_state:
            self._model.restore_kv_state(self._reset_cache_state)
        else:
            self._model.reset_kv()

    def _tokenize(self, text: str, role: str | None = None) -> list[int]:
        if not self._instruct_model or role is None:
            return self._tokenizer.encode(text).ids
        # Gemma 3 chat format: <start_of_turn>role\ntext<end_of_turn>\n
        # BOS is added once at warmup start; strip auto-prepended BOS here.
        if role == "model":
            ids = self._tokenizer.encode("<start_of_turn>model\n").ids
        else:
            ids = self._tokenizer.encode(
                "<start_of_turn>" + role + "\n" + text + "<end_of_turn>\n"
            ).ids
        if ids and ids[0] == self._bos_token_id:
            ids = ids[1:]
        return ids

    def _llm_step(self, token: int, seq_pos: int, *, sample: bool = True) -> int:
        if self._emb_buf is not None:
            self._emb_buf[0, 0, :] = self._token_embeddings[token]
            first = self._emb_buf
        else:
            self._pos_buf[0, 0] = token  # reuse pos_buf temporarily
            first = self._pos_buf.copy()  # need separate buffer for token

        self._pos_buf[0, 0] = seq_pos

        results = self._model.infer([first, self._pos_buf])
        self._logger.debug("LLM step time: %.3f ms", self._model.infer_time_ms)

        if not sample:
            return 0
        # Only bring logits to host for sampling
        return self._sample(results[0].to_host()[0, -1])

    def _sample(self, logits: np.ndarray) -> int:
        st = time.perf_counter_ns()
        logits = logits.astype(np.float32, copy=False)

        if self._temperature <= 0:
            token_id = int(logits.argmax())
            self._logger.debug("Token ID: %d, Token: %r, Sampling time: %.3f ms",
                               token_id, repr(self._tokenizer.decode([token_id], skip_special_tokens=False)),
                               (time.perf_counter_ns() - st) / 1e6)
            return token_id

        # Pre-select top-k candidates via O(n) partition to avoid
        # softmax / sort over the full 262 K vocabulary.
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
            self._llm_step(tok_id, pos, sample=False)
            pos += 1
            _raise_if_stopped(should_stop)
        if tokens:
            _raise_if_stopped(should_stop)
            tok = self._llm_step(tokens[-1], pos)
            _raise_if_stopped(should_stop)
        else:
            tok = 0
        pos += 1
        return tok, pos

    def _stop(self, token: int, gen: list[int]) -> bool:
        if token == self._eos_token_id:
            return True
        if self._end_of_turn_id is not None and token == self._end_of_turn_id:
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
        # Gemma3 format: <bos><start_of_turn>system\n{sys_prompt}<end_of_turn>\n
        sys_tokens = [self._bos_token_id] + self._tokenize(self._sys_prompt, "system")
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

        tokens = self._tokenize(user_input, "user")
        if self._instruct_model:
            tokens += self._tokenize("", "model")
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
        try:
            next_tok, pos = self._prefill(
                tokens,
                start=self._warmup_len,
                should_stop=should_stop,
            )
            self._time_to_first_token_ns = time.perf_counter_ns() - self._start_time_ns

            prev_text = self._tokenizer.decode([next_tok])
            yield prev_text

            gen = [next_tok]
            while not self._stop(next_tok, gen):
                _raise_if_stopped(should_stop)
                if pos >= self._max_seq_len:
                    if self._cache_keep_n is not None:
                        self._model.shift_kv(
                            self._cache_keep_n,
                            protect_first_n=self._warmup_len,
                        )
                        pos = self._warmup_len + self._cache_keep_n
                        self._logger.debug(
                            "Circular KV cache: shifted last %d entries after "
                            "%d protected prefix tokens",
                            self._cache_keep_n,
                            self._warmup_len,
                        )
                    else:
                        self._logger.warning("Max generation tokens reached")
                        break
                next_tok = self._llm_step(next_tok, pos)
                _raise_if_stopped(should_stop)
                gen.append(next_tok)
                pos += 1
                # Incremental decode: decode full sequence, emit only the new chars
                full_text = self._tokenizer.decode(gen)
                yield full_text[len(prev_text):]
                prev_text = full_text
        finally:
            self._n_tokens_gen = len(gen)
            self._last_infer_ns = time.perf_counter_ns() - self._start_time_ns


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
