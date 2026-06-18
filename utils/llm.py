# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sized
from pathlib import Path

import ml_dtypes
import numpy as np
from tokenizers import Tokenizer

from utils.inference import ManagedSelfAttnCacheRunner, SplitLMHeadRunner

StopCheck = Callable[[], bool]


class InferenceInterrupted(Exception):
    """Raised when interactive inference is cancelled by the user."""


def _raise_if_stopped(should_stop: StopCheck | None) -> None:
    if should_stop is not None and should_stop():
        raise InferenceInterrupted


def discover_lm_head_path(model_path: str | os.PathLike) -> Path | None:
    """Find a sibling LM head VMFB for *model_path*, when unambiguous."""
    model_path = Path(model_path).resolve()
    candidates = []
    for path in sorted(model_path.parent.glob("*.vmfb*")):
        if path.resolve() == model_path:
            continue
        normalized_stem = path.stem.lower().replace("-", "_")
        if "lm_head" in normalized_stem:
            candidates.append(path)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    candidate_list = ", ".join(str(path) for path in candidates)
    raise ValueError(
        "Found multiple LM head candidates next to model. "
        f"Please pass --lm-head explicitly. Candidates: {candidate_list}"
    )


def resolve_lm_head_path(
    model_path: str | os.PathLike,
    lm_head_path: str | os.PathLike | None = None,
    *,
    disable_lm_head: bool = False,
    logger: logging.Logger | None = None,
) -> Path | None:
    """Resolve explicit, disabled, or auto-discovered LM head selection."""
    if lm_head_path is not None and disable_lm_head:
        raise ValueError("--lm-head and --no-lm-head cannot be used together.")
    if lm_head_path is not None:
        return Path(lm_head_path)
    if disable_lm_head:
        return None

    discovered = discover_lm_head_path(model_path)
    if discovered is not None and logger is not None:
        logger.info("Auto-discovered LM head '%s'", str(discovered))
    return discovered


def resolve_token_id_lut(
    logits_size: int | None,
    vocab_size: int | None,
    token_id_lut: Sized | None,
    logger: logging.Logger | None = None,
) -> Sized | None:
    """Validate and choose the token ID LUT for compact-vocab LLM logits."""
    if logits_size is None:
        if token_id_lut is not None and logger is not None:
            logger.warning(
                "Cannot validate token ID LUT because logits shape is unavailable; "
                "using the LUT as provided."
            )
        return token_id_lut

    if vocab_size is None:
        if token_id_lut is not None and len(token_id_lut) != logits_size:
            raise ValueError(
                "Invalid token ID LUT: length "
                f"{len(token_id_lut)} does not match logits size {logits_size}."
            )
        if logger is not None:
            logger.warning(
                "Cannot determine vocab_size from config.json; token ID LUT "
                "requirement could not be fully validated."
            )
        return token_id_lut

    if logits_size == vocab_size:
        if token_id_lut is not None and logger is not None:
            logger.warning(
                "token_id_lut.npy exists, but logits size %d matches config "
                "vocab_size %d; ignoring the LUT.",
                logits_size,
                vocab_size,
            )
        return None

    if token_id_lut is None:
        raise ValueError(
            "token_id_lut.npy is required because logits size "
            f"{logits_size} does not match config vocab_size {vocab_size}."
        )

    if len(token_id_lut) != logits_size:
        raise ValueError(
            "Invalid token_id_lut.npy: length "
            f"{len(token_id_lut)} does not match logits size {logits_size}."
        )

    return token_id_lut


class DecoderOnlyLLMRunner(ABC):
    """Shared runner for decoder-only LLM VMFBs with managed KV cache."""

    __slots__ = (
        "_logger",
        "_debug_logging",
        "_model",
        "_model_dir",
        "_config",
        "_tokenizer",
        "_max_prompt_tokens",
        "_max_seq_len",
        "_max_user_tokens",
        "_temperature",
        "_top_p",
        "_top_k",
        "_bos_token_id",
        "_eos_token_id",
        "_pad_token_id",
        "_bos_token",
        "_eos_token",
        "_reset_cache_state",
        "_warmup_len",
        "_token_embeddings",
        "_token_id_lut",
        "_pos_buf",
        "_emb_buf",
        "_cache_keep_n",
        "_n_tokens_gen",
        "_last_infer_ns",
        "_time_to_first_token_ns",
        "_start_time_ns",
    )

    def __init__(
        self,
        model_path: str | os.PathLike,
        max_seq_len: int | None = None,
        max_prompt_tokens: int | None = None,
        n_threads: int | None = None,
        *,
        cache_keep_n: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 64,
        runtime_flags: list[str] | None = None,
        device_io: bool = False,
        lm_head_path: str | os.PathLike | None = None,
        disable_lm_head: bool = False,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._debug_logging = self._logger.isEnabledFor(logging.DEBUG)

        lm_head_path = resolve_lm_head_path(
            model_path,
            lm_head_path,
            disable_lm_head=disable_lm_head,
            logger=self._logger,
        )
        self._model = ManagedSelfAttnCacheRunner(
            model_path,
            n_threads=n_threads,
            runtime_flags=runtime_flags,
            device_io=device_io,
        )
        if lm_head_path is not None:
            self._model = SplitLMHeadRunner(
                self._model,
                lm_head_path,
                n_threads=n_threads,
                runtime_flags=runtime_flags,
            )

        model_seq_len = self._query_model_seq_len()
        if max_seq_len is not None and model_seq_len is not None:
            if max_seq_len != model_seq_len:
                self._logger.warning(
                    "max_seq_len=%d does not match model KV cache dim=%d; using %d",
                    max_seq_len,
                    model_seq_len,
                    model_seq_len,
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
            self._config = json.load(f)

        self._bos_token_id = self._config["bos_token_id"]
        self._eos_token_id = self._config["eos_token_id"]
        self._pad_token_id = self._config.get("pad_token_id") or 0
        self._tokenizer = Tokenizer.from_file(str(self._model_dir / "tokenizer.json"))
        self._bos_token = self._tokenizer.decode(
            [self._bos_token_id], skip_special_tokens=False
        )
        self._eos_token = self._tokenizer.decode(
            [self._eos_token_id], skip_special_tokens=False
        )

        self._max_prompt_tokens = max_prompt_tokens
        self._max_seq_len = max_seq_len
        self._max_user_tokens = None
        self._cache_keep_n = cache_keep_n
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k

        self._on_model_config_loaded(self._config)

        self._token_embeddings = self._load_embeddings()
        self._token_id_lut = resolve_token_id_lut(
            self._query_logits_size(),
            self._config.get("vocab_size"),
            self._load_token_id_lut(),
            self._logger,
        )
        if self._token_id_lut is not None:
            self._logger.info(
                "Loaded token ID LUT (%d entries) for trimmed vocab remap",
                len(self._token_id_lut),
            )

        self._pos_buf = np.zeros((1, 1), dtype=np.int32)
        if self._token_embeddings is not None:
            self._emb_buf = np.zeros(
                (1, 1, self._token_embeddings.shape[-1]),
                dtype=self._token_embeddings.dtype,
            )
        else:
            self._emb_buf = None

        self._warmup_len = self._warmup()
        if self._warmup_len > 0:
            self._reset_cache_state = self._model.save_kv_state()
        else:
            self._reset_cache_state = []

        self._n_tokens_gen = 0
        self._last_infer_ns = 0
        self._time_to_first_token_ns = 0
        self._start_time_ns = 0

        self._logger.info("Loaded model '%s'", str(model_path))
        if self._debug_logging:
            self._logger.warning(
                "DEBUG logging enabled: inference time metrics may be inflated"
            )

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

    @abstractmethod
    def tokenize(self, text: str, role: str | None = None) -> list[int]:
        """Tokenize text, optionally using a model-specific chat role."""
        ...

    def _on_model_config_loaded(self, cfg: dict) -> None:
        """Hook for subclasses to derive model-specific fields from config."""

    def _build_prompt_tokens(self, user_input: str) -> list[int]:
        """Build prompt tokens for a single user request."""
        return self.tokenize(user_input)

    def _build_warmup_tokens(self) -> list[int]:
        """Build reusable prefix tokens to prefill after cache reset."""
        return []

    def _should_stop(self, token: int, gen: list[int]) -> bool:
        """Return True when generation should stop after *token*."""
        return token == self._eos_token_id

    def _stop(self, token: int, gen: list[int]) -> bool:
        """Compatibility alias for older subclasses/tests."""
        return self._should_stop(token, gen)

    def _load_embeddings(self) -> np.ndarray | None:
        paths = list(self._model_dir.glob("token_embeddings.npy"))
        if not paths:
            return None
        arr = np.load(paths[0], mmap_mode="r")
        if arr.dtype == np.dtype("V2"):
            arr = arr.view(ml_dtypes.bfloat16)
        return arr

    def _load_token_id_lut(self) -> np.ndarray | None:
        paths = list(self._model_dir.glob("token_id_lut.npy"))
        if not paths:
            return None
        return np.load(paths[0])

    def _query_model_seq_len(self) -> int | None:
        """Extract max sequence length from the KV cache input shape."""
        info = self._model.inputs_info
        if info is None or len(info) < 3:
            return None
        kv_shape = info[2].shape
        if len(kv_shape) >= 3 and isinstance(kv_shape[2], int):
            return kv_shape[2]
        return None

    def _query_logits_size(self) -> int | None:
        info = self._model.outputs_info
        if not info:
            return None
        logits_shape = info[0].shape
        if logits_shape and isinstance(logits_shape[-1], int):
            return logits_shape[-1]
        return None

    def _reset_cache(self) -> None:
        if self._reset_cache_state:
            self._model.restore_kv_state(self._reset_cache_state)
        else:
            self._model.reset_kv()

    def reset(self) -> None:
        """Reset the model to its post-warmup state."""
        self._reset_cache()

    def llm_step(
        self,
        token: int,
        seq_pos: int,
        *,
        compute_logits: bool = True,
        sample_next: bool = True,
    ) -> int:
        if sample_next and not compute_logits:
            raise ValueError("sample_next=True requires compute_logits=True")

        if self._emb_buf is not None:
            self._emb_buf[0, 0, :] = self._token_embeddings[token]
            first = self._emb_buf
        else:
            self._pos_buf[0, 0] = token
            first = self._pos_buf.copy()

        self._pos_buf[0, 0] = seq_pos

        if not compute_logits:
            if isinstance(self._model, SplitLMHeadRunner):
                self._model.infer([first, self._pos_buf], skip_lm_head=True)
            else:
                self._model.infer([first, self._pos_buf])
            self._logger.debug("LLM step time: %.3f ms", self._model.infer_time_ms)
            return 0

        results = self._model.infer([first, self._pos_buf])
        self._logger.debug("LLM step time: %.3f ms", self._model.infer_time_ms)

        if not sample_next:
            return 0

        logits = results[0].to_host() if hasattr(results[0], "to_host") else results[0]
        compact_idx = self._sample(np.asarray(logits)[0, -1])
        if self._token_id_lut is not None:
            token_id = int(self._token_id_lut[compact_idx])
        else:
            token_id = compact_idx
        if self._debug_logging:
            self._logger.debug(
                "Token ID: %d, Token: %r",
                token_id,
                self._tokenizer.decode([token_id], skip_special_tokens=False),
            )
        return token_id

    def _sample(self, logits: np.ndarray) -> int:
        st = time.perf_counter_ns()
        logits = logits.astype(np.float32, copy=False)

        if self._temperature <= 0:
            token_id = int(logits.argmax())
            self._logger.debug(
                "Sampling time: %.3f ms", (time.perf_counter_ns() - st) / 1e6
            )
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
        self._logger.debug(
            "Sampling time: %.3f ms", (time.perf_counter_ns() - st) / 1e6
        )
        return token_id

    def _prefill(
        self,
        tokens: list[int],
        start: int = 0,
        should_stop: StopCheck | None = None,
        *,
        produce_next_token: bool = True,
    ) -> tuple[int, int]:
        pos = start
        for tok_id in tokens[:-1]:
            _raise_if_stopped(should_stop)
            self.llm_step(tok_id, pos, compute_logits=False, sample_next=False)
            pos += 1
            _raise_if_stopped(should_stop)
        if tokens:
            _raise_if_stopped(should_stop)
            tok = self.llm_step(
                tokens[-1],
                pos,
                compute_logits=produce_next_token,
                sample_next=produce_next_token,
            )
            _raise_if_stopped(should_stop)
        else:
            tok = 0
        pos += 1
        return tok, pos

    def _warmup(self) -> int:
        tokens = self._build_warmup_tokens()
        if not tokens:
            return 0

        self._logger.info("Warm-up started...")
        if isinstance(self._max_prompt_tokens, int):
            tokens = tokens[: self._max_prompt_tokens]
            self._max_user_tokens = max(0, self._max_prompt_tokens - len(tokens))
        n = len(tokens)
        self._prefill(tokens, produce_next_token=False)
        self._logger.info(
            "Warm-up complete: system prompt consumed %d tokens, "
            "remaining capacity is %d tokens",
            n,
            self._max_seq_len - n,
        )
        return n

    def prefill_tokens(self, tokens: list[int]) -> tuple[int, int]:
        return self._prefill(
            tokens,
            start=self._warmup_len,
            should_stop=None,
        )

    def _apply_prompt_limit(self, tokens: list[int]) -> list[int]:
        limit = (
            self._max_user_tokens
            if self._max_user_tokens is not None
            else self._max_prompt_tokens
        )
        if isinstance(limit, int):
            if len(tokens) > limit:
                return tokens[:limit]
            if len(tokens) < limit:
                return tokens + [self._pad_token_id] * (limit - len(tokens))
        return tokens

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
    ) -> Iterator[str]:
        """Yield decoded text chunks as tokens are generated."""
        self._reset_cache()
        self._n_tokens_gen = 0
        self._last_infer_ns = 0
        self._time_to_first_token_ns = 0

        tokens = self._apply_prompt_limit(self._build_prompt_tokens(user_input))

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
            while not self._should_stop(next_tok, gen):
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
                next_tok = self.llm_step(next_tok, pos)
                _raise_if_stopped(should_stop)
                gen.append(next_tok)
                pos += 1
                full_text = self._tokenizer.decode(gen)
                chunk = full_text[len(prev_text) :]
                yield_start_ns = time.perf_counter_ns()
                yield chunk
                yield_ns += time.perf_counter_ns() - yield_start_ns
                prev_text = full_text
        finally:
            self._n_tokens_gen = max(0, len(gen) - 1)
            self._last_infer_ns = (
                time.perf_counter_ns() - self._start_time_ns - yield_ns
            )
