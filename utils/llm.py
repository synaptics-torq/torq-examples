# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Final

import ml_dtypes
import numpy as np
from tokenizers import Tokenizer

from utils.inference import (
    InferenceInterrupted,
    ManagedSelfAttnCacheRunner,
    SplitLMHeadRunner,
    StopCheck,
    _raise_if_stopped,
)


_SIDECAR_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".incomplete", ".part", ".tmp", ".swp", ".lock", ".metadata", ".md5", ".sha256"}
)

_PREFILL_ENTRY_RE: Final[re.Pattern] = re.compile(r"^main_(\d+)$")


def parse_prefill_entry_size(name: str) -> int | None:
    """Chunk size B of a native prefill entry named ``main_B``, else None.

    ``main`` itself and degenerate names (``main_1``, ``main_x``) are not
    prefill entries.
    """
    m = _PREFILL_ENTRY_RE.fullmatch(name)
    if not m:
        return None
    size = int(m.group(1))
    return size if size > 1 else None


def discover_lm_head_path(model_path: str | os.PathLike) -> Path | None:
    """Find a sibling LM head VMFB for *model_path*, when unambiguous.

    Model file names carry build suffixes after ``.vmfb`` (``lm_head.vmfb.trim``,
    ``lm_head.vmfb.w4a16``), so candidates are matched on the whole name rather
    than a fixed extension. Interrupted-download and editor leftovers share that
    shape, so they are excluded: they are not loadable modules, and handing one
    to the runtime is far worse than not finding a head at all.
    """
    model_path = Path(model_path).resolve()
    candidates = []
    for path in sorted(model_path.parent.glob("*.vmfb*")):
        if path.resolve() == model_path:
            continue
        if path.suffix.lower() in _SIDECAR_SUFFIXES:
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


def discover_prefill_model_path(model_path: str | os.PathLike) -> Path | None:
    """Find a sibling batched prefill VMFB for *model_path*, when unambiguous.

    Matches the same rules as :func:`discover_lm_head_path` but looks for
    ``prefill`` in the file name (e.g. ``transformer_prefill.vmfb``).
    """
    model_path = Path(model_path).resolve()
    candidates = []
    for path in sorted(model_path.parent.glob("*.vmfb*")):
        if path.resolve() == model_path:
            continue
        if path.suffix.lower() in _SIDECAR_SUFFIXES:
            continue
        normalized_stem = path.stem.lower().replace("-", "_")
        if "prefill" in normalized_stem:
            candidates.append(path)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    candidate_list = ", ".join(str(path) for path in candidates)
    raise ValueError(
        "Found multiple batched prefill candidates next to model. "
        f"Please pass --batch-prefill-model explicitly. Candidates: {candidate_list}"
    )


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


def _tensor_signature(info) -> tuple[tuple, object]:
    """Comparable (shape, dtype) signature of a tensor metadata entry."""
    try:
        dtype = np.dtype(info.dtype)
    except (AttributeError, TypeError, ValueError):
        dtype = str(info.dtype)
    return tuple(info.shape), dtype


def _tensor_signature_matches(a, b) -> bool:
    return _tensor_signature(a) == _tensor_signature(b)


def _tensor_info(info) -> str:
    """Short "shape/dtype" description of a tensor metadata entry."""
    shape, dtype = _tensor_signature(info)
    return f"shape={shape}, dtype={dtype}"


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
        "_max_gen_tokens",
        "_temperature",
        "_top_p",
        "_top_k",
        "_bos_token_id",
        "_eos_token_id",
        "_eos_token_ids",
        "_bos_token",
        "_eos_token",
        "_reset_cache_state",
        "_warmup_len",
        "_token_embeddings",
        "_token_id_lut",
        "_pos_buf",
        "_emb_buf",
        "_id_buf",
        "_extra_step_inputs",
        "_cache_keep_n",
        "_cache_tokens",
        "_turn_pos",
        "_pending_token",
        "_n_tokens_gen",
        "_last_infer_ns",
        "_time_to_first_token_ns",
        "_start_time_ns",
        "_prefill_model",
        "_prefill_size",
        "_prefill_emb_buf",
        "_prefill_id_buf",
        "_native_prefill_entries",
        "_native_prefill_emb_buf",
        "_native_prefill_id_buf",
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
        max_gen_tokens: int | None = None,
        runtime_flags: list[str] | None = None,
        device_io: bool = False,
        lm_head_path: str | os.PathLike | None = None,
        disable_lm_head: bool = False,
        prefill_model_path: str | os.PathLike | None = None,
        disable_prefill: bool = False,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._debug_logging = self._logger.isEnabledFor(logging.DEBUG)

        lm_head_path = resolve_lm_head_path(
            model_path,
            lm_head_path,
            disable_lm_head=disable_lm_head,
            logger=self._logger,
        )
        if prefill_model_path is not None and disable_prefill:
            raise ValueError(
                "--batch-prefill-model and --no-batch-prefill-model cannot be "
                "used together."
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

        self._validate_decode_model()
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
        # Checkpoints may list several stop IDs (FunctionGemma: [1, 50],
        # where 50 is <start_function_response>); keep the first for display
        # and honor every configured stop.
        eos_token_id = self._config["eos_token_id"]
        self._eos_token_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
        self._eos_token_id = self._eos_token_ids[0]
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
        self._max_gen_tokens = max_gen_tokens
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

        self._init_step_buffers()

        self._prefill_model = None
        self._prefill_size = None
        self._prefill_emb_buf = None
        self._prefill_id_buf = None
        self._native_prefill_entries: list[int] = []
        self._native_prefill_emb_buf = None
        self._native_prefill_id_buf = None
        if prefill_model_path is not None:
            # Explicit --batch-prefill-model: the separate fixed-size sibling
            # build (legacy exporter-provided family).
            self._setup_sibling_prefill(
                Path(prefill_model_path), n_threads, runtime_flags, device_io
            )
        elif not disable_prefill:
            if not self._setup_native_prefill():
                sibling = discover_prefill_model_path(model_path)
                if sibling is not None:
                    self._logger.info(
                        "Auto-discovered batched prefill model '%s'", str(sibling)
                    )
                    self._setup_sibling_prefill(
                        sibling, n_threads, runtime_flags, device_io
                    )
        if self._prefill_model is None and not self._native_prefill_entries:
            self._logger.debug(
                "No batched prefill available; prompts will be "
                "prefilled with single-token decode steps."
            )

        self._turn_pos = 0
        self._pending_token = None
        self._cache_tokens = []
        self._warmup_len = self._warmup()
        if self._warmup_len > 0:
            self._reset_cache_state = self._model.save_kv_state()
        else:
            self._reset_cache_state = []
        self._turn_pos = self._warmup_len
        if (
            self._cache_keep_n is not None
            and self._warmup_len + self._cache_keep_n >= self._max_seq_len
        ):
            self._logger.warning(
                "kv-cache-window (%d) plus warm-up (%d) fills the %d-token KV "
                "cache; fewer history tokens will be retained to make room "
                "for new tokens.",
                self._cache_keep_n,
                self._warmup_len,
                self._max_seq_len,
            )

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
    def prefill_size(self) -> int | None:
        """Fixed prompt chunk size when a batched prefill model is loaded."""
        return self._prefill_size

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

    def _decode_generated_tokens(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def _should_stop(self, token: int, gen: list[int]) -> bool:
        """Return True when generation should stop after *token*."""
        return token in self._eos_token_ids

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

    def _init_step_buffers(self) -> None:
        """Build the per-step input buffers from the model's input metadata.

        Input 0 is the token input (embedding vector or token id), input 1 is
        ``position_ids``; both shapes and dtypes are taken from the model's
        reflection data rather than assumed. Any further non-cache inputs
        (e.g. the fixed ``attention_mask`` of newer static exports) are
        constant per step and are built by :meth:`_build_extra_step_inputs`.
        """
        in_info = self._model.inputs_info
        if not in_info or len(in_info) < 2:
            raise ValueError(
                f"Model '{self._model.model_path}' has no input metadata; "
                "cannot build the per-step input buffers."
            )
        n_user = len(in_info) - self._model.n_cache_inputs
        if n_user < 2:
            raise ValueError(
                f"Model '{self._model.model_path}' has only {n_user} non-cache "
                "input(s); expected at least (tokens, position_ids)."
            )
        token_info, pos_info = in_info[0], in_info[1]
        self._pos_buf = np.zeros(pos_info.shape, dtype=np.dtype(pos_info.dtype))
        if self._token_embeddings is not None:
            emb_hidden = self._token_embeddings.shape[-1]
            tok_shape = tuple(token_info.shape)
            if (
                len(tok_shape) >= 3
                and isinstance(tok_shape[-1], int)
                and tok_shape[-1] != emb_hidden
            ):
                raise ValueError(
                    f"token_embeddings.npy has hidden size {emb_hidden} but the "
                    f"model's token input expects {tok_shape[-1]} "
                    f"(shape {tok_shape})."
                )
            self._emb_buf = np.zeros(
                (1, 1, emb_hidden),
                # The LUT may be stored at higher precision than the model's
                # token input (e.g. an f32 embedding file for a bf16 export);
                # the buffer follows the input the runtime actually expects
                # and the lookup is cast on assignment.
                dtype=np.dtype(token_info.dtype),
            )
            self._id_buf = None
        else:
            self._emb_buf = None
            self._id_buf = np.zeros(token_info.shape, dtype=np.dtype(token_info.dtype))
        self._extra_step_inputs = self._build_extra_step_inputs(in_info[2:n_user])

    def _build_extra_step_inputs(self, infos) -> list[np.ndarray]:
        """Constant per-step inputs beyond (tokens, position_ids).

        Newer static exports add a fixed ``attention_mask`` input sized at
        the compiled KV-cache length; every step feeds it all-ones. Override
        for models whose extra inputs are not constant masks.
        """
        return [np.ones(info.shape, dtype=np.dtype(info.dtype)) for info in infos]

    def _validate_decode_model(self) -> None:
        """Fail early when the main model is not a single-token decode build.

        A fixed-size batched build (for example a ``transformer_prefill.vmfb``
        export deployed over the main model path) only rejects a single
        token deep inside the runtime with an opaque shape assertion. Catch
        it at load time and say how to fix the deployment.
        """
        info = self._model.inputs_info
        if not info or not info[0].shape:
            return
        shape = tuple(info[0].shape)
        if isinstance(shape[0], int) and shape[0] != 1:
            raise ValueError(
                f"Model '{self._model.model_path}' expects batch size "
                f"{shape[0]} (first input shape {shape}); only batch size "
                "1 is supported."
            )
        if len(shape) > 1 and isinstance(shape[1], int) and shape[1] > 1:
            raise ValueError(
                f"Model '{self._model.model_path}' expects {shape[1]} tokens "
                f"per step (first input shape {shape}); it looks like a "
                "fixed-size batched build rather than the single-token "
                "decode model. Pass the decode model (e.g. "
                "'transformer.vmfb') via -m, and provide the batched build "
                "as a sibling 'transformer_prefill.vmfb' or via "
                "--batch-prefill-model."
            )

    def _query_model_seq_len(self) -> int | None:
        """Extract max sequence length from the KV cache input shape."""
        if (seq_len := self._model.kv_seq_len) is not None:
            return seq_len
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
        """Reset the model to its post-warmup state (a fresh conversation)."""
        self._reset_cache()
        self._turn_pos = self._warmup_len
        self._pending_token = None
        self._cache_tokens = self._cache_tokens[:self._warmup_len]

    def _roll_cache(self, keep: int, should_stop: StopCheck | None) -> None:
        # Cached keys carry positional encodings. Moving their tensors leaves
        # those encodings at the old positions; replay tokens at the new ones.
        tokens = self._cache_tokens[-keep:] if keep else []
        self._reset_cache()
        self._turn_pos = self._warmup_len
        self._cache_tokens = self._cache_tokens[:self._warmup_len]
        self._prefill(
            tokens,
            start=self._turn_pos,
            should_stop=should_stop,
            produce_next_token=False,
        )

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
        return self._llm_tokens_step(
            self._model,
            [token],
            seq_pos,
            compute_logits=compute_logits,
            sample_next=sample_next,
        )

    def _llm_tokens_step(
        self,
        model,
        tokens: list[int],
        seq_pos: int,
        *,
        compute_logits: bool = True,
        sample_next: bool = True,
        entry: str | None = None,
    ) -> int:
        """Run one or more consecutive tokens through *model*.

        A single token is the usual decode step on the main model's default
        entry. A multi-token chunk is a batched prefill step, on either the
        separate fixed-size prefill model or a native prefill entry of the
        main model (*entry*, e.g. ``main_16``): the chunk's graph expands the
        single *seq_pos* (the chunk's start position) across the whole chunk
        and returns the final position's sampleable output (its hidden state
        routed through the shared LM head, or logits when the head is baked
        in).
        """
        if sample_next and not compute_logits:
            raise ValueError("sample_next=True requires compute_logits=True")

        if entry is not None:
            # Native batch prefill: variable-size chunk on the main model.
            if self._native_prefill_emb_buf is not None:
                self._native_prefill_emb_buf[:1, : len(tokens), :] = (
                    self._token_embeddings[tokens]
                )
                first = self._native_prefill_emb_buf[:1, : len(tokens), :]
            else:
                self._native_prefill_id_buf[:1, : len(tokens)] = tokens
                first = self._native_prefill_id_buf[:1, : len(tokens)]
        elif len(tokens) == 1:
            if self._emb_buf is not None:
                self._emb_buf[0, 0, :] = self._token_embeddings[tokens[0]]
                first = self._emb_buf
            else:
                self._id_buf[0, 0] = tokens[0]
                first = self._id_buf
        else:
            if self._prefill_emb_buf is not None:
                self._prefill_emb_buf[...] = self._token_embeddings[tokens]
                first = self._prefill_emb_buf
            else:
                self._prefill_id_buf[0, :] = tokens
                first = self._prefill_id_buf

        self._pos_buf[0, 0] = seq_pos
        step_inputs = [first, self._pos_buf, *self._extra_step_inputs]

        if not compute_logits and isinstance(model, SplitLMHeadRunner):
            results = model.infer(step_inputs, skip_lm_head=True, entry=entry)
        else:
            results = model.infer(step_inputs, entry=entry)
        self._turn_pos = seq_pos + len(tokens)
        self._cache_tokens[seq_pos:] = tokens
        self._logger.debug("LLM step time: %.3f ms", model.infer_time_ms)

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

    def _setup_sibling_prefill(
        self,
        prefill_path: str | os.PathLike,
        n_threads: int | None,
        runtime_flags: list[str] | None,
        device_io: bool,
    ) -> None:
        """Load the separate fixed-size batched prefill model and its buffers."""
        self._setup_prefill_model(prefill_path, n_threads, runtime_flags, device_io)
        prefill_token_info = self._prefill_model.inputs_info[0]
        if self._token_embeddings is not None:
            self._prefill_emb_buf = np.zeros(
                (1, self._prefill_size, self._token_embeddings.shape[-1]),
                dtype=np.dtype(prefill_token_info.dtype),
            )
        else:
            self._prefill_id_buf = np.zeros(
                prefill_token_info.shape, dtype=np.dtype(prefill_token_info.dtype)
            )

    def _setup_native_prefill(self) -> bool:
        """Enable native batch prefill when the main model exports one.

        The compiler can generate the whole prefill family (``main_4``,
        ``main_16``, …) inside the decode model's own image, sharing its
        weight pool: entries are invoked by name on the already-loaded
        module, so a native family needs no second file and no second
        weight load. Returns True when entries were found.
        """
        body = (
            self._model._body
            if isinstance(self._model, SplitLMHeadRunner)
            else self._model
        )
        names = getattr(body, "function_names", None) or []
        entries = sorted(
            {
                size
                for name in names
                if (size := parse_prefill_entry_size(name))
                and size <= self._max_seq_len
            }
        )
        if not entries:
            return False
        token_info = self._model.inputs_info[0]
        if self._token_embeddings is not None:
            self._native_prefill_emb_buf = np.zeros(
                (1, entries[-1], self._token_embeddings.shape[-1]),
                dtype=np.dtype(token_info.dtype),
            )
        else:
            self._native_prefill_id_buf = np.zeros(
                (1, entries[-1], *token_info.shape[2:]),
                dtype=np.dtype(token_info.dtype),
            )
        self._native_prefill_entries = entries
        self._logger.info(
            "Native batch prefill: entries %s in '%s'",
            ", ".join(f"main_{e}" for e in entries),
            str(self._model.model_path),
        )
        return True

    def _setup_prefill_model(
        self,
        prefill_path: str | os.PathLike,
        n_threads: int | None,
        runtime_flags: list[str] | None,
        device_io: bool,
    ) -> None:
        """Load the fixed-size batched prefill model for prompt chunks."""
        decode_body = (
            self._model._body
            if isinstance(self._model, SplitLMHeadRunner)
            else self._model
        )
        prefill_body = ManagedSelfAttnCacheRunner(
            prefill_path,
            n_threads=n_threads,
            runtime_flags=runtime_flags,
            device_io=device_io,
        )
        prefill_size = self._query_prefill_size(prefill_body)
        if prefill_size > self._max_seq_len:
            raise ValueError(
                f"Batched prefill size {prefill_size} from '{prefill_path}' "
                f"exceeds the model max sequence length {self._max_seq_len}."
            )
        # The prefill model and the decode model have the same per-layer cache
        # layout; use one set of on-device buffers for both.
        prefill_body.share_kv_cache(decode_body)
        self._validate_prefill_inputs(prefill_body, prefill_size)
        self._prefill_size = prefill_size
        self._prefill_model = self._wrap_prefill_model(prefill_body)
        self._logger.info(
            "Loaded %d-token batched prefill model '%s'", prefill_size, str(prefill_path)
        )

    def _wrap_prefill_model(self, prefill_body):
        """Adapt the prefill body so its first output is sampleable logits.

        The decode model's first output is always logits (a fused head, or
        the split head applied on the way out), so the prefill model must
        land in the same space:

        * prefill emits the head's *input* (hidden states) -> route it
          through the standalone head.
        * prefill already emits logits (head baked in at export time) -> use
          it directly, no wrapper.
        """
        prefill_out = prefill_body.outputs_info[0]
        if isinstance(self._model, SplitLMHeadRunner):
            head_in = self._model._lm_head.inputs_info[0]
            if _tensor_signature_matches(prefill_out, head_in):
                self._logger.info(
                    "Prefill model emits hidden states; reusing the decode "
                    "model's compiled LM head."
                )
                return SplitLMHeadRunner(prefill_body, self._model._lm_head)
            if _tensor_signature_matches(prefill_out, self._model.outputs_info[0]):
                self._logger.info(
                    "Prefill model has its LM head baked in; using it directly."
                )
                return prefill_body
            raise ValueError(
                f"Prefill model's first output {_tensor_info(prefill_out)} "
                f"matches neither the LM head input {_tensor_info(head_in)} "
                f"nor the decode model's logits "
                f"{_tensor_info(self._model.outputs_info[0])}; cannot use it "
                "for batched prefill."
            )
        if _tensor_signature_matches(prefill_out, self._model.outputs_info[0]):
            return prefill_body
        raise ValueError(
            "Batched prefill model emits hidden states, but the decode model "
            "has no standalone LM head to sample from. Load the split decode "
            "pair (transformer.vmfb + lm_head.vmfb) or a prefill model with "
            "a fused head."
        )

    def _validate_prefill_inputs(self, prefill_body, prefill_size: int) -> None:
        """Prefill and decode models must take the same per-step inputs.

        The same buffers (positions, extra inputs) are fed to both models
        every step, so everything but the token input's sequence length must
        line up; a mismatch would otherwise surface as an opaque runtime
        shape assertion deep in inference.
        """
        decode_in = self._model.inputs_info
        prefill_in = prefill_body.inputs_info
        n_decode_user = len(decode_in) - self._model.n_cache_inputs
        n_prefill_user = len(prefill_in) - prefill_body.n_cache_inputs
        if n_prefill_user != n_decode_user:
            raise ValueError(
                f"Prefill model has {n_prefill_user} non-cache inputs, the "
                f"decode model has {n_decode_user}; they must match."
            )
        d_tok, p_tok = decode_in[0], prefill_in[0]
        d_shape, d_dtype = _tensor_signature(d_tok)
        p_shape, p_dtype = _tensor_signature(p_tok)
        if d_dtype != p_dtype:
            raise ValueError(
                f"Prefill token input {_tensor_info(p_tok)} does not match "
                f"the decode model's token input {_tensor_info(d_tok)}."
            )
        if (
            len(d_shape) >= 2
            and isinstance(d_shape[1], int)
            and isinstance(p_shape[1], int)
            and (d_shape[1] != 1 or p_shape[1] != prefill_size)
        ):
            raise ValueError(
                f"Prefill token input sequence length {p_shape[1]} does not "
                f"match its declared chunk size {prefill_size} (decode "
                f"model expects 1)."
            )
        if d_shape[2:] != p_shape[2:]:
            raise ValueError(
                f"Prefill token input {_tensor_info(p_tok)} does not match "
                f"the decode model's token input {_tensor_info(d_tok)}."
            )
        for i in range(1, n_decode_user):
            if not _tensor_signature_matches(decode_in[i], prefill_in[i]):
                raise ValueError(
                    f"Prefill non-cache input {i} {_tensor_info(prefill_in[i])} "
                    f"does not match the decode model's "
                    f"{_tensor_info(decode_in[i])}."
                )

    @staticmethod
    def _query_prefill_size(prefill_body) -> int:
        """Fixed prompt chunk size from the prefill model's token input shape."""
        info = prefill_body.inputs_info
        if not info:
            raise ValueError(
                "Prefill model has no input metadata; cannot determine its "
                "fixed prompt chunk size."
            )
        shape = info[0].shape
        if len(shape) < 2 or not isinstance(shape[1], int) or shape[1] < 1:
            raise ValueError(
                f"Cannot determine fixed prefill size from first input shape "
                f"{shape}; expected (1, N, ...)."
            )
        return shape[1]

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

    @staticmethod
    def _plan_prefill_chunks(n_tokens: int, entries: list[int]) -> list[int]:
        """Split *n_tokens* into native prefill chunk sizes.

        Takes the largest entry that fits at each step (the compiler emits
        4, 16, 64, …, so greedy is optimal); leftovers smaller than the
        smallest entry become single-token steps (size 1 = ``main``).
        """
        plan = []
        remaining = n_tokens
        while remaining > 0:
            size = next((e for e in reversed(entries) if e <= remaining), 1)
            plan.append(size)
            remaining -= size
        return plan

    def _prefill(
        self,
        tokens: list[int],
        start: int = 0,
        should_stop: StopCheck | None = None,
        *,
        produce_next_token: bool = True,
    ) -> tuple[int, int]:
        if not tokens:
            # Nothing to feed: return without consuming a cache slot
            return 0, start
        pos = start
        if self._native_prefill_entries:
            # Native batch prefill: chunk the prompt with the entries
            # compiled into the main model; every unit before the final one
            # is body-only (it only extends the KV cache), the final unit
            # samples the first generated token.
            plan = self._plan_prefill_chunks(
                len(tokens), self._native_prefill_entries
            )
            off = 0
            tok = 0
            for i, size in enumerate(plan):
                _raise_if_stopped(should_stop)
                chunk = tokens[off : off + size]
                off += size
                want_logits = produce_next_token and i == len(plan) - 1
                if size == 1:
                    tok = self.llm_step(
                        chunk[0],
                        pos,
                        compute_logits=want_logits,
                        sample_next=want_logits,
                    )
                else:
                    self._logger.debug(
                        "Native prefill: %d tokens at pos %d via main_%d",
                        size,
                        pos,
                        size,
                    )
                    tok = self._llm_tokens_step(
                        self._model,
                        chunk,
                        pos,
                        compute_logits=want_logits,
                        sample_next=want_logits,
                        entry=f"main_{size}",
                    )
                _raise_if_stopped(should_stop)
                pos += size
            return tok, pos
        if self._prefill_model is not None and len(tokens) >= self._prefill_size:
            chunk = self._prefill_size
            full_chunks = len(tokens) // chunk
            remainder = len(tokens) - full_chunks * chunk
            # Every complete chunk before the final unit is body-only: it only
            # extends the shared KV cache, so the LM head is skipped.
            body_only_chunks = full_chunks if remainder else full_chunks - 1
            for i in range(body_only_chunks):
                _raise_if_stopped(should_stop)
                self._llm_tokens_step(
                    self._prefill_model,
                    tokens[i * chunk:(i + 1) * chunk],
                    pos,
                    compute_logits=False,
                    sample_next=False,
                )
                pos += chunk
                _raise_if_stopped(should_stop)
            if remainder:
                # Fall back to single-token decode for any remainder.
                rem_tokens = tokens[full_chunks * chunk:]
                for tok_id in rem_tokens[:-1]:
                    _raise_if_stopped(should_stop)
                    self.llm_step(tok_id, pos, compute_logits=False, sample_next=False)
                    pos += 1
                    _raise_if_stopped(should_stop)
                _raise_if_stopped(should_stop)
                tok = self.llm_step(
                    rem_tokens[-1],
                    pos,
                    compute_logits=produce_next_token,
                    sample_next=produce_next_token,
                )
                _raise_if_stopped(should_stop)
                pos += 1
            else:
                _raise_if_stopped(should_stop)
                tok = self._llm_tokens_step(
                    self._prefill_model,
                    tokens[len(tokens) - chunk:],
                    pos,
                    compute_logits=produce_next_token,
                    sample_next=produce_next_token,
                )
                _raise_if_stopped(should_stop)
                pos += chunk
            return tok, pos
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
        """Prefill tokens at the current cache end; returns (next_token, new_pos)."""
        return self._prefill(tokens, start=self._turn_pos, should_stop=None)

    def _apply_prompt_limit(self, tokens: list[int]) -> list[int]:
        """Truncate the prompt to the configured maximum input length.

        Prompts shorter than the limit pass through unchanged: the static
        exports bake a causal mask into the graph, so padding tokens would
        be attended over and make the model emit EOS immediately.
        """
        limit = (
            self._max_user_tokens
            if self._max_user_tokens is not None
            else self._max_prompt_tokens
        )
        if isinstance(limit, int) and len(tokens) > limit:
            return tokens[:limit]
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
        """Yield decoded text chunks as tokens are generated.

        The KV cache is shared across turns: the conversation so far stays in
        the cache and this prompt is prefilled right after it, so later turns
        see the earlier ones. When the conversation (or this prompt alone) no
        longer fits, the cache rolls: the sliding window keeps the warm-up
        prefix plus up to ``cache_keep_n`` recent tokens, replayed at their new
        positions. Without a window the conversation restarts from the warm-up
        prefix.
        """
        self._n_tokens_gen = 0
        self._last_infer_ns = 0
        self._time_to_first_token_ns = 0

        tokens = self._apply_prompt_limit(self._build_prompt_tokens(user_input))
        self._start_time_ns = time.perf_counter_ns()
        # Sampling predicts a token; its KV entry is written on the next step.
        # Carry the last prediction across normal stops and closed streams.
        pending = [] if self._pending_token is None else [self._pending_token]
        if self._turn_pos + len(pending) + len(tokens) > self._max_seq_len:
            if self._cache_keep_n is not None:
                capacity = self._max_seq_len - self._warmup_len
                # The new prompt takes priority over previous-turn history.
                if len(pending) + len(tokens) > capacity:
                    pending = []
                keep = min(
                    self._cache_keep_n,
                    self._turn_pos - self._warmup_len,
                    max(0, capacity - len(pending) - len(tokens)),
                )
                self._roll_cache(keep, should_stop)
                self._logger.debug(
                    "Circular KV cache at turn start: replayed last %d entries "
                    "after %d protected prefix tokens",
                    keep,
                    self._warmup_len,
                )
            else:
                self.reset()
                pending = []
                self._logger.warning(
                    "KV cache full, starting a fresh conversation"
                )
            room = max(0, self._max_seq_len - self._turn_pos - len(pending))
            if len(tokens) > room:
                tokens = tokens[:room]
                self._logger.warning(
                    "Prompt truncated to fit the KV cache (%d tokens)", len(tokens)
                )

        if not tokens:
            self._logger.warning("No prompt tokens fit in the KV cache")
            return
        tokens = pending + tokens

        gen: list[int] = []
        yield_ns = 0
        try:
            self._pending_token = None
            next_tok, pos = self._prefill(
                tokens,
                start=self._turn_pos,
                should_stop=should_stop,
            )
            self._pending_token = next_tok
            self._time_to_first_token_ns = time.perf_counter_ns() - self._start_time_ns

            prev_text = self._decode_generated_tokens([next_tok])
            yield_start_ns = time.perf_counter_ns()
            yield prev_text
            yield_ns += time.perf_counter_ns() - yield_start_ns

            gen = [next_tok]
            while not self._should_stop(next_tok, gen):
                _raise_if_stopped(should_stop)
                if self._max_gen_tokens is not None and len(gen) >= self._max_gen_tokens:
                    self._logger.info(
                        "Max generation tokens reached (%d)", self._max_gen_tokens
                    )
                    break
                if pos >= self._max_seq_len:
                    if self._cache_keep_n is not None:
                        keep = min(
                            self._cache_keep_n,
                            max(0, self._max_seq_len - self._warmup_len - 1),
                        )
                        self._roll_cache(keep, should_stop)
                        pos = self._turn_pos
                        self._logger.debug(
                            "Circular KV cache: replayed last %d entries after "
                            "%d protected prefix tokens",
                            keep,
                            self._warmup_len,
                        )
                    if pos >= self._max_seq_len:
                        # No window, or the warm-up alone fills the cache.
                        # Another step would write past the last cache slot.
                        self._logger.warning("KV cache full; stopping generation")
                        break
                next_tok = self.llm_step(next_tok, pos)
                self._pending_token = next_tok
                _raise_if_stopped(should_stop)
                gen.append(next_tok)
                pos += 1
                full_text = self._decode_generated_tokens(gen)
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
