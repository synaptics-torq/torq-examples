# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sized
from pathlib import Path
from time import perf_counter_ns
from typing import Final

import ml_dtypes
import numpy as np
from tokenizers import Tokenizer

from utils import ptd as ptd_core
from utils.inference import (
    InferenceInterrupted,
    ManagedPTDCacheRunner,
    ManagedSelfAttnCacheRunner,
    SplitLMHeadRunner,
    StopCheck,
    _raise_if_stopped,
)


_SIDECAR_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".incomplete", ".part", ".tmp", ".swp", ".lock", ".metadata", ".md5", ".sha256"}
)


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


def resolve_prefill_model_path(
    model_path: str | os.PathLike,
    prefill_model_path: str | os.PathLike | None = None,
    *,
    disable_prefill: bool = False,
    logger: logging.Logger | None = None,
) -> Path | None:
    """Resolve explicit, disabled, or auto-discovered prefill model selection."""
    if prefill_model_path is not None and disable_prefill:
        raise ValueError(
            "--batch-prefill-model and --no-batch-prefill-model cannot be used together."
        )
    if prefill_model_path is not None:
        return Path(prefill_model_path)
    if disable_prefill:
        return None

    discovered = discover_prefill_model_path(model_path)
    if discovered is not None and logger is not None:
        logger.info("Auto-discovered batched prefill model '%s'", str(discovered))
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


def _ptd_user_inputs(in_info) -> dict | None:
    """Identify the (tok, pos, vis) user inputs of a PTD body by shape.

    Input names are not reflected by the runtime, so identification is by
    shape: the KV caches are the 4-D [1, H, M, D] inputs (M = the largest
    key dim among the 4-D inputs); the tree_vis input is the 4-D
    [1, 1, R, R-1] input; position_ids the 2-D [1, R] input; the token
    input the remaining [1, R, ...] input (the extracted-embeddings
    layout every PTD export uses). Returns None when no unambiguous
    (token, positions, tree_vis) triple exists.
    """
    m4_idx = [
        j for j, i in enumerate(in_info)
        if len(i.shape) == 4 and i.shape[0] == 1
    ]
    if not m4_idx:
        return None
    kv_len = max(
        (in_info[j].shape[2] for j in m4_idx
         if isinstance(in_info[j].shape[2], int) and in_info[j].shape[2] > 1),
        default=None,
    )
    if kv_len is None:
        return None
    cache = {j for j in m4_idx if in_info[j].shape[2] == kv_len}
    vis = next(
        (
            j for j in m4_idx
            if j not in cache
            and in_info[j].shape[1] == 1
            and isinstance(in_info[j].shape[2], int)
            and in_info[j].shape[2] >= 2
            and in_info[j].shape[3] == in_info[j].shape[2] - 1
        ),
        None,
    )
    if vis is None:
        return None
    R = in_info[vis].shape[2]
    user = [j for j in range(len(in_info)) if j not in cache and j != vis]
    if len(user) != 2:
        return None
    pos = [j for j in user if list(in_info[j].shape) == [1, R]]
    tok = [
        j for j in user
        if j not in pos and in_info[j].shape[:2] == [1, R]
    ]
    if not tok and len(pos) == 2:
        # 2-D ids token input + 2-D positions: the token input comes first.
        tok = [user[0]]
        pos = [j for j in pos if j != user[0]]
    if len(pos) != 1 or len(tok) != 1:
        return None
    return {"tok": in_info[tok[0]], "pos": in_info[pos[0]], "vis": in_info[vis]}


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
        "_n_tokens_gen",
        "_last_infer_ns",
        "_time_to_first_token_ns",
        "_start_time_ns",
        "_prefill_model",
        "_prefill_size",
        "_prefill_emb_buf",
        "_prefill_id_buf",
        "_ptd_model",
        "_ptd_body",
        "_ptd_cache",
        "_ptd_rows",
        "_ptd_width",
        "_ptd_depth",
        "_ptd_pool_max_nodes",
        "_ptd_pool_max_entries",
        "_ptd_min_draft_rows",
        "_ptd_pad_token",
        "_ptd_tok_buf",
        "_ptd_pos_buf",
        "_ptd_vis_buf",
        "_ptd_feed_order",
        "_ptd_tree",
        "_ptd_pool",
        "_ptd_log",
        "_ptd_rounds",
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
        ptd_width: int = 4,
        ptd_depth: int = 6,
        ptd_pool_max_nodes: int = 16,
        ptd_pool_max_entries: int = 64,
        ptd_min_draft_rows: int = 1,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._debug_logging = self._logger.isEnabledFor(logging.DEBUG)

        lm_head_path = resolve_lm_head_path(
            model_path,
            lm_head_path,
            disable_lm_head=disable_lm_head,
            logger=self._logger,
        )
        prefill_path = resolve_prefill_model_path(
            model_path,
            prefill_model_path,
            disable_prefill=disable_prefill,
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

        # PTD state (both modes); _setup_standalone_ptd fills in the rest
        # when the model is a standalone fixed-row PTD body.
        self._ptd_model = None
        self._ptd_body = None
        self._ptd_cache = None
        self._ptd_rows = None
        self._ptd_width = ptd_width
        self._ptd_depth = ptd_depth
        self._ptd_pool_max_nodes = ptd_pool_max_nodes
        self._ptd_pool_max_entries = ptd_pool_max_entries
        self._ptd_min_draft_rows = ptd_min_draft_rows
        self._ptd_pad_token = 0
        self._ptd_tok_buf = None
        self._ptd_pos_buf = None
        self._ptd_vis_buf = None
        self._ptd_tree = None
        self._ptd_pool = None
        self._ptd_log = []
        self._ptd_rounds = 0

        # A standalone PTD export is a fixed-row body that *is* the model:
        # the same file names, but the body takes per-row tokens, positions,
        # and a tree_vis input. Detect it before any single-token-specific
        # validation runs; the setup itself happens after the config, the
        # embedding LUT, and the token ID LUT are loaded.
        if self._is_ptd_body():
            self._check_ptd_compat(prefill_path, temperature, top_p)

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
        self._eos_token_id = self._config["eos_token_id"]
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

        if self._is_ptd_body():
            self._setup_standalone_ptd(n_threads, runtime_flags, device_io)

        self._init_step_buffers()

        self._prefill_model = None
        self._prefill_size = None
        self._prefill_emb_buf = None
        self._prefill_id_buf = None
        if prefill_path is not None:
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
        elif self._ptd_rows is None:
            self._logger.debug(
                "No batched prefill model available; prompts will be "
                "prefilled with single-token decode steps."
            )

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

    def _init_step_buffers(self) -> None:
        """Build the per-step input buffers from the model's input metadata.

        Input 0 is the token input (embedding vector or token id), input 1 is
        ``position_ids``; both shapes and dtypes are taken from the model's
        reflection data rather than assumed. Any further non-cache inputs
        (e.g. the fixed ``attention_mask`` of newer static exports) are
        constant per step and are built by :meth:`_build_extra_step_inputs`.
        """
        if getattr(self, "_ptd_rows", None) is not None:
            # The standalone PTD path feeds its own per-row buffers
            # (_ptd_tok_buf/_ptd_pos_buf/_ptd_vis_buf); the single-token
            # buffers are unused.
            self._emb_buf = self._id_buf = self._pos_buf = None
            self._extra_step_inputs = []
            return
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
        if getattr(self, "_ptd_rows", None) is not None or self._is_ptd_body():
            return  # a standalone PTD body is fixed-row by design
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
        """Extract max sequence length from a KV cache input shape.

        Input names are not reflected by the runtime: the KV caches are the
        4-D [1, H, M, D] inputs, so M is the largest key dim among the 4-D
        inputs (the PTD body's tree_vis input [1, 1, R, R-1] is smaller).
        """
        info = self._model.inputs_info
        if info is None:
            return None
        kv_len = None
        for i in info:
            s = i.shape
            if (
                len(s) == 4
                and s[0] == 1
                and isinstance(s[2], int)
                and s[2] > 1
                and (kv_len is None or s[2] > kv_len)
            ):
                kv_len = s[2]
        return kv_len

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
    ) -> int:
        """Run one or more consecutive tokens through *model*.

        A single token is the usual decode step on the main model. A
        fixed-size chunk is a batched prefill step: the prefill model's graph
        expands the single *seq_pos* (the chunk's start position) across the
        whole chunk and returns the final position's sampleable output (its
        hidden state routed through the shared LM head, or logits when the
        prefill model has the head baked in).
        """
        if sample_next and not compute_logits:
            raise ValueError("sample_next=True requires compute_logits=True")

        if len(tokens) == 1:
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

        if not compute_logits:
            if isinstance(model, SplitLMHeadRunner):
                model.infer(step_inputs, skip_lm_head=True)
            else:
                model.infer(step_inputs)
            self._logger.debug("LLM step time: %.3f ms", model.infer_time_ms)
            return 0

        results = model.infer(step_inputs)
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

    # -- PTD (Progressive Tree Drafting) ------------------------------------
    #
    # Standalone speculative decode: the fixed-row body (the ``--ptd-rows``
    # ONNX export of the torq pipeline, compiled to ``transformer.vmfb`` +
    # ``lm_head.vmfb``) is the model for every phase. Prompt prefill packs
    # causal chains into the row budget; every generation round runs the
    # anchor plus packed draft/verification tree rows in one forward. PTD
    # greedy output is token-identical to ordinary greedy decoding (anchor
    # row = the ordinary step; accepted continuations are the ordinary
    # greedy predictions for their prefix; the stop rule is scanned per
    # emitted token, including the padded anchor-only pass near cache
    # capacity).
    #
    # Progressive Tree Drafting: Gao et al. 2026,
    # https://arxiv.org/abs/2607.10661 (attribution and static-graph
    # adaptations: utils/ptd.py).

    def _check_ptd_compat(
        self,
        prefill_path: str | os.PathLike | None,
        temperature: float,
        top_p: float,
    ) -> None:
        """Standalone PTD bodies are incompatible with a batched prefill model
        (they prefill themselves) and with sampling (PTD verification is
        greedy-only)."""
        if prefill_path is not None:
            raise ValueError(
                "The model is a standalone PTD body: it prefills prompts "
                "itself with packed causal chains, so a batched prefill "
                "model (--batch-prefill-model) cannot be used with it."
            )
        if temperature != 0.0 or top_p != 1.0:
            raise ValueError(
                "PTD supports greedy decoding only (temperature=0, top_p=1)."
            )

    def _is_ptd_body(self) -> bool:
        """Whether the main model is a standalone fixed-row PTD body.

        A PTD body's user inputs (the non-KV-cache ones) are exactly
        (tokens [1, R, ...], position_ids [1, R], tree_vis [1, 1, R, R-1])
        with R >= 2; ordinary decode/prefill bodies have two user inputs
        and a one-token sequence dimension.
        """
        body = self._model._body if isinstance(self._model, SplitLMHeadRunner) else self._model
        in_info = body.inputs_info
        if not in_info or len(in_info) < 4:
            return False
        user = _ptd_user_inputs(in_info)
        if user is None:
            return False
        tok_shape = user["tok"].shape
        return (
            len(tok_shape) >= 2
            and isinstance(tok_shape[1], int)
            and tok_shape[1] >= 2
        )

    def _setup_standalone_ptd(
        self,
        n_threads: int | None,
        runtime_flags: list[str] | None,
        device_io: bool,
    ) -> None:
        """Adopt the PTD execution path: the body does prompt prefill and
        every generation round, so it is swapped for a cache-owning runner
        and per-row input buffers are built."""
        body = self._model._body if isinstance(self._model, SplitLMHeadRunner) else self._model
        ptd_body = ManagedPTDCacheRunner(
            body.model_path,
            n_threads=n_threads,
            runtime_flags=runtime_flags,
            device_io=device_io,
        )
        user = _ptd_user_inputs(ptd_body.inputs_info)
        if user is None:
            raise ValueError(
                f"PTD body '{ptd_body.model_path}' has no identifiable "
                "(token, position_ids, tree_vis) user inputs."
            )
        tok_info, pos_info, vis_info = user["tok"], user["pos"], user["vis"]
        rows = self._ptd_rows = tok_info.shape[1]
        # User-input order in the model (the caches are interleaved among
        # them); the feed helper expands this to the full input list.
        self._ptd_feed_order = [
            key
            for i in ptd_body.inputs_info
            for key in ("vis", "pos", "tok")
            if i is user[key]
        ]
        self._validate_ptd_outputs(ptd_body, rows)
        # The PTD loop owns the cache-update policy (scratch promotion on
        # hits), so the body must not roll the present caches back itself.
        if isinstance(self._model, SplitLMHeadRunner):
            # Reuse the same compiled head: the split head projects the full
            # [1, rows, hidden] stack in one call.
            self._model = SplitLMHeadRunner(ptd_body, self._model._lm_head)
        else:
            self._model = ptd_body
        self._ptd_body = ptd_body
        self._ptd_cache = ptd_body._kv_cache
        self._ptd_model = self._model

        tok_shape = tuple(tok_info.shape)
        if self._token_embeddings is not None:
            emb_hidden = self._token_embeddings.shape[-1]
            if (
                len(tok_shape) >= 3
                and isinstance(tok_shape[-1], int)
                and tok_shape[-1] != emb_hidden
            ):
                raise ValueError(
                    f"token_embeddings.npy has hidden size {emb_hidden} but the "
                    f"PTD body's token input expects {tok_shape[-1]} "
                    f"(shape {tok_shape})."
                )
            self._ptd_tok_buf = np.zeros(
                (1, rows, emb_hidden),
                dtype=np.dtype(tok_info.dtype),
            )
        else:
            self._ptd_tok_buf = np.zeros(tok_shape, dtype=np.dtype(tok_info.dtype))
        self._ptd_pos_buf = np.zeros(pos_info.shape, dtype=np.dtype(pos_info.dtype))
        self._ptd_vis_buf = np.zeros(vis_info.shape, dtype=np.dtype(vis_info.dtype))
        self._ptd_pad_token = self._config.get("pad_token_id") or 0
        self._logger.info(
            "Standalone PTD mode: '%s' is the model for prefill and decode "
            "(%d rows, width=%d, depth=%d)",
            str(body.model_path), rows, self._ptd_width, self._ptd_depth,
        )

    def _validate_ptd_outputs(self, ptd_body, rows: int) -> None:
        """Standalone PTD body outputs: (first, committed per layer, combined
        key/value scratch per layer), with the committed layout of the body's
        own cache inputs."""
        n_layers = ptd_body.n_cache_inputs
        out_info = ptd_body.outputs_info
        if len(out_info) != 1 + 2 * n_layers:
            raise ValueError(
                f"PTD body '{ptd_body.model_path}' output layout is not "
                "(first, committed per layer, combined key/value scratch per "
                f"layer): {len(out_info)} outputs for {n_layers} layers. Rebuild "
                "with the current torq-tools PTD export (combined scratch K/V)."
            )
        cache_input = ptd_body.inputs_info[ptd_body._cache_input_idx[0]]
        cache_sig = _tensor_signature(cache_input)
        for i in range(1, 1 + n_layers):
            if _tensor_signature(out_info[i]) != cache_sig:
                raise ValueError(
                    f"PTD body committed output {i} {_tensor_info(out_info[i])} "
                    f"does not match the cache layout {_tensor_info(cache_input)}."
                )
        scratch_sig = (
            (1, cache_sig[0][1], rows - 1, cache_sig[0][-1]),
            cache_sig[1],
        )
        for i in range(1 + n_layers, 1 + 2 * n_layers):
            if _tensor_signature(out_info[i]) != scratch_sig:
                raise ValueError(
                    f"PTD body scratch output {i} {_tensor_info(out_info[i])} "
                    f"does not match the expected {scratch_sig[0]}."
                )

    def _ptd_reset_request_state(self) -> None:
        """Per-request PTD state: a fresh draft tree; the candidate pool
        warms up from the first round (same request-local semantics as the
        torq-tools PTD runner)."""
        if self._ptd_model is None:
            return
        self._ptd_tree = ptd_core.PTDTree(
            width=self._ptd_width, max_depth=self._ptd_depth
        )
        self._ptd_pool = ptd_core.PTDPool(
            max_entry_nodes=self._ptd_pool_max_nodes,
            max_entries=self._ptd_pool_max_entries,
            width=self._ptd_width,
        )
        self._ptd_log = []
        self._ptd_rounds = 0

    def _ptd_fill_inputs(self, packing) -> None:
        if self._token_embeddings is not None:
            self._ptd_tok_buf[0] = self._token_embeddings[packing.tokens]
        else:
            self._ptd_tok_buf[0] = packing.tokens
        self._ptd_pos_buf[0] = packing.positions
        self._ptd_vis_buf[0, 0] = packing.vis

    def _ptd_feed(self) -> list:
        """The PTD user inputs in the body's model input order (the
        cache-owning runner reinserts the managed caches)."""
        return [
            {"tok": self._ptd_tok_buf, "pos": self._ptd_pos_buf, "vis": self._ptd_vis_buf}[k]
            for k in self._ptd_feed_order
        ]

    def _ptd_prefill(
        self,
        tokens: list[int],
        start: int,
        should_stop: StopCheck | None,
        produce_next_token: bool,
    ) -> tuple[int, int]:
        """Prompt prefill with packed causal chains: chunk the prompt into
        row-budget chains (anchor = the chunk's first token), promote every
        real row into the committed cache, and run the head only on the
        final chain (and only when its token is wanted)."""
        rows = self._ptd_rows
        if start + len(tokens) > self._max_seq_len:
            raise ValueError("PTD prompt exceeds the remaining cache capacity")
        if not tokens:
            return 0, start
        pos = start
        next_token = 0
        for offset in range(0, len(tokens), rows):
            chunk = tokens[offset:offset + rows]
            nodes = []
            for depth, token in enumerate(chunk[1:], start=1):
                nodes.append(ptd_core.PTDNode(token, depth, parent=nodes[-1] if nodes else None))
            packing = ptd_core.pack_round(
                pos, chunk[0], nodes, [], rows, self._ptd_pad_token
            )
            self._ptd_fill_inputs(packing)
            want_logits = offset + len(chunk) == len(tokens) and produce_next_token
            step_inputs = self._ptd_feed()
            if isinstance(self._ptd_model, SplitLMHeadRunner):
                results = self._ptd_model.infer(step_inputs, skip_lm_head=not want_logits)
            else:
                results = self._ptd_model.infer(step_inputs)
            self._ptd_commit_cache(results, packing, nodes, pos)
            pos += len(chunk)
            if want_logits:
                first = self._host(results[0])
                row = np.asarray(first).reshape(rows, -1)[len(chunk) - 1]
                compact = int(np.argmax(row))
                next_token = (
                    int(self._token_id_lut[compact])
                    if self._token_id_lut is not None
                    else compact
                )
            _raise_if_stopped(should_stop)
        return next_token, pos

    def _ptd_round(
        self, anchor_token: int, L: int, gen: list[int], max_emit: int | None
    ) -> tuple[list[int], bool]:
        """One PTD round. Returns (emitted tokens, stopped) and commits the
        round's KVs into the body's cache. Near capacity the round shrinks to
        as many rows as fit (anchor-only at L = M-1); there is no ordinary
        decode fallback."""
        rows = self._ptd_rows
        free = min(rows, self._max_seq_len - L) - 1

        # 1. Retrieve candidates (key = this round's anchor = the last emitted
        # token) and allocate the row budget: candidates first, keeping at
        # least one draft row while the draft tree is non-empty.
        candidate_tree = self._ptd_pool.retrieve(anchor_token) if self._ptd_pool else None
        cand_nodes = candidate_tree.bfs_nodes() if candidate_tree is not None else []
        draft_nodes = self._ptd_tree.bfs_nodes()
        if draft_nodes and self._ptd_min_draft_rows > 0:
            n_cand = min(len(cand_nodes), free - self._ptd_min_draft_rows)
        else:
            n_cand = min(len(cand_nodes), free)
        n_draft = min(len(draft_nodes), free - max(n_cand, 0))
        cand_nodes = cand_nodes[: max(n_cand, 0)]
        draft_nodes = draft_nodes[:n_draft]

        # 2. Pack the round (slot 0 = anchor; candidate rows; draft rows;
        # invalid slots padded with zero visibility).
        packing = ptd_core.pack_round(
            L, anchor_token, cand_nodes, draft_nodes, rows, self._ptd_pad_token
        )

        # 3. Body (+ block head): per-row tokens.
        self._ptd_fill_inputs(packing)
        t0 = perf_counter_ns()
        results = self._ptd_model.infer(self._ptd_feed())
        body_ms = (perf_counter_ns() - t0) / 1e6
        first = self._host(results[0])
        row_tokens = np.argmax(np.asarray(first).reshape(rows, -1), axis=-1)
        if self._token_id_lut is not None:
            row_tokens = self._token_id_lut[row_tokens]

        # 4. Greedy verification. The stop adapter matches ordinary greedy:
        # _should_stop expects the candidate included as the last element of
        # the generated-token list; the PTD scan hands over the tokens
        # committed *before* the candidate. The accepted budget is the lesser
        # of the cache room and the --max-gen-tokens remainder.
        max_accepted = min(free, (max_emit - 1) if max_emit is not None else free)
        result = ptd_core.greedy_verify(
            row_tokens,
            cand_nodes,
            packing.candidate_slots,
            gen,
            lambda token, committed: self._should_stop(token, list(committed) + [token]),
            max_accepted,
        )

        # 5. Commit: the present caches carry the anchor at L; promote the
        # accepted path's scratch rows to L+1.. before rolling them into the
        # cache.
        self._ptd_commit_cache(results, packing, result.accepted, L)

        # 6. Evolve the draft tree and pool (bounded, request-local):
        # packed draft nodes grow their argmax children, the tree steps and
        # re-anchors on the round's first emitted token, and the pool absorbs
        # the packed nodes' subtrees keyed by token value.
        for node in packing.draft_slots:
            self._ptd_tree.add_child(node, int(row_tokens[packing.draft_slots[node]]))
        self._ptd_tree.step_and_prune()
        self._ptd_tree.reroot(result.emitted[0])
        for node in packing.draft_slots:
            self._ptd_pool.merge_node(node)

        self._ptd_rounds += 1
        self._ptd_log.append({
            "round": self._ptd_rounds,
            "L": L,
            "rows": rows,
            "n_candidates": packing.n_candidates,
            "n_drafts": packing.n_drafts,
            "emitted": len(result.emitted),
            "accepted": len(result.accepted),
            "hit": result.hit,
            "pool_entries": len(self._ptd_pool) if self._ptd_pool else 0,
            "body_ms": body_ms,
        })
        return result.emitted, result.stopped

    def _ptd_commit_cache(self, results: list, packing, accepted: list, L: int) -> None:
        """Roll the round's present caches into the body's KV cache.

        Miss (no accepted nodes): the present outputs *are* the new committed
        state (only row L changed), so they replace the cache entries in
        place — no host transfer. Hit: the accepted path's combined scratch
        rows are spliced into rows L+1.. on the host, then the merged cache
        is uploaded (the only per-round cache transfer, and only on hits).
        """
        n_layers = self._ptd_body.n_cache_inputs
        if not accepted:
            for i in range(n_layers):
                self._ptd_cache[i] = results[1 + i]
            return
        for l in range(n_layers):
            committed = self._host(results[1 + l])
            scratch = self._host(results[1 + n_layers + l])
            for j, node in enumerate(accepted):
                s = packing.candidate_slots[node] - 1  # scratch row index
                committed[0, :, L + 1 + j, :] = scratch[0, :, s, :]
            self._ptd_cache[l] = self._ptd_body.allocate_device_array(committed)

    @staticmethod
    def _host(x):
        """Host view of an inference output (DeviceArray -> NumPy)."""
        return x.to_host() if hasattr(x, "to_host") else np.asarray(x)

    @property
    def ptd_stats(self) -> dict | None:
        """Per-answer PTD summary (None when no PTD round ran)."""
        if not self._ptd_log:
            return None
        rounds = len(self._ptd_log)
        hits = sum(1 for e in self._ptd_log if e["hit"])
        accept = sum(e["accepted"] for e in self._ptd_log)
        return {
            "rounds": rounds,
            "committed": sum(e["emitted"] for e in self._ptd_log),
            "hit_rate": hits / rounds,
            "avg_accept": accept / rounds,
        }

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
        if self._ptd_rows is not None:
            return self._ptd_prefill(tokens, start, should_stop, produce_next_token)
        pos = start
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
        return self._prefill(
            tokens,
            start=self._warmup_len,
            should_stop=None,
        )

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
            self._ptd_reset_request_state()
            while not self._should_stop(next_tok, gen):
                _raise_if_stopped(should_stop)
                if self._max_gen_tokens is not None and len(gen) >= self._max_gen_tokens:
                    self._logger.info(
                        "Max generation tokens reached (%d)", self._max_gen_tokens
                    )
                    break
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
                remaining = (
                    self._max_gen_tokens - len(gen)
                    if self._max_gen_tokens is not None
                    else None
                )
                # Standalone PTD: every step is a PTD round; near capacity
                # the round shrinks to the rows that fit (anchor-only at the
                # last slot), so there is no plain-step fallback.
                if self._ptd_model is not None:
                    emitted, stopped = self._ptd_round(next_tok, pos, gen, remaining)
                    next_tok = emitted[-1]
                    for tok in emitted:
                        _raise_if_stopped(should_stop)
                        gen.append(tok)
                        pos += 1
                        full_text = self._tokenizer.decode(gen)
                        chunk = full_text[len(prev_text) :]
                        yield_start_ns = time.perf_counter_ns()
                        yield chunk
                        yield_ns += time.perf_counter_ns() - yield_start_ns
                        prev_text = full_text
                    if stopped:
                        break
                    continue
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
