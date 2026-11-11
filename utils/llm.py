# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

"""Chat LLMs on the Torq NPU: ``LLM`` (turns, warm-up prefix, rolling KV
window, text streaming) on top of ``utils.decoder.Decoder``, and
``ChatTemplate``, the data that tells chat formats apart.
"""

import json
import logging
import os
import time
from collections.abc import Iterator, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from tokenizers import Tokenizer

from utils.cache import KVCache
from utils.decoder import Decoder, generate, sample
from utils.inference import ContextOverflow, StopCheck
from utils.runtime import Model, RuntimeConfig, load_lut


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



@dataclass(frozen=True)
class ChatTemplate:
    """How an instruct model's chat turns are written.

    ``turn`` formats one message from ``{role}`` and ``{text}``;
    ``generation_prompt`` opens the model's answer. ``bos_prefix`` starts the
    warm-up (system prompt) with the BOS token, ``strip_bos`` drops a BOS the
    tokenizer prepends to every encode, and ``stop_tokens`` end an answer in
    addition to the config's ``eos_token_id``.
    """

    turn: str
    generation_prompt: str
    bos_prefix: bool = False
    strip_bos: bool = False
    stop_tokens: tuple[str, ...] = ()


GEMMA = ChatTemplate(
    turn="<start_of_turn>{role}\n{text}<end_of_turn>\n",
    generation_prompt="<start_of_turn>model\n",
    bos_prefix=True,
    strip_bos=True,
    stop_tokens=("<end_of_turn>",),
)
CHATML = ChatTemplate(
    turn="<|im_start|>{role}\n{text}<|im_end|>\n",
    generation_prompt="<|im_start|>assistant\n",
    stop_tokens=("<|im_end|>", "<|endoftext|>"),
)


class LLM:
    """A chat LLM: a ``Decoder`` plus conversation state.

    The model files next to *model_path* are picked up automatically: a split
    LM head (``lm_head.vmfb``), a batched prefill model
    (``transformer_prefill.vmfb``), the CPU embedding table
    (``token_embeddings.npy``), the trimmed-vocabulary map
    (``token_id_lut.npy``), ``config.json`` and ``tokenizer.json``.

    *template* is the model family's chat format. With *chat* (an instruct
    model) prompts are wrapped in chat turns and the system prompt is
    prefilled once as a warm-up prefix that every conversation reuses.
    Otherwise (a base model) the text is completed as is, and an answer also
    stops at a blank line; the template's stop tokens apply either way.

    The KV cache is shared across turns, so later turns see the earlier ones.
    When the conversation no longer fits, the cache rolls: it keeps the
    warm-up prefix plus the last *cache_keep_n* tokens, replayed at their new
    positions (cached keys carry positional encodings, so moving them is not
    enough). Without a window the conversation restarts after the prefix.
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        runtime: RuntimeConfig,
        *,
        template: ChatTemplate | None = None,
        chat: bool = True,
        sys_prompt: str | None = None,
        max_seq_len: int | None = None,
        max_prompt_tokens: int | None = None,
        cache_keep_n: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 64,
        max_gen_tokens: int | None = None,
        max_cache_rolls: int | None = None,
        truncate_prompts: bool = True,
        lm_head_path: str | os.PathLike | None = None,
        disable_lm_head: bool = False,
        prefill_model_path: str | os.PathLike | None = None,
        disable_prefill: bool = False,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._template = template
        self._chat = chat and template is not None
        self._sys_prompt = sys_prompt
        self._max_prompt_tokens = max_prompt_tokens
        self._max_user_tokens = None
        self._max_gen_tokens = max_gen_tokens
        self._max_cache_rolls = max_cache_rolls
        self._truncate_prompts = truncate_prompts
        self._rolls = 0  # mid-answer window rolls in the current answer
        self._cache_keep_n = cache_keep_n
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k

        model_dir = Path(model_path).parent
        with open(model_dir / "config.json") as f:
            self._config = json.load(f)
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._bos_token_id = self._config["bos_token_id"]
        # Checkpoints may list several stop IDs (FunctionGemma: [1, 50]).
        eos = self._config["eos_token_id"]
        self._eos_token_ids = eos if isinstance(eos, list) else [eos]
        stop_ids = [self._tokenizer.token_to_id(t) for t in (template.stop_tokens if template else ())]
        self._stop_ids = set(self._eos_token_ids) | {i for i in stop_ids if i is not None}
        self._nl_token_id = self._tokenizer.encode("\n").ids[-1]
        self._double_nl_token_id = self._tokenizer.encode("\n\n").ids[-1]

        self._decoder, self._cache = self._load_decoder(
            model_path, model_dir, runtime, lm_head_path, disable_lm_head,
            prefill_model_path, disable_prefill,
        )
        self._max_seq_len = self._resolve_max_seq_len(max_seq_len, self._decoder.seq_len)

        self._pending_token = None  # predicted, but not yet in the KV cache
        self._cache_tokens: list[int] = []  # the tokens in the KV cache, in order
        self._turn_pos = 0
        self._warmup_len = self._warmup()
        self._warmup_state = self._cache.snapshot() if self._warmup_len else None
        self._turn_pos = self._warmup_len
        if cache_keep_n is not None and self._warmup_len + cache_keep_n >= self._max_seq_len:
            self._logger.warning(
                "kv-cache-window (%d) plus warm-up (%d) fills the %d-token KV "
                "cache; fewer history tokens will be retained to make room "
                "for new tokens.",
                cache_keep_n, self._warmup_len, self._max_seq_len,
            )

        self._n_tokens_gen = 0
        self._last_infer_ns = 0
        self._time_to_first_token_ns = 0
        self._logger.info("Loaded model '%s'", str(model_path))
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.warning("DEBUG logging enabled: inference time metrics may be inflated")

    def _load_decoder(
        self, model_path, model_dir, runtime, lm_head_path, disable_lm_head,
        prefill_model_path, disable_prefill,
    ) -> tuple[Decoder, KVCache]:
        head_path = resolve_lm_head_path(
            model_path, lm_head_path, disable_lm_head=disable_lm_head, logger=self._logger
        )
        prefill_path = resolve_prefill_model_path(
            model_path, prefill_model_path, disable_prefill=disable_prefill, logger=self._logger
        )
        body = Model(model_path, runtime, device_outputs=True)
        head = Model(head_path, runtime, device_outputs=True) if head_path else None
        prefill = Model(prefill_path, runtime, device_outputs=True) if prefill_path else None

        emb_path = model_dir / "token_embeddings.npy"
        lut_path = model_dir / "token_id_lut.npy"
        logits = (head or body).outputs_info
        logits_size = logits[0].shape[-1] if logits and isinstance(logits[0].shape[-1], int) else None
        vocab_lut = resolve_token_id_lut(
            logits_size,
            self._config.get("vocab_size"),
            np.load(lut_path) if lut_path.exists() else None,
            self._logger,
        )
        if vocab_lut is not None:
            self._logger.info("Loaded token ID LUT (%d entries) for trimmed vocab remap", len(vocab_lut))

        cache = KVCache(body)
        decoder = Decoder(
            body, cache, head=head, prefill=prefill,
            embed_lut=load_lut(emb_path) if emb_path.exists() else None,
            vocab_lut=vocab_lut,
        )
        if prefill is not None:
            self._logger.info(
                "Loaded %d-token batched prefill model '%s'", decoder.prefill_size, str(prefill_path)
            )
        return decoder, cache

    def _resolve_max_seq_len(self, requested: int | None, model_seq_len: int | None) -> int:
        if model_seq_len is None:
            if requested is None:
                raise ValueError(
                    "Cannot determine max_seq_len: model has no reflection metadata. "
                    "Pass max_seq_len explicitly."
                )
            return requested
        if requested is not None and requested != model_seq_len:
            self._logger.warning(
                "max_seq_len=%d does not match model KV cache dim=%d; using %d",
                requested, model_seq_len, model_seq_len,
            )
        return model_seq_len

    @property
    def is_instruct_model(self) -> bool:
        return self._chat

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    @property
    def prefill_size(self) -> int | None:
        """Fixed prompt chunk size when a batched prefill model is loaded."""
        return self._decoder.prefill_size

    @property
    def last_infer_time(self) -> float:
        return self._last_infer_ns / 1e6

    @property
    def time_to_first_token(self) -> float:
        return self._time_to_first_token_ns / 1e6

    @property
    def generated_tokens(self) -> int:
        return self._n_tokens_gen

    # ── Text <-> tokens (the template is data; these are the hooks) ──────────

    def tokenize(self, text: str, role: str | None = None) -> list[int]:
        """Token ids of *text*; with a *role*, of one chat turn.

        The role ``"model"`` / ``"assistant"`` gives the generation prompt
        that opens an answer (*text* is ignored).
        """
        if not self._chat or role is None:
            return self._tokenizer.encode(text).ids
        if role in ("model", "assistant"):
            text = self._template.generation_prompt
        else:
            text = self._template.turn.format(role=role, text=text)
        ids = self._tokenizer.encode(text).ids
        if self._template.strip_bos and ids and ids[0] == self._bos_token_id:
            ids = ids[1:]
        return ids

    def _build_prompt_tokens(self, user_input: str) -> list[int]:
        if not self._chat:
            return self.tokenize(user_input)
        return self.tokenize(user_input, "user") + self.tokenize("", "model")

    def _build_warmup_tokens(self) -> list[int]:
        if not self._chat:
            return []
        bos = [self._bos_token_id] if self._template.bos_prefix else []
        return bos + self.tokenize(self._sys_prompt or "", "system")

    def _should_stop(self, token: int, gen: list[int]) -> bool:
        if token in self._stop_ids:
            return True
        if not self._chat and len(gen) > 2:
            # A base model has no end-of-turn token: stop at a blank line.
            if token == self._double_nl_token_id:
                return True
            return all(t == self._nl_token_id for t in gen[-2:])
        return False

    def _decode_generated_tokens(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def _apply_prompt_limit(self, tokens: list[int]) -> list[int]:
        """Truncate the prompt to the configured maximum input length.

        Shorter prompts pass through unchanged: the static exports bake a
        causal mask into the graph, so padding tokens would be attended over
        and make the model emit EOS immediately. Without *truncate_prompts*,
        a prompt that would be cut (here or to fit the KV cache) raises
        ``ContextOverflow`` instead.
        """
        limit = self._max_user_tokens if self._max_user_tokens is not None else self._max_prompt_tokens
        if isinstance(limit, int) and len(tokens) > limit:
            if not self._truncate_prompts:
                raise ContextOverflow(f"prompt of {len(tokens)} tokens exceeds --max-inp-len {limit}")
            return tokens[:limit]
        capacity = self._max_seq_len - self._warmup_len
        if not self._truncate_prompts and len(tokens) > capacity:
            raise ContextOverflow(f"prompt of {len(tokens)} tokens exceeds the {capacity}-token KV cache")
        return tokens

    # ── KV cache bookkeeping ─────────────────────────────────────────────────

    def _next_token(self, logits: np.ndarray) -> int:
        index = sample(logits, self._temperature, self._top_p, self._top_k)
        return self._decoder.token_id(index)

    def _prefill(
        self, tokens: list[int], should_stop: StopCheck | None = None, *, want_next: bool = True
    ) -> int | None:
        """Prefill *tokens* at the end of the cache; the next token, or None."""
        start = self._turn_pos
        logits = self._decoder.prefill(tokens, start, want_logits=want_next, should_stop=should_stop)
        self._cache_tokens[start:] = tokens
        self._turn_pos = start + len(tokens)
        return self._next_token(logits) if want_next and tokens else None

    def _warmup(self) -> int:
        tokens = self._build_warmup_tokens()
        if not tokens:
            return 0
        self._logger.info("Warm-up started...")
        if isinstance(self._max_prompt_tokens, int):
            tokens = tokens[: self._max_prompt_tokens]
            self._max_user_tokens = max(0, self._max_prompt_tokens - len(tokens))
        self._prefill(tokens, want_next=False)
        self._logger.info(
            "Warm-up complete: system prompt consumed %d tokens, remaining capacity is %d tokens",
            len(tokens), self._max_seq_len - len(tokens),
        )
        return len(tokens)

    def _reset_cache(self) -> None:
        if self._warmup_state is not None:
            self._cache.restore(self._warmup_state)
        else:
            self._cache.reset()
        self._turn_pos = self._warmup_len
        del self._cache_tokens[self._warmup_len:]

    def reset(self) -> None:
        """Back to the post-warm-up state (a fresh conversation)."""
        self._reset_cache()
        self._pending_token = None

    def _roll_cache(self, keep: int, should_stop: StopCheck | None) -> None:
        """Keep the warm-up prefix plus the last *keep* tokens, replayed at
        their new positions."""
        tokens = self._cache_tokens[-keep:] if keep else []
        self._reset_cache()
        self._prefill(tokens, should_stop, want_next=False)

    def _on_cache_full(self, should_stop: StopCheck | None) -> int | None:
        """Roll the window mid-answer; the new position, or None to stop."""
        if self._cache_keep_n is not None:
            if self._max_cache_rolls is not None and self._rolls >= self._max_cache_rolls:
                # A greedy model stuck in a loop never emits EOS; without this
                # bound the window would roll on indefinitely.
                self._logger.warning("KV window rolled %d times; stopping generation", self._rolls)
                return None
            self._rolls += 1
            keep = min(self._cache_keep_n, max(0, self._max_seq_len - self._warmup_len - 1))
            self._roll_cache(keep, should_stop)
            self._logger.debug(
                "Circular KV cache: replayed last %d entries after %d protected prefix tokens",
                keep, self._warmup_len,
            )
        if self._turn_pos >= self._max_seq_len:
            # No window, or the warm-up alone fills the cache.
            self._logger.warning("KV cache full; stopping generation")
            return None
        return self._turn_pos

    def prefill_tokens(self, tokens: list[int]) -> tuple[int, int]:
        """Prefill tokens at the current cache end; returns (next_token, new_pos)."""
        return self._prefill(tokens), self._turn_pos

    # ── Answers ──────────────────────────────────────────────────────────────

    def run(self, user_input: str, should_stop: StopCheck | None = None) -> str:
        return "".join(self.run_stream(user_input, should_stop=should_stop))

    def run_stream(self, user_input: str, should_stop: StopCheck | None = None) -> Iterator[str]:
        """Yield the answer as decoded text chunks.

        The time spent in the caller between chunks is not counted in
        ``last_infer_time``.
        """
        self._n_tokens_gen = 0
        self._last_infer_ns = 0
        self._time_to_first_token_ns = 0
        self._rolls = 0
        tokens = self._apply_prompt_limit(self._build_prompt_tokens(user_input))
        start_ns = time.perf_counter_ns()
        # The last predicted token goes into the cache first.
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
                    "after %d protected prefix tokens", keep, self._warmup_len,
                )
            else:
                self.reset()
                pending = []
                self._logger.warning("KV cache full, starting a fresh conversation")
            room = max(0, self._max_seq_len - self._turn_pos - len(pending))
            if len(tokens) > room:
                tokens = tokens[:room]
                self._logger.warning("Prompt truncated to fit the KV cache (%d tokens)", len(tokens))
        if not tokens:
            self._logger.warning("No prompt tokens fit in the KV cache")
            return

        gen: list[int] = []
        yield_ns = 0
        text = ""
        try:
            self._pending_token = None
            first = self._prefill(pending + tokens, should_stop)
            self._time_to_first_token_ns = time.perf_counter_ns() - start_ns
            for token in generate(
                self._decoder, first, self._turn_pos,
                next_token=self._next_token,
                stop=self._should_stop,
                max_new=self._max_gen_tokens,
                on_full=lambda pos: self._on_cache_full(should_stop),
                should_stop=should_stop,
            ):
                if gen:  # the step that produced *token* wrote the previous one
                    self._cache_tokens.append(gen[-1])
                    self._turn_pos += 1
                gen.append(token)
                self._pending_token = token
                full = self._decode_generated_tokens(gen)
                chunk, text = full[len(text):], full
                t = time.perf_counter_ns()
                yield chunk
                yield_ns += time.perf_counter_ns() - t
            if self._max_gen_tokens is not None and len(gen) >= self._max_gen_tokens:
                self._logger.info("Max generation tokens reached (%d)", self._max_gen_tokens)
        finally:
            self._n_tokens_gen = max(0, len(gen) - 1)
            self._last_infer_ns = time.perf_counter_ns() - start_ns - yield_ns
