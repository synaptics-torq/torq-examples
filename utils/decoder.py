# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""The standalone decoding core: single-token steps, batched prompt prefill,
sampling, and the autoregressive loop.

``Decoder`` runs a single-token decode model (a ``utils.runtime.Model``)
against a shared ``utils.cache.KVCache``, optionally with a split LM head and
a fixed-size batched prefill model for prompt chunks. ``sample`` maps host
logits to a token index; ``generate`` is the one autoregressive loop LLM,
VLM and STT demos run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from time import perf_counter_ns

import numpy as np
import numpy.typing as npt

from utils.cache import KVCache, Layout
from utils.inference import StopCheck, _raise_if_stopped
from utils.runtime import Model, to_host

logger = logging.getLogger(__name__)


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


def _query_prefill_size(prefill: Model) -> int:
    """Fixed prompt chunk size from the prefill model's token input shape."""
    info = prefill.inputs_info
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


def sample(logits: npt.NDArray, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 64) -> int:
    """Sample a token index from host logits ``[vocab]``.

    Greedy (argmax) when ``temperature <= 0``; otherwise top-k, then top-p,
    with the global NumPy random state (seed it for reproducibility).
    """
    logits = logits.astype(np.float32, copy=False)

    if temperature <= 0:
        return int(logits.argmax())

    k = min(top_k, logits.shape[-1])
    top_k_idx = np.argpartition(logits, -k)[-k:]
    x = logits[top_k_idx]

    x /= temperature
    x -= x.max()
    np.exp(x, out=x)
    x /= x.sum()

    order = np.argsort(x)[::-1]
    cdf = np.cumsum(x[order])
    cut = int(np.searchsorted(cdf, top_p)) + 1
    keep = order[:cut]
    p = x[keep]
    p /= p.sum()
    return int(np.random.choice(top_k_idx[keep], p=p))


class Decoder:
    """Single-token decode steps and batched prompt prefill for one model.

    Composed of ``utils.runtime.Model`` instances: *model* (the single-token
    decode body), *cache* (its KV cache), *head* (optional split LM head),
    *prefill* (optional fixed-size batched prompt-chunk model sharing the
    cache). *embed_lut* is the host ``[vocab, hidden]`` token-embedding table
    (``load_lut`` output) feeding the token input; without it the token id is
    fed directly. *vocab_lut* remaps sampled compact-vocab logits back to
    token ids (``token_id``).

    Step inputs are ``[token_or_embedding, position, *extra_constant_inputs,
    *cache.inputs]``; every buffer is preallocated from the input metadata.
    """

    def __init__(
        self,
        model: Model,
        cache: KVCache,
        *,
        head: Model | None = None,
        prefill: Model | None = None,
        embed_lut: npt.NDArray | None = None,
        vocab_lut: npt.NDArray | None = None,
    ):
        self._model = model
        self._cache = cache
        self._head = head
        self._embed_lut = embed_lut
        self._vocab_lut = vocab_lut

        if head is not None:
            self._validate_head(head)
        self._validate_decode_model()
        self._layout = cache.attach(model)
        self._init_step_buffers()

        self._prefill = None
        self._prefill_size = None
        self._prefill_head = None
        self._prefill_layout = None
        self._prefill_emb_buf = None
        self._prefill_id_buf = None
        if prefill is not None:
            self._setup_prefill(prefill)

        # Wall time of the last step, in ns (debug logging only).
        self.last_step_ns = 0

    @property
    def seq_len(self) -> int | None:
        """Cache length; :func:`generate` stops (or rolls) here."""
        return self._cache.seq_len

    @property
    def prefill_size(self) -> int | None:
        """Fixed prompt chunk size when a batched prefill model is loaded."""
        return self._prefill_size

    def step(self, token: int, pos: int, want_logits: bool = True) -> npt.NDArray | None:
        """One single-token decode step at *pos*; host logits, or None.

        The token goes in as its embedding row when ``embed_lut`` was given,
        else as the raw id. Returns the head's (or fused) logits for the last
        position — ``[vocab]`` float32 — or None when *want_logits* is False
        (the LM head is skipped).
        """
        if self._emb_buf is not None:
            self._emb_buf[0, 0, :] = self._embed_lut[token]
            first = self._emb_buf
        else:
            self._id_buf[0, 0] = token
            first = self._id_buf
        return self._run_step(first, pos, self._model, self._layout, want_logits, self._head)

    def _emb_row(self, row: npt.NDArray, pos: int, want_logits: bool) -> npt.NDArray | None:
        self._emb_buf[0, 0, :] = row
        return self._run_step(self._emb_buf, pos, self._model, self._layout, want_logits, self._head)

    def _run_step(
        self,
        first: npt.NDArray,
        pos: int,
        model: Model,
        layout: Layout,
        want_logits: bool,
        head: Model | None,
    ) -> npt.NDArray | None:
        """One model call over the step buffers; updates the shared cache."""
        self._pos_buf[0, 0] = pos
        start = perf_counter_ns()
        results = model.infer([first, self._pos_buf, *self._extra, *self._cache.inputs])
        self._cache.update(layout, results, pos)
        if not want_logits:
            self.last_step_ns = perf_counter_ns() - start
            return None
        logits = head.infer([results[0]])[0] if head is not None else results[0]
        self.last_step_ns = perf_counter_ns() - start
        return np.asarray(to_host(logits))[0, -1].astype(np.float32, copy=False)

    def prefill(
        self,
        tokens: list[int],
        start: int,
        want_logits: bool = True,
        should_stop: StopCheck | None = None,
    ) -> npt.NDArray | None:
        """Prefill *tokens* at *start*; host logits of the last token, or None.

        Complete chunks of ``prefill_size`` run through the batched prefill
        model; only the final unit computes logits (the LM head is skipped
        for the body-only chunks). A short prompt (or a chunk remainder)
        falls back to single decode steps. Raises
        ``InferenceInterrupted`` between steps when *should_stop* fires.
        """
        if not tokens:
            # Nothing to feed: return without consuming a cache slot.
            return None
        pos = start
        if self._prefill is not None and len(tokens) >= self._prefill_size:
            chunk = self._prefill_size
            full_chunks = len(tokens) // chunk
            remainder = len(tokens) - full_chunks * chunk
            # Every complete chunk before the final unit is body-only: it only
            # extends the shared KV cache, so the LM head is skipped.
            body_only_chunks = full_chunks if remainder else full_chunks - 1
            for i in range(body_only_chunks):
                _raise_if_stopped(should_stop)
                self._chunk_step(tokens[i * chunk:(i + 1) * chunk], pos, want_logits=False)
                pos += chunk
                _raise_if_stopped(should_stop)
            if remainder:
                # Fall back to single-token decode for any remainder.
                rem_tokens = tokens[full_chunks * chunk:]
                for tok_id in rem_tokens[:-1]:
                    _raise_if_stopped(should_stop)
                    self.step(tok_id, pos, want_logits=False)
                    pos += 1
                    _raise_if_stopped(should_stop)
                _raise_if_stopped(should_stop)
                logits = self.step(rem_tokens[-1], pos, want_logits=want_logits)
                _raise_if_stopped(should_stop)
                pos += 1
            else:
                _raise_if_stopped(should_stop)
                logits = self._chunk_step(tokens[-chunk:], pos, want_logits=want_logits)
                _raise_if_stopped(should_stop)
                pos += chunk
            return logits
        for tok_id in tokens[:-1]:
            _raise_if_stopped(should_stop)
            self.step(tok_id, pos, want_logits=False)
            pos += 1
            _raise_if_stopped(should_stop)
        _raise_if_stopped(should_stop)
        logits = self.step(tokens[-1], pos, want_logits=want_logits)
        _raise_if_stopped(should_stop)
        return logits

    def prefill_embeddings(
        self,
        embs: npt.NDArray,
        start: int,
        want_logits: bool = True,
        should_stop: StopCheck | None = None,
    ) -> npt.NDArray | None:
        """Feed precomputed embedding rows ``[n, hidden]`` as single decode
        steps; host logits of the last row, or None."""
        if self._emb_buf is None:
            raise ValueError("prefill_embeddings needs a Decoder built with an embedding LUT")
        if len(embs) == 0:
            return None
        pos = start
        for row in embs[:-1]:
            _raise_if_stopped(should_stop)
            self._emb_row(row, pos, want_logits=False)
            pos += 1
            _raise_if_stopped(should_stop)
        _raise_if_stopped(should_stop)
        logits = self._emb_row(embs[-1], pos, want_logits=want_logits)
        _raise_if_stopped(should_stop)
        return logits

    def _chunk_step(self, tokens: list[int], pos: int, want_logits: bool) -> npt.NDArray | None:
        """One batched prefill step: the whole chunk, positioned at the
        chunk's start *pos* (the graph expands it across the chunk)."""
        if self._prefill_emb_buf is not None:
            self._prefill_emb_buf[...] = self._embed_lut[tokens]
            first = self._prefill_emb_buf
        else:
            self._prefill_id_buf[0, :] = tokens
            first = self._prefill_id_buf
        return self._run_step(
            first, pos, self._prefill, self._prefill_layout, want_logits, self._prefill_head
        )

    def token_id(self, index: int) -> int:
        """Map a sampled logits index to a token id (``vocab_lut`` remap)."""
        return int(self._vocab_lut[index]) if self._vocab_lut is not None else int(index)

    # ── Setup-time validation ──────────────────────────────────────────────

    def _init_step_buffers(self) -> None:
        """Build the per-step input buffers from the model's input metadata.

        Input 0 is the token input (embedding vector or token id), input 1 is
        ``position_ids``; shapes and dtypes come from the model's reflection
        data. Any further non-cache inputs (e.g. the fixed all-ones
        ``attention_mask`` of newer static exports) are constant per step.
        """
        in_info = self._model.inputs_info
        if not in_info or len(in_info) < 2:
            raise ValueError(
                f"Model '{self._model.path}' has no input metadata; "
                "cannot build the per-step input buffers."
            )
        n_user = len(in_info) - len(self._cache.inputs)
        if n_user < 2:
            raise ValueError(
                f"Model '{self._model.path}' has only {n_user} non-cache "
                "input(s); expected at least (tokens, position_ids)."
            )
        token_info, pos_info = in_info[0], in_info[1]
        self._pos_buf = np.zeros(pos_info.shape, dtype=np.dtype(pos_info.dtype))
        if self._embed_lut is not None:
            emb_hidden = self._embed_lut.shape[-1]
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
        self._extra = [np.ones(info.shape, dtype=np.dtype(info.dtype)) for info in in_info[2:n_user]]

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
                f"Model '{self._model.path}' expects batch size "
                f"{shape[0]} (first input shape {shape}); only batch size "
                "1 is supported."
            )
        if len(shape) > 1 and isinstance(shape[1], int) and shape[1] > 1:
            raise ValueError(
                f"Model '{self._model.path}' expects {shape[1]} tokens "
                f"per step (first input shape {shape}); it looks like a "
                "fixed-size batched build rather than the single-token "
                "decode model. Pass the decode model (e.g. "
                "'transformer.vmfb') via -m, and provide the batched build "
                "as a sibling 'transformer_prefill.vmfb' or via "
                "--batch-prefill-model."
            )

    def _validate_head(self, head: Model) -> None:
        body_outputs = self._model.outputs_info
        lm_head_inputs = head.inputs_info
        if not body_outputs:
            raise ValueError(
                f"Body model '{self._model.path}' is missing output metadata "
                "required to validate split LM head compatibility."
            )
        if not lm_head_inputs:
            raise ValueError(
                f"LM head model '{head.path}' is missing input metadata "
                "required to validate split LM head compatibility."
            )
        body_hidden, lm_head_input = body_outputs[0], lm_head_inputs[0]
        if tuple(body_hidden.shape) != tuple(lm_head_input.shape):
            raise ValueError(
                "Split LM head input shape does not match body output shape: "
                f"body output[0] {_tensor_info(body_hidden)}, "
                f"LM head input[0] {_tensor_info(lm_head_input)}."
            )
        if not _tensor_signature_matches(body_hidden, lm_head_input):
            # Shape matched above; only the dtype can differ.
            raise ValueError(
                "Split LM head input dtype does not match body output dtype: "
                f"body output[0] {_tensor_info(body_hidden)}, "
                f"LM head input[0] {_tensor_info(lm_head_input)}."
            )

    def _setup_prefill(self, prefill: Model) -> None:
        """Wire up the fixed-size batched prefill model for prompt chunks."""
        prefill_size = _query_prefill_size(prefill)
        if self._cache.seq_len is not None and prefill_size > self._cache.seq_len:
            raise ValueError(
                f"Batched prefill size {prefill_size} from '{prefill.path}' "
                f"exceeds the model max sequence length {self._cache.seq_len}."
            )
        # The prefill model has the same per-layer cache layout as the
        # decode model; both run against the shared on-device buffers.
        self._prefill_layout = self._cache.attach(prefill)
        self._validate_prefill_inputs(prefill, prefill_size)
        self._prefill = prefill
        self._prefill_size = prefill_size
        self._prefill_head = self._prefill_head_for(prefill)
        prefill_token_info = prefill.inputs_info[0]
        if self._embed_lut is not None:
            self._prefill_emb_buf = np.zeros(
                (1, prefill_size, self._embed_lut.shape[-1]),
                dtype=np.dtype(prefill_token_info.dtype),
            )
        else:
            self._prefill_id_buf = np.zeros(
                prefill_token_info.shape, dtype=np.dtype(prefill_token_info.dtype)
            )

    def _prefill_head_for(self, prefill: Model) -> Model | None:
        """The LM head a prefill chunk's first output goes through.

        The decode model's first output is always logits (a fused head, or
        the split head applied on the way out), so the prefill model must
        land in the same space:

        * prefill emits the head's *input* (hidden states) -> route it
          through the standalone head;
        * prefill already emits logits (head baked in at export time) -> use
          it directly, no head.

        Returns the head to apply, or None.
        """
        prefill_out = prefill.outputs_info[0]
        if self._head is not None:
            head_in = self._head.inputs_info[0]
            if _tensor_signature_matches(prefill_out, head_in):
                logger.info(
                    "Prefill model emits hidden states; reusing the decode "
                    "model's compiled LM head."
                )
                return self._head
            if _tensor_signature_matches(prefill_out, self._head.outputs_info[0]):
                logger.info(
                    "Prefill model has its LM head baked in; using it directly."
                )
                return None
            raise ValueError(
                f"Prefill model's first output {_tensor_info(prefill_out)} "
                f"matches neither the LM head input {_tensor_info(head_in)} "
                f"nor the decode model's logits "
                f"{_tensor_info(self._head.outputs_info[0])}; cannot use it "
                "for batched prefill."
            )
        if _tensor_signature_matches(prefill_out, self._model.outputs_info[0]):
            return None
        raise ValueError(
            "Batched prefill model emits hidden states, but the decode model "
            "has no standalone LM head to sample from. Load the split decode "
            "pair (transformer.vmfb + lm_head.vmfb) or a prefill model with "
            "a fused head."
        )

    def _validate_prefill_inputs(self, prefill: Model, prefill_size: int) -> None:
        """Prefill and decode models must take the same per-step inputs.

        The same buffers (positions, extra inputs) are fed to both models
        every step, so everything but the token input's sequence length must
        line up; a mismatch would otherwise surface as an opaque runtime
        shape assertion deep in inference.
        """
        decode_in = self._model.inputs_info
        prefill_in = prefill.inputs_info
        n_cache = len(self._cache.inputs)
        n_decode_user = len(decode_in) - n_cache
        n_prefill_user = len(prefill_in) - n_cache
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


def generate(
    decoder: Decoder,
    token: int,
    pos: int,
    *,
    next_token: Callable[[npt.NDArray], int],
    stop: Callable[[int, list[int]], bool],
    max_new: int | None = None,
    on_full: Callable[[int], int | None] | None = None,
    should_stop: StopCheck | None = None,
) -> Iterator[int]:
    """Run the autoregressive loop; yields each new token id.

    *token* was produced by a prefill and is yielded first. *next_token* maps
    a step's host logits to a token id (``sample`` plus any vocab remap);
    *stop* sees the last token and the generation so far. When *pos* reaches
    ``decoder.seq_len``, *on_full(pos)* (the KV window roll) returns the new
    position or None to stop; without *on_full* generation stops there.
    Raises ``InferenceInterrupted`` when *should_stop* fires.
    """
    gen = [token]
    yield token
    while not stop(token, gen):
        _raise_if_stopped(should_stop)
        if max_new is not None and len(gen) >= max_new:
            return
        if decoder.seq_len is not None and pos >= decoder.seq_len:
            pos = on_full(pos) if on_full is not None else None
            if pos is None or pos >= decoder.seq_len:
                return
        token = next_token(decoder.step(token, pos))
        _raise_if_stopped(should_stop)
        gen.append(token)
        pos += 1
        yield token
