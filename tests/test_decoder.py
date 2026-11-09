# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Tests for the decoding core: Decoder steps (id / embedding / mask inputs,
split head), batched prefill chunking, sampling, and the generate loop."""

from types import SimpleNamespace

import numpy as np
import pytest

from utils.cache import KVCache
from utils.decoder import Decoder, generate, sample
from utils.inference import InferenceInterrupted


F32 = "float32"
I32 = "int32"
I64 = "int64"
HIDDEN = 4
VOCAB = 8
SEQ = 8


class _Info:
    def __init__(self, shape, dtype=F32):
        self.shape = list(shape)
        self.dtype = dtype


def info(shape, dtype=F32):
    return _Info(shape, dtype)


class _Array:
    """Device array stand-in."""

    def __init__(self, a):
        # Copy: staging a host buffer uploads a fresh device buffer per call,
        # so recorded call inputs keep the value they had at call time.
        self.a = np.array(a, copy=True)
        self.shape, self.dtype = self.a.shape, np.dtype(self.a.dtype)
        self._buffer_view = SimpleNamespace(get_buffer=lambda: self)

    def to_host(self):
        return self.a

    def __array__(self, dtype=None):
        return self.a


class _Device:
    def create_semaphore(self, value):
        return SimpleNamespace(wait=lambda v: None)

    def queue_execute(self, cb, waits, signals):
        pass


class _Model:
    """Fake utils.runtime.Model: metadata, device staging, recorded calls."""

    def __init__(self, path, inputs, outputs, result_fn=None):
        self.path = path
        self.inputs_info = inputs
        self.outputs_info = outputs
        self.device = _Device()
        self.calls = []
        self._result_fn = result_fn

    def to_device(self, x):
        return _Array(x) if isinstance(x, np.ndarray) else x

    def allocate_device_array(self, x):
        return _Array(x)

    def infer(self, inputs):
        # Stage like the real Model.infer: host buffers upload fresh device
        # arrays per call, so recorded calls keep their call-time values.
        staged = [self.to_device(x) for x in inputs]
        self.calls.append(staged)
        if self._result_fn is not None:
            return self._result_fn(staged)
        return [_Array(np.zeros(o.shape, np.dtype(o.dtype))) for o in self.outputs_info]


def _decode_model(seq=SEQ, token=None, extra=(), primary=None, result_fn=None):
    """A single-token decode body: (token, position, *extra, [2kv, seq, D])."""
    return _Model(
        "transformer.vmfb",
        [token or info((1, 1), I32), info((1, 1), I64), *extra, info((1, 2, seq, HIDDEN))],
        [primary or info((1, 1, VOCAB)), info((1, 2, seq, HIDDEN))],
        result_fn=result_fn,
    )


def _token_body(seq=SEQ, extra=(), cache_outs=None):
    """Fused-head body: the logits' argmax is the fed token id (mod VOCAB)."""
    def fn(inputs):
        tok = int(np.asarray(inputs[0]).reshape(-1)[0])
        logits = np.zeros((1, 1, VOCAB), F32)
        logits[0, 0, tok % VOCAB] = 1.0
        cache_out = _Array(np.zeros((1, 2, seq, HIDDEN), F32))
        if cache_outs is not None:
            cache_outs.append(cache_out)
        return [_Array(logits), cache_out]
    return _decode_model(seq=seq, extra=extra, result_fn=fn)


def _hidden_body(seq=SEQ, token=None, extra=(), cache_outs=None):
    """Split body: hidden row[0] = the fed value, plus a cache output."""
    def fn(inputs):
        value = float(np.asarray(inputs[0]).reshape(-1)[0])
        hidden = np.zeros((1, 1, HIDDEN), F32)
        hidden[0, 0, 0] = value
        cache_out = _Array(np.zeros((1, 2, seq, HIDDEN), F32))
        if cache_outs is not None:
            cache_outs.append(cache_out)
        return [_Array(hidden), cache_out]
    return _decode_model(seq=seq, token=token, extra=extra,
                         primary=info((1, 1, HIDDEN)), result_fn=fn)


def _head_model():
    """Split LM head: the logits' argmax is the hidden row's value (mod VOCAB)."""
    def fn(inputs):
        value = float(np.asarray(inputs[0])[0, 0, 0])
        logits = np.zeros((1, 1, VOCAB), F32)
        logits[0, 0, int(value) % VOCAB] = 1.0
        return [_Array(logits)]
    return _Model("lm_head.vmfb", [info((1, 1, HIDDEN))], [info((1, 1, VOCAB))], result_fn=fn)


def _prefill_model(chunk=4, fused=False, cache_outs=None):
    """A fixed-size batched prefill body; the value is the chunk's last token."""
    def fn(inputs):
        value = float(np.asarray(inputs[0]).reshape(-1)[-1])
        if fused:
            out = np.zeros((1, 1, VOCAB), F32)
            out[0, 0, int(value) % VOCAB] = 1.0
        else:
            out = np.zeros((1, 1, HIDDEN), F32)
            out[0, 0, 0] = value
        cache_out = _Array(np.zeros((1, 2, SEQ, HIDDEN), F32))
        if cache_outs is not None:
            cache_outs.append(cache_out)
        return [_Array(out), cache_out]
    primary = info((1, 1, VOCAB)) if fused else info((1, 1, HIDDEN))
    return _Model(
        "transformer_prefill.vmfb",
        [info((1, chunk), I32), info((1, 1), I64), info((1, 2, SEQ, HIDDEN))],
        [primary, info((1, 2, SEQ, HIDDEN))],
        result_fn=fn,
    )


# ── step ──────────────────────────────────────────────────────────────────────


def test_step_id_input_feeds_token_position_and_cache():
    cache_outs = []
    model = _token_body(cache_outs=cache_outs)
    cache = KVCache(model)
    decoder = Decoder(model, cache)
    assert decoder.seq_len == SEQ
    assert decoder.prefill_size is None

    out = decoder.step(3, 7)

    inp = model.calls[0]
    assert np.asarray(inp[0])[0, 0] == 3
    assert np.asarray(inp[1])[0, 0] == 7
    # The shared cache input trails the step inputs; the update swaps in the
    # model's fresh output.
    assert inp[2] is not cache_outs[0]
    assert cache.inputs[0] is cache_outs[0]
    assert out.shape == (VOCAB,) and out.dtype == np.dtype(F32)
    assert int(out.argmax()) == 3


def test_step_embed_lut_cast_to_input_dtype():
    lut = np.arange(16 * HIDDEN, dtype=np.float16).reshape(16, HIDDEN)
    model = _hidden_body(token=info((1, 1, HIDDEN), F32))
    decoder = Decoder(model, KVCache(model), embed_lut=lut)

    out = decoder.step(5, 2)

    tok_in = np.asarray(model.calls[0][0])
    assert tok_in.dtype == np.dtype(F32)  # the buffer follows the input's dtype
    np.testing.assert_array_equal(tok_in[0, 0], lut[5].astype(F32))
    assert out.shape == (HIDDEN,) and out[0] == lut[5][0].astype(F32)


def test_step_feeds_all_ones_mask():
    model = _token_body(extra=[info((1, SEQ), I64)])
    decoder = Decoder(model, KVCache(model))
    decoder.step(1, 0)
    mask = np.asarray(model.calls[0][2])
    assert mask.dtype == np.dtype(I64) and mask.shape == (1, SEQ)
    assert np.all(mask == 1)


def test_split_head_runs_and_skips_without_logits():
    model = _hidden_body()
    head = _head_model()
    decoder = Decoder(model, KVCache(model), head=head)

    out = decoder.step(3, 0)

    assert len(head.calls) == 1
    assert len(head.calls[0]) == 1  # the head sees only the hidden row
    np.testing.assert_array_equal(np.asarray(head.calls[0][0])[0, 0, 0], 3.0)
    assert out.shape == (VOCAB,) and int(out.argmax()) == 3

    model2 = _hidden_body()
    head2 = _head_model()
    decoder2 = Decoder(model2, KVCache(model2), head=head2)
    assert decoder2.step(3, 0, want_logits=False) is None
    assert head2.calls == []  # body-only step
    assert len(model2.calls) == 1


def test_head_metadata_mismatches_rejected():
    model = _hidden_body()
    bad_head = _Model("lm_head.vmfb", [info((1, 1, HIDDEN * 2))], [info((1, 1, VOCAB))])
    with pytest.raises(ValueError, match="Split LM head input shape"):
        Decoder(model, KVCache(model), head=bad_head)

    model = _hidden_body()
    bad_dtype = _Model("lm_head.vmfb", [info((1, 1, HIDDEN), "float16")], [info((1, 1, VOCAB))])
    with pytest.raises(ValueError, match="Split LM head input dtype"):
        Decoder(model, KVCache(model), head=bad_dtype)


def test_embed_lut_hidden_mismatch_rejected():
    model = _decode_model(token=info((1, 1, HIDDEN), F32))
    lut = np.zeros((16, HIDDEN * 2), F32)
    with pytest.raises(ValueError, match="token_embeddings.npy has hidden size"):
        Decoder(model, KVCache(model), embed_lut=lut)


def test_decode_model_batched_build_rejected():
    model = _decode_model(token=info((1, 4), I32))
    with pytest.raises(ValueError, match="fixed-size batched build"):
        Decoder(model, KVCache(model))

    model = _decode_model(token=info((2, 1), I32))
    with pytest.raises(ValueError, match="only batch size 1 is supported"):
        Decoder(model, KVCache(model))


# ── prefill ───────────────────────────────────────────────────────────────────


def test_prefill_chunks_with_head_skipped_except_final_unit():
    model = _hidden_body()
    head = _head_model()
    prefill = _prefill_model(4)
    decoder = Decoder(model, KVCache(model), head=head, prefill=prefill)
    assert decoder.prefill_size == 4

    out = decoder.prefill([1, 2, 3, 4, 5, 6, 7], 0)

    # One body-only chunk, then the 3-token remainder as single decode steps.
    assert len(prefill.calls) == 1
    np.testing.assert_array_equal(np.asarray(prefill.calls[0][0])[0], [1, 2, 3, 4])
    assert int(np.asarray(prefill.calls[0][1])[0, 0]) == 0
    assert [int(np.asarray(c[0])[0, 0]) for c in model.calls] == [5, 6, 7]
    assert [int(np.asarray(c[1])[0, 0]) for c in model.calls] == [4, 5, 6]
    # The head ran exactly once — for the final unit only.
    assert len(head.calls) == 1
    assert out.shape == (VOCAB,) and int(out.argmax()) == 7


def test_prefill_exact_chunks_run_head_only_on_last_chunk():
    model = _hidden_body()
    head = _head_model()
    prefill = _prefill_model(4)
    decoder = Decoder(model, KVCache(model), head=head, prefill=prefill)

    out = decoder.prefill([1, 2, 3, 4, 5, 6, 7, 8], 0)

    assert len(prefill.calls) == 2
    assert model.calls == []  # no remainder: no single decode steps
    assert int(np.asarray(prefill.calls[0][1])[0, 0]) == 0
    assert int(np.asarray(prefill.calls[1][1])[0, 0]) == 4
    assert len(head.calls) == 1
    assert int(out.argmax()) == 0  # the second chunk's last token is 8


def test_prefill_short_prompt_falls_back_to_single_steps():
    model = _hidden_body()
    head = _head_model()
    prefill = _prefill_model(4)
    decoder = Decoder(model, KVCache(model), head=head, prefill=prefill)

    out = decoder.prefill([1, 2, 3], 2)

    assert prefill.calls == []
    assert [int(np.asarray(c[0])[0, 0]) for c in model.calls] == [1, 2, 3]
    assert [int(np.asarray(c[1])[0, 0]) for c in model.calls] == [2, 3, 4]
    assert len(head.calls) == 1
    assert int(out.argmax()) == 3


def test_prefill_empty_is_a_no_op():
    model = _token_body()
    decoder = Decoder(model, KVCache(model), prefill=_prefill_model(4, fused=True))
    assert decoder.prefill([], 0) is None
    assert model.calls == []


def test_prefill_with_fused_head_returns_its_own_logits():
    model = _token_body()
    prefill = _prefill_model(4, fused=True)
    decoder = Decoder(model, KVCache(model), prefill=prefill)
    out = decoder.prefill([4, 5, 6, 7], 0)
    assert len(prefill.calls) == 1
    assert int(out.argmax()) == 7


def test_hidden_prefill_without_split_head_rejected():
    model = _token_body()  # fused-head decode body, no standalone head
    prefill = _prefill_model(4)  # emits hidden states
    with pytest.raises(ValueError, match="no standalone LM head"):
        Decoder(model, KVCache(model), prefill=prefill)


def test_prefill_output_matching_neither_rejected():
    model = _hidden_body()
    head = _head_model()
    bad = _Model(
        "transformer_prefill.vmfb",
        [info((1, 4), I32), info((1, 1), I64), info((1, 2, SEQ, HIDDEN))],
        [info((1, 1, VOCAB + 1)), info((1, 2, SEQ, HIDDEN))],
    )
    with pytest.raises(ValueError, match="matches neither"):
        Decoder(model, KVCache(model), head=head, prefill=bad)


def test_prefill_size_exceeding_seq_len_rejected():
    model = _decode_model(seq=2)  # tiny cache: seq_len 2
    prefill = _prefill_model(4)  # chunk 4 > 2
    with pytest.raises(ValueError, match="exceeds the model max sequence length"):
        Decoder(model, KVCache(model), prefill=prefill)


def test_prefill_input_mismatches_rejected():
    # Non-cache input count: the decode body has a mask the prefill lacks.
    model = _decode_model(extra=[info((1, SEQ), I64)])
    with pytest.raises(ValueError, match="non-cache inputs"):
        Decoder(model, KVCache(model), prefill=_prefill_model(4))

    # Token input dtype.
    model = _token_body()
    bad = _Model(
        "transformer_prefill.vmfb",
        [info((1, 4), I64), info((1, 1), I64), info((1, 2, SEQ, HIDDEN))],
        [info((1, 1, HIDDEN)), info((1, 2, SEQ, HIDDEN))],
    )
    with pytest.raises(ValueError, match="does not match"):
        Decoder(model, KVCache(model), prefill=bad)

    # A non-token non-cache input (position) shape.
    bad = _Model(
        "transformer_prefill.vmfb",
        [info((1, 4), I32), info((1, 4), I64), info((1, 2, SEQ, HIDDEN))],
        [info((1, 1, HIDDEN)), info((1, 2, SEQ, HIDDEN))],
    )
    with pytest.raises(ValueError, match="non-cache input 1"):
        Decoder(model, KVCache(model), prefill=bad)


def test_prefill_embeddings_single_steps_with_last_logits_only():
    model = _hidden_body(token=info((1, 1, HIDDEN), F32))
    head = _head_model()
    decoder = Decoder(model, KVCache(model), head=head, embed_lut=np.zeros((16, HIDDEN), F32))
    embs = np.arange(12, dtype=F32).reshape(3, HIDDEN)

    out = decoder.prefill_embeddings(embs, 5)

    assert [np.asarray(c[0])[0, 0].tolist() for c in model.calls] == embs.tolist()
    assert [int(np.asarray(c[1])[0, 0]) for c in model.calls] == [5, 6, 7]
    assert len(head.calls) == 1  # logits only for the last row
    assert out.shape == (VOCAB,) and int(out.argmax()) == int(embs[-1][0]) % VOCAB

    # Empty input consumes no slot.
    assert decoder.prefill_embeddings(np.zeros((0, HIDDEN), F32), 5) is None
    assert len(model.calls) == 3

    # Without a LUT there is no embedding path.
    decoder2 = Decoder(_token_body(), KVCache(_token_body()))
    with pytest.raises(ValueError, match="embedding LUT"):
        decoder2.prefill_embeddings(embs, 0)


# ── token_id / sample ─────────────────────────────────────────────────────────


def test_token_id_vocab_lut_remap():
    decoder = Decoder(_token_body(), KVCache(_token_body()),
                      vocab_lut=np.arange(VOCAB) + 1000)
    assert decoder.token_id(3) == 1003
    assert decoder.token_id(0) == 1000
    assert Decoder(_token_body(), KVCache(_token_body())).token_id(3) == 3


def test_sample_greedy_at_zero_temperature():
    logits = np.array([0.1, 3.0, -2.0, 1.5], np.float16)
    assert sample(logits) == 1
    assert sample(logits, temperature=0.0, top_p=0.1, top_k=1) == 1


def test_sample_top_p_keeps_only_the_dominant_token():
    logits = np.array([10.0, 0.0, 0.0], F32)
    for seed in (0, 1, 42):
        np.random.seed(seed)
        assert sample(logits, temperature=1.0, top_p=0.6, top_k=3) == 0


def test_sample_seeded_top_k_top_p_matches_reference():
    # Pinned against the pre-restructure DecoderOnlyLLMRunner._sample output
    # (same algorithm, same global random state).
    np.random.seed(123)
    logits = np.array([0.5, 0.1, 0.7, 0.2], F32)
    assert sample(logits, temperature=1.0, top_p=1.0, top_k=4) == 3


# ── generate ──────────────────────────────────────────────────────────────────


def _walk_decoder(seq=SEQ):
    """Fused-head body whose argmax walks token + 1 (mod VOCAB)."""
    def fn(inputs):
        tok = int(np.asarray(inputs[0]).reshape(-1)[0])
        logits = np.zeros((1, 1, VOCAB), F32)
        logits[0, 0, (tok + 1) % VOCAB] = 1.0
        return [_Array(logits), _Array(np.zeros((1, 2, seq, HIDDEN), F32))]
    model = _decode_model(seq=seq, result_fn=fn)
    return Decoder(model, KVCache(model))


def _next(decoder):
    return lambda logits: decoder.token_id(sample(logits))


def test_generate_yields_prefill_token_and_stops_on_callback():
    decoder = _walk_decoder()
    out = list(generate(decoder, 0, 0, next_token=_next(decoder),
                        stop=lambda token, gen: token == 7))
    assert out == [0, 1, 2, 3, 4, 5, 6, 7]


def test_generate_max_new_caps_generation():
    decoder = _walk_decoder()
    out = list(generate(decoder, 0, 0, next_token=_next(decoder),
                        stop=lambda token, gen: False, max_new=3))
    assert out == [0, 1, 2]


def test_generate_on_full_rolls_the_window():
    decoder = _walk_decoder(seq=2)
    rolls = []

    def on_full(pos):
        rolls.append(pos)
        return 0

    out = list(generate(decoder, 0, 0, next_token=_next(decoder),
                        stop=lambda token, gen: token == 3, on_full=on_full))
    assert out == [0, 1, 2, 3]
    assert rolls == [2]


def test_generate_on_full_returning_none_stops():
    decoder = _walk_decoder(seq=2)
    out = list(generate(decoder, 0, 0, next_token=_next(decoder),
                        stop=lambda token, gen: False, on_full=lambda pos: None))
    assert out == [0, 1, 2]


def test_generate_without_on_full_stops_at_seq_len():
    decoder = _walk_decoder(seq=2)
    out = list(generate(decoder, 0, 0, next_token=_next(decoder),
                        stop=lambda token, gen: False))
    assert out == [0, 1, 2]


def test_generate_should_stop_interrupts():
    decoder = _walk_decoder()
    with pytest.raises(InferenceInterrupted):
        list(generate(decoder, 0, 0, next_token=_next(decoder),
                      stop=lambda token, gen: False, should_stop=lambda: True))
