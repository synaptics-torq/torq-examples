# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Standalone PTD runner tests with fake VMFB models.

A standalone PTD export is a fixed-row body that *is* the model
(``transformer.vmfb`` + ``lm_head.vmfb``): the same file names as an ordinary
build, but the body takes per-row tokens, positions, and a float32
``tree_vis`` input, and emits per-row logits or hidden states plus the
committed cache and per-layer combined key/value scratch. The fakes mimic
that contract exactly; the fake models predict ``succ(t)`` per input row
(context-free), so ordinary greedy and PTD greedy produce the same token
stream. The fake's "device arrays" are plain NumPy; the runner's ``_host``
helper accepts both.
"""

import logging

import numpy as np
import pytest

from gemma3.src.runner import Gemma3Static
from utils.inference import (
    BaseManagedCacheRunner,
    ManagedPTDCacheRunner,
    SplitLMHeadRunner,
)
from utils import llm

VOCAB = 4
EOS = 3
ROWS = 4
N_LAYERS = 2
H_KV = 2
HEAD_DIM = 3
M = 14
PROMPT = [1]
HIDDEN = 8


def _cycle(t: int) -> int:
    return (t + 1) % 3


def _plain_greedy(succ, prompt, eos, max_seq_len, max_gen_tokens=None):
    """The reference greedy chain a standalone PTD run must reproduce."""
    gen = [succ(prompt[-1])]
    pos = len(prompt)
    while (
        gen[-1] != eos
        and pos < max_seq_len
        and (max_gen_tokens is None or len(gen) < max_gen_tokens)
    ):
        gen.append(succ(gen[-1]))
        pos += 1
    return gen


class _TensorInfo:
    def __init__(self, shape, dtype, name=""):
        self.shape = list(shape)
        self.dtype = dtype
        self.name = name


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


class _FakePTDBody:
    """Standalone PTD body stand-in with the compiled output layout
    (first, committed per layer, combined key/value scratch per layer)."""

    def __init__(self, cache, succ, rows, vocab=VOCAB, hidden=False):
        self._cache = cache
        self._succ = succ
        self.rows = rows
        self.vocab = vocab
        self._hidden = hidden
        self.n_cache_inputs = N_LAYERS
        self.model_path = "transformer.vmfb"
        self.calls = []
        self.uploads = 0
        self.last_results = None
        self._kv_cache = list(cache)
        self._kv_init = list(cache)
        first_shape = (1, rows, HIDDEN) if hidden else (1, rows, vocab)
        self.inputs_info = (
            [
                _TensorInfo((1, rows), "int64", "input_ids"),
                _TensorInfo((1, rows), "int64", "position_ids"),
            ]
            + [
                _TensorInfo(cache[0].shape, "float32", f"past_key_values.{i}.key_value")
                for i in range(N_LAYERS)
            ]
            + [
                _TensorInfo((1, 1, rows, rows - 1), "float32", "tree_vis"),
            ]
        )
        self.outputs_info = (
            [_TensorInfo(first_shape, "float32", "last_hidden_states" if hidden else "logits")]
            + [
                _TensorInfo(cache[0].shape, "float32", f"present.{i}.key_value")
                for i in range(N_LAYERS)
            ]
            + [
                _TensorInfo(
                    (1, 2 * H_KV, rows - 1, cache[0].shape[-1]),
                    "float32",
                    f"present.{i}.key_value_scratch",
                )
                for i in range(N_LAYERS)
            ]
        )

    def _row_predictions(self, toks):
        return [self._succ(int(t)) for t in toks]

    def infer(self, user_inputs):
        toks = np.asarray(user_inputs[0]).reshape(-1)
        pos = np.asarray(user_inputs[1]).reshape(-1)
        vis = np.asarray(user_inputs[2])
        L = int(pos[0])
        self.calls.append((toks.copy(), pos.copy(), vis.copy()))
        preds = self._row_predictions(toks)
        if self._hidden:
            # Column 0 carries the input token id so the fake head can
            # recompute the same prediction.
            first = np.zeros((1, self.rows, HIDDEN), dtype=np.float32)
            first[0, :, 0] = toks
        else:
            first = np.zeros((1, self.rows, self.vocab), dtype=np.float32)
            for i, p in enumerate(preds):
                first[0, i, p] = 1.0
        results = [first]
        for c in self._cache:
            committed = c.copy()
            committed[0, :, L, 0] = toks[0]
            results.append(committed)
        for c in self._cache:
            scratch = np.zeros((1, 2 * H_KV, self.rows - 1, c.shape[-1]), dtype=np.float32)
            scratch[0, :, np.arange(self.rows - 1), 0] = toks[1:][:, None]
            results.append(scratch)
        self.last_results = results
        return results

    def allocate_device_array(self, array):
        self.uploads += 1
        return array


class _FakeSplitHead(SplitLMHeadRunner):
    """SplitLMHeadRunner stand-in: body hidden [1, R, H] (token id in
    column 0) -> per-row logits via the same succ function."""

    def __init__(self, body, succ, vocab):
        self._body = body
        self._infer_time_ms = 0.0
        self._lm_head = _TensorInfo((1, body.rows, vocab), "float32", "logits")
        self._succ = succ
        self._vocab = vocab
        self.head_calls = 0

    def infer(self, inputs, *, skip_lm_head=False):
        results = self._body.infer(inputs)
        if not skip_lm_head:
            self.head_calls += 1
            hidden = np.asarray(results[0])
            logits = np.zeros((1, hidden.shape[1], self._vocab), dtype=np.float32)
            for i, t in enumerate(hidden[0, :, 0]):
                logits[0, i, self._succ(int(t))] = 1.0
            results = [logits] + results[1:]
        return results

    @property
    def outputs_info(self):
        return [_TensorInfo((1, self._body.rows, self._vocab), "float32", "logits")]


class _PtdGemma(Gemma3Static):
    """Gemma3Static with __init__ skipped; the (fake) models are wired per
    run, matching what _setup_standalone_ptd produces."""

    def __init__(
        self,
        succ,
        *,
        max_seq_len: int = M,
        max_gen_tokens: int | None = None,
        rows: int = ROWS,
        vocab: int = VOCAB,
        eos: int = EOS,
        split: bool = False,
        prompt: list[int] | None = None,
    ):
        self._prompt = list(PROMPT if prompt is None else prompt)
        self._cache = [
            np.zeros((1, 2 * H_KV, max_seq_len, HEAD_DIM), dtype=np.float32)
            for _ in range(N_LAYERS)
        ]
        body = _FakePTDBody(self._cache, succ, rows, vocab=vocab, hidden=split)
        model = _FakeSplitHead(body, succ, vocab) if split else body
        self._logger = logging.getLogger("test_ptd_runner")
        self._model = model
        self._max_seq_len = max_seq_len
        self._max_gen_tokens = max_gen_tokens
        self._warmup_len = 0
        self._cache_keep_n = None
        self._reset_cache_state = []
        self._tokenizer = _FakeTokenizer()
        self._eos_token_id = eos
        self._instruct_model = False
        self._max_prompt_tokens = None
        self._max_user_tokens = None
        self._config = {}
        self._n_tokens_gen = 0
        self._last_infer_ns = 0
        self._time_to_first_token_ns = 0
        self._start_time_ns = 0
        self._token_embeddings = None
        self._token_id_lut = None
        self._prefill_model = None
        self._prefill_size = None

        self._ptd_width = 4
        self._ptd_depth = 6
        self._ptd_pool_max_nodes = 16
        self._ptd_pool_max_entries = 64
        self._ptd_min_draft_rows = 1
        self._ptd_pad_token = 0
        self._ptd_tree = None
        self._ptd_pool = None
        self._ptd_log = []
        self._ptd_rounds = 0
        self._ptd_body = body
        self._ptd_model = model
        self._ptd_rows = rows
        self._ptd_cache = self._cache
        self._ptd_feed_order = ["tok", "pos", "vis"]
        self._ptd_tok_buf = np.zeros((1, rows), dtype=np.int64)
        self._ptd_pos_buf = np.zeros((1, rows), dtype=np.int64)
        self._ptd_vis_buf = np.zeros((1, 1, rows, rows - 1), dtype=np.float32)

    def _should_stop(self, token, gen):
        return token == self._eos_token_id

    def _reset_cache(self):
        pass

    def _build_prompt_tokens(self, user_input):
        return list(self._prompt)


def test_ptd_cache_inputs_keep_layer_order(monkeypatch):
    cache_indices = list(range(2, 20))
    device_io = True
    body = ManagedPTDCacheRunner.__new__(ManagedPTDCacheRunner)
    body._device_io = device_io
    body._cache_input_idx = cache_indices
    n_inputs = len(cache_indices) + 3
    monkeypatch.setattr(
        ManagedPTDCacheRunner, "inputs_info", property(lambda self: [None] * n_inputs),
        raising=False,
    )
    monkeypatch.setattr(
        BaseManagedCacheRunner, "_infer", lambda self, inputs: inputs, raising=False,
    )
    body.allocate_device_array = lambda array: array
    user_inputs = [np.array([i]) for i in range(3)]
    # Start with zero caches, then feed distinct per-layer outputs back into
    # the next call: reversing initially identical caches hides the defect.
    for step in range(2):
        body._kv_cache = [np.array([step * (i + 1)]) for i in range(len(cache_indices))]
        actual = body._infer(user_inputs)
        caches = dict(zip(cache_indices, body._kv_cache))
        users = iter(user_inputs)
        expected = [caches[i] if i in caches else next(users) for i in range(n_inputs)]
        assert all(got is want for got, want in zip(actual, expected))
        assert len(actual) == len(expected)


def test_ptd_runtime_detection_ignores_sibling_prefill(tmp_path, monkeypatch):
    model_path = tmp_path / "transformer.vmfb"
    model_path.touch()
    (tmp_path / "transformer_prefill.vmfb").touch()
    runner = _PtdGemma(_cycle, split=True)
    assert runner._is_ptd_body()
    monkeypatch.setattr(llm, "ManagedSelfAttnCacheRunner", lambda *a, **kw: runner._model._body)

    class SelectionChecked(Exception):
        pass

    def check_prefill(self, prefill_path, temperature, top_p):
        assert prefill_path is None
        raise SelectionChecked

    monkeypatch.setattr(llm.DecoderOnlyLLMRunner, "_check_ptd_compat", check_prefill)
    with pytest.raises(SelectionChecked):
        llm.DecoderOnlyLLMRunner.__init__(
            Gemma3Static.__new__(Gemma3Static), model_path, disable_lm_head=True,
        )


def test_ptd_split_prefill_and_decode_match_greedy():
    succ = lambda t: (t + 2) % 4  # noqa: E731
    prompt = [1, 2, 0, 1, 2, 0]  # 6 tokens, ROWS=4 -> two units
    runner = _PtdGemma(succ, split=True, prompt=prompt, max_seq_len=16)
    expected = _plain_greedy(succ, prompt, EOS, 16)
    assert runner.run("x") == " ".join(str(t) for t in expected)

    body = runner._ptd_body
    # Two prefill units: a full chain, then the remainder chain.
    first_toks, first_pos, first_vis = body.calls[0]
    assert list(first_toks) == [1, 2, 0, 1]
    assert list(first_pos) == [0, 1, 2, 3]
    # Chain visibility: row i sees slots 1..i; the anchor row sees none.
    assert (first_vis[0, 0, 0, :] == 0).all()
    assert first_vis[0, 0, 1, :].tolist() == [1, 0, 0]
    assert first_vis[0, 0, 2, :].tolist() == [1, 1, 0]
    assert first_vis[0, 0, 3, :].tolist() == [1, 1, 1]
    second_toks, second_pos, second_vis = body.calls[1]
    assert list(second_toks[:2]) == [2, 0]
    assert list(second_pos[:2]) == [4, 5]
    # Padding row: zero visibility, anchor position.
    assert (second_vis[0, 0, 2, :] == 0).all() and second_pos[2] == 4

    # Every prompt token is committed (the chain rows were promoted).
    for i, t in enumerate(prompt):
        assert runner._cache[0][0, 0, i, 0] == t
    # The split head ran once in prefill (final unit only), then once per
    # round.
    assert runner._ptd_model.head_calls == 1 + runner._ptd_rounds
    assert any(round_["hit"] for round_ in runner._ptd_log)
    for layer in runner._cache:
        assert layer[0, 0, :, 0].tolist() == (prompt + expected[:-1])
        assert layer[0, H_KV, :, 0].tolist() == (prompt + expected[:-1])


def test_ptd_generation_stops_at_eos_and_cache_capacity():
    eos_runner = _PtdGemma(lambda token: token + 1, rows=8, vocab=16, eos=10)
    assert eos_runner.run("x") == "2 3 4 5 6 7 8 9 10"
    assert eos_runner._ptd_cache[0][0, 0, 9, 0] == 0  # EOS has no committed KV.

    full_runner = _PtdGemma(_cycle, rows=8, eos=99)
    expected = _plain_greedy(_cycle, PROMPT, 99, M)
    assert full_runner.run("x") == " ".join(str(t) for t in expected)
    final = full_runner._ptd_log[-1]
    assert final["L"] == M - 1
    assert final["n_candidates"] == final["n_drafts"] == 0
    assert (full_runner._ptd_body.calls[-1][2] == 0).all()


def test_ptd_lookup_verifies_candidates_with_generation_limit():
    prompt = [0, 1, 2, 0, 1]
    runner = _PtdGemma(_cycle, prompt=prompt, max_gen_tokens=8)
    runner._ptd_lookup_ngram = 4
    runner._ptd_min_draft_rows = 0
    expected = _plain_greedy(_cycle, prompt, EOS, M, max_gen_tokens=8)
    assert runner.run("x") == " ".join(str(t) for t in expected)
    assert runner._ptd_log[0]["accepted"] > 0
    assert runner._ptd_cache[0][0, 0, :12, 0].tolist() == prompt + expected[:-1]

    wrong = _PtdGemma(_cycle, prompt=[1, 1], max_gen_tokens=4)
    wrong._ptd_lookup_ngram = 4
    assert wrong.run("x") == "2 0 1 2"
    assert wrong._ptd_log[0]["accepted"] == 0
