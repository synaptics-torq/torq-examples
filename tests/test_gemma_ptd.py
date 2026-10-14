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
from utils import ptd as ptd_core
from utils.inference import (
    BaseManagedCacheRunner,
    ManagedPTDCacheRunner,
    SplitLMHeadRunner,
    print_llm_stats,
)

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

    # Interface used by the runner.
    @property
    def model_path(self):
        return self._body.model_path

    @property
    def outputs_info(self):
        return [_TensorInfo((1, self._body.rows, self._vocab), "float32", "logits")]

    @property
    def inputs_info(self):
        return self._body.inputs_info

    @property
    def n_cache_inputs(self):
        return self._body.n_cache_inputs

    def allocate_device_array(self, array):
        return self._body.allocate_device_array(array)

    def save_kv_state(self):
        return []

    def restore_kv_state(self, state):
        pass

    def reset_kv(self):
        pass

    def shift_kv(self, *args, **kwargs):
        pass


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
        ptd: bool = True,
        vocab: int = VOCAB,
        eos: int = EOS,
        split: bool = False,
        prompt: list[int] | None = None,
    ):
        self._succ = succ
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
        self._ptd_body = None
        self._ptd_model = None
        self._ptd_rows = None
        self._ptd_cache = self._cache
        self._ptd_tok_buf = self._ptd_pos_buf = self._ptd_vis_buf = None
        self._ptd_feed_order = ["tok", "pos", "vis"]
        if ptd:
            self._ptd_rows = rows
            self._ptd_body = body
            self._ptd_model = model
            self._ptd_tok_buf = np.zeros((1, rows), dtype=np.int64)
            self._ptd_pos_buf = np.zeros((1, rows), dtype=np.int64)
            self._ptd_vis_buf = np.zeros((1, 1, rows, rows - 1), dtype=np.float32)

    def _should_stop(self, token, gen):
        return token == self._eos_token_id

    def _reset_cache(self):
        pass

    def _build_prompt_tokens(self, user_input):
        return list(self._prompt)

    def gen_tokens(self):
        """Generated tokens from the committed cache (row i holds gen[i-1])."""
        return [
            int(self._cache[0][0, 0, i, 0])
            for i in range(1, self._max_seq_len)
            if self._cache[0][0, 0, i, 0] != 0
        ]


# ── detection & setup guards ─────────────────────────────────────────────────


@pytest.mark.parametrize("cache_indices", [list(range(2, 20)), [0, 2, 4]])
@pytest.mark.parametrize("device_io", [False, True])
def test_ptd_cache_inputs_keep_layer_order(monkeypatch, cache_indices, device_io):
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


def test_detects_standalone_ptd_body():
    fused = _PtdGemma(_cycle)
    assert fused._is_ptd_body()
    split = _PtdGemma(_cycle, split=True)
    assert split._is_ptd_body()


def test_detection_rejects_ordinary_and_malformed_bodies():
    runner = _PtdGemma(_cycle, ptd=False)
    cache = _TensorInfo(runner._cache[0].shape, "float32", "past_key_values.0")
    # Ordinary decode body: two user inputs, one-token sequence dim.
    runner._model.inputs_info = (
        [_TensorInfo((1, 1), "int64", "input_ids"),
         _TensorInfo((1, 1), "int64", "position_ids"),
         cache]
    )
    assert not runner._is_ptd_body()
    # Fixed-row token input but no tree_vis.
    runner._model.inputs_info = (
        [_TensorInfo((1, ROWS), "int64", "input_ids"),
         _TensorInfo((1, ROWS), "int64", "position_ids"),
         cache]
    )
    assert not runner._is_ptd_body()
    # R == 1 is not a PTD body.
    runner._model.inputs_info = (
        [_TensorInfo((1, 1), "int64", "input_ids"),
         _TensorInfo((1, 1), "int64", "position_ids"),
         cache,
         _TensorInfo((1, 1, 1, 0), "float32", "tree_vis")]
    )
    assert not runner._is_ptd_body()
    # tree_vis with the wrong shape for the row budget.
    runner._model.inputs_info = (
        [_TensorInfo((1, ROWS), "int64", "input_ids"),
         _TensorInfo((1, ROWS), "int64", "position_ids"),
         cache,
         _TensorInfo((1, 1, ROWS, ROWS), "float32", "tree_vis")]
    )
    assert not runner._is_ptd_body()


def test_setup_rejects_prefill_model_and_sampling():
    runner = _PtdGemma(_cycle)
    with pytest.raises(ValueError, match="standalone PTD body"):
        runner._check_ptd_compat("prefill.vmfb", 0.0, 1.0)
    with pytest.raises(ValueError, match="greedy"):
        runner._check_ptd_compat(None, 0.7, 1.0)
    with pytest.raises(ValueError, match="greedy"):
        runner._check_ptd_compat(None, 0.0, 0.9)
    runner._check_ptd_compat(None, 0.0, 1.0)  # no error


def _validate_shell():
    r = Gemma3Static.__new__(Gemma3Static)
    r._logger = logging.getLogger("test_ptd_setup")
    return r


def _ptd_io(rows, first_shape, scratch=(1, 2 * H_KV, None, HEAD_DIM)):
    cache = _TensorInfo((1, 2 * H_KV, M, HEAD_DIM), "float32", "past_key_values.0")
    inputs = [
        _TensorInfo((1, rows), "int64", "input_ids"),
        _TensorInfo((1, rows), "int64", "position_ids"),
        *[_TensorInfo(cache.shape, cache.dtype, "past_key_values.0") for _ in range(N_LAYERS)],
        _TensorInfo((1, 1, rows, rows - 1), "float32", "tree_vis"),
    ]
    outputs = [
        _TensorInfo(first_shape, "float32", "logits"),
        *[_TensorInfo(cache.shape, cache.dtype, "present") for _ in range(N_LAYERS)],
        *[
            _TensorInfo(
                (scratch[0], scratch[1], scratch[2] or rows - 1, scratch[3]),
                "float32",
                "scratch",
            )
            for _ in range(N_LAYERS)
        ],
    ]
    return inputs, outputs


def _meta_body(inputs, outputs, n_cache_inputs=N_LAYERS):
    body = _FakePTDBody.__new__(_FakePTDBody)
    body.inputs_info = inputs
    body.outputs_info = outputs
    body.n_cache_inputs = n_cache_inputs
    body._cache_input_idx = list(range(2, 2 + n_cache_inputs))
    body.model_path = "transformer.vmfb"
    return body


def test_validate_ptd_outputs_accepts_standalone_layout():
    r = _validate_shell()
    inputs, outputs = _ptd_io(ROWS, (1, ROWS, HIDDEN))
    body = _meta_body(inputs, outputs)
    r._validate_ptd_outputs(body, ROWS)  # no error


def test_validate_ptd_outputs_rejects_separate_scratch_layout():
    r = _validate_shell()
    # The old ptd_transformer layout: 1 + 3L outputs (k/v scratch separate).
    cache = _TensorInfo((1, 2 * H_KV, M, HEAD_DIM), "float32", "past")
    inputs = [
        _TensorInfo((1, ROWS), "int64", "input_ids"),
        _TensorInfo((1, ROWS), "int64", "position_ids"),
        *[_TensorInfo(cache.shape, cache.dtype, "past_key_values.0") for _ in range(N_LAYERS)],
        _TensorInfo((1, 1, ROWS, ROWS - 1), "float32", "tree_vis"),
    ]
    outputs = (
        [_TensorInfo((1, ROWS, HIDDEN), "float32", "logits")]
        + [_TensorInfo(cache.shape, cache.dtype, "present") for _ in range(N_LAYERS)]
        + [
            _TensorInfo((1, H_KV, ROWS - 1, HEAD_DIM), "float32", "scratch")
            for _ in range(2 * N_LAYERS)
        ]
    )
    with pytest.raises(ValueError, match="combined key/value scratch"):
        r._validate_ptd_outputs(_meta_body(inputs, outputs), ROWS)


def test_validate_ptd_outputs_rejects_bad_committed_or_scratch():
    r = _validate_shell()
    inputs, outputs = _ptd_io(ROWS, (1, ROWS, HIDDEN))
    outputs[1].shape = [1, 2 * H_KV, M + 1, HEAD_DIM]
    with pytest.raises(ValueError, match="committed output"):
        r._validate_ptd_outputs(_meta_body(inputs, outputs), ROWS)

    inputs, outputs = _ptd_io(ROWS, (1, ROWS, HIDDEN))
    outputs[1 + N_LAYERS].shape = [1, 2 * H_KV, ROWS, HEAD_DIM]
    with pytest.raises(ValueError, match="scratch output"):
        r._validate_ptd_outputs(_meta_body(inputs, outputs), ROWS)


# ── prefill ──────────────────────────────────────────────────────────────────


def test_ptd_prefill_promotes_full_chain_head_once():
    succ = lambda t: (t + 2) % 4  # noqa: E731
    prompt = [1, 2, 0, 1, 2, 0]  # 6 tokens, ROWS=4 -> two units
    runner = _PtdGemma(succ, split=True, prompt=prompt, max_seq_len=16)
    runner.run("x")

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


def test_ptd_prefill_prompt_exceeding_capacity_raises():
    runner = _PtdGemma(_cycle, max_seq_len=6, prompt=list(range(6)))
    with pytest.raises(ValueError, match="exceeds the remaining cache capacity"):
        runner._prefill(list(range(6)), start=1)


# ── decode loop ──────────────────────────────────────────────────────────────


def test_ptd_output_token_identical_to_plain_greedy():
    runner = _PtdGemma(_cycle)
    runner.run("x")

    plain = _plain_greedy(_cycle, PROMPT, EOS, M)
    # Committed cache rows 1.. hold the plain chain minus the final bonus
    # (which has no KV yet).
    for i, t in enumerate(plain[:-1]):
        assert runner._cache[0][0, 0, 1 + i, 0] == t, (i, t)
    assert runner._n_tokens_gen == len(plain) - 1

    # Real tree activity: multiple rounds and at least one candidate hit.
    assert runner._ptd_rounds > 1
    hits = [e for e in runner._ptd_log if e["hit"]]
    assert hits, "expected candidate hits with a warmed pool"
    for e in hits:
        assert e["accepted"] >= 1
        assert e["emitted"] == e["accepted"] + 1
        assert e["accepted"] <= min(e["rows"], M - e["L"]) - 1


def test_ptd_round_positions_track_committed_length():
    runner = _PtdGemma(_cycle)
    runner.run("x")
    for toks, pos, vis in runner._ptd_body.calls[2:]:  # after prefill units
        L = int(pos[0])
        assert 1 <= L < M, (L, "PTD round ran without cache room")
        for p in pos[1:]:
            assert L <= p < M, (pos, "row position outside [L, M)")


def test_ptd_miss_path_commits_anchor_without_transfer():
    runner = _PtdGemma(_cycle)
    runner._ptd_reset_request_state()
    emitted, stopped = runner._ptd_round(2, 1, [2], None)
    assert not stopped and emitted == [_cycle(2)]
    assert not runner._ptd_log[0]["hit"]
    # Miss: the present outputs replace the cache entries in place (zero-copy).
    for i in range(N_LAYERS):
        assert runner._ptd_cache[i] is runner._ptd_model.last_results[1 + i]
        assert runner._ptd_cache[i][0, 0, 1, 0] == 2  # anchor committed at L
        assert runner._ptd_cache[i][0, H_KV, 1, 0] == 2
    assert runner._ptd_body.uploads == 0


def test_ptd_commit_cache_hit_splices_combined_scratch():
    runner = _PtdGemma(_cycle)
    body = runner._ptd_body
    toks = np.array([[2, 5, 6, 0]], dtype=np.int64)
    pos = np.array([[3, 4, 5, 3]], dtype=np.int64)
    results = body.infer([toks, pos, np.zeros((1, 1, ROWS, ROWS - 1), dtype=np.float32)])
    node1 = ptd_core.PTDNode(5, 1)
    node2 = ptd_core.PTDNode(6, 2, parent=node1)
    packing = ptd_core.pack_round(3, 2, [node1, node2], [], ROWS, 0)
    assert packing.candidate_slots[node1] == 1 and packing.candidate_slots[node2] == 2

    runner._ptd_commit_cache(results, packing, [node1, node2], 3)
    cache = runner._cache[0]
    assert cache[0, 0, 3, 0] == 2  # anchor at L
    assert cache[0, 0, 4, 0] == 5  # scratch row 0 (slot 1)
    assert cache[0, 0, 5, 0] == 6  # scratch row 1 (slot 2)
    assert cache[0, H_KV, 4, 0] == 5  # combined: value half spliced too
    assert body.uploads == N_LAYERS  # one upload per layer, hit path


def test_ptd_stop_check_in_tail_region():
    """EOS emitted by a PTD round must stop the run exactly like ordinary
    greedy: the emission is truncated at the EOS, which never gets committed."""

    def inc(t: int) -> int:
        return t + 1

    runner = _PtdGemma(inc, max_seq_len=14, rows=8, vocab=16, eos=10)
    runner.run("x")

    # Prompt 1 -> 2..10 (EOS): the cache holds 2..9, the final 10 has no KV.
    assert runner.gen_tokens() == [2, 3, 4, 5, 6, 7, 8, 9]
    assert runner._ptd_cache[0][0, 0, 9, 0] == 0
    assert runner._n_tokens_gen == 8
    # Every step after prefill was a PTD round (no ordinary steps exist in
    # standalone mode), and each used at most the rows that fit at its
    # committed length.
    assert len(runner._ptd_log) == 8
    for e in runner._ptd_log:
        assert e["n_candidates"] + e["n_drafts"] <= min(8, 14 - e["L"]) - 1


def test_ptd_anchor_only_at_capacity_tail():
    """Without a stop token, generation runs to the cache limit: the rounds
    shrink as the row budget (free = min(rows, M - L) - 1) shrinks, and the
    final round at L = M - 1 is a padded anchor-only pass on the same body."""
    runner = _PtdGemma(_cycle, max_seq_len=14, rows=8, eos=99, vocab=16)
    runner.run("x")

    logs = runner._ptd_log
    assert logs[-1]["L"] == 13  # anchor-only round at the last slot
    assert logs[-1]["n_candidates"] + logs[-1]["n_drafts"] == 0
    final = runner._ptd_body.calls[-1]
    assert int(final[1][0]) == 13 and (final[1] == 13).all() and (final[2] == 0).all()
    # The shrinking: later rounds used fewer rows than the earlier ones.
    budgets = [min(8, 14 - e["L"]) - 1 for e in logs]
    for e, budget in zip(logs, budgets):
        assert e["n_candidates"] + e["n_drafts"] <= budget
    assert budgets[-1] == 0 and budgets[0] == 7
    # The run emitted M - len(prompt) + 1 tokens; all but the final bonus
    # are committed, so gen length is M - len(prompt) + 1.
    assert runner._n_tokens_gen == 13


def test_ptd_max_gen_tokens_caps_generation():
    capped = _PtdGemma(_cycle, max_gen_tokens=3)
    plain = _plain_greedy(_cycle, PROMPT, EOS, M, max_gen_tokens=3)
    assert capped.run("x") == " ".join(str(t) for t in plain)
    assert capped._n_tokens_gen == len(plain) - 1

    # remaining == 1 is a budget-zero PTD round (anchor only), not a skip.
    one = _PtdGemma(_cycle, max_gen_tokens=2)
    one.run("x")
    assert one._ptd_log and all(e["emitted"] == 1 and e["accepted"] == 0 for e in one._ptd_log)


def test_ptd_lut_remaps_row_tokens():
    lut = np.array([5, 9, 12, 20])  # compact index -> model token id
    const2 = lambda t: 2  # noqa: E731
    runner = _PtdGemma(const2, vocab=3, eos=12)
    runner._token_id_lut = lut
    runner._ptd_reset_request_state()
    emitted, stopped = runner._ptd_round(5, 1, [5], None)
    # argmax(compact) = 2 -> model id 12 = EOS: emitted and stopped.
    assert emitted == [12] and stopped


# ── stats ────────────────────────────────────────────────────────────────────


def test_ptd_stats_and_summary_line(capsys):
    runner = _PtdGemma(_cycle)
    runner.run("x")
    stats = runner.ptd_stats
    assert stats["rounds"] == runner._ptd_rounds
    assert 0 <= stats["hit_rate"] <= 1
    assert stats["committed"] == sum(e["emitted"] for e in runner._ptd_log)

    print_llm_stats(runner)
    out = capsys.readouterr().out
    assert f"PTD: {stats['rounds']} rounds" in out


def test_ptd_stats_none_without_rounds():
    runner = _PtdGemma(_cycle, eos=1, prompt=[0])  # prefill token is EOS: no rounds
    assert runner.ptd_stats is None
