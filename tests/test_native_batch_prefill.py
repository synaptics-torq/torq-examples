# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Tests for native batch prefill: a prefill family (main_4, main_16, …)
compiled into the decode model's own VMFB image, invoked by name on the
same loaded weights with the managed KV cache.

Covers entry-name parsing, prompt chunk planning, native setup (entry
discovery, buffer allocation), chunk steps through named entries, prefill
orchestration, and the multi-entry invocation on the managed cache runner.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LIQUID_SRC = _REPO_ROOT / "LiquidAI" / "LiquidAI-LFM2.5" / "src"
if str(_LIQUID_SRC) not in sys.path:
    sys.path.insert(0, str(_LIQUID_SRC))

from runner import LiquidStatic  # noqa: E402
from utils import llm as llm_module  # noqa: E402
from utils.inference import (  # noqa: E402
    ManagedSelfAttnCacheRunner,
    SplitLMHeadRunner,
)
from utils.llm import parse_prefill_entry_size  # noqa: E402

# The stub numpy installed by tests/conftest.py has no array APIs; the buffer
# tests below exercise real numpy semantics and are skipped where numpy is
# only stubbed (they run for real once tests/requirements.txt is installed).
REAL_NUMPY = hasattr(np, "zeros")

F32 = "float32"
I64 = "int64"
HIDDEN = 8
LOGITS = 32
KV = ((1, 2, 16, 4), F32)


class TensorInfo:
    def __init__(self, shape, dtype):
        self.shape = list(shape)
        self.dtype = dtype


class FakeBody:
    """Stands in for ManagedSelfAttnCacheRunner (the decode model)."""

    model_path = "transformer.vmfb"

    def __init__(self, function_names, inputs_info=None, outputs_info=None):
        self.function_names = function_names
        self.inputs_info = inputs_info or [
            TensorInfo((1, 1, HIDDEN), F32),
            TensorInfo((1, 1), I64),
            TensorInfo(*KV),
        ]
        self.outputs_info = outputs_info or [
            TensorInfo((1, 1, HIDDEN), F32),
            TensorInfo(*KV),
        ]
        self.n_cache_inputs = 1
        self.infer_calls = []

    @property
    def infer_time_ms(self):
        return 0.0

    def infer(self, inputs, **kwargs):
        self.infer_calls.append((inputs, kwargs))
        return ["primary"]


class FakeLMHead:
    model_path = "lm_head.vmfb"
    inputs_info = [TensorInfo((1, 1, HIDDEN), F32)]
    outputs_info = [TensorInfo((1, 1, LOGITS), F32)]

    def __init__(self):
        self.infer_calls = []

    def infer(self, inputs, **kwargs):
        self.infer_calls.append(inputs)
        return ["logits"]


def _liquid(function_names=("main",)):
    runner = LiquidStatic.__new__(LiquidStatic)
    runner._logger = logging.getLogger("test_native_prefill")
    runner._debug_logging = False
    runner._cache_tokens = []
    runner._model = FakeBody(function_names)
    return runner


def _split_model(body):
    head = FakeLMHead()
    model = SplitLMHeadRunner.__new__(SplitLMHeadRunner)
    model._body = body
    model._lm_head = head
    model._infer_time_ms = 0.0
    return model, head


# ── entry name parsing ────────────────────────────────────────────────────────


class TestParsePrefillEntrySize:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("main", None),
            ("main_4", 4),
            ("main_16", 16),
            ("main_256", 256),
            ("main_1", None),
            ("main_0", None),
            ("main_x", None),
            ("main_4x", None),
            ("other_16", None),
        ],
    )
    def test_parse(self, name, expected):
        assert parse_prefill_entry_size(name) == expected


# ── chunk planning ────────────────────────────────────────────────────────────


class TestPlanPrefillChunks:
    plan = staticmethod(llm_module.DecoderOnlyLLMRunner._plan_prefill_chunks)

    def test_plan_covers_the_prompt_exactly(self):
        for n in range(0, 400):
            entries = [4, 16, 64, 256]
            assert sum(self.plan(n, entries)) == n

    def test_greedy_largest_first(self):
        assert self.plan(21, [4, 16, 64]) == [16, 4, 1]

    def test_exact_entry_single_chunk(self):
        assert self.plan(64, [4, 16, 64, 256]) == [64]

    def test_entry_plus_single_token(self):
        assert self.plan(65, [4, 16, 64, 256]) == [64, 1]

    def test_below_smallest_entry_all_single_tokens(self):
        assert self.plan(3, [4, 16, 64]) == [1, 1, 1]

    def test_empty_prompt_empty_plan(self):
        assert self.plan(0, [4, 16, 64]) == []

    def test_no_oversized_entries_used(self):
        # 40 = 16 + 16 + 4 + 4, never 64.
        assert self.plan(40, [4, 16, 64]) == [16, 16, 4, 4]

    def test_small_entries_with_remainder(self):
        assert self.plan(7, [4, 16]) == [4, 1, 1, 1]


# ── native setup ──────────────────────────────────────────────────────────────


@pytest.mark.skipif(not REAL_NUMPY, reason="requires real numpy, not the conftest stub")
class TestNativePrefillSetup:
    def _runner(self, function_names, token_embeddings=None, max_seq_len=64):
        runner = _liquid(function_names)
        runner._token_embeddings = token_embeddings
        runner._max_seq_len = max_seq_len
        # Mirror DecoderOnlyLLMRunner.__init__, which the fakes skip.
        runner._native_prefill_entries = []
        runner._native_prefill_emb_buf = None
        runner._native_prefill_id_buf = None
        return runner

    def test_entries_detected_and_embedding_buffer_allocated(self):
        runner = self._runner(
            ["main", "main_4", "main_16", "main_64"],
            token_embeddings=np.zeros((10, HIDDEN), dtype=F32),
        )

        assert runner._setup_native_prefill() is True
        assert runner._native_prefill_entries == [4, 16, 64]
        assert runner._native_prefill_emb_buf is not None
        assert runner._native_prefill_emb_buf.shape == (1, 64, HIDDEN)
        assert runner._native_prefill_emb_buf.dtype == np.dtype(F32)
        assert runner._native_prefill_id_buf is None

    def test_entries_above_max_seq_len_are_dropped(self):
        runner = self._runner(
            ["main", "main_4", "main_16", "main_64", "main_256"],
            token_embeddings=np.zeros((10, HIDDEN), dtype=F32),
            max_seq_len=16,
        )

        assert runner._setup_native_prefill() is True
        assert runner._native_prefill_entries == [4, 16]
        assert runner._native_prefill_emb_buf.shape == (1, 16, HIDDEN)

    def test_no_entries(self):
        runner = self._runner(["main"])
        assert runner._setup_native_prefill() is False
        assert runner._native_prefill_entries == []

    def test_body_without_function_names(self):
        runner = _liquid(None)
        del runner._model.function_names
        runner._token_embeddings = None
        runner._max_seq_len = 16
        runner._native_prefill_emb_buf = None
        runner._native_prefill_id_buf = None
        assert runner._setup_native_prefill() is False

    def test_id_input_buffer_without_embedding_lut(self):
        runner = _liquid(["main", "main_4", "main_16"])
        runner._token_embeddings = None
        runner._max_seq_len = 16
        runner._native_prefill_emb_buf = None
        runner._native_prefill_id_buf = None
        runner._model.inputs_info = [
            TensorInfo((1, 1), I64),
            TensorInfo((1, 1), I64),
            TensorInfo(*KV),
        ]

        assert runner._setup_native_prefill() is True
        assert runner._native_prefill_id_buf.shape == (1, 16)
        assert runner._native_prefill_id_buf.dtype == np.dtype(I64)
        assert runner._native_prefill_emb_buf is None

    def test_split_model_uses_body_entries(self):
        body = FakeBody(["main", "main_4", "main_16"])
        model, _ = _split_model(body)
        runner = _liquid(None)
        runner._model = model
        runner._token_embeddings = np.zeros((10, HIDDEN), dtype=F32)
        runner._max_seq_len = 16
        runner._native_prefill_emb_buf = None
        runner._native_prefill_id_buf = None

        assert runner._setup_native_prefill() is True
        assert runner._native_prefill_entries == [4, 16]


# ── chunk steps through named entries ─────────────────────────────────────────


@pytest.mark.skipif(not REAL_NUMPY, reason="requires real numpy, not the conftest stub")
class TestNativeChunkStep:
    def _runner(self, token_embeddings=None):
        runner = _liquid(["main", "main_4", "main_16"])
        runner._token_embeddings = token_embeddings
        runner._max_seq_len = 16
        runner._init_step_buffers()
        if token_embeddings is not None:
            runner._native_prefill_emb_buf = np.zeros((1, 16, HIDDEN), dtype=F32)
            runner._native_prefill_id_buf = None
        else:
            runner._native_prefill_emb_buf = None
            runner._native_prefill_id_buf = np.zeros((1, 16), dtype=I64)
        return runner

    def test_embedding_chunk_feeds_entry_and_position(self):
        runner = self._runner(np.zeros((64, HIDDEN), dtype=F32))
        runner._pos_buf[0, 0] = -1
        runner._model.infer_calls = []

        runner._llm_tokens_step(
            runner._model, [1, 2, 3, 4], 8,
            compute_logits=False, sample_next=False, entry="main_4",
        )

        inputs, kwargs = runner._model.infer_calls[0]
        assert kwargs == {"entry": "main_4"}
        token_in, pos_in = inputs[0], inputs[1]
        assert token_in.shape == (1, 4, HIDDEN)
        assert token_in.base is runner._native_prefill_emb_buf
        assert np.array_equal(token_in, runner._token_embeddings[[1, 2, 3, 4]][None])
        assert pos_in[0, 0] == 8

    def test_id_chunk_feeds_entry_and_position(self):
        runner = self._runner(None)
        runner._pos_buf[0, 0] = -1

        runner._llm_tokens_step(
            runner._model, [5, 6], 3,
            compute_logits=False, sample_next=False, entry="main_4",
        )

        inputs, kwargs = runner._model.infer_calls[0]
        assert kwargs == {"entry": "main_4"}
        assert np.array_equal(inputs[0], np.array([[5, 6]], dtype=I64))
        assert inputs[1][0, 0] == 3

    def test_split_model_body_only_chunk_skips_head(self):
        body = FakeBody(["main", "main_4", "main_16"])
        model, head = _split_model(body)
        runner = self._runner(np.zeros((64, HIDDEN), dtype=F32))
        runner._model = model
        runner._init_step_buffers()

        runner._llm_tokens_step(
            model, [1, 2], 0,
            compute_logits=False, sample_next=False, entry="main_4",
        )

        assert body.infer_calls[0][1] == {"entry": "main_4"}
        assert head.infer_calls == []

    def test_split_model_chunk_routes_last_hidden_through_head(self):
        body = FakeBody(["main", "main_4", "main_16"])
        head = FakeLMHead()
        model = SplitLMHeadRunner.__new__(SplitLMHeadRunner)
        model._body = body
        model._lm_head = head
        model._infer_time_ms = 0.0

        assert model.infer(["h"], entry="main_4") == ["logits"]
        assert body.infer_calls == [(["h"], {"entry": "main_4"})]
        assert head.infer_calls == [["primary"]]
        assert model.infer(["h"], skip_lm_head=True, entry="main_4") == ["primary"]
        assert head.infer_calls == [["primary"]]


# ── prefill orchestration ─────────────────────────────────────────────────────


class RecordingRunner(LiquidStatic):
    """Records native chunk/token calls; skips __init__."""

    __slots__ = ("calls",)

    def __init__(self, entries):
        self.calls = []
        self._logger = logging.getLogger("test_native_recording")
        self._native_prefill_entries = entries
        self._model = FakeBody(tuple(f"main_{e}" for e in entries))
        self._warmup_len = 0

    def _llm_tokens_step(self, model, tokens, seq_pos, **kwargs):
        self.calls.append(("chunk", kwargs.get("entry"), tuple(tokens), seq_pos, kwargs))
        return sum(tokens) + 100 if kwargs.get("sample_next") else 0

    def llm_step(self, token, seq_pos, **kwargs):
        self.calls.append(("token", None, (token,), seq_pos, kwargs))
        return token + 100 if kwargs.get("sample_next") else 0


class TestNativePrefillOrchestration:
    def _recording(self, entries):
        return RecordingRunner(entries)

    def _body_only(self):
        return {"compute_logits": False, "sample_next": False}

    def _sample(self):
        return {"compute_logits": True, "sample_next": True}

    def test_mixed_entries_and_single_token_tail(self):
        runner = self._recording([4, 16, 64])

        next_token, pos = runner._prefill(
            list(range(1, 22)), start=0
        )

        assert (next_token, pos) == (121, 21)
        entries = [c[1] for c in runner.calls]
        assert entries == ["main_16", "main_4", None]
        assert runner.calls[0][4] == {**self._body_only(), "entry": "main_16"}
        assert runner.calls[1][4] == {**self._body_only(), "entry": "main_4"}
        assert runner.calls[2][4] == self._sample()
        # Chunks carry the full window; positions advance per unit.
        assert runner.calls[0][2] == tuple(range(1, 17))
        assert runner.calls[0][3] == 0
        assert runner.calls[1][2] == (17, 18, 19, 20)
        assert runner.calls[1][3] == 16
        assert runner.calls[2][2] == (21,)
        assert runner.calls[2][3] == 20

    def test_exact_entry_single_chunk(self):
        runner = self._recording([4, 16, 64])

        next_token, pos = runner._prefill(list(range(64)), start=5)

        assert (next_token, pos) == (sum(range(64)) + 100, 69)
        assert [c[1] for c in runner.calls] == ["main_64"]
        assert runner.calls[0][3] == 5
        assert runner.calls[0][4] == {**self._sample(), "entry": "main_64"}

    def test_entry_plus_single_token(self):
        runner = self._recording([4, 16, 64])

        next_token, pos = runner._prefill(list(range(65)), start=0)

        assert (next_token, pos) == (164, 65)
        assert [c[1] for c in runner.calls] == ["main_64", None]
        assert runner.calls[0][4] == {**self._body_only(), "entry": "main_64"}
        assert runner.calls[1][4] == self._sample()
        assert runner.calls[1][3] == 64

    def test_small_prompt_single_tokens_only(self):
        runner = self._recording([4, 16, 64])

        next_token, pos = runner._prefill([1, 2, 3], start=0)

        assert (next_token, pos) == (103, 3)
        assert all(c[1] is None for c in runner.calls)
        assert runner.calls[-1][4] == self._sample()

    def test_warmup_produces_no_token(self):
        runner = self._recording([4, 16, 64])

        next_token, pos = runner._prefill(
            list(range(20)), start=0, produce_next_token=False
        )

        assert (next_token, pos) == (0, 20)
        assert [c[1] for c in runner.calls] == ["main_16", "main_4"]
        assert runner.calls[0][4] == {**self._body_only(), "entry": "main_16"}
        assert runner.calls[1][4] == {**self._body_only(), "entry": "main_4"}

    def test_native_takes_precedence_over_sibling_size(self):
        runner = self._recording([4, 16])
        # A sibling prefill model set up at the same time must be ignored.
        runner._prefill_model = object()
        runner._prefill_size = 4

        next_token, pos = runner._prefill([1, 2, 3, 4, 5], start=0)

        assert (next_token, pos) == (5 + 100, 5)
        assert [c[1] for c in runner.calls] == ["main_4", None]


# ── multi-entry invocation on the managed cache runner ────────────────────────


class _FakeInvoker:
    def __init__(self, tag, cache_out="kvout"):
        self.tag = tag
        self.cache_out = cache_out
        self.calls = []

    def __call__(self, *inputs):
        self.calls.append(list(inputs))
        return ("out0", self.cache_out)


def _managed_runner(n_kv=1):
    runner = ManagedSelfAttnCacheRunner.__new__(ManagedSelfAttnCacheRunner)
    runner._model_path = "transformer.vmfb"
    runner._logger = logging.getLogger("test_managed_entry")
    runner._device_io = False
    runner._device_outputs = True
    runner._cache_start_idx = 1
    runner._n_kv = n_kv
    runner._kv_seq_len = None
    runner._rows_axis = [None] * n_kv
    runner._kv_cache = ["kvin"]
    runner._invoker = _FakeInvoker("main")
    runner._bound_module = {"main": runner._invoker}
    runner._function_names = ["main"]
    return runner


class TestManagedRunnerEntryInvoke:
    def test_entry_invokes_named_export_and_reads_back_cache(self):
        runner = _managed_runner()
        entry = _FakeInvoker("main_16", cache_out="kv16")
        runner._bound_module["main_16"] = entry
        runner._function_names = ["main", "main_16"]

        out = runner.infer(["tok", "pos"], entry="main_16")

        assert out == ["out0"]
        assert entry.calls == [["tok", "pos", "kvin"]]
        assert runner._kv_cache == ["kv16"]
        assert runner._invoker.calls == []
        assert runner._infer_time_ms >= 0.0

    def test_default_entry_still_used_without_entry(self):
        runner = _managed_runner()

        runner.infer(["tok", "pos"])

        assert runner._invoker.calls == [["tok", "pos", "kvin"]]

    def test_multiple_entries_share_one_cache_set(self):
        runner = _managed_runner()
        main_4 = _FakeInvoker("main_4", cache_out="kv4")
        runner._bound_module["main_4"] = main_4
        runner._function_names = ["main", "main_4"]

        runner.infer(["a"], entry="main_4")
        runner.infer(["b"])  # back to main, continuing from the updated cache

        assert main_4.calls[0] == ["a", "kvin"]
        assert runner._invoker.calls[0] == ["b", "kv4"]

    def test_unknown_entry_raises_with_valid_names(self):
        runner = _managed_runner()
        runner._function_names = ["main", "main_4"]

        with pytest.raises(ValueError, match="Function 'main_16' not found"):
            runner.infer(["a"], entry="main_16")
        with pytest.raises(ValueError, match="main_4"):
            runner.infer(["a"], entry="main_16")


class TestSplitRunnerEntryForwarding:
    def test_entry_forwarded_to_body(self):
        body = FakeBody(["main", "main_16"])
        model, head = _split_model(body)

        model.infer(["h"], entry="main_16")

        assert body.infer_calls == [(["h"], {"entry": "main_16"})]
        assert head.infer_calls == [["primary"]]
