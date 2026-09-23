# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Tests for the shared runner's support of the new liquid-style exports:
metadata-driven per-step inputs (position dtype, fixed attention mask) and
batched prefill models whose first output is either hidden states (shared
split head) or already-fused logits (head baked in at export time).
"""

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LIQUID_SRC = _REPO_ROOT / "LiquidAI" / "LiquidAI-LFM2.5" / "src"
if str(_LIQUID_SRC) not in sys.path:
    sys.path.insert(0, str(_LIQUID_SRC))

from runner import LiquidStatic  # noqa: E402
from utils import llm as llm_module  # noqa: E402
from utils.inference import SplitLMHeadRunner, VMFBInferenceRunner  # noqa: E402

# The stub numpy installed by tests/conftest.py has no array APIs; the buffer
# tests below exercise real numpy semantics and are skipped where numpy is
# only stubbed (they run for real once tests/requirements.txt is installed).
REAL_NUMPY = hasattr(np, "zeros")

F32 = "float32"
I64 = "int64"
HIDDEN = 8
LOGITS = 32
KV = ((1, 2, 16, 4), F32)  # one rank-4 KV cache per layer, 2 layers
CONV = ((1, 4, 3), F32)  # one rank-3 conv state per layer, 2 layers
MASK = ((1, 16), I64)  # fixed attention mask, compiled KV window = 16


class TensorInfo:
    def __init__(self, shape, dtype):
        self.shape = list(shape)
        self.dtype = dtype


def _decode_inputs(token_shape=((1, 1, HIDDEN), F32), mask=MASK):
    return [
        TensorInfo(*token_shape),
        TensorInfo((1, 1), I64),
        *([TensorInfo(*mask)] if mask else []),
        TensorInfo(*CONV),
        TensorInfo(*KV),
        TensorInfo(*CONV),
        TensorInfo(*KV),
    ]


def _decode_outputs(primary=((1, 1, HIDDEN), F32)):
    return [
        TensorInfo(*primary),
        TensorInfo(*CONV),
        TensorInfo(*KV),
        TensorInfo(*CONV),
        TensorInfo(*KV),
    ]


class FakeBody:
    """Stands in for ManagedSelfAttnCacheRunner (decode or prefill model)."""

    model_path = "transformer.vmfb"

    def __init__(self, inputs_info, outputs_info):
        self.inputs_info = inputs_info
        self.outputs_info = outputs_info
        self.n_cache_inputs = 4
        self.shared_with = None
        self.infer_calls = []

    @property
    def infer_time_ms(self):
        return 0.0

    def share_kv_cache(self, owner):
        self.shared_with = owner

    def infer(self, inputs, **kwargs):
        self.infer_calls.append((inputs, kwargs))
        return ["primary"]


class FakeLMHead(VMFBInferenceRunner):
    model_path = "lm_head.vmfb"

    def __init__(self):
        self.inputs_info = [TensorInfo((1, 1, HIDDEN), F32)]
        self.outputs_info = [TensorInfo((1, 1, LOGITS), F32)]
        self.infer_calls = []

    def infer(self, inputs, **kwargs):
        self.infer_calls.append(inputs)
        return ["logits"]


def _split_model(body):
    head = FakeLMHead()
    model = SplitLMHeadRunner.__new__(SplitLMHeadRunner)
    model._body = body
    model._lm_head = head
    model._infer_time_ms = 0.0
    return model, head


def _liquid():
    runner = LiquidStatic.__new__(LiquidStatic)
    runner._logger = logging.getLogger("test_liquid_prefill")
    return runner


@pytest.mark.skipif(not REAL_NUMPY, reason="requires real numpy, not the conftest stub")
class TestStepBuffers:
    def test_metadata_driven_position_dtype_and_mask_input(self):
        runner = _liquid()
        body = FakeBody(_decode_inputs(), _decode_outputs())
        runner._model = body
        runner._token_embeddings = np.zeros((4, HIDDEN), dtype=F32)
        runner._init_step_buffers()

        assert runner._pos_buf.dtype == np.dtype(I64)
        assert runner._pos_buf.shape == (1, 1)
        assert runner._emb_buf.dtype == np.dtype(F32)
        assert runner._id_buf is None
        # The fixed attention mask is built from metadata: all-ones.
        assert len(runner._extra_step_inputs) == 1
        mask = runner._extra_step_inputs[0]
        assert mask.dtype == np.dtype(I64)
        assert mask.shape == (1, 16)
        assert np.all(mask == 1)

    def test_two_input_legacy_models_get_no_extra_inputs(self):
        runner = _liquid()
        body = FakeBody(_decode_inputs(mask=None), _decode_outputs())
        runner._model = body
        runner._token_embeddings = None
        runner._init_step_buffers()

        assert runner._extra_step_inputs == []
        # No embedding LUT: the token id buffer follows input 0's metadata.
        assert runner._id_buf.dtype == np.dtype(F32)
        assert runner._id_buf.shape == (1, 1, HIDDEN)

    def test_chunk_step_feeds_token_position_and_mask(self):
        runner = _liquid()
        body = FakeBody(_decode_inputs(), _decode_outputs())
        runner._model = body
        runner._token_embeddings = np.zeros((64, HIDDEN), dtype=F32)
        runner._init_step_buffers()
        runner._prefill_size = 4
        runner._prefill_emb_buf = np.zeros((1, 4, HIDDEN), dtype=F32)
        runner._prefill_id_buf = None

        runner._llm_tokens_step(body, [1, 2, 3, 4], 8, compute_logits=False, sample_next=False)

        inputs, kwargs = body.infer_calls[0]
        assert kwargs == {}
        token_in, pos_in, mask_in = inputs
        assert token_in is runner._prefill_emb_buf
        assert np.array_equal(token_in[0], runner._token_embeddings[[1, 2, 3, 4]])
        assert pos_in[0, 0] == 8
        assert mask_in is runner._extra_step_inputs[0]

    def test_single_token_step_feeds_token_position_and_mask(self):
        runner = _liquid()
        body = FakeBody(_decode_inputs(), _decode_outputs())
        runner._model = body
        runner._token_embeddings = np.zeros((64, HIDDEN), dtype=F32)
        runner._init_step_buffers()
        runner._prefill_model = None
        runner._prefill_size = None
        runner._prefill_emb_buf = None
        runner._prefill_id_buf = None

        runner._llm_tokens_step(body, [7], 3, compute_logits=False, sample_next=False)

        inputs, kwargs = body.infer_calls[0]
        assert len(inputs) == 3
        assert np.array_equal(inputs[0][0, 0], runner._token_embeddings[7])
        assert inputs[1][0, 0] == 3
        assert inputs[2] is runner._extra_step_inputs[0]


class TestPrefillModelSetup:
    def _setup(self, runner, prefill_body):
        with patch.object(
            llm_module, "ManagedSelfAttnCacheRunner", return_value=prefill_body
        ):
            runner._setup_prefill_model("transformer_prefill.vmfb", None, None, False)

    def test_hidden_state_prefill_reuses_shared_head(self):
        runner = _liquid()
        decode_body = FakeBody(_decode_inputs(), _decode_outputs())
        model, head = _split_model(decode_body)
        runner._model = model
        runner._max_seq_len = 16
        prefill_body = FakeBody(
            _decode_inputs(token_shape=((1, 4, HIDDEN), F32)),
            _decode_outputs(),
        )
        prefill_body.model_path = "transformer_prefill.vmfb"

        self._setup(runner, prefill_body)

        assert runner._prefill_size == 4
        assert prefill_body.shared_with is decode_body
        # Wrapped with a real SplitLMHeadRunner sharing the compiled head.
        assert isinstance(runner._prefill_model, SplitLMHeadRunner)
        assert runner._prefill_model._body is prefill_body
        assert runner._prefill_model._lm_head is head

    def test_fused_head_prefill_used_directly_with_split_decode(self):
        runner = _liquid()
        decode_body = FakeBody(_decode_inputs(), _decode_outputs())
        model, _ = _split_model(decode_body)
        runner._model = model
        runner._max_seq_len = 16
        prefill_body = FakeBody(
            _decode_inputs(token_shape=((1, 4, HIDDEN), F32)),
            _decode_outputs(primary=((1, 1, LOGITS), F32)),
        )

        self._setup(runner, prefill_body)

        assert runner._prefill_model is prefill_body
        assert runner._prefill_size == 4

    def test_fused_head_prefill_with_fused_decode(self):
        runner = _liquid()
        runner._model = FakeBody(
            _decode_inputs(), _decode_outputs(primary=((1, 1, LOGITS), F32))
        )
        runner._max_seq_len = 16
        prefill_body = FakeBody(
            _decode_inputs(token_shape=((1, 4, HIDDEN), F32)),
            _decode_outputs(primary=((1, 1, LOGITS), F32)),
        )

        self._setup(runner, prefill_body)

        assert runner._prefill_model is prefill_body

    def test_hidden_state_prefill_without_split_head_rejected(self):
        runner = _liquid()
        runner._model = FakeBody(
            _decode_inputs(), _decode_outputs(primary=((1, 1, LOGITS), F32))
        )
        runner._max_seq_len = 16
        prefill_body = FakeBody(
            _decode_inputs(token_shape=((1, 4, HIDDEN), F32)),
            _decode_outputs(),
        )

        with pytest.raises(ValueError, match="no standalone LM head"):
            self._setup(runner, prefill_body)

    def test_prefill_output_matching_neither_rejected(self):
        runner = _liquid()
        decode_body = FakeBody(_decode_inputs(), _decode_outputs())
        model, _ = _split_model(decode_body)
        runner._model = model
        runner._max_seq_len = 16
        prefill_body = FakeBody(
            _decode_inputs(token_shape=((1, 4, HIDDEN), F32)),
            _decode_outputs(primary=((1, 1, LOGITS + 1), F32)),
        )

        with pytest.raises(ValueError, match="matches neither"):
            self._setup(runner, prefill_body)

    def test_prefill_size_exceeding_max_seq_len_rejected(self):
        runner = _liquid()
        decode_body = FakeBody(_decode_inputs(), _decode_outputs())
        runner._model = decode_body
        runner._max_seq_len = 4
        prefill_body = FakeBody(
            _decode_inputs(token_shape=((1, 8, HIDDEN), F32)),
            _decode_outputs(primary=((1, 1, LOGITS), F32)),
        )

        with pytest.raises(ValueError, match="exceeds the model max sequence length"):
            self._setup(runner, prefill_body)

    def test_prefill_mask_mismatch_rejected(self):
        runner = _liquid()
        decode_body = FakeBody(_decode_inputs(), _decode_outputs())
        runner._model = decode_body
        runner._max_seq_len = 16
        prefill_inputs = _decode_inputs(token_shape=((1, 4, HIDDEN), F32))
        prefill_inputs[2] = TensorInfo((1, 8), I64)  # wrong mask window
        prefill_body = FakeBody(
            prefill_inputs, _decode_outputs(primary=((1, 1, LOGITS), F32))
        )

        with pytest.raises(ValueError, match="non-cache input 2"):
            self._setup(runner, prefill_body)

    def test_prefill_input_count_mismatch_rejected(self):
        runner = _liquid()
        decode_body = FakeBody(_decode_inputs(mask=None), _decode_outputs())
        runner._model = decode_body
        runner._max_seq_len = 16
        prefill_body = FakeBody(
            _decode_inputs(token_shape=((1, 4, HIDDEN), F32)),
            _decode_outputs(primary=((1, 1, LOGITS), F32)),
        )

        with pytest.raises(ValueError, match="non-cache inputs"):
            self._setup(runner, prefill_body)


class RecordingPrefill(LiquidStatic):
    """Records _llm_tokens_step / llm_step calls; skips __init__."""

    __slots__ = ("calls",)

    def __init__(self):
        self.calls = []
        # _prefill() branches on these.
        self._prefill_model = None
        self._prefill_size = None

    def _llm_tokens_step(self, model, tokens, seq_pos, **kwargs):
        self.calls.append(("chunk", tuple(tokens), seq_pos, kwargs))
        return sum(tokens) + 100 if kwargs.get("sample_next") else 0

    def llm_step(self, token, seq_pos, **kwargs):
        self.calls.append(("token", token, seq_pos, kwargs))
        return token + 100 if kwargs.get("sample_next") else 0


class TestChunkedPrefillOrchestration:
    def _recording(self, prefill_size):
        runner = RecordingPrefill()
        runner._warmup_len = 0
        runner._prefill_size = prefill_size
        runner._prefill_model = object()
        return runner

    def test_full_chunks_plus_remainder(self):
        runner = self._recording(4)

        next_token, pos = runner._prefill([1, 2, 3, 4, 5, 6, 7], start=0)

        assert (next_token, pos) == (107, 7)
        assert runner.calls == [
            ("chunk", (1, 2, 3, 4), 0,
             {"compute_logits": False, "sample_next": False}),
            ("token", 5, 4, {"compute_logits": False, "sample_next": False}),
            ("token", 6, 5, {"compute_logits": False, "sample_next": False}),
            ("token", 7, 6, {"compute_logits": True, "sample_next": True}),
        ]

    def test_exact_chunks_sample_last_chunk_only(self):
        runner = self._recording(4)

        next_token, pos = runner._prefill([1, 2, 3, 4, 5, 6, 7, 8], start=0)

        assert (next_token, pos) == (126, 8)
        assert runner.calls == [
            ("chunk", (1, 2, 3, 4), 0,
             {"compute_logits": False, "sample_next": False}),
            ("chunk", (5, 6, 7, 8), 4,
             {"compute_logits": True, "sample_next": True}),
        ]

    def test_remainder_falls_back_to_single_tokens(self):
        runner = self._recording(4)

        next_token, pos = runner._prefill([1, 2, 3, 4, 5, 6], start=2)

        assert (next_token, pos) == (106, 8)
        assert runner.calls == [
            ("chunk", (1, 2, 3, 4), 2,
             {"compute_logits": False, "sample_next": False}),
            ("token", 5, 6, {"compute_logits": False, "sample_next": False}),
            ("token", 6, 7, {"compute_logits": True, "sample_next": True}),
        ]

    def test_short_prompt_stays_single_token(self):
        runner = self._recording(4)

        next_token, pos = runner._prefill([1, 2], start=0)

        assert (next_token, pos) == (102, 2)
        assert runner.calls == [
            ("token", 1, 0, {"compute_logits": False, "sample_next": False}),
            ("token", 2, 1, {"compute_logits": True, "sample_next": True}),
        ]
