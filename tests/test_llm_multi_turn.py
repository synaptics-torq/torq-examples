# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from gemma3.src.runner import Gemma3Static
from qwen2_5.src.runner import QwenStatic
from qwen3.src.runner import Qwen3Static


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LiquidStatic = _load("liquid_multi_turn", "LiquidAI/LiquidAI-LFM2.5/src/runner.py").LiquidStatic
pytestmark = pytest.mark.skipif(not hasattr(np, "zeros"), reason="Requires real numpy")


class Tokenizer:
    def encode(self, text):
        ids = [12] if "model\n" in text or "assistant\n" in text else [10, 11]
        return SimpleNamespace(ids=ids)

    def decode(self, ids):
        return " ".join(map(str, ids))


class Device:
    def __init__(self, array):
        self.array = array

    def to_host(self):
        return self.array


class Cache:
    infer_time_ms = 0

    def __init__(self, capacity):
        self._kv_cache = [Device(np.zeros((1, 1, capacity, 1), dtype=np.int64))]
        self.calls = []

    @property
    def tokens(self):
        return self._kv_cache[0].to_host().reshape(-1)

    def infer(self, inputs, *, entry=None):
        tokens = inputs[0][0].tolist()
        if entry is not None:
            assert entry == f"main_{len(tokens)}"
        pos = int(inputs[1][0, 0])
        assert 0 <= pos and pos + len(tokens) <= len(self.tokens)
        self.calls.append((tokens, pos))
        self.tokens[pos:pos + len(tokens)] = tokens
        logits = np.zeros((1, 1, 100), dtype=np.float32)
        logits[0, 0, {20: 21, 21: 99}.get(tokens[-1], 20)] = 1
        return [logits]

    def restore_kv_state(self, state):
        self._kv_cache[0] = Device(state[0].copy())

    def reset_kv(self):
        self.tokens[:] = 0


class PositionCache(Cache):
    """Store each key's encoded position alongside its token."""

    def __init__(self, capacity):
        super().__init__(capacity)
        self._kv_cache = [Device(np.zeros((1, 1, capacity, 2), dtype=np.int64))]

    @property
    def tokens(self):
        return self._kv_cache[0].to_host()[0, 0, :, 0]

    def infer(self, inputs, *, entry=None):
        pos = int(inputs[1][0, 0])
        positions = self._kv_cache[0].to_host()[0, 0, :, 1]
        np.testing.assert_array_equal(positions[:pos], np.arange(pos))
        result = super().infer(inputs, entry=entry)
        positions[pos:pos + inputs[0].shape[1]] = np.arange(pos, pos + inputs[0].shape[1])
        return result

    def reset_kv(self):
        self._kv_cache[0].to_host()[:] = 0


def _runner(cls=Gemma3Static, instruct=False, batched=False, cap=None, capacity=128):
    runner = cls.__new__(cls)
    runner._logger = logging.getLogger("test_multi_turn")
    runner._instruct_model = instruct
    runner._bos_token_id = 1
    runner._eos_token_id = 99
    runner._eos_token_ids = [99]
    runner._nl_token_id = 8
    runner._double_nl_token_id = 9
    if cls is Gemma3Static:
        runner._end_of_turn_id = 99
    if cls in (QwenStatic, Qwen3Static):
        runner._im_end_id = 99
        runner._endoftext_id = 98
    if cls is Qwen3Static:
        runner._thinking = True
    runner._warmup_len = runner._turn_pos = 2 if instruct else 0
    runner._pending_token = None
    runner._max_seq_len = capacity
    runner._max_gen_tokens = cap
    runner._cache_keep_n = runner._max_user_tokens = runner._max_prompt_tokens = None
    runner._tokenizer = Tokenizer()
    runner._model = Cache(capacity)
    runner._model.tokens[:runner._warmup_len] = 7
    runner._cache_tokens = runner._model.tokens[:runner._warmup_len].tolist()
    runner._reset_cache_state = [runner._model._kv_cache[0].to_host().copy()] if instruct else []
    runner._prefill_model = runner._model if batched is True else None
    runner._native_prefill_entries = [2] if batched == "native" else []
    runner._native_prefill_emb_buf = None
    runner._native_prefill_id_buf = np.zeros((1, 2), dtype=np.int64)
    runner._prefill_size = 2 if batched else None
    runner._prefill_emb_buf = runner._emb_buf = None
    runner._prefill_id_buf = np.zeros((1, 2), dtype=np.int64)
    runner._id_buf = np.zeros((1, 1), dtype=np.int64)
    runner._pos_buf = np.zeros((1, 1), dtype=np.int64)
    runner._extra_step_inputs = []
    runner._token_id_lut = None
    runner._temperature = 0
    runner._debug_logging = False
    return runner


@pytest.mark.parametrize("cls", [Gemma3Static, QwenStatic, Qwen3Static, LiquidStatic])
@pytest.mark.parametrize("instruct", [False, True])
@pytest.mark.parametrize("batched", [False, True, "native"])
@pytest.mark.parametrize("cap", [None, 1])
def test_next_turn_preserves_final_prediction(cls, instruct, batched, cap):
    runner = _runner(cls, instruct, batched, cap)
    assert runner.run("first") == ("20 21 99" if cap is None else "20")
    end = runner._turn_pos
    history = runner._model.tokens[:end].tolist()
    final = 99 if cap is None else 20
    runner.run("second")
    prompt = runner._build_prompt_tokens("second")
    assert runner._model.tokens[:end + 1 + len(prompt)].tolist() == history + [final] + prompt


def test_closing_stream_preserves_last_yielded_token():
    runner = _runner()
    stream = runner.run_stream("first")
    assert next(stream) == "20"
    stream.close()
    end = runner._turn_pos
    runner.run("second")
    assert runner._model.tokens[end] == 20


@pytest.mark.parametrize("instruct", [False, True])
def test_turn_start_rollover_uses_occupied_cache(instruct):
    runner = _runner(instruct=instruct, cap=1, capacity=8)
    runner._cache_keep_n = 3
    runner._turn_pos = 6
    runner._model.tokens[runner._warmup_len:6] = [1, 2, 3, 4, 5, 6][runner._warmup_len:]
    runner._cache_tokens = runner._model.tokens[:6].tolist()
    runner._pending_token = 99
    prefix = runner._model.tokens[:runner._warmup_len].tolist()
    runner.run("next")
    prompt = runner._build_prompt_tokens("next")
    keep = min(3, 8 - len(prefix) - 1 - len(prompt))
    expected = prefix + [4, 5, 6][-keep:] + [99] + prompt
    assert runner._model.tokens[:len(expected)].tolist() == expected


def test_reset_discards_pending_prediction_and_history():
    runner = _runner(instruct=True, cap=1)
    runner.run("first")
    runner.reset()
    assert runner._pending_token is None
    assert runner._turn_pos == 2
    runner.run("next")
    assert runner._model.tokens[:5].tolist() == [7, 7, 10, 11, 12]


def test_full_cache_without_window_discards_previous_prediction():
    runner = _runner(cap=1, capacity=8)
    runner._turn_pos = 7
    runner._model.tokens[:7] = 6
    runner._cache_tokens = runner._model.tokens[:7].tolist()
    runner._pending_token = 99
    runner.run("next")
    assert runner._model.tokens.tolist() == [10, 11, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("cls", [Gemma3Static, QwenStatic, Qwen3Static])
@pytest.mark.parametrize("instruct", [False, True])
@pytest.mark.parametrize("batched", [False, True, "native"])
def test_full_cache_reserves_room_for_complete_prompt(cls, instruct, batched):
    runner = _runner(cls, instruct, batched, cap=1, capacity=8)
    runner._turn_pos = 8
    runner._cache_keep_n = 8
    runner._pending_token = 99
    runner._model.tokens[runner._warmup_len:] = list(range(8))[runner._warmup_len:]
    runner._cache_tokens = runner._model.tokens.tolist()
    assert runner.run("next") == "20"
    prompt = runner._build_prompt_tokens("next")
    assert runner._model.tokens[-len(prompt) - 1:].tolist() == [99] + prompt


@pytest.mark.parametrize("instruct", [False, True])
@pytest.mark.parametrize("keep", [6, 8, 9])
def test_large_window_frees_space_on_every_rollover(instruct, keep):
    runner = _runner(instruct=instruct, cap=20, capacity=8)
    runner._cache_keep_n = keep
    runner._eos_token_id = runner._end_of_turn_id = 98
    runner._eos_token_ids = [98]
    assert len(runner.run("next").split()) == 20
    assert runner._turn_pos <= 8


def test_oversized_prompt_takes_priority_over_history():
    runner = _runner(cap=1, capacity=8)
    runner._cache_keep_n = 5
    runner._turn_pos = 2
    runner._model.tokens[:2] = [4, 5]
    runner._cache_tokens = [4, 5]
    runner._pending_token = 99
    runner._tokenizer = Mock()
    runner._tokenizer.encode.return_value = SimpleNamespace(ids=[10] * 8)
    runner._tokenizer.decode.return_value = "answer"
    runner.run("long")
    assert runner._model.tokens.tolist() == [10] * 8


def test_in_turn_rollover_then_next_turn_preserves_pending_token():
    runner = _runner(capacity=8)
    runner._turn_pos = 6
    runner._model.tokens[:6] = [1, 2, 3, 4, 5, 6]
    runner._cache_tokens = runner._model.tokens[:6].tolist()
    runner._cache_keep_n = 3
    assert runner.run("next") == "20 21 99"
    assert runner._model.tokens[:5].tolist() == [6, 10, 11, 20, 21]
    runner._max_gen_tokens = 1
    runner.run("next")
    assert runner._model.tokens[:8].tolist() == [6, 10, 11, 20, 21, 99, 10, 11]


@pytest.mark.parametrize("cls", [Gemma3Static, QwenStatic, Qwen3Static])
@pytest.mark.parametrize("batched", [False, True, "native"])
@pytest.mark.parametrize("instruct", [False, True])
def test_default_window_recomputes_key_positions_across_turns(cls, batched, instruct):
    runner = _runner(cls, instruct=instruct, batched=batched, capacity=8)
    runner._model = PositionCache(8)
    runner._model.tokens[:runner._warmup_len] = 7
    runner._model._kv_cache[0].to_host()[0, 0, :runner._warmup_len, 1] = np.arange(runner._warmup_len)
    runner._reset_cache_state = [runner._model._kv_cache[0].to_host().copy()]
    runner._prefill_model = runner._model if batched is True else None
    runner._cache_keep_n = 2
    for _ in range(4):
        assert runner.run("next") == "20 21 99"
        assert runner._cache_tokens == runner._model.tokens[:runner._turn_pos].tolist()


def test_failed_step_does_not_advance_cache_position():
    runner = _runner()
    runner._model.infer = Mock(side_effect=RuntimeError("inference failed"))
    with pytest.raises(RuntimeError, match="inference failed"):
        runner.llm_step(10, 0)
    assert runner._turn_pos == 0


def test_generation_validation_resets_between_samples(monkeypatch):
    monkeypatch.setitem(sys.modules, "runner", sys.modules[Gemma3Static.__module__])
    validate = _load("gemma_multi_turn_validate", "gemma3/src/validate.py")
    validator = validate.TextGenerationValidator.__new__(validate.TextGenerationValidator)
    validator.dataset = [{"src": "a", "tgt": "b"}, {"src": "c", "tgt": "d"}]
    validator.n = 2
    validator.few_shot_prompt = None
    validator.tgt_lang = "Spanish"
    validator.verbose = False
    validator.output_csv = None
    validator.gemma3 = Mock(last_infer_time=1, time_to_first_token=1, generated_tokens=1)
    validator.gemma3.run.return_value = "b"
    validator.run()
    assert [call[0] for call in validator.gemma3.method_calls] == ["reset", "run", "reset", "run"]
