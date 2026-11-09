# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Batched prefill through LiquidStatic over the fake models: long prompts
run in prefill-size chunks with the remainder as single decode steps, short
prompts stay single steps, and prefill size > max_seq_len is rejected at
construction. The step buffer metadata, prefill/decode input validation, and
chunk orchestration the old tests pinned now live in tests/test_decoder.py
(``test_step_*``, ``test_prefill_input_mismatches_rejected``,
``test_prefill_*``)."""

import importlib.util
from pathlib import Path

import pytest

from llm_fakes import CharTokenizer, make_llm, scripted

# The LFM2.5 folder name is not importable as a package; load by path.
_spec = importlib.util.spec_from_file_location(
    "_lfm2_5_liquid", Path(__file__).resolve().parents[1] / "LiquidAI" / "LiquidAI-LFM2.5" / "src" / "runner.py"
)
_lfm2_5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lfm2_5)
LiquidStatic = _lfm2_5.LiquidStatic


def _liquid(tmp_path, monkeypatch, **kwargs):
    return make_llm(
        tmp_path, monkeypatch, LiquidStatic, tokenizer=CharTokenizer({}),
        config={"bos_token_id": 0, "eos_token_id": 1, "vocab_size": 4096},
        **kwargs,
    )


def test_prefill_chunks_then_remainder_falls_back(tmp_path, monkeypatch):
    llm, models = _liquid(tmp_path, monkeypatch, prefill_size=4, next_fn=scripted([], then=2))
    assert llm.prefill_size == 4

    next_token, pos = llm.prefill_tokens(list(range(1000, 1009)))

    assert (next_token, pos) == (2, 9)
    assert models["prefill"].calls == [
        ([1000, 1001, 1002, 1003], 0),
        ([1004, 1005, 1006, 1007], 4),
    ]
    assert models["body"].calls == [([1008], 8)]  # only the remainder decodes singly


def test_short_prompt_runs_single_steps(tmp_path, monkeypatch):
    llm, models = _liquid(tmp_path, monkeypatch, prefill_size=4, next_fn=scripted([], then=2))

    next_token, pos = llm.prefill_tokens([1000, 1001])

    assert (next_token, pos) == (2, 2)
    assert models["prefill"].calls == []
    assert models["body"].calls == [([1000], 0), ([1001], 1)]


def test_prefill_size_exceeding_max_seq_len_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="exceeds the model max sequence length"):
        _liquid(tmp_path, monkeypatch, prefill_size=32, seq=16)
