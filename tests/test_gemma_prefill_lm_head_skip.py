# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Gemma chat behaviour on the shared LLM: prefill after the warm-up prefix,
Gemma turn tokenization (the tokenizer's auto-BOS stripped), and the stop
policy. The LM-head skip, head shape/dtype validation, and single-step
chunking the old SplitLMHeadRunner/Gemma3Static tests pinned now live in
tests/test_decoder.py (``test_split_head_runs_and_skips_without_logits``,
``test_head_metadata_mismatches_rejected``, ``test_prefill_*``)."""

from gemma3.src.runner import Gemma3Static
from llm_fakes import CHAR_BASE, CharTokenizer, cached_tokens, make_llm, scripted
from utils.llm import LLM


def _gemma_tok():
    return CharTokenizer({"<start_of_turn>": 10, "<end_of_turn>": 11}, bos_id=2)


def _gemma(tmp_path, monkeypatch, **kwargs):
    return make_llm(
        tmp_path, monkeypatch, Gemma3Static, tokenizer=_gemma_tok(),
        config={"bos_token_id": 2, "eos_token_id": 1, "vocab_size": 4096},
        **kwargs,
    )


def test_prefill_tokens_starts_after_warmup(tmp_path, monkeypatch):
    tok = _gemma_tok()
    llm, models = _gemma(tmp_path, monkeypatch, instruct_model=True, sys_prompt="S",
                         next_fn=scripted([7], then=1))
    warm = [2] + tok.ids("<start_of_turn>system\nS<end_of_turn>\n")
    assert cached_tokens(llm) == warm

    next_token, pos = llm.prefill_tokens([1000, 1001])

    assert (next_token, pos) == (7, len(warm) + 2)
    assert models["body"].calls[-2:] == [([1000], len(warm)), ([1001], len(warm) + 1)]
    assert cached_tokens(llm) == warm + [1000, 1001]


def test_gemma_tokenize_strips_bos_and_builds_turns(tmp_path, monkeypatch):
    tok = _gemma_tok()
    llm, _ = _gemma(tmp_path, monkeypatch, instruct_model=True)
    assert llm.tokenize("hello", "user") == [10, *tok.ids("user\nhello"), 11, *tok.ids("\n")]
    assert llm.tokenize("", "model") == [10, *tok.ids("model\n")]
    assert llm._build_prompt_tokens("hello") == (
        llm.tokenize("hello", "user") + llm.tokenize("", "model")
    )


def test_gemma_stops_at_end_of_turn(tmp_path, monkeypatch):
    llm, _ = _gemma(tmp_path, monkeypatch, instruct_model=True, sys_prompt="S",
                    next_fn=scripted([CHAR_BASE + ord("a"), 11], then=1))
    assert llm._should_stop(1, [1])          # the config eos
    assert llm._should_stop(11, [11])        # the template's end-of-turn
    assert not llm._should_stop(1105, [1105])
    assert llm.run("hi") == "a"              # end-of-turn ends the answer, unemitted
    assert llm._pending_token == 11


def test_base_model_stops_at_blank_line(tmp_path, monkeypatch):
    llm, _ = make_llm(
        tmp_path, monkeypatch, cls=LLM, tokenizer=CharTokenizer({}),
        config={"bos_token_id": 0, "eos_token_id": 1, "vocab_size": 4096},
    )
    nl = CHAR_BASE + ord("\n")
    assert llm._should_stop(nl, [1001, 1002, nl])  # an answer, then a blank line
    assert llm._should_stop(nl, [1001, nl, nl])
    assert not llm._should_stop(nl, [nl, nl])      # the stop needs earlier history
