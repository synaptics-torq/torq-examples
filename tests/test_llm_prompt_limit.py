# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Prompt-length limiting (``--max-inp-len``): long prompts are truncated to
the limit, short prompts pass through unchanged — the limit is a maximum,
never a pad target (the static exports bake the causal mask in, so padding
tokens would be attended over and the model emits EOS immediately)."""

from llm_fakes import CharTokenizer, make_llm


def _llm(tmp_path, monkeypatch, **kwargs):
    llm, _ = make_llm(
        tmp_path, monkeypatch, tokenizer=CharTokenizer({}),
        config={"bos_token_id": 0, "eos_token_id": 1, "vocab_size": 4096},
        **kwargs,
    )
    return llm


def test_long_prompt_truncated_to_max_prompt_tokens(tmp_path, monkeypatch):
    llm = _llm(tmp_path, monkeypatch, max_prompt_tokens=32)
    assert llm._apply_prompt_limit(list(range(40))) == list(range(32))


def test_prompt_at_or_under_the_limit_unchanged(tmp_path, monkeypatch):
    llm = _llm(tmp_path, monkeypatch, max_prompt_tokens=32)
    assert llm._apply_prompt_limit(list(range(32))) == list(range(32))
    assert llm._apply_prompt_limit([7, 9, 1]) == [7, 9, 1]  # no padding


def test_no_limit_leaves_prompt_unchanged(tmp_path, monkeypatch):
    llm = _llm(tmp_path, monkeypatch)
    assert llm._apply_prompt_limit([1, 2, 3]) == [1, 2, 3]


def test_user_limit_takes_precedence_over_prompt_limit(tmp_path, monkeypatch):
    # With a warm-up prefix the user prompt is limited to the remaining
    # capacity; it is still a maximum, never a pad target.
    llm = _llm(tmp_path, monkeypatch, max_prompt_tokens=32)
    llm._max_user_tokens = 10
    assert llm._apply_prompt_limit(list(range(15))) == list(range(10))
    assert llm._apply_prompt_limit(list(range(4))) == list(range(4))
