# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Tests for the shared runner's prompt-length limit (``--max-inp-len``)."""

import logging

from utils.llm import DecoderOnlyLLMRunner


class _Probe(DecoderOnlyLLMRunner):
    """Minimal concrete runner; __init__ is skipped, fields set per test."""

    def tokenize(self, text: str, role: str | None = None) -> list[int]:
        raise NotImplementedError


def _runner(max_prompt_tokens, max_user_tokens=None):
    runner = _Probe.__new__(_Probe)
    runner._logger = logging.getLogger("test_llm_prompt_limit")
    runner._max_prompt_tokens = max_prompt_tokens
    runner._max_user_tokens = max_user_tokens
    return runner


def test_long_prompt_truncated_to_max_inp_len():
    runner = _runner(32)
    assert runner._apply_prompt_limit(list(range(40))) == list(range(32))


def test_exact_length_prompt_unchanged():
    runner = _runner(32)
    assert runner._apply_prompt_limit(list(range(32))) == list(range(32))


def test_short_prompt_not_padded():
    # Regression (bugs-and-todos B3): the old code padded short prompts
    # with pad token 0 at the end of the sequence. The static exports have
    # a baked-in causal mask (no runtime attention-mask input), so the
    # model attended over the padding and emitted EOS immediately — the
    # demo printed an empty answer.
    runner = _runner(32)
    tokens = [7, 9, 1]
    assert runner._apply_prompt_limit(tokens) == tokens


def test_no_limit_leaves_prompt_unchanged():
    runner = _runner(None)
    assert runner._apply_prompt_limit([1, 2, 3]) == [1, 2, 3]


def test_warmup_user_limit_truncates_without_padding():
    # With a warm-up prefix the user prompt is limited to the remaining
    # capacity; it is still a maximum, never a pad target.
    runner = _runner(32, max_user_tokens=10)
    assert runner._apply_prompt_limit(list(range(15))) == list(range(10))
    assert runner._apply_prompt_limit(list(range(4))) == list(range(4))
