# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Multi-turn KV cache behaviour of the shared ``utils.llm.LLM``, driven end
to end over fake models: the pending token handoff between turns, reset back
to the warm-up prefix, turn-start and in-turn window rolls (kept tokens
replayed at their new positions), the no-window restart, prompt
truncation, and batched-prefill parity.
"""

import logging
from unittest.mock import Mock

import pytest

from gemma3.src.runner import Gemma3Static
from llm_fakes import CHAR_BASE, CharTokenizer, cached_tokens, make_llm, scripted
from qwen2_5.src.runner import QwenStatic
from qwen3.src.runner import Qwen3Static
from utils.tasks.text_generation import run_prompts, run_teacher_forced

# Built from pieces, not literals, so the strings match the templates'.
IM_START, IM_END = "<|" + "im_start|>", "<|" + "im_end|>"
ENDOFTEXT = "<|" + "endoftext|>"


def _chars(text: str) -> list[int]:
    return [CHAR_BASE + ord(ch) for ch in text]


#: The answer's stop token: <end_of_turn> for Gemma, im_end for Qwen.
STOP = 106

#: (tokenizer specials, auto-BOS, config, answer stop token) per binding.
SPECS = {
    Gemma3Static: (
        {"<bos>": 2, "<eos>": 1, "<start_of_turn>": 105, "<end_of_turn>": 106},
        2,
        {"bos_token_id": 2, "eos_token_id": 1, "vocab_size": 4096},
        106,
    ),
    QwenStatic: (
        {IM_START: 105, IM_END: 106, ENDOFTEXT: 98, "<eos>": 1},
        None,
        {"bos_token_id": 1, "eos_token_id": 1, "vocab_size": 4096},
        106,
    ),
    Qwen3Static: (
        {IM_START: 105, IM_END: 106, ENDOFTEXT: 98, "<eos>": 1},
        None,
        {"bos_token_id": 1, "eos_token_id": 1, "vocab_size": 4096},
        106,
    ),
}


def _build(cls, tmp, monkeypatch, *, instruct, next_fn, seq=64, keep=None, cap=None,
           prefill_size=None, sys_prompt="S"):
    specials, bos_id, config, stop = SPECS[cls]
    kwargs = dict(instruct_model=instruct, sys_prompt=sys_prompt, max_gen_tokens=cap)
    if keep is not None:
        kwargs["cache_keep_n"] = keep
    llm, models = make_llm(
        tmp, monkeypatch, cls, tokenizer=CharTokenizer(specials, bos_id=bos_id),
        config=config, seq=seq, prefill_size=prefill_size, next_fn=next_fn, **kwargs,
    )
    return llm, models, stop


def fill(llm, tokens) -> None:
    """Occupy the first ``len(tokens)`` cache positions as prior history."""
    llm._cache.inputs[0].to_host()[0, 0, : len(tokens), 0] = tokens
    llm._cache_tokens = list(tokens)
    llm._turn_pos = len(tokens)


@pytest.mark.parametrize("cls", [Gemma3Static, QwenStatic, Qwen3Static])
@pytest.mark.parametrize("instruct", [False, True])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("cap", [None, 1])
def test_next_turn_prefills_right_after_first_answer(
    cls, instruct, batched, cap, tmp_path, monkeypatch
):
    # The first answer's final prediction is not in the cache yet; the second
    # turn prefills it right after the first turn, then its own prompt. Qwen
    # chat turns are long, so the cache is big.
    A = _chars("Hi")
    llm, models, stop = _build(
        cls, tmp_path, monkeypatch, instruct=instruct, seq=128,
        next_fn=scripted((A + [STOP]) * 2, then=STOP),
        cap=cap, prefill_size=2 if batched else None,
    )
    warm = cached_tokens(llm)
    p1 = llm._build_prompt_tokens("first")
    p2 = llm._build_prompt_tokens("second")
    if cap is None:
        assert llm.run("first") == "Hi"
        assert cached_tokens(llm) == warm + p1 + A
        assert llm._pending_token == stop
        assert llm.run("second") == "Hi"
        assert cached_tokens(llm) == warm + p1 + A + [stop] + p2 + A
        assert llm._pending_token == stop
    else:
        assert llm.run("first") == "H"
        assert cached_tokens(llm) == warm + p1
        assert llm._pending_token == A[0]
        assert llm.run("second") == "i"
        assert cached_tokens(llm) == warm + p1 + [A[0]] + p2
        assert llm._pending_token == A[1]
    # The bookkeeping list mirrors the cache at every step.
    assert llm._cache_tokens == cached_tokens(llm)


def test_closing_stream_preserves_last_yielded_token(tmp_path, monkeypatch):
    A = _chars("Hi")
    llm, _, stop = _build(Gemma3Static, tmp_path, monkeypatch, instruct=False,
                          next_fn=scripted((A + [STOP]) * 2, then=STOP))
    stream = llm.run_stream("q")
    assert next(stream) == "H"
    stream.close()
    assert llm._pending_token == A[0]
    end = llm._turn_pos
    llm.run("r")
    assert cached_tokens(llm)[end] == A[0]
    assert cached_tokens(llm) == [2, *_chars("q"), A[0], 2, *_chars("r"), A[1]]
    assert llm._pending_token == stop


@pytest.mark.parametrize("instruct", [False, True])
def test_turn_start_rollover_keeps_warmup_and_last_n(instruct, tmp_path, monkeypatch):
    # The window rolls before the new prompt: warm-up prefix plus the last
    # kept history tokens, replayed at their new positions.
    seq = 36 if instruct else 8
    a = CHAR_BASE + ord("x")
    llm, models, stop = _build(Gemma3Static, tmp_path, monkeypatch, instruct=instruct,
                               seq=seq, keep=3, next_fn=scripted([a, STOP], then=STOP))
    warm = cached_tokens(llm)
    hist = [31 + i for i in range(11 if instruct else 7)]
    fill(llm, warm + hist)
    llm._pending_token = stop
    assert llm.run("q") == "x"
    p1 = llm._build_prompt_tokens("q")
    kept = hist[-3:]
    assert cached_tokens(llm) == warm + kept + [stop] + p1 + [a]
    assert llm._pending_token == stop
    # The kept tokens were re-fed at the head of the cache, not their old spots.
    assert ([kept[0]], len(warm)) in models["body"].calls


@pytest.mark.parametrize("instruct", [False, True])
def test_reset_returns_to_warmup_prefix(instruct, tmp_path, monkeypatch):
    A = _chars("Hi")
    llm, _, stop = _build(Gemma3Static, tmp_path, monkeypatch, instruct=instruct,
                          next_fn=scripted((A + [STOP]) * 2, then=STOP))
    warm = cached_tokens(llm)
    assert llm.run("q") == "Hi"
    llm.reset()
    assert cached_tokens(llm) == warm
    assert llm._pending_token is None
    assert llm._turn_pos == len(warm)
    p = llm._build_prompt_tokens("next")
    assert llm.run("next") == "Hi"
    assert cached_tokens(llm) == warm + p + A
    assert llm._pending_token == stop


@pytest.mark.parametrize("instruct", [False, True])
def test_full_cache_without_window_restarts_conversation(instruct, tmp_path, monkeypatch, caplog):
    # No sliding window: a full cache restarts the conversation at the
    # warm-up prefix and drops the pending prediction.
    seq = 40 if instruct else 8
    a = CHAR_BASE + ord("x")
    llm, _, stop = _build(Gemma3Static, tmp_path, monkeypatch, instruct=instruct,
                          seq=seq, next_fn=scripted([a, STOP], then=STOP))
    warm = cached_tokens(llm)
    fill(llm, warm + [30 + i for i in range(seq - len(warm))])
    llm._pending_token = stop
    with caplog.at_level(logging.WARNING):
        assert llm.run("q") == "x"
    assert "KV cache full, starting a fresh conversation" in caplog.text
    p1 = llm._build_prompt_tokens("q")
    assert cached_tokens(llm) == warm + p1 + [a]
    assert llm._pending_token == stop


def test_oversized_prompt_drops_history_and_truncates(tmp_path, monkeypatch, caplog):
    # A prompt bigger than the whole window keeps nothing of the previous
    # conversation (nor its pending token) and is truncated to fit.
    a = CHAR_BASE + ord("x")
    llm, _, stop = _build(Gemma3Static, tmp_path, monkeypatch, instruct=False, seq=8, keep=5,
                          next_fn=scripted([a, STOP] * 2, then=STOP))
    assert llm.run("q") == "x"
    assert llm._pending_token == stop
    prompt = "0123456789"  # 11 tokens with the auto-BOS: bigger than the 8-token cache
    with caplog.at_level(logging.WARNING):
        assert llm.run(prompt) == "x"
    assert "Prompt truncated to fit the KV cache (8 tokens)" in caplog.text
    chars = _chars(prompt)
    assert cached_tokens(llm) == chars[2:7] + [a]
    assert llm._pending_token == stop


@pytest.mark.parametrize("keep", [6, 8])
def test_large_window_keeps_generating_on_rollover(keep, tmp_path, monkeypatch, caplog):
    # A window bigger than the remaining room is clamped, and mid-answer
    # rolls keep going until the answer completes.
    a = [CHAR_BASE + 97 + i for i in range(20)]
    llm, _, stop = _build(Gemma3Static, tmp_path, monkeypatch, instruct=False, seq=8, keep=keep,
                          next_fn=scripted(a + [STOP], then=STOP))
    with caplog.at_level(logging.WARNING):
        assert llm.run("q") == "".join(chr(t - CHAR_BASE) for t in a)
    assert llm._turn_pos <= 8
    assert "stopping generation" not in caplog.text


def test_in_turn_rollover_then_next_turn_preserves_pending(tmp_path, monkeypatch):
    # The cache fills mid-answer; the roll keeps the last window tokens, and
    # the next turn still picks up the pending prediction.
    a, b, c = 120, 121, 122
    llm, _, stop = _build(Gemma3Static, tmp_path, monkeypatch, instruct=False, seq=8, keep=3,
                          next_fn=scripted([a, b, STOP, c], then=STOP))
    fill(llm, [30, 31, 32, 33, 34, 35, 36])
    llm.run("q")
    p1 = llm._build_prompt_tokens("q")
    assert cached_tokens(llm) == [34, 35, 36] + p1 + [a, b]
    assert llm._pending_token == stop
    llm._max_gen_tokens = 1
    llm.run("q")
    p2 = llm._build_prompt_tokens("q")
    assert cached_tokens(llm) == [_chars("q")[0], a, b] + [stop] + p2
    assert llm._pending_token == c


def test_batched_prefill_matches_single_step_cache(tmp_path_factory, monkeypatch):
    A = _chars("Hi")
    spec = SPECS[Gemma3Static]
    specials, bos_id, config, stop = spec
    tok = CharTokenizer(specials, bos_id=bos_id)

    def build(name, prefill_size):
        return make_llm(
            tmp_path_factory.mktemp(name), monkeypatch, Gemma3Static, tokenizer=tok,
            config=config, next_fn=scripted((A + [STOP]) * 2, then=STOP),
            prefill_size=prefill_size, instruct_model=True, sys_prompt="S",
        )

    single, _ = build("single", None)
    batched, models = build("batched", 2)
    assert models["prefill"] is not None
    assert cached_tokens(single) == cached_tokens(batched)
    for question in ("first", "second"):
        assert single.run(question) == batched.run(question) == "Hi"
        assert cached_tokens(single) == cached_tokens(batched)
    assert models["prefill"].calls


def test_repeated_turns_keep_bookkeeping_in_sync_with_cache(tmp_path, monkeypatch):
    # Several turns over a small window: the turn-start roll (turn 4) and the
    # mid-answer roll (turn 2) both leave _cache_tokens matching the cache.
    a = CHAR_BASE + ord("x")
    llm, _, stop = _build(Gemma3Static, tmp_path, monkeypatch, instruct=False, seq=12, keep=3,
                          next_fn=scripted([a, STOP] * 4, then=STOP))
    for _ in range(4):
        assert llm.run("next") == "x"
        assert llm._cache_tokens == cached_tokens(llm)
        assert llm._turn_pos < 12


def test_failed_prefill_does_not_advance_position(tmp_path, monkeypatch):
    llm, models, _ = _build(Gemma3Static, tmp_path, monkeypatch, instruct=False,
                            next_fn=scripted([], then=1))

    def boom(inputs):
        raise RuntimeError("inference failed")

    models["body"].infer = boom
    with pytest.raises(RuntimeError, match="inference failed"):
        llm.run("q")
    assert llm._turn_pos == 0
    assert llm._cache_tokens == []
    assert llm._pending_token is None


def test_generation_validation_resets_between_samples(tmp_path, capsys):
    runner = Mock(last_infer_time=1, time_to_first_token=1, generated_tokens=1)
    runner.run.return_value = "b"
    items = [
        {"name": "a", "prompt": "a", "expected": "b", "check": lambda answer: answer == "b"},
        {"name": "c", "prompt": "c", "expected": "b", "check": lambda answer: answer == "b"},
    ]
    assert run_prompts(runner, items) == 1.0
    assert [call[0] for call in runner.method_calls] == ["reset", "run", "reset", "run"]


def test_teacher_forced_validation_steps(tmp_path, monkeypatch, capsys):
    specials, bos_id, config, _ = SPECS[Gemma3Static]
    llm, _ = make_llm(tmp_path, monkeypatch, Gemma3Static,
                      tokenizer=CharTokenizer(specials, bos_id=bos_id), config=config,
                      next_fn=scripted([], then=1), instruct_model=True, sys_prompt="S")
    run_teacher_forced(llm, [{"src": "hello", "tgt": "hola mundo", "tgt_lang": "Spanish"}])
