# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Tests for the FunctionGemma function-calling support in the functiongemma demo.

The golden strings below are rendered by the published
``google/functiongemma-270m-it`` checkpoint's ``chat_template.jinja``
(transformers ``apply_chat_template`` with the ``DEFAULT_TOOLS`` schemas), so
they pin the hand-rolled rendering to the real template. The runner tests
build a real ``FunctionGemmaStatic`` over the fake models from ``llm_fakes``;
the chat loop runs ``infer.run_function_loop`` with scripted stdin.
"""

import builtins
import logging
from contextlib import nullcontext
from unittest import mock

import pytest

from functiongemma.src import function_tools
from functiongemma.src.function_tools import (
    DEFAULT_TOOLS,
    FUNCTIONGEMMA_DEV_INSTRUCTION,
    START_FUNCTION_RESPONSE,
    format_developer_block,
    format_function_declaration,
    format_function_response,
    parse_function_call,
)
from functiongemma.src.runner import FunctionGemmaStatic
from llm_fakes import CHAR_BASE, CharTokenizer, cached_tokens, make_llm

# ── Golden strings from the checkpoint's chat template ───────────────────────

DECLARATION = (
    "declaration:get_temperature"
    "{description:<escape>Get the temperature in a city.<escape>"
    ",parameters:{properties:{city:{description:<escape>The city name.<escape>,"
    "type:<escape>STRING<escape>}},"
    "required:[<escape>city<escape>],type:<escape>OBJECT<escape>}}"
)

#: The developer message body: instruction + tool declarations.
DEV_BLOCK_BODY = (
    f"{FUNCTIONGEMMA_DEV_INSTRUCTION}"
    f"<start_function_declaration>{DECLARATION}<end_function_declaration>"
)

#: The developer message as rendered by the checkpoint's template.
DEV_BLOCK = f"<start_of_turn>developer\n{DEV_BLOCK_BODY}<end_of_turn>\n"

#: Template output for [developer, user], add_generation_prompt=True.
FIRST_PROMPT = (
    "<bos>"
    f"{DEV_BLOCK}"
    "<start_of_turn>user\nWhat is the temperature in London?<end_of_turn>\n"
    "<start_of_turn>model\n"
)

#: What the model generates for a tool call: it stops on
#: <start_function_response> (id 50, in the config's eos_token_id list).
MODEL_CALL = (
    "<start_function_call>call:get_temperature{city:<escape>London<escape>}"
    "<end_function_call><start_function_response>"
)


class TestRendering:
    def test_declaration_matches_checkpoint_template(self):
        assert format_function_declaration(DEFAULT_TOOLS[0]) == DECLARATION

    def test_developer_block_matches_checkpoint_template(self):
        assert format_developer_block(DEFAULT_TOOLS) == DEV_BLOCK_BODY

    def test_first_prompt_matches_checkpoint_template(self):
        # The demo splits the template output as warm-up (BOS + developer
        # block) plus the first user turn; together they are FIRST_PROMPT.
        text = (
            "<bos>"
            + DEV_BLOCK
            + "<start_of_turn>user\nWhat is the temperature in London?<end_of_turn>\n"
            + "<start_of_turn>model\n"
        )
        assert text == FIRST_PROMPT

    def test_function_response_scalars(self):
        assert (
            format_function_response("get_temperature", "18")
            == "response:get_temperature{value:<escape>18<escape>}<end_function_response>"
        )
        assert (
            format_function_response("get_temperature", 18)
            == "response:get_temperature{value:18}<end_function_response>"
        )
        assert (
            format_function_response("f", True)
            == "response:f{value:true}<end_function_response>"
        )
        assert (
            format_function_response("f", "18.5")
            == "response:f{value:<escape>18.5<escape>}<end_function_response>"
        )

    def test_function_response_mapping_and_sequence(self):
        assert (
            format_function_response("f", {"b": 1, "a": "x"})
            == "response:f{a:<escape>x<escape>,b:1}<end_function_response>"
        )
        assert (
            format_function_response("f", [1, "two"])
            == "response:f{value:[1,<escape>two<escape>]}<end_function_response>"
        )

    def test_developer_block_has_no_declared_tools(self):
        assert format_developer_block([]) == FUNCTIONGEMMA_DEV_INSTRUCTION


class TestParsing:
    def test_parses_model_call(self):
        assert (
            parse_function_call(f"sure! {MODEL_CALL}")
            == ("get_temperature", {"city": "London"})
        )

    def test_parses_number_bool_and_nested_arguments(self):
        text = (
            "<start_function_call>call:f{a:{b:1},c:[1,<escape>two<escape>],d:true,"
            "e:1.5}<end_function_call><start_function_response>"
        )
        assert parse_function_call(text) == (
            "f",
            {"a": {"b": 1}, "c": [1, "two"], "d": True, "e": 1.5},
        )

    def test_empty_arguments(self):
        assert (
            parse_function_call("<start_function_call>call:ping{}<end_function_call>")
            == ("ping", {})
        )

    def test_no_call(self):
        assert parse_function_call("It is 18 degrees in London.") is None

    def test_malformed_arguments_are_rejected(self):
        assert (
            parse_function_call("<start_function_call>call:f{a:}x}<end_function_call>")
            is None
        )
        assert (
            parse_function_call("<start_function_call>call:f{a [1}<end_function_call>")
            is None
        )

    def test_rejects_multiple_calls(self):
        text = (
            "<start_function_call>call:a{}<end_function_call>x"
            "<start_function_call>call:b{y:2}<end_function_call>"
        )
        assert parse_function_call(text) is None

    def test_round_trip(self):
        call = (
            "<start_function_call>call:get_temperature"
            "{city:<escape>New York<escape>}<end_function_call>"
        )
        assert parse_function_call(call) == ("get_temperature", {"city": "New York"})


class TestBuiltInFunctions:
    def test_get_temperature(self):
        assert function_tools.get_temperature("London") == 18

    def test_get_temperature_is_case_insensitive(self):
        assert function_tools.get_temperature("new york") == 22

    def test_get_temperature_unknown_city(self):
        with pytest.raises(ValueError, match="unknown city"):
            function_tools.get_temperature("Atlantis")


@pytest.mark.parametrize(
    "body",
    [
        "f", "f{", "{a:1}", "f{a:1", "f{a:}", "f{a:[1}",
        "f{a:{b:1}", "f{a:1,}", "f{a:1,a:2}", "f{a:<escape>oops}",
        "f{a:1}junk", "f{a:word}", "f{a:[1,]}",
    ],
)
def test_rejects_incomplete_or_ambiguous_calls(body):
    assert parse_function_call(f"<start_function_call>call:{body}<end_function_call>") is None


def test_parses_whitespace_in_containers():
    text = "<start_function_call>call:f{a:[1, 2 ],b:{c:true }}<end_function_call>"
    assert parse_function_call(text) == ("f", {"a": [1, 2], "b": {"c": True}})


@pytest.mark.parametrize(
    "name,args",
    [
        ("not_registered", {}), ("get_temperature", {}),
        ("get_temperature", {"city": "Atlantis"}), ("get_temperature", {"city": 123}),
    ],
)
def test_execution_errors_become_tool_results(name, args):
    from functiongemma.src.infer import _execute
    assert "error" in _execute(name, args)


# ── Runner: FunctionGemmaStatic over the fake models ─────────────────────────

USER_1 = "<start_of_turn>user\nWhat is the temperature in London?<end_of_turn>\n"
MODEL_HEADER = "<start_of_turn>model\n"
CALL_BODY = MODEL_CALL.removesuffix(START_FUNCTION_RESPONSE)

#: <eos> 1, developer block 10, <start_function_response> 50,
#: <end_of_turn> 106, the four protocol markers 20-23.
BASE_SPECIALS = {
    "<eos>": 1,
    DEV_BLOCK: 10,
    START_FUNCTION_RESPONSE: 50,
    "<end_of_turn>": 106,
    "<start_function_call>": 20,
    "<end_function_call>": 21,
    "<end_function_response>": 22,
    "<escape>": 23,
}

#: The FunctionGemma checkpoint's stop ids: eos plus <start_function_response>.
CONFIG = {"bos_token_id": 2, "eos_token_id": [1, 50], "vocab_size": 4096}


def _build(tmp_path, monkeypatch, specials, mapping, **kwargs):
    """A FunctionGemmaStatic whose next token follows *mapping*: id of the
    last cached token -> id to emit (default <eos>)."""
    return make_llm(
        tmp_path, monkeypatch, FunctionGemmaStatic,
        tokenizer=CharTokenizer({**BASE_SPECIALS, **specials}, bos_id=2),
        config=kwargs.pop("config", CONFIG),
        seq=kwargs.pop("seq", 64),
        next_fn=lambda cached: mapping.get(cached[-1] if cached else None, 1),
        **kwargs,
    )


def fill(llm, tokens):
    """Occupy the first ``len(tokens)`` cache positions as prior history."""
    llm._cache.inputs[0].to_host()[0, 0, : len(tokens), 0] = tokens
    llm._cache_tokens = list(tokens)
    llm._turn_pos = len(tokens)


def _prepare_loop(tmp_path, monkeypatch, specials, mapping):
    from functiongemma.src import infer
    llm, _ = _build(tmp_path, monkeypatch, specials, mapping)
    inputs = iter(["What is the temperature in London?", "exit"])
    monkeypatch.setattr(builtins, "input", lambda *_: next(inputs))
    monkeypatch.setattr(infer, "InferenceStopInput", lambda _: nullcontext(None))
    monkeypatch.setattr(infer, "_think", lambda *_: None)
    return infer, llm


def test_warmup_is_bos_plus_developer_block(tmp_path, monkeypatch):
    llm, _ = _build(tmp_path, monkeypatch, {}, {})
    assert cached_tokens(llm) == [2, 10]
    assert llm._warmup_len == 2


def test_construction_rejects_export_without_function_tokens(tmp_path, monkeypatch):
    # E.g. a plain gemma3 export: the protocol markers are not in the tokenizer.
    missing = dict(BASE_SPECIALS)
    for token in ("<start_function_call>", "<end_function_call>", "<end_function_response>", "<escape>"):
        del missing[token]
    with pytest.raises(ValueError, match="FunctionGemma export"):
        make_llm(tmp_path, monkeypatch, FunctionGemmaStatic,
                 tokenizer=CharTokenizer(missing, bos_id=2), config=CONFIG, seq=64)


def test_construction_rejects_start_function_response_not_a_stop(tmp_path, monkeypatch):
    # The tokens exist but the config's stop ids lack <start_function_response>.
    with pytest.raises(ValueError, match="FunctionGemma export"):
        make_llm(tmp_path, monkeypatch, FunctionGemmaStatic,
                 tokenizer=CharTokenizer(dict(BASE_SPECIALS), bos_id=2),
                 config={"bos_token_id": 2, "eos_token_id": 1, "vocab_size": 4096}, seq=64)


def test_warmup_limit_rejects_oversized_developer_block(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="--max-inp-len"):
        _build(tmp_path, monkeypatch, {}, {}, max_prompt_tokens=1)


def test_warmup_must_leave_cache_capacity(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="no room"):
        _build(tmp_path, monkeypatch, {}, {}, seq=2)


def test_function_call_flow(tmp_path, monkeypatch):
    # Warm-up (BOS + developer block), then: the user turn stops the model on
    # <start_function_response> (pending, not in the cache); the response
    # continuation replays it and generates the answer; the next user turn
    # replays the pending <end_of_turn> plus the turn's newline.
    response = format_function_response("get_temperature", 18)
    answer = "It is 18 degrees in London."
    user_2 = "<start_of_turn>user\nAnd in Paris?<end_of_turn>\n"
    specials = {USER_1: 30, MODEL_HEADER: 31, CALL_BODY: 32, response: 33, answer: 34, user_2: 35}
    mapping = {31: 32, 32: 50, 33: 34, 34: 106}
    llm, _ = _build(tmp_path, monkeypatch, specials, mapping)

    assert not llm.awaiting_function_response
    call_text = llm.run("What is the temperature in London?")
    # The protocol tokens survive decoding, so the call parses out of the text.
    assert call_text == CALL_BODY + START_FUNCTION_RESPONSE
    assert parse_function_call(call_text) == ("get_temperature", {"city": "London"})
    assert llm._pending_token == 50
    assert llm.awaiting_function_response
    assert cached_tokens(llm) == [2, 10, 30, 31, 32]

    final = "".join(llm.run_function_response(response))
    assert final == answer + "<end_of_turn>"
    assert llm._pending_token == 106
    assert cached_tokens(llm) == [2, 10, 30, 31, 32, 50, 33, 34]

    llm.run("And in Paris?")
    assert cached_tokens(llm) == [2, 10, 30, 31, 32, 50, 33, 34, 106,
                                   CHAR_BASE + ord("\n"), 35, 31, 32]
    assert llm._pending_token == 50


def test_function_response_requires_pending_call(tmp_path, monkeypatch):
    response = format_function_response("f", 1)
    llm, _ = _build(tmp_path, monkeypatch, {response: 33}, {})
    with pytest.raises(ValueError, match="pending function call"):
        list(llm.run_function_response(response))
    assert not llm._function_response_next
    assert cached_tokens(llm) == [2, 10]


def test_prompt_limits_reject_instead_of_truncating(tmp_path, monkeypatch):
    # --max-inp-len leaves 1 token for the user after the 2-token warm-up.
    llm, _ = _build(tmp_path, monkeypatch, {USER_1: 30, MODEL_HEADER: 31}, {},
                    max_prompt_tokens=3)
    with pytest.raises(ValueError, match="--max-inp-len"):
        llm.run("What is the temperature in London?")
    assert cached_tokens(llm) == [2, 10]
    assert llm._pending_token is None


def test_response_capacity_error_clears_continuation_state(tmp_path, monkeypatch):
    response = format_function_response("f", 1)
    llm, _ = _build(tmp_path, monkeypatch, {response: 33}, {}, seq=8)
    fill(llm, [2, 10, 40, 41, 42, 43, 44])  # one short of full
    llm._pending_token = 50
    with pytest.raises(ValueError, match="KV cache"):
        list(llm.run_function_response(response))
    assert not llm._function_response_next
    assert llm._pending_token == 50


def test_unconsumed_response_does_not_change_next_user_prompt(tmp_path, monkeypatch):
    response = format_function_response("f", 1)
    llm, _ = _build(tmp_path, monkeypatch, {response: 33}, {})
    llm._pending_token = 50
    stream = llm.run_function_response(response)
    assert not llm._function_response_next  # the generator is lazy: nothing ran
    stream.close()
    assert not llm._function_response_next


def test_closing_response_stream_restores_user_prompt_mode(tmp_path, monkeypatch):
    response = format_function_response("f", 1)
    answer = "It is 18."
    llm, _ = _build(tmp_path, monkeypatch, {response: 33, answer: 34}, {33: 34, 34: 106})
    llm._pending_token = 50
    stream = llm.run_function_response(response)
    next(stream)
    assert llm._function_response_next
    stream.close()
    assert not llm._function_response_next


def test_decode_keeps_protocol_tokens(tmp_path, monkeypatch):
    llm, _ = _build(tmp_path, monkeypatch, {}, {})
    call, escape = 20, 23  # <start_function_call>, <escape>
    assert llm._tokenizer.decode([call, escape]) == ""
    assert "<start_function_call>" in llm._decode_generated_tokens([call, escape])
    assert "<escape>" in llm._decode_generated_tokens([call, escape])


def test_all_configured_stop_ids_are_honored(tmp_path, monkeypatch):
    llm, _ = _build(tmp_path, monkeypatch, {}, {})
    for token in (1, 50, 106):  # the config's eos list plus the template's <end_of_turn>
        assert llm._should_stop(token, [])
    assert not llm._should_stop(42, [])


def test_full_cache_restart_has_no_extra_turn_separator(tmp_path, monkeypatch, caplog):
    # FunctionGemma never rolls: a full cache restarts at the warm-up prefix,
    # drops the pending token, and the new turn has no end-of-turn newline.
    answer = "It is 18 degrees in Paris."
    llm, _ = _build(tmp_path, monkeypatch,
                    {USER_1: 30, MODEL_HEADER: 31, answer: 34},
                    {31: 34, 34: 106}, seq=8)
    fill(llm, [2, 10, 40, 41, 42, 43, 44, 45])
    llm._pending_token = 106
    with caplog.at_level(logging.WARNING):
        llm.run("What is the temperature in London?")
    assert "KV cache full, starting a fresh conversation" in caplog.text
    assert cached_tokens(llm) == [2, 10, 30, 31, 34]
    assert llm._pending_token == 106


def test_chat_loop_executes_call_and_prints_answer(tmp_path, monkeypatch, capsys):
    response = format_function_response("get_temperature", 18)
    answer = "It is 18 degrees in London."
    infer, llm = _prepare_loop(
        tmp_path, monkeypatch,
        {USER_1: 30, MODEL_HEADER: 31, CALL_BODY: 32, response: 33, answer: 34},
        {31: 32, 32: 50, 33: 34, 34: 106},
    )
    infer.run_function_loop(llm)
    out = capsys.readouterr().out
    assert "[function] get_temperature(city='London')" in out
    assert "[result] 18" in out
    assert answer in out
    assert "<end_of_turn>" not in out
    assert llm._pending_token == 106


def test_chat_loop_resets_malformed_call_without_execution(tmp_path, monkeypatch):
    malformed = (
        "<start_function_call>call:get_temperature{city:<escape>London<escape>"
        "<end_function_call>"
    )
    infer, llm = _prepare_loop(
        tmp_path, monkeypatch,
        {USER_1: 30, MODEL_HEADER: 31, malformed: 32},
        {31: 32, 32: 50},
    )
    with mock.patch.object(infer, "_execute") as execute:
        infer.run_function_loop(llm)
    execute.assert_not_called()
    assert llm._pending_token is None
    assert llm._turn_pos == llm._warmup_len


def test_chat_loop_caps_repeated_calls_and_resets(tmp_path, monkeypatch, capsys):
    response = format_function_response("get_temperature", 18)
    infer, llm = _prepare_loop(
        tmp_path, monkeypatch,
        {USER_1: 30, MODEL_HEADER: 31, CALL_BODY: 32, response: 33},
        {31: 32, 32: 50, 33: 32},
    )
    monkeypatch.setattr(infer, "_MAX_TOOL_CALLS", 1)
    infer.run_function_loop(llm)
    assert capsys.readouterr().out.count("[function]") == 1
    assert llm._pending_token is None
    assert not llm._function_response_next


def test_chat_loop_interruption_resets_partial_turn(tmp_path, monkeypatch):
    infer, llm = _prepare_loop(tmp_path, monkeypatch, {}, {})
    with mock.patch.object(FunctionGemmaStatic, "run", side_effect=infer.InferenceInterrupted):
        infer.run_function_loop(llm)
    assert llm._pending_token is None
    assert llm._turn_pos == llm._warmup_len


def test_call_without_response_stop_is_not_executed_and_discards_history(tmp_path, monkeypatch, caplog):
    infer, llm = _prepare_loop(
        tmp_path, monkeypatch,
        {USER_1: 30, MODEL_HEADER: 31, CALL_BODY: 32},
        {31: 32, 32: 106},
    )
    with mock.patch.object(infer, "_execute") as execute:
        infer.run_function_loop(llm)
    execute.assert_not_called()
    assert llm._pending_token is None
    assert llm._turn_pos == llm._warmup_len
    assert "Function call did not stop on <start_function_response>" in caplog.text
