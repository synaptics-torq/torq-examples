# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Tests for the FunctionGemma function-calling support in the functiongemma demo.

The golden strings below are rendered by the published
``google/functiongemma-270m-it`` checkpoint's ``chat_template.jinja``
(transformers ``apply_chat_template`` with the ``DEFAULT_TOOLS`` schemas), so
they pin the hand-rolled rendering to the real template.
"""

import logging
import re
from types import SimpleNamespace

import numpy as np
import pytest

from functiongemma.src import function_tools
from functiongemma.src.function_tools import (
    DEFAULT_TOOLS,
    FUNCTIONGEMMA_DEV_INSTRUCTION,
    format_developer_block,
    format_function_declaration,
    format_function_response,
    parse_function_call,
)
from functiongemma.src.runner import FunctionGemmaStatic

pytestmark = pytest.mark.skipif(not hasattr(np, "zeros"), reason="Requires real numpy")

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

#: The full developer turn the runner's tokenize() renders (== DEV_BLOCK).
DEV_TURN = DEV_BLOCK

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


# ── Runner tools mode ─────────────────────────────────────────────────────────


class TextTokenizer:
    """One token per unique string; mirrors the auto-prepended BOS."""

    def __init__(self, bos_id: int = 2):
        self._bos_id = bos_id
        self._ids: dict[str, int] = {"<start_function_response>": 50, "<end_of_turn>": 106, "<eos>": 1}
        self._next = 100

    def _id(self, text: str) -> int:
        if text not in self._ids:
            self._ids[text] = self._next
            self._next += 1
            while self._next in self._ids.values():
                self._next += 1
        return self._ids[text]

    def encode(self, text: str):
        ids = [self._bos_id]
        if text:
            ids.append(self._id(text))
        return SimpleNamespace(ids=ids)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        inverse = {value: key for key, value in self._ids.items()}
        text = "".join(inverse.get(i, str(i)) for i in ids)
        return re.sub(r"<[^>]+>", "", text) if skip_special_tokens else text

    def id_of(self, text: str) -> int:
        return self._id(text)

    def token_to_id(self, token: str) -> int | None:
        return self._ids.get(token)


class Device:
    def __init__(self, array):
        self.array = array

    def to_host(self):
        return self.array


class ScriptedCache:
    """Records every fed token; emits the scripted successor of the last one."""

    infer_time_ms = 0

    def __init__(self, capacity: int, script: dict[int, int]):
        self._kv_cache = [Device(np.zeros((1, 1, capacity, 1), dtype=np.int64))]
        self._script = script

    @property
    def tokens(self):
        return self._kv_cache[0].to_host().reshape(-1)

    def infer(self, inputs, *, entry=None):
        tokens = inputs[0][0].tolist()
        pos = int(inputs[1][0, 0])
        assert 0 <= pos and pos + len(tokens) <= len(self.tokens)
        self.tokens[pos : pos + len(tokens)] = tokens
        logits = np.zeros((1, 1, 1000), dtype=np.float32)
        logits[0, 0, self._script.get(tokens[-1], 0)] = 1
        return [logits]

    def save_kv_state(self):
        return [self._kv_cache[0].to_host().copy()]

    def restore_kv_state(self, state):
        self._kv_cache[0] = Device(state[0].copy())

    def reset_kv(self):
        self.tokens[:] = 0


def _tools_runner(script: dict[int, int], capacity: int = 64, tokenizer: TextTokenizer | None = None) -> FunctionGemmaStatic:
    runner = FunctionGemmaStatic.__new__(FunctionGemmaStatic)
    runner._logger = logging.getLogger("test_functiongemma")
    runner._debug_logging = False
    runner._tokenizer = tokenizer or TextTokenizer(bos_id=2)
    runner._bos_token_id = 2
    runner._eos_token_id = 1
    runner._eos_token_ids = [1, 50]  # <eos> and <start_function_response>
    runner._instruct_model = True
    runner._tools = DEFAULT_TOOLS
    runner._function_response_next = False
    runner._sys_prompt = None
    runner._nl_token_id = 8
    runner._double_nl_token_id = 9
    runner._end_of_turn_id = 106
    model = ScriptedCache(capacity, script)
    runner._model = model
    runner._max_seq_len = capacity
    runner._max_prompt_tokens = runner._max_user_tokens = None
    runner._max_gen_tokens = None
    runner._cache_keep_n = None
    runner._temperature = 0
    runner._token_embeddings = None
    runner._token_id_lut = None
    runner._emb_buf = None
    runner._prefill_emb_buf = None
    runner._id_buf = np.zeros((1, 1), dtype=np.int64)
    runner._prefill_id_buf = None
    runner._pos_buf = np.zeros((1, 1), dtype=np.int64)
    runner._extra_step_inputs = []
    runner._prefill_model = None
    runner._native_prefill_entries = []
    runner._prefill_size = None
    runner._pending_token = None
    runner._cache_tokens = []
    runner._turn_pos = 0
    return runner


def test_tools_mode_warmup_is_bos_plus_developer_block():
    tok = TextTokenizer(bos_id=2)
    runner = _tools_runner({}, tokenizer=tok)
    n = runner._warmup()
    assert n == 2
    assert runner._model.tokens[:2].tolist() == [2, tok.id_of(DEV_TURN)]
    runner._warmup_len = n
    runner._turn_pos = n
    runner._reset_cache_state = runner._model.save_kv_state()


def test_tools_mode_function_call_flow():
    tok = TextTokenizer(bos_id=2)
    call_body = MODEL_CALL.removesuffix("<start_function_response>")
    user_turn = "<start_of_turn>user\nWhat is the temperature in London?<end_of_turn>\n"
    model_start = "<start_of_turn>model\n"
    answer = "It is 18 degrees in London."
    response = "response:get_temperature{value:18}<end_function_response>"
    user_turn2 = "<start_of_turn>user\nAnd in Paris?<end_of_turn>\n"
    script = {
        tok.id_of(model_start): tok.id_of(call_body),
        tok.id_of(call_body): 50,
        tok.id_of(response): tok.id_of(answer),
        tok.id_of(answer): 106,
    }
    runner = _tools_runner(script, tokenizer=tok)
    runner._warmup()
    runner._warmup_len = runner._turn_pos = 2
    runner._reset_cache_state = runner._model.save_kv_state()

    # Turn 1: user turn stops the model on <start_function_response> (50).
    question = "What is the temperature in London?"
    answer1 = runner.run(question)
    assert ("get_temperature", {"city": "London"}) == parse_function_call(answer1)
    assert runner._pending_token == 50
    assert runner._model.tokens[:8].tolist() == [
        2, tok.id_of(DEV_TURN), tok.id_of(user_turn), tok.id_of(model_start),
        tok.id_of(call_body),
    ] + [0, 0, 0]
    assert 50 not in runner._model.tokens[:6].tolist()

    # The function response continues with no turn header; the pending 50 is
    # replayed first, then the model answers and stops on <end_of_turn>.
    answer2 = "".join(runner.run_function_response(format_function_response("get_temperature", 18)))
    assert answer2.startswith(answer)
    assert runner._pending_token == 106
    assert runner._model.tokens[:8].tolist() == [
        2,
        tok.id_of(DEV_TURN),
        tok.id_of(user_turn),
        tok.id_of(model_start),
        tok.id_of(call_body),
        50,
        tok.id_of(response),
        tok.id_of(answer),
    ]

    # The next user turn replays the pending <end_of_turn> before its header.
    runner.run("And in Paris?")
    assert runner._model.tokens[:12].tolist() == [
        2,
        tok.id_of(DEV_TURN),
        tok.id_of(user_turn),
        tok.id_of(model_start),
        tok.id_of(call_body),
        50,
        tok.id_of(response),
        tok.id_of(answer),
        106,
        tok.id_of("\n"),
        tok.id_of(user_turn2),
        tok.id_of(model_start),
    ]


def test_function_response_requires_pending_call():
    runner = _tools_runner({})
    with pytest.raises(ValueError, match="pending function call"):
        list(runner.run_function_response("response:f{value:1}<end_function_response>"))
    assert not runner._function_response_next


def test_tools_mode_requires_functiongemma_export():
    dummy_cfg = {"num_hidden_layers": 1, "num_key_value_heads": 1, "head_dim": 1}
    # A tokenizer without the function tokens (e.g. the plain gemma3 export).
    runner = _tools_runner({})
    runner._tokenizer = TextTokenizer(bos_id=2)
    with pytest.raises(ValueError, match="FunctionGemma export"):
        runner._on_model_config_loaded(dummy_cfg)
    # The token must also be a configured stop token.
    tok = TextTokenizer(bos_id=2)
    sfr = tok.id_of("<start_function_response>")
    runner._tokenizer = tok
    runner._eos_token_ids = [1]
    with pytest.raises(ValueError, match="FunctionGemma export"):
        runner._on_model_config_loaded(dummy_cfg)
    runner._eos_token_ids = [1, sfr]
    for marker in ("<start_function_call>", "<end_function_call>", "<end_function_response>", "<escape>"):
        tok.id_of(marker)
    runner._on_model_config_loaded(dummy_cfg)  # no raise


def test_tools_mode_warmup_limit_raises():
    runner = _tools_runner({})
    runner._max_prompt_tokens = 1
    with pytest.raises(ValueError, match="--max-inp-len"):
        runner._build_warmup_tokens()


@pytest.mark.parametrize("body", [
    "f", "f{", "{a:1}", "f{a:1", "f{a:}", "f{a:[1}",
    "f{a:{b:1}", "f{a:1,}", "f{a:1,a:2}", "f{a:<escape>oops}",
    "f{a:1}junk", "f{a:word}", "f{a:[1,]}",
])
def test_rejects_incomplete_or_ambiguous_calls(body):
    assert parse_function_call(f"<start_function_call>call:{body}<end_function_call>") is None


def test_parses_whitespace_in_containers():
    text = "<start_function_call>call:f{a:[1, 2 ],b:{c:true }}<end_function_call>"
    assert parse_function_call(text) == ("f", {"a": [1, 2], "b": {"c": True}})


def test_real_tokenizer_preserves_protocol_tokens():
    from tokenizers import AddedToken, Tokenizer, models

    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "<start_function_call>": 1, "<escape>": 2}, unk_token="<unk>"))
    tokenizer.add_special_tokens([AddedToken("<start_function_call>", special=True), AddedToken("<escape>", special=True)])
    runner = _tools_runner({})
    runner._tokenizer = tokenizer
    assert tokenizer.decode([1, 2]) == ""
    assert "<start_function_call>" in runner._decode_generated_tokens([1, 2])
    assert "<escape>" in runner._decode_generated_tokens([1, 2])


def test_prompt_limits_reject_instead_of_truncating():
    runner = _tools_runner({})
    runner._warmup_len = 2
    runner._max_user_tokens = 1
    with pytest.raises(ValueError, match="--max-inp-len"):
        runner.run("question")
    assert not runner._model.tokens.any()


def test_response_capacity_error_clears_continuation_state():
    runner = _tools_runner({}, capacity=8)
    runner._warmup_len = 2
    runner._turn_pos = 8
    runner._pending_token = 50
    with pytest.raises(ValueError, match="KV cache"):
        list(runner.run_function_response(format_function_response("f", 1)))
    assert not runner._function_response_next
    assert runner._pending_token == 50


def test_unconsumed_response_does_not_change_next_user_prompt():
    runner = _tools_runner({})
    runner._warmup_len = 2
    runner._pending_token = 50
    stream = runner.run_function_response(format_function_response("f", 1))
    assert not runner._function_response_next
    stream.close()
    assert not runner._function_response_next


def test_warmup_must_leave_cache_capacity():
    runner = _tools_runner({}, capacity=2)
    with pytest.raises(ValueError, match="no room"):
        runner._build_warmup_tokens()


def _prepare_loop(monkeypatch, script):
    from contextlib import nullcontext
    from functiongemma.src import infer

    runner = _tools_runner(script)
    runner._warmup_len = runner._turn_pos = runner._warmup()
    runner._reset_cache_state = runner._model.save_kv_state()
    inputs = iter(["What is the temperature in London?", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(infer, "InferenceStopInput", lambda _: nullcontext(None))
    monkeypatch.setattr(infer, "_think", lambda _: None)
    return infer, runner


def test_chat_loop_executes_call_and_prints_answer(monkeypatch, capsys):
    infer, runner = _prepare_loop(monkeypatch, {})
    tok = runner._tokenizer
    call_body = MODEL_CALL.removesuffix("<start_function_response>")
    answer = "It is 18 degrees in London."
    runner._model._script.update({
        tok.id_of("<start_of_turn>model\n"): tok.id_of(call_body),
        tok.id_of(call_body): 50,
        tok.id_of(format_function_response("get_temperature", 18)): tok.id_of(answer),
        tok.id_of(answer): 106,
    })
    infer.run_function_loop(runner)
    output = capsys.readouterr().out
    assert "[function] get_temperature(city='London')" in output
    assert "[result] 18" in output
    assert answer in output
    assert "<end_of_turn>" not in output
    assert runner._pending_token == 106


def test_chat_loop_resets_malformed_call_without_execution(monkeypatch):
    infer, runner = _prepare_loop(monkeypatch, {})
    tok = runner._tokenizer
    malformed = "<start_function_call>call:get_temperature{city:<escape>London<escape><end_function_call>"
    runner._model._script.update({
        tok.id_of("<start_of_turn>model\n"): tok.id_of(malformed),
        tok.id_of(malformed): 50,
    })
    from unittest import mock
    with mock.patch.object(infer, "_execute") as execute:
        infer.run_function_loop(runner)
    execute.assert_not_called()
    assert runner._pending_token is None
    assert runner._turn_pos == runner._warmup_len


def test_chat_loop_caps_repeated_calls_and_resets(monkeypatch, capsys):
    infer, runner = _prepare_loop(monkeypatch, {})
    monkeypatch.setattr(infer, "_MAX_TOOL_CALLS", 1)
    tok = runner._tokenizer
    call_body = MODEL_CALL.removesuffix("<start_function_response>")
    runner._model._script.update({
        tok.id_of("<start_of_turn>model\n"): tok.id_of(call_body),
        tok.id_of(call_body): 50,
        tok.id_of(format_function_response("get_temperature", 18)): tok.id_of(call_body),
    })
    infer.run_function_loop(runner)
    assert capsys.readouterr().out.count("[function]") == 1
    assert runner._pending_token is None
    assert not runner._function_response_next


def test_chat_loop_interruption_resets_partial_turn(monkeypatch):
    infer, runner = _prepare_loop(monkeypatch, {})
    from unittest import mock
    with mock.patch.object(FunctionGemmaStatic, "run", side_effect=infer.InferenceInterrupted):
        infer.run_function_loop(runner)
    assert runner._pending_token is None
    assert runner._turn_pos == runner._warmup_len


@pytest.mark.parametrize("name,args", [
    ("not_registered", {}), ("get_temperature", {}),
    ("get_temperature", {"city": "Atlantis"}), ("get_temperature", {"city": 123}),
])
def test_execution_errors_become_tool_results(name, args):
    from functiongemma.src.infer import _execute
    assert "error" in _execute(name, args)


def test_all_configured_stop_ids_are_honored():
    from utils.llm import DecoderOnlyLLMRunner
    runner = _tools_runner({})
    for token in [1, 50]:
        assert runner._should_stop(token, [token])
        assert DecoderOnlyLLMRunner._should_stop(runner, token, [token])
    assert not runner._should_stop(42, [42])


def test_next_user_after_capacity_reset_has_no_extra_turn_separator():
    runner = _tools_runner({}, capacity=8)
    runner._warmup_len = runner._turn_pos = runner._warmup()
    runner._reset_cache_state = runner._model.save_kv_state()
    runner._turn_pos = 8
    runner._pending_token = 106
    runner._max_gen_tokens = 1
    runner.run("next")
    assert runner._model.tokens[2:4].tolist() == [
        runner._tokenizer.id_of("<start_of_turn>user\nnext<end_of_turn>\n"),
        runner._tokenizer.id_of("<start_of_turn>model\n"),
    ]


def test_closing_response_stream_restores_user_prompt_mode():
    runner = _tools_runner({})
    runner._warmup_len = runner._turn_pos = runner._warmup()
    runner._pending_token = 50
    stream = runner.run_function_response(format_function_response("f", 1))
    next(stream)
    assert runner._function_response_next
    stream.close()
    assert not runner._function_response_next


def test_call_without_response_stop_is_not_executed_and_discards_history(monkeypatch, caplog):
    infer, runner = _prepare_loop(monkeypatch, {})
    tok = runner._tokenizer
    call_body = MODEL_CALL.removesuffix("<start_function_response>")
    runner._model._script.update({
        tok.id_of("<start_of_turn>model\n"): tok.id_of(call_body),
        tok.id_of(call_body): 106,
    })
    from unittest import mock
    with mock.patch.object(infer, "_execute") as execute:
        infer.run_function_loop(runner)
    execute.assert_not_called()
    assert runner._pending_token is None
    assert runner._turn_pos == runner._warmup_len
    assert "Function call did not stop on <start_function_response>" in caplog.text
