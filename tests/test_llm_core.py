# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""The shared LLM end to end over fake models: chat templates, warm-up
prefix, multi-turn KV cache reuse with the pending token, and reset."""

import importlib.util
from pathlib import Path

from gemma3.src.runner import Gemma3Static
from llm_fakes import CHAR_BASE, CharTokenizer, cached_tokens, make_llm, scripted
from qwen3.src.runner import CHATML_NO_THINKING
from utils.llm import CHATML, GEMMA

# The LFM2.5 folder name is not importable as a package; load by path.
_spec = importlib.util.spec_from_file_location(
    "_lfm2_5_runner", Path(__file__).resolve().parents[1] / "LiquidAI" / "LiquidAI-LFM2.5" / "src" / "runner.py"
)
_lfm2_5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lfm2_5)
LFM2_CHATML = _lfm2_5.LFM2_CHATML

def test_gemma_chat_turns_share_the_cache(tmp_path, monkeypatch):
    tok = CharTokenizer({"<bos>": 2, "<eos>": 1, "<start_of_turn>": 105, "<end_of_turn>": 106}, bos_id=2)
    A = [CHAR_BASE + ord(c) for c in "Hi"]
    llm, models = make_llm(tmp_path, monkeypatch, Gemma3Static, tokenizer=tok,
        config={"bos_token_id": 2, "eos_token_id": 1, "vocab_size": 4096},
        next_fn=scripted(A + [106] + A + [106], then=1), instruct_model=True, sys_prompt="S")
    warm = [2] + tok.ids("<start_of_turn>system\nS<end_of_turn>\n")
    assert cached_tokens(llm) == warm
    assert llm.run("q") == "Hi"
    p1 = tok.ids("<start_of_turn>user\nq<end_of_turn>\n<start_of_turn>model\n")
    assert cached_tokens(llm) == warm + p1 + A          # 106 is pending
    assert llm._pending_token == 106
    assert llm.run("r") == "Hi"
    p2 = tok.ids("<start_of_turn>user\nr<end_of_turn>\n<start_of_turn>model\n")
    assert cached_tokens(llm) == warm + p1 + A + [106] + p2 + A
    assert llm.generated_tokens == 2
    llm.reset()
    assert cached_tokens(llm) == warm


def test_templates_reproduce_the_chat_formats():
    # Built from pieces, not literals, so the strings match the old runners'.
    im_start, im_end = "<|" + "im_start|>", "<|" + "im_end|>"
    assert GEMMA.turn.format(role="user", text="x") == "<start_of_turn>user\nx<end_of_turn>\n"
    assert GEMMA.generation_prompt == "<start_of_turn>model\n"
    assert CHATML.turn.format(role="user", text="x") == f"{im_start}user\nx{im_end}\n"
    assert CHATML.generation_prompt == f"{im_start}assistant\n"
    assert CHATML.stop_tokens == (im_end, "<|" + "endoftext|>")
    assert CHATML_NO_THINKING.generation_prompt == f"{im_start}assistant\n<think>\n\n</think>\n\n"
    assert (LFM2_CHATML.bos_prefix, LFM2_CHATML.strip_bos, LFM2_CHATML.stop_tokens) == (True, True, ())
