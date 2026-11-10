# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Host tests for the ``text_generation`` task category: the shared LLM
constructor kwargs, the degenerate-output detector, the ported corpus
BLEU, the answer extractors and dataset loaders, and the validate modes end
to end (exit code, printed accuracy, metrics JSON) with fake dataset rows in
place of the download."""

import argparse
import json
import sys
from unittest import mock

import pytest

from utils.inference import add_llm_inference_args
from utils.tasks import text_generation as tg
from utils.tasks.text_generation import corpus_bleu, is_degenerate, llm_kwargs, normalize_translation


class _FakeRunner:
    """A scripted LLM runner: ``run`` returns the next answer in order."""

    last_infer_time = 123.4
    time_to_first_token = 10.0
    generated_tokens = 5

    def __init__(self, answers):
        self._answers = iter(answers)
        self.prompts = []

    def reset(self):
        pass

    def run(self, prompt, should_stop=None):
        self.prompts.append(prompt)
        return next(self._answers)


# ── llm_kwargs ───────────────────────────────────────────────────────────────


def _args(**overrides):
    parser = argparse.ArgumentParser()
    add_llm_inference_args(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_llm_kwargs_mapping():
    args = _args(
        max_seq_len=512, max_inp_len=128, max_gen_tokens=64,
        temperature=0.7, top_p=0.9, top_k=40,
        lm_head="h.vmfb", batch_prefill_model="p.vmfb", kv_cache_window=3,
    )
    assert llm_kwargs(args) == dict(
        max_seq_len=512,
        max_prompt_tokens=128,
        cache_keep_n=3,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_gen_tokens=64,
        lm_head_path="h.vmfb",
        disable_lm_head=False,
        prefill_model_path="p.vmfb",
        disable_prefill=False,
    )


def test_llm_kwargs_no_kv_window_disables_cache_keep_n():
    assert llm_kwargs(_args(no_kv_cache_window=True))["cache_keep_n"] is None


def test_llm_kwargs_flags_a_demo_disabled_fall_back_to_defaults():
    # functiongemma disables the instruct/kv-window/max-inp-len flags.
    parser = argparse.ArgumentParser()
    add_llm_inference_args(parser, with_instruct=False, with_kv_window=False,
                           with_max_inp_len=False)
    kwargs = llm_kwargs(parser.parse_args([]))
    assert kwargs["max_prompt_tokens"] is None
    assert kwargs["cache_keep_n"] is None
    assert kwargs["temperature"] == 0.0
    assert kwargs["disable_lm_head"] is False


# ── degenerate detector ──────────────────────────────────────────────────────


@pytest.mark.parametrize("answer", [
    "the the the the the the the the the the",  # one 6-gram, all of them
    "Paris " * 20,  # a repeated word loop
    "a" * 20,  # one character 20 times in a row
    "no\n" * 25,  # a repeated line
])
def test_is_degenerate_repetition(answer):
    assert is_degenerate(answer)


@pytest.mark.parametrize("answer", [
    "",
    "The capital of France is Paris.",  # 6 words: a single, unrepeated 6-gram
    "It was written by a famous playwright from England.",
    "a" * 19,
])
def test_is_degenerate_normal_answers(answer):
    assert not is_degenerate(answer)


# ── corpus BLEU (ported from gemma3/src/validate.py) ─────────────────────────


def test_corpus_bleu_perfect_match_is_100():
    assert corpus_bleu(["the quick brown fox"], ["the quick brown fox"]) == pytest.approx(100.0)


def test_corpus_bleu_zero_on_empty_or_no_overlap():
    assert corpus_bleu([], ["a reference sentence here"]) == 0.0
    assert corpus_bleu(["a b c d"], ["w x y z"]) == 0.0


def test_corpus_bleu_short_hypothesis_is_zero():
    # The single-reference implementation has no order-skipping: fewer than
    # max_n tokens in the whole corpus leaves a zero order.
    assert corpus_bleu(["hello world foo"], ["hello world foo"]) == 0.0


def test_corpus_bleu_partial_overlap_between_zero_and_100():
    assert 0 < corpus_bleu(
        ["the cat sat on the mat"], ["the cat sat on the rug"]
    ) < 100


def test_normalize_translation_strips_preamble_and_wrappers():
    assert normalize_translation('Here is the translation: "Buenos días."') == "Buenos días."
    assert normalize_translation("**Hola**") == "Hola"


# ── answer extractors and loaders ────────────────────────────────────────────


@pytest.mark.parametrize("answer,expected", [
    ("B", "B"), ("b) because", "B"), ("(C) the third", "C"), ("The answer is D.", "D"),
    ("Answer: A", "A"), ("I think A is right", "A"), ("none of them", None),
])
def test_first_choice_letters(answer, expected):
    assert tg.first_choice(answer, "ABCD") == expected


def test_first_choice_digits():
    assert tg.first_choice("2. Use a fork", "12") == "2"
    assert tg.first_choice("Solution 1 works", "12") == "1"
    assert tg.first_choice("3", "12") is None


def test_last_number():
    assert tg.last_number("2 + 3 = 5, so she earns $1,234.50") == 1234.5
    assert tg.last_number("no digits") is None


def _fetch(rows_by_dataset):
    def fetch_rows(dataset, *, config, split, samples, offset=0):
        return rows_by_dataset[dataset][:samples]
    return fetch_rows


def test_piqa_mmlu_pro_and_opus100_loaders():
    rows = {
        "baber/piqa": [{"goal": "Dry a phone", "sol1": "Rice", "sol2": "Water", "label": 0}],
        "TIGER-Lab/MMLU-Pro": [{"question_id": 7, "question": "2+2?", "options": ["3", "4", "5"], "answer": "B"}],
        "Helsinki-NLP/opus-100": [{"translation": {"en": "Hello", "es": "Hola"}}],
    }
    with mock.patch.object(tg, "fetch_rows", _fetch(rows)):
        (piqa,) = tg.LOADERS["baber/piqa"]({"dataset": "baber/piqa", "config": "default", "split": "validation", "samples": 5})
        (mmlu,) = tg.LOADERS["TIGER-Lab/MMLU-Pro"]({"dataset": "TIGER-Lab/MMLU-Pro", "config": "default", "split": "test", "samples": 5})
        (opus,) = tg.LOADERS["Helsinki-NLP/opus-100"]({"dataset": "Helsinki-NLP/opus-100", "config": "es-en", "split": "test", "samples": 5})
    assert "1. Rice\n2. Water" in piqa["prompt"] and piqa["expected"] == "1"
    assert piqa["check"]("1") and not piqa["check"]("2")
    assert "A. 3\nB. 4\nC. 5" in mmlu["prompt"] and mmlu["check"]("B") and not mmlu["check"]("D")
    assert opus == {"src": "Hola", "tgt": "Hello", "tgt_lang": "English"}


# ── validate modes end to end ────────────────────────────────────────────────

VALIDATION = [{"dataset": "openai/gsm8k", "config": "main", "split": "test", "samples": 10}]
GSM8K = [
    {"question": "Tom has 2 apples and buys 3. How many?", "answer": "2 + 3 = 5\n#### 5"},
    {"question": "A shirt costs $1,000 twice. Total?", "answer": "2 * 1000\n#### 2,000"},
]


def _run_validate_main(tmp_path, answers, min_accuracy, argv=(), rows=None, loaders=None):
    metrics_out = tmp_path / "metrics.json"
    runner = _FakeRunner(answers)
    made = {}

    def make_runner(args, runtime):
        made["temperature"] = args.temperature
        made["mode"] = args.mode
        return runner

    rows = rows or {"openai/gsm8k": GSM8K}
    with mock.patch.object(tg, "fetch_rows", _fetch(rows)), mock.patch.object(sys, "argv", [
        "validate.py",
        "-m", str(tmp_path / "model.vmfb"),
        "--no-refresh",
        "--metrics-out", str(metrics_out),
        *argv,
    ]):
        try:
            tg.validate_main(
                make_runner,
                demo="test",
                validation=VALIDATION,
                min_accuracy=min_accuracy,
                local_model=lambda: None,
                ensure_models=lambda model_dir, **kwargs: None,
                loaders=loaders,
            )
            code = 0
        except SystemExit as e:
            code = e.code
    return code, metrics_out, made, runner


def test_validate_prompts_fail_exits_1_and_writes_metrics(tmp_path, capsys):
    code, metrics_out, made, runner = _run_validate_main(
        tmp_path, ["2 + 3 = 5\nThe answer is 5.", "It costs a lot."], min_accuracy=2.0,
    )
    out = capsys.readouterr().out
    assert code == 1
    assert made["temperature"] == 0.0  # greedy decoding forced
    assert made["mode"] == "prompts"
    assert runner.prompts[0].startswith("Tom has 2 apples")
    assert "  [PASS] openai/gsm8k #0" in out
    assert "  [FAIL] openai/gsm8k #1" in out
    assert "(expected '2000')" in out
    assert "accuracy=0.5000 min=2.0" in out
    assert out.rstrip().endswith("FAIL")
    doc = json.loads(metrics_out.read_text())
    assert doc["measurements"]["time_to_first_token"] == 10_000_000.0  # 10 ms in ns
    assert doc["measurements"]["tokens_per_second"] == pytest.approx(5 / (123.4 - 10.0) * 1000)
    assert "cpu_load" in doc["measurements"]


def test_validate_prompts_pass_exits_0(tmp_path, capsys):
    code, _, _, _ = _run_validate_main(tmp_path, ["5", "So the total is 2,000"], min_accuracy=0.5)
    out = capsys.readouterr().out
    assert code == 0
    assert "accuracy=1.0000 min=0.5" in out
    assert out.rstrip().endswith("PASS")


def test_validate_prompts_degenerate_answer_counts_as_fail(tmp_path, capsys):
    code, _, _, _ = _run_validate_main(tmp_path, ["5 " * 12, "2000"], min_accuracy=1.0)
    out = capsys.readouterr().out
    assert code == 1
    assert out.count("[FAIL]") == 1  # only the repeated-word answer
    assert "accuracy=0.5000 min=1.0" in out


def test_validate_prompts_samples(tmp_path):
    code, _, _, runner = _run_validate_main(tmp_path, ["5", "unused"], min_accuracy=1.0, argv=["--samples", "1"])
    assert code == 0
    assert len(runner.prompts) == 1


def test_validate_bleu_defaults_to_opus100(tmp_path, capsys):
    rows = {"Helsinki-NLP/opus-100": [{"translation": {"en": "Good morning", "es": "Buenos días amigo mío"}}]}
    code, _, made, runner = _run_validate_main(
        tmp_path, ["Buenos días amigo mío"], min_accuracy=0.5, argv=["--mode", "bleu"], rows=rows,
    )
    out = capsys.readouterr().out
    assert made["mode"] == "bleu"
    assert runner.prompts == ['Translate to Spanish: "Good morning"']
    assert "Corpus BLEU-4: 100.00" in out
    assert code == 0


def test_validate_mode_needs_a_matching_dataset(tmp_path, capsys):
    code, _, _, _ = _run_validate_main(
        tmp_path, [], min_accuracy=1.0, argv=["--mode", "bleu", "--dataset", "openai/gsm8k", "--config", "main", "--split", "test"],
    )
    assert code == 2
    assert "--mode bleu needs a translation dataset" in capsys.readouterr().err


def test_validate_prepare_value_error_fails_the_item(tmp_path, capsys):
    def too_big(runner):
        raise ValueError("Developer tool block exceeds --max-inp-len")

    loaders = {"x/tools": lambda spec: [
        {"name": "big", "prompt": "p", "expected": "-", "check": lambda a: True, "prepare": too_big},
        {"name": "ok", "prompt": "p", "expected": "-", "check": lambda a: True},
    ]}
    code, _, _, runner = _run_validate_main(
        tmp_path, ["fine"], min_accuracy=0.5, argv=["--dataset", "x/tools"], loaders=loaders,
    )
    out = capsys.readouterr().out
    assert "  [FAIL] big (not run: Developer tool block exceeds --max-inp-len)" in out
    assert "  [PASS] ok" in out
    assert len(runner.prompts) == 1
    assert code == 0
