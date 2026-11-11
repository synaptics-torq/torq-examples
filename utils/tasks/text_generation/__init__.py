# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""The ``text_generation`` task category (CONTRIBUTING.md).

``infer_main`` is the shared command-line main of the LLM demos'
``src/infer.py`` scripts: the standard LLM flags, default-model resolution
and refresh, one NPU-boosted session, and an interactive loop (the plain
chat loop, or the demo's own). ``validate_main`` is the shared main of the
``src/validate.py`` scripts: the same model setup with greedy decoding
forced, a Hugging Face dataset (``LOADERS``) and one of three modes —
``prompts`` (each prompt's answer is checked against the dataset's label: an
option letter or a final number), ``bleu`` (corpus BLEU-4 over a translation
dataset) and ``teacher-forced`` (next-token accuracy over the same dataset). Both record the
CONTRIBUTING.md metrics; validate also records its score.
"""

import argparse
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from utils.inference import ContextOverflow, add_llm_inference_args, print_llm_stats, run_chat_loop
from utils.log import configure_logging
from utils.metrics import MetricsRun
from utils.model_setup import resolve_default_model
from utils.datasets import fetch_rows
from utils.runtime import RuntimeConfig, npu_clock
from utils.tasks import add_dataset_args, load_items, scored, skip_item

__all__ = [
    "llm_kwargs",
    "infer_main",
    "validate_main",
    "run_prompts",
    "run_bleu",
    "run_teacher_forced",
    "corpus_bleu",
    "normalize_translation",
    "is_degenerate",
    "first_choice",
    "choice_check",
    "last_number",
    "LOADERS",
    "TRANSLATION_DATASET",
]

# ── infer_main ───────────────────────────────────────────────────────────────


def llm_kwargs(args) -> dict:
    """The shared LLM constructor kwargs of the standard LLM args.

    The demo's ``make_runner`` spreads these into its runner, adding the
    demo-specific options (``instruct_model``, ``thinking``). ``getattr``
    defaults cover flags a demo's ``llm_args`` disabled.
    """
    return dict(
        max_seq_len=args.max_seq_len,
        max_prompt_tokens=getattr(args, "max_inp_len", None),
        cache_keep_n=(
            None
            if getattr(args, "no_kv_cache_window", False)
            or not hasattr(args, "kv_cache_window")
            else args.kv_cache_window
        ),
        temperature=getattr(args, "temperature", 0.0),
        top_p=getattr(args, "top_p", 1.0),
        top_k=getattr(args, "top_k", 64),
        max_gen_tokens=getattr(args, "max_gen_tokens", None),
        max_cache_rolls=getattr(args, "max_cache_rolls", None),
        truncate_prompts=getattr(args, "truncate_prompts", True),
        lm_head_path=getattr(args, "lm_head", None),
        disable_lm_head=getattr(args, "no_lm_head", False),
        prefill_model_path=getattr(args, "batch_prefill_model", None),
        disable_prefill=getattr(args, "no_batch_prefill_model", False),
    )


def _resolve_and_ensure(parser, args, *, demo, local_model, ensure_models, download) -> None:
    """Default ``args.model`` to the demo's local model, then refresh it."""
    if args.model is None:
        args.model = str(
            resolve_default_model(
                parser,
                local_model,
                download=download,
                no_refresh=args.no_refresh,
                message=(
                    f"no local {demo} model found; pass -m/--model or run "
                    f"`python setup_demos.py {demo}` from torq-examples root"
                ),
            )
        )
    ensure_models(Path(args.model).parent, refresh=not args.no_refresh)


def _default_chat(runner, args, m) -> None:
    run_chat_loop(
        lambda text, should_stop: runner.run(text, should_stop=should_stop),
        runner.run_stream,
        lambda: print_llm_stats(runner, m),
        debug=args.logging.upper() == "DEBUG",
    )


def infer_main(
    make_runner: Callable,
    *,
    description: str,
    demo: str,
    local_model: Callable[[], Path | None],
    ensure_models: Callable,
    download: Callable | None = None,
    llm_args: dict | None = None,
    add_args: Callable[[argparse.ArgumentParser], None] | None = None,
    chat: Callable | None = None,
) -> None:
    """Shared main of the LLM demos' ``src/infer.py``.

    *make_runner(args, runtime)* builds the demo's runner; *chat(runner,
    args, metrics)* is the interactive loop (the plain chat loop by
    default). *demo* is the setup_demos.py name used in the no-local-model
    error; *download* (optional) fetches the model on a first run without
    one.
    """
    parser = argparse.ArgumentParser(description=description)
    add_llm_inference_args(parser, **(llm_args or {}))
    if add_args is not None:
        add_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)
    _resolve_and_ensure(
        parser, args, demo=demo, local_model=local_model,
        ensure_models=ensure_models, download=download,
    )
    runtime = RuntimeConfig.from_args(args)
    with npu_clock():
        runner = make_runner(args, runtime)
        with MetricsRun(args.metrics_out) as m:
            (chat or _default_chat)(runner, args, m)


# ── validate_main ────────────────────────────────────────────────────────────

# Regex: preamble the model sometimes prepends before the actual translation.
_PREAMBLE_RE = re.compile(
    r'^\s*'
    r'(?:'
    r"(?:okay[,.]?\s*)?here(?:'s|\s+is|\s+are)\s+(?:the|a|your)?\s*translation[s]?[^:]*[:\-]?"
    r"|traducción[^:]*[:\-]?"
    r'|of\s+["“].+?["”]\s+into\s+\w+\s*[:\-]?'
    r')\s*',
    re.IGNORECASE,
)
# Regex: markdown bold/italic wrappers.
_MD_BOLD_RE = re.compile(r'\*{1,2}(.+?)\*{1,2}')
# Regex: leading bullet / list marker.
_BULLET_RE = re.compile(r'^\s*[\*\-•]\s*')

#: A 6-gram of words repeated until it is the majority of the answer's 6-grams,
#: or one character repeated 20+ times in a row, marks degenerate output.
_NGRAM_N = 6
_CHAR_RUN_RE = re.compile(r"(.)\1{19,}")


def is_degenerate(answer: str) -> bool:
    """Whether *answer* looks like broken model output rather than an answer.

    The n-gram rule needs the dominant 6-gram to occur more than once, so a
    short answer with a single (unrepeated) 6-gram is not degenerate.
    """
    words = answer.split()
    if len(words) >= _NGRAM_N:
        top = _ngram_counts(words, _NGRAM_N).most_common(1)[0][1]
        if top > 1 and top * 2 > len(words) - _NGRAM_N + 1:
            return True
    return bool(_CHAR_RUN_RE.search(answer))


def _tokenize_bleu(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def corpus_bleu(hypotheses: list[str], references: list[str], max_n: int = 4) -> float:
    """Return corpus BLEU-4 score in [0, 100].

    Implements the original Papineni et al. (2002) formula with a single
    reference per sentence.  Returns 0.0 if any n-gram order has zero
    clipped count.
    """
    if not hypotheses:
        return 0.0

    total_hyp_len = 0
    total_ref_len = 0
    clipped: list[int] = [0] * max_n
    total: list[int] = [0] * max_n

    for hyp, ref in zip(hypotheses, references):
        h = _tokenize_bleu(hyp)
        r = _tokenize_bleu(ref)
        total_hyp_len += len(h)
        total_ref_len += len(r)
        for n in range(1, max_n + 1):
            h_ng = _ngram_counts(h, n)
            r_ng = _ngram_counts(r, n)
            clipped[n - 1] += sum(min(c, r_ng[g]) for g, c in h_ng.items())
            total[n - 1] += max(0, len(h) - n + 1)

    precisions: list[float] = []
    for n in range(max_n):
        if total[n] == 0 or clipped[n] == 0:
            return 0.0
        precisions.append(clipped[n] / total[n])

    log_avg = sum(math.log(p) for p in precisions) / max_n
    if total_hyp_len == 0:
        return 0.0
    bp = 1.0 if total_hyp_len >= total_ref_len else math.exp(1.0 - total_ref_len / total_hyp_len)
    return bp * math.exp(log_avg) * 100.0


def normalize_translation(text: str) -> str:
    """Strip preamble and wrapper artifacts from model output."""
    parts = text.strip().split("\n\n")
    if len(parts) > 1 and _PREAMBLE_RE.search(parts[0]):
        text = "\n\n".join(parts[1:])
    else:
        text = text.strip()
    text = _PREAMBLE_RE.sub("", text).strip()

    first_line = text.split("\n")[0].strip()
    if first_line:
        text = first_line

    text = _BULLET_RE.sub("", text).strip()
    text = _MD_BOLD_RE.sub(r"\1", text).strip()
    for q_open, q_close in [('"', '"'), ('“', '”'), ("'", "'")]:
        if text.startswith(q_open) and text.endswith(q_close) and len(text) > 1:
            text = text[len(q_open):-len(q_close)]
            break

    return text.strip()


def first_choice(answer: str, choices: str) -> str | None:
    """The option an answer picks: its leading option (``B``, ``B.``, ``(B)``),
    else the one after "answer is", else its first standalone option."""
    opts = re.escape(choices)
    for pattern in (rf"^\W*([{opts}])(?![A-Za-z0-9])", rf"answer\s*(?:is|:)\s*\W*([{opts}])(?![A-Za-z0-9])",
                    rf"(?<![A-Za-z0-9])([{opts}])(?![A-Za-z0-9])"):
        match = re.search(pattern, answer.strip(), re.IGNORECASE if choices.isalpha() else 0)
        if match:
            return match.group(1).upper()
    return None


def choice_check(expected: str, choices: str) -> Callable[[str], bool]:
    """A check passing answers that pick option *expected* out of *choices*."""
    return lambda answer: first_choice(answer, choices) == expected


def last_number(answer: str) -> float | None:
    """The last number in *answer* (thousands separators ignored)."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def _rows(spec: dict) -> list[dict]:
    return fetch_rows(spec["dataset"], config=spec["config"], split=spec["split"],
                      samples=spec["samples"], offset=spec["offset"])


def _piqa(spec: dict) -> list[dict]:
    items = []
    for i, row in enumerate(_rows(spec), spec["offset"]):
        expected = str(int(row["label"]) + 1)
        items.append({
            "name": f"{spec['dataset']} #{i}",
            "prompt": f"Goal: {row['goal']}\n1. {row['sol1']}\n2. {row['sol2']}\n"
                      "Which solution achieves the goal? Answer with 1 or 2.",
            "expected": expected,
            "check": choice_check(expected, "12"),
        })
    return items


def _gsm8k(spec: dict) -> list[dict]:
    items = []
    for i, row in enumerate(_rows(spec), spec["offset"]):
        expected = row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
        items.append({
            "name": f"{spec['dataset']} #{i}",
            "prompt": row["question"] + "\nSolve it step by step, then give the final answer as a number on the last line.",
            "expected": expected,
            "check": lambda answer, want=float(expected): last_number(answer) == want,
        })
    return items


def _mmlu_pro(spec: dict) -> list[dict]:
    items = []
    for row in _rows(spec):
        letters = "ABCDEFGHIJ"[: len(row["options"])]
        options = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, row["options"]))
        items.append({
            "name": f"{spec['dataset']} #{row['question_id']}",
            "prompt": f"{row['question']}\n{options}\nAnswer with the letter of the correct option only.",
            "expected": row["answer"],
            "check": choice_check(row["answer"], letters),
        })
    return items


_LANGUAGES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German"}


def _opus100(spec: dict) -> list[dict]:
    src, tgt = spec["config"].split("-")
    return [
        {"name": f"{spec['dataset']} #{i}", "src": row["translation"][src], "tgt": row["translation"][tgt],
         "tgt_lang": _LANGUAGES.get(tgt, tgt)}
        for i, row in enumerate(_rows(spec), spec["offset"])
    ]


#: Supported datasets. ``prompts`` items: ``{"name", "prompt", "expected", "check"}``
#: (plus an optional ``prepare(runner)`` hook run before the prompt); translation
#: items (``bleu`` / ``teacher-forced``): ``{"src", "tgt", "tgt_lang"}``.
LOADERS: dict[str, Callable[[dict], list[dict]]] = {
    "baber/piqa": _piqa,
    "openai/gsm8k": _gsm8k,
    "TIGER-Lab/MMLU-Pro": _mmlu_pro,
    "Helsinki-NLP/opus-100": _opus100,
}

#: The translation modes' default dataset (English to Spanish OPUS-100).
TRANSLATION_DATASET = [{"dataset": "Helsinki-NLP/opus-100", "config": "en-es", "split": "test", "samples": 50}]


def _answer_stats(runner, metrics: MetricsRun | None) -> str:
    """The per-answer stats suffix; records TTFT/tok-s in *metrics*, if any."""
    decode_ms = runner.last_infer_time - runner.time_to_first_token
    tps = runner.generated_tokens / decode_ms * 1000 if decode_ms > 0 else 0
    if metrics is not None:
        metrics.add("time_to_first_token", int(runner.time_to_first_token * 1e6))
        metrics.add("tokens_per_second", tps)
    return (
        f"({runner.last_infer_time:.0f} ms, "
        f"TTFT: {runner.time_to_first_token:.0f} ms, "
        f"{tps:.1f} tok/s)"
    )


def run_prompts(runner, items: list[dict], *, metrics: MetricsRun | None = None) -> float:
    """Answer each item's prompt; returns the fraction passing ``item["check"]``.

    A degenerate answer (see :func:`is_degenerate`) always fails. An item
    that does not fit the model's context (``prepare(runner)`` or the run
    raises ``ContextOverflow``) is skipped: left out of the score and replaced
    by the dataset's next row (see :func:`skip_item`).
    """
    passed = skipped = 0
    for item in items:
        try:
            if "prepare" in item:
                item["prepare"](runner)
            runner.reset()
            answer = runner.run(item["prompt"])
        except ContextOverflow as exc:
            skip_item(items, item, exc)
            skipped += 1
            continue
        ok = bool(answer.strip()) and not is_degenerate(answer) and item["check"](answer)
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {item['name']} {_answer_stats(runner, metrics)}")
        if not ok:
            print(f"         answer: {answer.strip()[-200:]!r} (expected {item['expected']!r})")
    return passed / scored(items, skipped)


def _translation_prompt(item: dict) -> str:
    return f"Translate to {item['tgt_lang']}: \"{item['src']}\""


def run_bleu(runner, items: list[dict], *, metrics: MetricsRun | None = None) -> float:
    """Corpus BLEU-4 over translation items; returns BLEU/100.

    Each sample is translated with the direct prompt
    ``Translate to <tgt>: "<src>"``; the corpus score is computed on the
    normalized outputs.
    """
    n = len(items)
    hypotheses: list[str] = []
    references: list[str] = []
    for i, item in enumerate(items):
        runner.reset()
        try:
            answer = runner.run(_translation_prompt(item))
        except ContextOverflow as exc:
            skip_item(items, item, exc)
            continue
        hypotheses.append(normalize_translation(answer))
        references.append(item["tgt"])
        print(f"  [{i + 1}/{n}] {_answer_stats(runner, metrics)}")
    n = scored(items, len(items) - len(hypotheses))
    bleu = corpus_bleu(hypotheses, references)
    print(f"Corpus BLEU-4: {bleu:.2f}  (n={n})")
    return bleu / 100


def run_teacher_forced(runner, items: list[dict], *, metrics: MetricsRun | None = None) -> float:
    """Teacher-forced next-token accuracy over translation items.

    After prefilling the direct translation prompt, the reference tokens are
    fed one by one (the gold token, not the prediction) and each prediction
    is compared against the next reference token. Returns the fraction of
    correct tokens.
    """
    n = len(items)
    total_tokens = 0
    correct_tokens = 0
    for i, item in enumerate(items):
        prompt_ids = runner.tokenize(_translation_prompt(item), "user")
        if runner.is_instruct_model:
            prompt_ids += runner.tokenize("", "model")
        target_ids = runner.tokenize(item["tgt"])
        if not target_ids:
            continue
        if target_ids[0] == runner._bos_token_id:
            target_ids = target_ids[1:]
        runner.reset()
        next_tok, pos = runner.prefill_tokens(prompt_ids)
        steps = min(len(target_ids), runner.max_seq_len - pos)
        sample_correct = 0
        for idx in range(steps):
            gold = target_ids[idx]
            if next_tok == gold:
                sample_correct += 1
            if idx < steps - 1:
                next_tok, pos = runner.prefill_tokens([gold])
        total_tokens += steps
        correct_tokens += sample_correct
        print(f"  [{i + 1}/{n}] {sample_correct}/{steps} tokens correct")
    print(
        f"Teacher-forced token accuracy: "
        f"{100.0 * correct_tokens / max(1, total_tokens):.2f}%  (n={n}, tokens={total_tokens})"
    )
    return correct_tokens / max(1, total_tokens)


_MODES: dict[str, Callable] = {
    "prompts": run_prompts,
    "bleu": run_bleu,
    "teacher-forced": run_teacher_forced,
}
#: The metric each mode's score is recorded as (CONTRIBUTING.md metrics JSON, percent).
_SCORES = {"prompts": "accuracy", "bleu": "bleu", "teacher-forced": "token_accuracy"}


def validate_main(
    make_runner: Callable,
    *,
    demo: str,
    validation: list[dict],
    local_model: Callable[[], Path | None],
    ensure_models: Callable,
    download: Callable | None = None,
    llm_args: dict | None = None,
    add_args: Callable[[argparse.ArgumentParser], None] | None = None,
    loaders: dict[str, Callable[[dict], list[dict]]] | None = None,
) -> None:
    """Shared main of the LLM demos' ``src/validate.py``.

    Same model setup as :func:`infer_main` (with greedy decoding forced)
    plus the validation dataset flags (defaults: *validation*, the demo's
    ``info.json`` list; :data:`TRANSLATION_DATASET` for the translation
    modes) and ``--mode {prompts,bleu,teacher-forced}`` (default
    ``prompts``). *loaders* adds demo-specific datasets to :data:`LOADERS`.
    Prints one line per item, then the score, recorded as the ``accuracy``,
    ``bleu`` or ``token_accuracy`` metric (percent) by mode.
    """
    parser = argparse.ArgumentParser(description=f"Validate the {demo} demo.")
    add_llm_inference_args(parser, **(llm_args or {}))
    parser.add_argument(
        "--mode", choices=sorted(_MODES), default="prompts",
        help=(
            "prompts: score each answer against the dataset's label; bleu / teacher-forced: "
            f"translation score (default dataset {TRANSLATION_DATASET[0]['dataset']}) (default: %(default)s)"
        ),
    )
    add_dataset_args(parser, validation)
    if add_args is not None:
        add_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)
    args.temperature = 0.0  # greedy decoding for reproducibility
    args.truncate_prompts = False  # a truncated item is skipped, not scored
    _resolve_and_ensure(
        parser, args, demo=demo, local_model=local_model,
        ensure_models=ensure_models, download=download,
    )
    translation = args.mode != "prompts"
    items, datasets = load_items(args, TRANSLATION_DATASET if translation else validation, {**LOADERS, **(loaders or {})})
    if translation != ("src" in items[0]):
        parser.error(f"--mode {args.mode} needs a {'translation' if translation else 'prompts'} dataset")
    runtime = RuntimeConfig.from_args(args)
    with npu_clock():
        runner = make_runner(args, runtime)
        with MetricsRun(args.metrics_out, info={"validation": datasets}) as m:
            score = _MODES[args.mode](runner, items, metrics=m)
            m.add(_SCORES[args.mode], score * 100)
    print(f"{_SCORES[args.mode]}={score:.4f}")
