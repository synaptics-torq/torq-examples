# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Validate the FunctionGemma model.

Scores function calling on the Berkeley Function Calling Leaderboard's simple
category (``info.json``; raw files of the dataset repo, which the dataset
viewer does not serve): each question comes with its own function
declarations, and the call the model emits must name the expected function
with argument values among the leaderboard's possible answers.
"""

import json
from pathlib import Path

from functiongemma.src.runner import FunctionGemmaStatic
from functiongemma.setup_demo import (
    download_functiongemma,
    ensure_functiongemma_models,
    local_functiongemma_model_path,
)
from functiongemma.src.function_tools import parse_function_call
from utils.datasets import fetch_file
from utils.tasks import demo_validation
from utils.tasks.text_generation import llm_kwargs, validate_main

MIN_ACCURACY = 2.0  # DUMMY until a measured baseline exists: above 1.0, so the accuracy check always FAILS.


def make_runner(args, runtime):
    return FunctionGemmaStatic(args.model, runtime=runtime, **llm_kwargs(args))


def _json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _same(value, accepted) -> bool:
    if isinstance(value, str) and isinstance(accepted, str):
        return value.strip().lower() == accepted.strip().lower()
    if isinstance(value, (int, float)) and isinstance(accepted, (int, float)) and not isinstance(value, bool):
        return float(value) == float(accepted)
    return value == accepted


def call_matches(answer: str, expected: dict[str, dict[str, list]]) -> bool:
    """Whether *answer* holds one call to the expected function whose every
    argument value is among the possible answers ("" marks an optional one)."""
    call = parse_function_call(answer)
    if call is None:
        return False
    name, args = call
    if name not in expected:
        return False
    possible = expected[name]
    if set(args) - set(possible):
        return False
    return all(
        any(_same(args[arg], ok) for ok in accepted) if arg in args else "" in accepted
        for arg, accepted in possible.items()
    )


def _bfcl(spec: dict) -> list[dict]:
    """BFCL v3 questions (``BFCL_v3_<config>.json``) with ``possible_answer/``.

    Function names are declared with ``.`` as ``_``: the FunctionGemma call
    syntax has no dots in names.
    """
    filename = f"BFCL_v3_{spec['config']}.json"
    answers = {a["id"]: a["ground_truth"][0] for a in _json_lines(fetch_file(spec["dataset"], f"possible_answer/{filename}"))}
    items = []
    for question in _json_lines(fetch_file(spec["dataset"], filename))[: spec["samples"]]:
        tools = [{"function": dict(f, name=f["name"].replace(".", "_"))} for f in question["function"]]
        expected = {name.replace(".", "_"): args for name, args in answers[question["id"]].items()}
        items.append({
            "name": question["id"],
            "prompt": question["question"][0][-1]["content"],
            "expected": json.dumps(expected),
            "check": lambda answer, expected=expected: call_matches(answer, expected),
            "prepare": lambda runner, tools=tools: runner.set_tools(tools),
        })
    return items


if __name__ == "__main__":
    validate_main(
        make_runner,
        demo="functiongemma",
        validation=demo_validation(__file__),
        min_accuracy=MIN_ACCURACY,
        local_model=local_functiongemma_model_path,
        ensure_models=ensure_functiongemma_models,
        download=download_functiongemma,
        # Same standard-flag set as infer.py (the runner forces instruct mode
        # and never rolls the KV cache).
        llm_args=dict(with_instruct=False, with_kv_window=False, with_max_inp_len=False),
        loaders={"gorilla-llm/Berkeley-Function-Calling-Leaderboard": _bfcl},
    )
