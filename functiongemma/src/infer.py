# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Function-calling chat loop for the FunctionGemma demo.

The interactive loop is demo-specific: each user turn generates until the
model emits a function call or a final answer, executes the call, and feeds
the result back. Everything else (flags, model resolution and refresh,
runtime, metrics) is the shared text_generation infer main.
"""

import logging
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from functiongemma.src import function_tools
from functiongemma.src.function_tools import FUNCTIONS
from functiongemma.src.runner import FunctionGemmaStatic, format_answer
from functiongemma.setup_demo import (
    download_functiongemma,
    ensure_functiongemma_models,
    local_functiongemma_model_path,
)
from utils.inference import InferenceInterrupted, finish_interrupted_output
from utils.terminal import InferenceStopInput
from utils.tasks.text_generation import infer_main, llm_kwargs

#: Cap on function calls per user turn, against runaway call/response loops.
_MAX_TOOL_CALLS: Final[int] = 5
#: Spinner while a generation runs; cleared before the first printed line.
_THINKING: Final[str] = "\033[2m[thinking...]\033[0m"


def _format_args(args: dict) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


def _execute(name: str, args: dict):
    """Run a model-generated call; failures become tool errors for the model."""
    if name not in FUNCTIONS:
        return {"error": f"unknown function '{name}'"}
    try:
        return FUNCTIONS[name](**args)
    except TypeError as exc:
        return {"error": f"invalid arguments for {name}: {exc}"}
    except Exception as exc:  # surfaced to the model as the tool result
        return {"error": str(exc) or exc.__class__.__name__}


def _think(show: bool) -> None:
    sys.stdout.write(_THINKING if show else "\r" + " " * 13 + "\r")
    sys.stdout.flush()


def run_function_loop(runner: FunctionGemmaStatic, metrics=None) -> None:
    """Function-calling chat loop for FunctionGemma exports.

    Each user turn generates until the model emits a function call (stopping
    on ``<start_function_response>``) or a final answer. Calls execute from
    :data:`function_tools.FUNCTIONS`; their results are fed back until the
    model answers without a call. A call is only executed when the turn
    ended on the function-response stop token: a call-shaped answer that
    stopped elsewhere would replay a mismatched pending token.
    """
    logger = logging.getLogger("FunctionGemma")
    while True:
        try:
            inp = input("You (type 'exit' or 'quit' to stop): ").strip()
        except EOFError:
            break
        if not inp:
            continue
        if inp.lower() in ("exit", "quit"):
            break

        started_output = False
        try:
            _think(True)
            with InferenceStopInput(sys.stdin) as should_stop:
                answer = runner.run(inp, should_stop=should_stop)
            calls = 0
            while runner.awaiting_function_response:
                call = function_tools.parse_function_call(answer)
                if call is None:
                    logger.warning(
                        "Turn stopped on %s but no function call parsed; "
                        "showing raw output",
                        function_tools.START_FUNCTION_RESPONSE,
                    )
                    break
                calls += 1
                if calls > _MAX_TOOL_CALLS:
                    logger.warning(
                        "Stopping after %d function calls in one turn",
                        calls - 1,
                    )
                    break
                name, call_args = call
                _think(False)
                print(f"[function] {name}({_format_args(call_args)})")
                result = _execute(name, call_args)
                print(f"[result] {result!r}")
                started_output = True
                _think(True)
                with InferenceStopInput(sys.stdin) as should_stop:
                    answer = "".join(
                        runner.run_function_response(
                            function_tools.format_function_response(name, result),
                            should_stop=should_stop,
                        )
                    )
            if "<start_function_call>" in answer and not runner.awaiting_function_response:
                logger.warning(
                    "Function call did not stop on %s; discarding conversation without executing it",
                    function_tools.START_FUNCTION_RESPONSE,
                )
                runner.reset()
            elif runner.awaiting_function_response or not runner._should_stop(runner._pending_token, []):
                runner.reset()
            _think(False)
            decode_ms = runner.last_infer_time - runner.time_to_first_token
            tps = runner.generated_tokens / decode_ms * 1000 if decode_ms > 0 else 0
            if metrics is not None:
                metrics.add("time_to_first_token", int(runner.time_to_first_token * 1e6))
                metrics.add("tokens_per_second", tps)
            print(
                format_answer(
                    answer.replace("<end_of_turn>", "").replace("<eos>", ""),
                    runner.last_infer_time,
                    runner.time_to_first_token,
                    stats=[f"{tps:.1f} tok/s"],
                )
            )
            started_output = True
        except (InferenceInterrupted, KeyboardInterrupt):
            finish_interrupted_output(started_output)
            _think(False)
            runner.reset()
            logger.debug("Function-calling turn interrupted", exc_info=True)
        except ValueError as exc:
            _think(False)
            runner.reset()
            logger.warning("Turn discarded: %s", exc)



def add_max_inp_len_args(parser):
    parser.add_argument(
        "--max-inp-len", type=int,
        help="Maximum prompt tokens including the developer block; oversized prompts are rejected",
    )


def make_runner(args, runtime):
    return FunctionGemmaStatic(args.model, runtime=runtime, **llm_kwargs(args))


if __name__ == "__main__":
    infer_main(
        make_runner,
        description="Run FunctionGemma function-calling inference.",
        demo="functiongemma",
        local_model=local_functiongemma_model_path,
        ensure_models=ensure_functiongemma_models,
        download=download_functiongemma,
        llm_args=dict(with_instruct=False, with_kv_window=False, with_max_inp_len=False),
        add_args=add_max_inp_len_args,
        chat=lambda runner, args, m: run_function_loop(runner, m),
    )
