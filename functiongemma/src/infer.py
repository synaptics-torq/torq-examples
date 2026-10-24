# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from functiongemma.src import function_tools
from functiongemma.src.function_tools import FUNCTIONS
from functiongemma.src.runner import FunctionGemmaStatic
from gemma3.src.runner import format_answer
from functiongemma.setup_demo import (
    download_functiongemma,
    ensure_functiongemma_models,
    local_functiongemma_model_path,
)
from utils.inference import (
    InferenceInterrupted,
    add_llm_inference_args,
    finish_interrupted_output,
)
from utils.terminal import InferenceStopInput
from utils.log import configure_logging
from utils.model_setup import resolve_default_model
from utils.runtime import (
    build_runtime_flags,
    cleanup_npu_after_inference,
    setup_npu_for_inference,
)

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


def run_function_loop(runner: FunctionGemmaStatic) -> None:
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


def main(args: argparse.Namespace):

    logging.getLogger("FunctionGemma").info("Starting assistant...")
    ensure_functiongemma_models(Path(args.model).parent, refresh=not args.no_refresh)

    setup_npu_for_inference()

    try:
        runner = FunctionGemmaStatic(
            args.model,
            args.max_seq_len,
            max_prompt_tokens=args.max_inp_len,
            n_threads=args.threads,
            cache_keep_n=None,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_gen_tokens=args.max_gen_tokens,
            runtime_flags=build_runtime_flags(args.tda, args.runtime_flags),
            device_io=args.device_io,
            lm_head_path=args.lm_head,
            disable_lm_head=args.no_lm_head,
            prefill_model_path=args.batch_prefill_model,
            disable_prefill=args.no_batch_prefill_model,
        )
        run_function_loop(runner)
    finally:
        cleanup_npu_after_inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FunctionGemma function-calling inference.")
    add_llm_inference_args(parser, with_instruct=False, with_kv_window=False, with_max_inp_len=False)
    parser.add_argument("--max-inp-len", type=int, help="Maximum prompt tokens including the developer block; oversized prompts are rejected")
    args = parser.parse_args()
    configure_logging(args.logging)
    if args.model is None:
        args.model = str(
            resolve_default_model(
                parser,
                local_functiongemma_model_path,
                download=download_functiongemma,
                no_refresh=args.no_refresh,
                message="no local FunctionGemma model found; pass -m/--model or run `python setup_demos.py functiongemma`",
            )
        )
    main(args)
