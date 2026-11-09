# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.


import argparse
import signal
import sys
from collections.abc import Callable, Iterator

from utils.runtime import add_common_inference_args
from utils.terminal import InferenceStopInput

StopCheck = Callable[[], bool]


class InferenceInterrupted(Exception):
    """Raised when interactive inference is cancelled by the user."""


def _raise_if_stopped(should_stop: StopCheck | None) -> None:
    if should_stop is not None and should_stop():
        raise InferenceInterrupted


# ── Shared inference CLI ─────────────────────────────────────────────────────


_LLM_MODEL_HELP: str = "Path to VMFB model (default: the one setup_demo.py downloaded)"


def add_llm_inference_args(
    parser: argparse.ArgumentParser,
    *,
    model_help: str | None = None,
    with_lm_head: bool = True,
    with_prefill: bool = True,
    with_max_inp_len: bool = True,
    with_max_gen_tokens: bool = True,
    with_instruct: bool = True,
    with_kv_window: bool = True,
    with_sampling: bool = True,
    with_allocator: bool = True,
    with_no_refresh: bool = True,
    device_io_default: bool = True,
) -> None:
    """The standard LLM inference flag set, shared by the LLM demos.

    Adds ``-m/--model`` (optional; the demo defaults it to the model its setup
    downloaded), the LM-head and batched-prefill selection pairs, the
    sequence/generation limits, the sampling options, and
    :func:`add_common_inference_args` (threads, no-refresh, allocator, runtime
    flags, logging). Demo-specific flags are added by the caller.
    """
    add_common_inference_args(
        parser,
        with_allocator=with_allocator,
        with_no_refresh=with_no_refresh,
        device_io_default=device_io_default,
    )
    parser.add_argument(
        "-m", "--model", type=str, default=None,
        help=model_help or _LLM_MODEL_HELP,
    )
    if with_lm_head:
        lm_head_group = parser.add_mutually_exclusive_group()
        lm_head_group.add_argument(
            "--lm-head", type=str, default=None, metavar="PATH",
            help=(
                "Path to a separately compiled LM head .vmfb. "
                "Overrides sibling LM head auto-discovery."
            ),
        )
        lm_head_group.add_argument(
            "--no-lm-head", action="store_true", default=False,
            help="Disable sibling LM head auto-discovery and run only --model.",
        )
    if with_prefill:
        prefill_group = parser.add_mutually_exclusive_group()
        prefill_group.add_argument(
            "--batch-prefill-model", type=str, default=None, metavar="PATH",
            help=(
                "Path to a batched prefill .vmfb that runs complete fixed-size "
                "prompt chunks. Overrides sibling prefill model auto-discovery. "
                "Requires a split LM head (--lm-head or sibling lm_head)."
            ),
        )
        prefill_group.add_argument(
            "--no-batch-prefill-model", action="store_true", default=False,
            help=(
                "Disable sibling batched prefill model auto-discovery and prefill "
                "the prompt with single-token decode steps only."
            ),
        )
    parser.add_argument(
        "--max-seq-len", type=int, default=None,
        help="Maximum sequence length (prompt + generation); auto-detected from model if omitted",
    )
    if with_max_inp_len:
        parser.add_argument(
            "--max-inp-len", type=int,
            help="Maximum input (prompt) length in tokens; longer prompts are "
                 "truncated, shorter ones pass through unchanged",
        )
    if with_max_gen_tokens:
        parser.add_argument(
            "--max-gen-tokens", type=int, default=None,
            help="Maximum number of generated tokens per answer (default: no limit)",
        )
    if with_instruct:
        parser.add_argument(
            "--instruct-model", action="store_true", default=False,
            help="Is instruct model",
        )
    inference_group = (
        parser.add_argument_group("inference")
        if with_kv_window or with_sampling
        else parser
    )
    if with_kv_window:
        inference_group.add_argument(
            "--kv-cache-window",
            type=int,
            default=2,
            metavar="N",
            help=(
                "Enable sliding-window KV cache: when the cache is full, keep the most "
                "recent N entries and discard older ones before continuing generation "
                "(default: %(default)s)"
            ),
        )
        inference_group.add_argument(
            "--no-kv-cache-window",
            action="store_true",
            default=False,
            help=(
                "Disable sliding-window KV cache behavior. "
                "Once the KV cache reaches its maximum length, no further tokens can be generated."
            ),
        )
    if with_sampling:
        inference_group.add_argument(
            "--temperature", type=float, default=0.0,
            help="Sampling temperature (0.0 = greedy) (default: %(default)s)",
        )
        inference_group.add_argument(
            "--top-p", type=float, default=1.0,
            help="Top-p (nucleus) sampling threshold (default: %(default)s)",
        )
        inference_group.add_argument(
            "--top-k", type=int, default=64,
            help="Top-k pre-filter size for sampling (default: %(default)s)",
        )


# ── Shared chat loop ─────────────────────────────────────────────────────────


YELLOW = "\033[33m"
RESET = "\033[0m"


def finish_interrupted_output(started_output: bool) -> None:
    """Print the [Interrupt] marker after a cancelled answer.

    *started_output* says whether any answer text was already printed on the
    line: a fresh ``\\r``-based clear is only safe before the first chunk.
    """
    marker = f"{YELLOW}[Interrupt]{RESET}"
    if started_output:
        sys.stdout.write(f" {marker} \n")
    else:
        sys.stdout.write("\r" + " " * 80 + f"\r{marker} \n")
    sys.stdout.flush()


def print_llm_stats(runner, metrics=None) -> None:
    """Print the per-answer stats line for a LLM runner.

    Uses the runner's ``last_infer_time`` (total ms), ``time_to_first_token``
    (ms) and ``generated_tokens`` — the same fields every LLM demo reports.
    With a *metrics* run, the printed TTFT (ns) and tok/s are recorded too.
    """
    decode_ms = runner.last_infer_time - runner.time_to_first_token
    tps = runner.generated_tokens / decode_ms * 1000 if decode_ms > 0 else 0
    if metrics is not None:
        metrics.add("time_to_first_token", int(runner.time_to_first_token * 1e6))
        metrics.add("tokens_per_second", tps)
    print(
        f"  ({runner.last_infer_time:.0f} ms, "
        f"TTFT: {runner.time_to_first_token:.0f} ms, "
        f"{tps:.1f} tok/s)\n"
    )


def run_chat_loop(
    run_fn: Callable[[str, StopCheck | None], str],
    stream_fn: Callable[[str, StopCheck | None], Iterator[str]],
    stats_fn: Callable[[], None],
    *,
    prompt: str = "You (type 'exit' or 'quit' to stop): ",
    thinking: str = "\033[2m[thinking...]\033[0m",
    erase_width: int = 40,
    debug: bool = False,
) -> None:
    """The interactive answer loop shared by the LLM demos.

    Prompts until EOF or 'exit'/'quit'. Each input runs one answer: streamed
    chunk-by-chunk via *stream_fn* (with the *thinking* spinner) or fully
    buffered via *run_fn* in *debug* mode. Ctrl+C/D during an answer is caught
    by :class:`InferenceStopInput` (typed while inference runs) or a plain
    KeyboardInterrupt; either way the [Interrupt] marker and *stats_fn* are
    printed and the loop continues. *stats_fn* is printed after every answer.
    """
    try:
        while True:
            try:
                inp = input(prompt).strip()
            except EOFError:
                break
            if not inp:
                continue
            if inp.lower() in ("exit", "quit"):
                break

            if debug:
                started_output = False
                try:
                    with InferenceStopInput(sys.stdin) as should_stop:
                        answer = run_fn(inp, should_stop)
                    sys.stdout.write(f"Agent: {answer}")
                    started_output = True
                except (InferenceInterrupted, KeyboardInterrupt):
                    finish_interrupted_output(started_output)
                    stats_fn()
                    continue
            else:
                sys.stdout.write(thinking)
                sys.stdout.flush()
                first = True
                started_output = False
                try:
                    with InferenceStopInput(sys.stdin) as should_stop:
                        for chunk in stream_fn(inp, should_stop):
                            if first:
                                sys.stdout.write("\r" + " " * erase_width + "\rAgent: ")
                                first = False
                                started_output = True
                            sys.stdout.write(chunk)
                            sys.stdout.flush()
                except (InferenceInterrupted, KeyboardInterrupt):
                    finish_interrupted_output(started_output)
                    stats_fn()
                    continue
            stats_fn()
    except KeyboardInterrupt:
        # A second Ctrl+C must not interrupt runtime/interpreter cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print()
