# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.


import argparse
import os
import signal
import sys
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping

import numpy as np
import numpy.typing as npt

from torq.runtime import VMFBInferenceRunner
from iree.runtime import DeviceArray

from utils.runtime import add_common_inference_args
from utils.terminal import InferenceStopInput

StopCheck = Callable[[], bool]


class InferenceInterrupted(Exception):
    """Raised when interactive inference is cancelled by the user."""


def _raise_if_stopped(should_stop: StopCheck | None) -> None:
    if should_stop is not None and should_stop():
        raise InferenceInterrupted


class BaseManagedCacheRunner(VMFBInferenceRunner):
    """Abstract base for inference runners with managed KV caches.
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        cache_start_idx: int = 1,
        **kwargs,
    ) -> None:
        kwargs["device_outputs"] = True
        super().__init__(model_path, **kwargs)

        if self.inputs_info is None or self.outputs_info is None:
            raise ValueError(
                f"Model '{model_path}' is missing input/output metadata "
                "required for KV cache management."
            )

        self._cache_start_idx = cache_start_idx

    @abstractmethod
    def reset_kv(self) -> None:
        """Reset mutable KV caches to their initial state."""
        ...

    @abstractmethod
    def save_kv_state(self):
        """Snapshot the current KV-cache state to host NumPy arrays."""
        ...

    @abstractmethod
    def restore_kv_state(self, state) -> None:
        """Restore KV caches from a previously saved snapshot."""
        ...


class ManagedEncDecCacheRunner(BaseManagedCacheRunner):
    """VMFBInferenceRunner with managed self + cross attention KV cache.

    For encoder-decoder architectures (e.g. Moonshine, Opus-MT).

    Assumes model inputs layout: [..., self_v0, self_k0, cross_v0, cross_k0, ..., cross_vN, cross_kN].

    Assumes model outputs layout: [..., self_v0, self_k0, self_v1, self_k1, ..., self_vN, self_kN].

    Args:
        model_path: Path to the ``.vmfb`` file.
        initial_cache: Cross-attn and initial self-attn cache values.
            When ``None`` (the default), caches are zero-initialised from
            the model's input metadata.
        cache_start_idx: Index of the first cache output in the model's
            output list. Defaults to 1 ([logits, self_v0, ...]).
        input_cache_start_idx: Index of the first cache input in the
            model's input list.  Defaults to *cache_start_idx*.
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        initial_cache: list[npt.NDArray | DeviceArray] | None = None,
        cache_start_idx: int = 1,
        input_cache_start_idx: int | None = None,
        device_io: bool = False,
        **kwargs,
    ) -> None:
        self._device_io = device_io
        super().__init__(model_path, cache_start_idx=cache_start_idx, **kwargs)

        self._input_cache_start: int = (
            input_cache_start_idx if input_cache_start_idx is not None
            else cache_start_idx
        )

        n_self_cache_outputs = len(self.outputs_info) - self._cache_start_idx
        if n_self_cache_outputs <= 0 or n_self_cache_outputs % 2 != 0:
            raise ValueError(
                f"Expected self-attn cache outputs (from index {cache_start_idx}) "
                f"to be a multiple of 2 (v, k per layer), "
                f"got {n_self_cache_outputs}."
            )
        self._n_layers = n_self_cache_outputs // 2

        n_cache_inputs = len(self.inputs_info) - self._input_cache_start
        if n_cache_inputs != 4 * self._n_layers:
            raise ValueError(
                f"Expected {4 * self._n_layers} interleaved cache inputs "
                f"({self._n_layers} layers * 4), got {n_cache_inputs}."
            )

        if initial_cache is not None:
            expected = 4 * self._n_layers
            if len(initial_cache) != expected:
                raise ValueError(
                    f"Expected {expected} cache tensors "
                    f"({self._n_layers} layers * 4), got {len(initial_cache)}."
                )
            initial_cache = [
                self.allocate_device_array(c) if isinstance(c, np.ndarray) else c
                for c in initial_cache
            ]
        else:
            in_info = self.inputs_info
            initial_cache = [
                self.allocate_device_array(
                    np.zeros(
                        in_info[self._input_cache_start + i].shape,
                        dtype=np.dtype(in_info[self._input_cache_start + i].dtype),
                    )
                )
                for i in range(4 * self._n_layers)
            ]
        self._self_cache: list = [None] * (2 * self._n_layers)
        self._cross_cache: list = [None] * (2 * self._n_layers)
        for layer in range(self._n_layers):
            base = layer * 4
            self._self_cache[2 * layer] = initial_cache[base]
            self._self_cache[2 * layer + 1] = initial_cache[base + 1]
            self._cross_cache[2 * layer] = initial_cache[base + 2]
            self._cross_cache[2 * layer + 1] = initial_cache[base + 3]

    def set_cache(
        self,
        interleaved_cache: list[npt.NDArray | DeviceArray],
        *,
        pad_axis: int = 2,
    ) -> None:
        """Set self and cross caches from an interleaved list.

        Self-attention caches are zero-padded along *pad_axis* when their
        shape is smaller than the model expects (e.g. a first decoder step
        producing ``[B, H, 1, D]`` while the model needs ``[B, H, L, D]``).
        """
        expected = 4 * self._n_layers
        if len(interleaved_cache) != expected:
            raise ValueError(
                f"Expected {expected} cache tensors "
                f"({self._n_layers} layers * 4), got {len(interleaved_cache)}."
            )

        in_info = self.inputs_info
        for layer in range(self._n_layers):
            base = layer * 4
            for j in range(2):  # self v, k
                c = interleaved_cache[base + j]
                if isinstance(c, DeviceArray):
                    c = c.to_host()
                else:
                    c = np.asarray(c)
                target = tuple(in_info[self._input_cache_start + base + j].shape)
                if c.shape != target:
                    padded = np.zeros(target, dtype=c.dtype)
                    slices = [slice(None)] * c.ndim
                    slices[pad_axis] = slice(0, c.shape[pad_axis])
                    padded[tuple(slices)] = c
                    c = padded
                self._self_cache[2 * layer + j] = self.allocate_device_array(c)

            for j in range(2):  # cross v, k
                c = interleaved_cache[base + 2 + j]
                if isinstance(c, DeviceArray):
                    c = c.to_host()
                else:
                    c = np.asarray(c)
                self._cross_cache[2 * layer + j] = self.allocate_device_array(c)

    def _infer(self, inputs: Iterable[npt.NDArray] | Mapping[str, npt.NDArray]) -> list:
        if isinstance(inputs, Mapping):
            user_inputs = list(inputs.values())
        else:
            user_inputs = list(inputs)
        if self._device_io:
            user_inputs = [
                self.allocate_device_array(x) if isinstance(x, np.ndarray) else x
                for x in user_inputs
            ]

        # Rebuild interleaved cache: self_v, self_k, cross_v, cross_k per layer.
        interleaved = []
        for layer in range(self._n_layers):
            interleaved.append(self._self_cache[2 * layer])
            interleaved.append(self._self_cache[2 * layer + 1])
            interleaved.append(self._cross_cache[2 * layer])
            interleaved.append(self._cross_cache[2 * layer + 1])

        full_inputs = user_inputs + interleaved

        results = super()._infer(full_inputs)

        # Outputs from cache_start_idx onward are self-attn only (2 per layer).
        n_self_outputs = 2 * self._n_layers
        for i in range(n_self_outputs):
            self._self_cache[i] = results[self._cache_start_idx + i]

        return results[:self._cache_start_idx]

    def reset_kv(self) -> None:
        """Reset self-attention caches to zeros; cross-attention is unchanged."""
        in_info = self.inputs_info
        for layer in range(self._n_layers):
            for j in range(2):  # v, k
                idx = self._input_cache_start + layer * 4 + j
                info = in_info[idx]
                z = np.zeros(info.shape, dtype=np.dtype(info.dtype))
                self._self_cache[2 * layer + j] = self.allocate_device_array(z)

    def set_cross_cache(self, cross_tensors: list[npt.NDArray | DeviceArray]) -> None:
        """Set cross-attention caches from a flat list (2 per layer: key, value).

        Typically called with encoder outputs that produce the cross-attn
        KV caches directly.
        """
        expected = 2 * self._n_layers
        if len(cross_tensors) != expected:
            raise ValueError(
                f"Expected {expected} cross-attn cache tensors "
                f"({self._n_layers} layers * 2), got {len(cross_tensors)}."
            )
        for i, c in enumerate(cross_tensors):
            if isinstance(c, np.ndarray):
                c = self.allocate_device_array(c)
            self._cross_cache[i] = c

    def save_kv_state(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Snapshot self and cross caches to host."""
        self_state = [c.to_host().copy() for c in self._self_cache]
        cross_state = [c.to_host().copy() for c in self._cross_cache]
        return self_state, cross_state

    def restore_kv_state(
        self, state: tuple[list[np.ndarray], list[np.ndarray]]
    ) -> None:
        """Restore both cache types from a previously saved snapshot."""
        self_state, cross_state = state
        self._self_cache = [self.allocate_device_array(a) for a in self_state]
        self._cross_cache = [self.allocate_device_array(a) for a in cross_state]


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
