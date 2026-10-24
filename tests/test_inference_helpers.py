# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Tests for the shared inference helpers (T6): the standard LLM argparser,
the chat loop, and the ``max_gen_tokens`` generation cap."""

import argparse
import builtins
import logging
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from utils.inference import (
    InferenceInterrupted,
    add_common_inference_args,
    add_llm_inference_args,
    finish_interrupted_output,
    print_llm_stats,
    run_chat_loop,
)
from utils import llm as llm_module
from utils.llm import DecoderOnlyLLMRunner


# ── add_llm_inference_args / add_common_inference_args ─────────────────────────────────


def _llm_parser(**kwargs) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_llm_inference_args(parser, **kwargs)
    return parser


def test_llm_args_defaults():
    args = _llm_parser().parse_args([])
    assert args.model is None
    assert args.lm_head is None and args.no_lm_head is False
    assert args.batch_prefill_model is None and args.no_batch_prefill_model is False
    assert args.max_seq_len is None
    assert args.max_inp_len is None
    assert args.max_gen_tokens is None
    assert args.instruct_model is False
    assert args.threads is None
    assert args.kv_cache_window == 2 and args.no_kv_cache_window is False
    assert args.temperature == 0.0 and args.top_p == 1.0 and args.top_k == 64
    assert args.no_refresh is False
    assert args.tda == "dmabuf" and args.device_io is True
    assert args.runtime_flags is None
    assert args.logging == "INFO"


def test_llm_args_values():
    args = _llm_parser().parse_args(
        [
            "-m", "x.vmfb",
            "--lm-head", "h.vmfb",
            "--batch-prefill-model", "p.vmfb",
            "--max-seq-len", "512",
            "--max-inp-len", "128",
            "--max-gen-tokens", "64",
            "--instruct-model",
            "-j", "4",
            "--no-kv-cache-window",
            "--temperature", "0.7",
            "--top-p", "0.9",
            "--top-k", "40",
            "--no-refresh",
            "--tda", "cpu",
            "--no-device-io",
            "--logging", "DEBUG",
            "--runtime-flags", "--torq_hw_type=sim", "--other=1",
        ]
    )
    assert args.model == "x.vmfb"
    assert args.lm_head == "h.vmfb"
    assert args.batch_prefill_model == "p.vmfb"
    assert args.max_seq_len == 512
    assert args.max_inp_len == 128
    assert args.max_gen_tokens == 64
    assert args.instruct_model is True
    assert args.threads == 4
    assert args.no_kv_cache_window is True
    assert args.temperature == 0.7 and args.top_p == 0.9 and args.top_k == 40
    assert args.no_refresh is True
    assert args.tda == "cpu" and args.device_io is False
    assert args.runtime_flags == ["--torq_hw_type=sim", "--other=1"]
    assert args.logging == "DEBUG"


def test_llm_args_lm_head_pair_is_mutually_exclusive():
    with pytest.raises(SystemExit):
        _llm_parser().parse_args(["--lm-head", "a.vmfb", "--no-lm-head"])


def test_llm_args_prefill_pair_is_mutually_exclusive():
    with pytest.raises(SystemExit):
        _llm_parser().parse_args(
            ["--batch-prefill-model", "a.vmfb", "--no-batch-prefill-model"]
        )


@pytest.mark.parametrize(
    "kwargs,flag",
    [
        ({"with_lm_head": False}, "--lm-head"),
        ({"with_prefill": False}, "--batch-prefill-model"),
        ({"with_max_inp_len": False}, "--max-inp-len"),
        ({"with_max_gen_tokens": False}, "--max-gen-tokens"),
        ({"with_instruct": False}, "--instruct-model"),
        ({"with_kv_window": False}, "--kv-cache-window"),
        ({"with_sampling": False}, "--temperature"),
        ({"with_allocator": False}, "--tda"),
        ({"with_allocator": False}, "--device-io"),
        ({"with_no_refresh": False}, "--no-refresh"),
    ],
)
def test_llm_args_knobs_remove_flags(kwargs, flag):
    with pytest.raises(SystemExit):
        _llm_parser(**kwargs).parse_args([flag])


def test_common_inference_args_threads_off():
    parser = argparse.ArgumentParser()
    add_common_inference_args(parser, with_threads=False, with_allocator=False)
    args = parser.parse_args(["--runtime-flags", "--a=1"])
    assert not hasattr(args, "threads")
    assert not hasattr(args, "tda")
    assert not hasattr(args, "device_io")
    assert args.runtime_flags == ["--a=1"]
    assert args.no_refresh is False


def test_common_inference_args_device_io_default_false():
    parser = argparse.ArgumentParser()
    add_common_inference_args(parser, device_io_default=False)
    args = parser.parse_args([])
    assert args.device_io is False


def test_common_inference_args_tda_default():
    parser = argparse.ArgumentParser()
    add_common_inference_args(parser)
    assert parser.parse_args([]).tda == "dmabuf"

    # Moonshine and the other demos use the dmabuf allocator by default.
    parser = argparse.ArgumentParser()
    add_common_inference_args(parser, tda_default="dmabuf")
    assert parser.parse_args([]).tda == "dmabuf"


# ── chat loop ────────────────────────────────────────────────────────────────


class _FakeStatsRunner:
    last_infer_time = 123.4
    time_to_first_token = 10.0
    generated_tokens = 5


def _fake_stream(text, should_stop=None):
    if text == "boom":
        raise InferenceInterrupted()
    yield "Hello"
    yield " world"


def test_run_chat_loop_answers_interrupts_and_exits(monkeypatch, capsys):
    inputs = iter(["hi", "boom", ""])
    stats_calls = []
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs, "quit"))

    run_chat_loop(
        lambda text, should_stop: "unused",
        _fake_stream,
        lambda: stats_calls.append(1),
    )

    out = capsys.readouterr().out
    assert "Agent: Hello world" in out
    assert "[Interrupt]" in out
    assert stats_calls == [1, 1]  # one stats line per answered/aborted prompt


def test_run_chat_loop_empty_input_is_skipped(monkeypatch, capsys):
    inputs = iter(["", "hi", "quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs, "quit"))

    run_chat_loop(lambda t, s: "", _fake_stream, lambda: None)

    out = capsys.readouterr().out
    assert "Agent: Hello world" in out
    assert out.count("Agent:") == 1


def test_run_chat_loop_debug_mode_uses_run_fn(monkeypatch, capsys):
    inputs = iter(["hi", "exit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs, "exit"))

    run_chat_loop(
        lambda text, should_stop: f"full answer to {text}",
        _fake_stream,
        lambda: None,
        debug=True,
    )

    out = capsys.readouterr().out
    assert "Agent: full answer to hi" in out
    assert "[thinking" not in out


def test_run_chat_loop_prompt_and_terminate(monkeypatch, capsys):
    inputs = iter(["hi", "QUIT"])
    seen_prompts = []
    monkeypatch.setattr(
        builtins, "input", lambda prompt="": (seen_prompts.append(prompt), next(inputs, ""))[1]
    )

    run_chat_loop(lambda t, s: "", _fake_stream, lambda: None, prompt="Q: ")

    assert seen_prompts == ["Q: ", "Q: "]
    assert "Agent: Hello world" in capsys.readouterr().out


def test_finish_interrupted_output_variants(capsys):
    finish_interrupted_output(True)
    out = capsys.readouterr().out
    assert "[Interrupt]" in out and "\r" not in out

    finish_interrupted_output(False)
    out = capsys.readouterr().out
    assert "\r" in out and "[Interrupt]" in out


@pytest.mark.parametrize("debug", [False, True])
def test_ctrl_c_during_generation_keeps_prompt_interruptible(monkeypatch, capsys, debug):
    inputs = iter(["hi", "quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt: next(inputs))

    def interrupt(*args):
        raise KeyboardInterrupt

    stats = mock.Mock()
    with mock.patch("utils.inference.signal.signal") as set_handler:
        run_chat_loop(interrupt, interrupt, stats, debug=debug)
    set_handler.assert_not_called()
    stats.assert_called_once_with()
    assert "[Interrupt]" in capsys.readouterr().out


def test_repeated_ctrl_c_during_thread_shutdown_exits_cleanly():
    script = textwrap.dedent('''
        import builtins
        import runpy
        import sys
        import threading

        runpy.run_path("tests/conftest.py")
        from utils.inference import run_chat_loop

        def interrupt(prompt):
            raise KeyboardInterrupt

        builtins.input = interrupt
        threading.Thread(target=lambda: sys.stdin.read()).start()
        run_chat_loop(None, None, None)
        print("shutting down", flush=True)
    ''')
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=Path(__file__).resolve().parents[1],
    )
    try:
        assert child.stdout.readline() == "\n"
        assert child.stdout.readline() == "shutting down\n"
        child.send_signal(signal.SIGINT)
        stdout, stderr = child.communicate("\n", timeout=10)
        assert child.returncode == 0
        assert stdout == stderr == ""
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()


def test_print_llm_stats_line():
    with mock.patch("builtins.print") as print_mock:
        print_llm_stats(_FakeStatsRunner())
    line = print_mock.call_args.args[0]
    assert line == "  (123 ms, TTFT: 10 ms, 44.1 tok/s)\n"


def test_inference_interrupted_shared_across_modules():
    # utils.llm must re-export the very class the chat loop catches.
    assert llm_module.InferenceInterrupted is InferenceInterrupted


# ── max_gen_tokens cap (T2 plumbing) ─────────────────────────────────────────


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


class _GenProbe(DecoderOnlyLLMRunner):
    """Concrete runner with __init__ skipped; generation is faked per step."""

    def __init__(self, max_gen_tokens):
        self._logger = logging.getLogger("test_max_gen_tokens")
        self._max_gen_tokens = max_gen_tokens
        self._n_tokens_gen = 0
        self._last_infer_ns = 0
        self._time_to_first_token_ns = 0
        self._start_time_ns = 0
        self._warmup_len = 0
        self._max_seq_len = 1000
        self._max_prompt_tokens = None
        self._max_user_tokens = None
        self._cache_keep_n = None
        self._turn_pos = 0
        self._pending_token = None
        self._reset_cache_state = []
        self._tokenizer = _FakeTokenizer()
        self._eos_token_id = 99
        self._eos_token_ids = [99]

    def tokenize(self, text: str, role: str | None = None) -> list[int]:
        raise NotImplementedError

    def _reset_cache(self) -> None:
        pass

    def _build_prompt_tokens(self, user_input: str) -> list[int]:
        return [1]

    def _prefill(self, tokens, start=0, should_stop=None, *, produce_next_token=True):
        return 2, start + 1

    def llm_step(self, token, seq_pos, **kwargs):
        return 3

    def _should_stop(self, token, gen):
        return len(gen) >= 6  # natural stop, so the uncapped case terminates


def test_max_gen_tokens_caps_generation():
    assert _GenProbe(max_gen_tokens=3).run("hi") == "2 3 3"


def test_max_gen_tokens_none_keeps_old_behaviour():
    # Without a cap, generation runs to the runner's own stop condition (6 here).
    assert _GenProbe(max_gen_tokens=None).run("hi") == "2 3 3 3 3 3"


# ── ManagedSelfAttnCacheRunner: read-only KV cache rows ───────────────────────


def test_managed_cache_runner_writes_new_rows_at_the_step_position(monkeypatch):
    import numpy as np
    from types import SimpleNamespace

    from utils import inference

    class _Array:
        """Device array stand-in; its own buffer for the copies."""

        def __init__(self, a):
            self.a = np.ascontiguousarray(a)
            self.shape, self.dtype = self.a.shape, self.a.dtype
            self._buffer_view = SimpleNamespace(get_buffer=lambda: self)

    class _CommandBuffer:
        def __init__(self, device):
            pass

        def copy(self, src, dst, src_offset, dst_offset, length):
            dst.a.reshape(-1).view(np.uint8)[dst_offset:dst_offset + length] = (
                src.a.reshape(-1).view(np.uint8)[src_offset:src_offset + length]
            )

        def end(self):
            pass

    info = lambda shape: SimpleNamespace(shape=shape, dtype="float32")  # noqa: E731
    # Time-major [1, L, 2kv, D] and head-major [1, 2kv, L, D] row caches, and a whole-cache state.
    caches = [info([1, 4, 2, 3]), info([1, 2, 4, 3]), info([1, 5])]
    rows = [info([1, 1, 2, 3]), info([1, 2, 1, 3]), info([1, 5])]

    def fake_infer(self, inputs):
        pos = int(np.asarray(inputs[1]).reshape(-1)[0])
        return [_Array(np.zeros(1)), *(_Array(np.full(r.shape, pos + 1, np.float32)) for r in rows)]

    monkeypatch.setattr(inference, "HalCommandBuffer", _CommandBuffer)
    monkeypatch.setattr(inference.VMFBInferenceRunner, "__init__", lambda self, *a, **k: None, raising=False)
    monkeypatch.setattr(inference.VMFBInferenceRunner, "_infer", fake_infer, raising=False)
    semaphore = SimpleNamespace(wait=lambda value: None)
    monkeypatch.setattr(inference.ManagedSelfAttnCacheRunner, "inputs_info", [info([1, 1]), info([1, 1]), *caches], raising=False)
    monkeypatch.setattr(inference.ManagedSelfAttnCacheRunner, "outputs_info", [info([1]), *rows], raising=False)
    monkeypatch.setattr(inference.ManagedSelfAttnCacheRunner, "device", SimpleNamespace(
        create_semaphore=lambda value: semaphore, queue_execute=lambda cb, wait, signal: None,
    ), raising=False)
    monkeypatch.setattr(inference.ManagedSelfAttnCacheRunner, "allocate_device_array", lambda self, a: _Array(a), raising=False)

    runner = inference.ManagedSelfAttnCacheRunner("model.vmfb")
    assert runner.kv_seq_len == 4
    for pos in range(3):
        runner._infer([np.zeros((1, 1)), np.array([[pos]])])

    time_major, head_major, state = (c.a for c in runner._kv_cache)
    np.testing.assert_array_equal(time_major[0, :, 0, 0], [1, 2, 3, 0])
    np.testing.assert_array_equal(head_major[0, 1, :, 2], [1, 2, 3, 0])
    np.testing.assert_array_equal(state, np.full((1, 5), 3))
