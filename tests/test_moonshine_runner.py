# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Tests for moonshine's ``MoonshineRunner`` on the shared decoding core:
cross tensors land in the fixed slots, self caches update, generation stops
at END / max_tokens, and the output token array matches the script."""

import os
from types import SimpleNamespace

import numpy as np

from moonshine.src import runner as ms_runner
from moonshine.src.runner import MoonshineRunner
from utils.runtime import RuntimeConfig

START, END = 1, 2
VOCAB = 16
HIDDEN = 4
SEQ = 4  # self-cache length (decoder.seq_len)
MEM = 3  # cross-cache length
HEADS, DIM = 2, 2
N_LAYERS = 2
ENC_LEN = 16000  # 1 s at the default 16 kHz → default max_tokens = 6

AUDIO = np.zeros((1, ENC_LEN), "float32")


class _Info:
    def __init__(self, shape, dtype):
        self.shape = list(shape)
        self.dtype = dtype


class _Array:
    """Device array stand-in."""

    def __init__(self, a):
        self.a = np.array(a, copy=True)
        self.shape, self.dtype = self.a.shape, np.dtype(self.a.dtype)
        self._buffer_view = SimpleNamespace(get_buffer=lambda: self)

    def to_host(self):
        return self.a

    def __array__(self, dtype=None):
        return self.a


class _Device:
    def create_semaphore(self, value):
        return SimpleNamespace(wait=lambda v: None)

    def queue_execute(self, cb, waits, signals):
        pass


class _Model:
    """Fake utils.runtime.Model: metadata, device staging, recorded calls."""

    def __init__(self, path, inputs, outputs, result_fn=None):
        self.path = path
        self.inputs_info = inputs
        self.outputs_info = outputs
        self.device = _Device()
        self.calls = []
        self.results = []
        self.infer_time_ms = 0.0
        self._result_fn = result_fn

    def to_device(self, x):
        return _Array(x) if isinstance(x, np.ndarray) else x

    def allocate_device_array(self, x):
        return _Array(x)

    def infer(self, inputs):
        staged = [self.to_device(x) for x in inputs]
        self.calls.append(staged)
        out = (
            self._result_fn(staged)
            if self._result_fn is not None
            else [_Array(np.zeros(o.shape, np.dtype(o.dtype))) for o in self.outputs_info]
        )
        self.results.append(out)
        return out


def _token_of(token_input, token_shape):
    """The fed token id: the embedding row index, or the raw id."""
    value = int(np.asarray(token_input).reshape(-1)[0])
    return value // HIDDEN if len(token_shape) >= 3 else value


def _decoder_model(replies, no_end, token_shape, cross_dtype="float16"):
    """Scripted decoder body: logits argmax walks *replies*, then *no_end*.

    The self-cache outputs carry the fed position, so tests can read the
    update back out of the cache.
    """
    it = iter(replies)

    def fn(inputs):
        pos = int(np.asarray(inputs[1]).reshape(-1)[0])
        reply = next(it, no_end)
        logits = np.zeros((1, 1, VOCAB), "float32")
        logits[0, 0, reply] = 1.0
        outs = [_Array(logits)]
        outs.extend(
            _Array(np.full((1, HEADS, SEQ, DIM), pos, "float32"))
            for _ in range(2 * N_LAYERS)
        )
        return outs

    inputs = [_Info(token_shape, "float32" if len(token_shape) >= 3 else "int32"),
              _Info((1, 1), "int64")]
    for _ in range(N_LAYERS):
        inputs += [
            _Info((1, HEADS, SEQ, DIM), "float32"),  # self_v
            _Info((1, HEADS, SEQ, DIM), "float32"),  # self_k
            _Info((1, HEADS, MEM, DIM), cross_dtype),  # cross_v
            _Info((1, HEADS, MEM, DIM), cross_dtype),  # cross_k
        ]
    outputs = [_Info((1, 1, VOCAB), "float32")] + [
        _Info((1, HEADS, SEQ, DIM), "float32") for _ in range(2 * N_LAYERS)
    ]
    return _Model("decoder.vmfb", inputs, outputs, result_fn=fn)


def _encoder_model(device_outputs):
    """Encoder emitting 2 * N_LAYERS distinct cross tensors."""
    base = np.arange(HEADS * MEM * DIM).reshape((1, HEADS, MEM, DIM)).astype("float32")

    def fn(_inputs):
        outs = [base + i for i in range(2 * N_LAYERS)]
        return [o if not device_outputs else _Array(o) for o in outs]

    return _Model(
        "encoder.vmfb",
        [_Info((1, ENC_LEN), "float32")],
        [_Info((1, HEADS, MEM, DIM), "float32")] * (2 * N_LAYERS),
        result_fn=fn,
    )


def make_runner(tmp_path, monkeypatch, *, replies, no_end=9,
                with_embeddings=True, device_cross=False):
    """A MoonshineRunner over fake models; returns (runner, models)."""
    (tmp_path / "encoder.vmfb").write_bytes(b"")
    (tmp_path / "decoder.vmfb").write_bytes(b"")
    if with_embeddings:
        np.save(
            tmp_path / "decoder_token_embeddings.npy",
            np.arange(VOCAB * HIDDEN).reshape(VOCAB, HIDDEN).astype("float32"),
        )

    token_shape = (1, 1, HIDDEN) if with_embeddings else (1, 1)
    models = {
        "encoder": _encoder_model(device_cross),
        "decoder": _decoder_model(replies, no_end, token_shape),
    }

    def fake_model(path, runtime, *, function="main", device_outputs=False):
        return models["encoder" if "encoder" in os.path.basename(str(path)) else "decoder"]

    monkeypatch.setattr(ms_runner, "Model", fake_model)
    runner = MoonshineRunner(str(tmp_path), runtime=RuntimeConfig(tda="cpu", device_io=False))
    return runner, models


def test_run_generates_until_end(tmp_path, monkeypatch):
    runner, models = make_runner(tmp_path, monkeypatch, replies=[10, 11, END])

    tokens = runner.run(AUDIO)

    assert tokens.shape == (1, 3)
    np.testing.assert_array_equal(tokens[0], [10, 11, END])
    assert runner.generated_tokens == 3
    decoder = models["decoder"]
    token_shape = (1, 1, HIDDEN)
    # START at position 0, then each generated token at the next position.
    assert [int(np.asarray(c[1])[0, 0]) for c in decoder.calls] == [0, 1, 2]
    assert [_token_of(c[0], token_shape) for c in decoder.calls] == [START, 10, 11]


def test_run_stops_at_max_tokens_without_end(tmp_path, monkeypatch):
    runner, models = make_runner(tmp_path, monkeypatch,
                                 replies=[10, 11, 12, 13], no_end=99)

    tokens = runner.run(AUDIO, max_tokens=3)

    np.testing.assert_array_equal(tokens[0], [10, 11, 12])
    assert runner.generated_tokens == 3
    assert len(models["decoder"].calls) == 3


def test_cross_tensors_landed_in_fixed_slots(tmp_path, monkeypatch):
    runner, models = make_runner(tmp_path, monkeypatch, replies=[END])

    runner.run(AUDIO)

    base = np.arange(HEADS * MEM * DIM).reshape((1, HEADS, MEM, DIM))
    for out_idx, slot in enumerate([2, 3, 6, 7]):  # cross_v/k of each layer
        staged = runner._cache.inputs[slot]
        assert staged.dtype == np.dtype("float16")  # cast to the fixed slots' dtype
        np.testing.assert_array_equal(
            np.asarray(staged), (base + out_idx).astype("float16")
        )
    # The self caches were reset to zeros before the first step.
    for i in (2, 3, 6, 7):
        assert np.all(np.asarray(models["decoder"].calls[0][i]) == 0)


def test_device_cross_tensors_pass_through(tmp_path, monkeypatch):
    runner, models = make_runner(tmp_path, monkeypatch,
                                 replies=[END], device_cross=True)

    runner.run(AUDIO)

    for out_idx, slot in enumerate([2, 3, 6, 7]):
        # No re-upload: the encoder's device arrays land in the slots as-is.
        assert runner._cache.inputs[slot] is models["encoder"].results[0][out_idx]


def test_self_caches_update_from_model_outputs(tmp_path, monkeypatch):
    runner, models = make_runner(tmp_path, monkeypatch, replies=[10, END])

    runner.run(AUDIO)

    results = models["decoder"].results[-1]  # the step at position 1
    for j, slot in enumerate([0, 1, 4, 5]):  # self_v/k of each layer
        assert runner._cache.inputs[slot] is results[1 + j]
        np.testing.assert_array_equal(np.asarray(runner._cache.inputs[slot]), 1)


def test_run_without_token_embeddings_feeds_raw_ids(tmp_path, monkeypatch):
    runner, models = make_runner(tmp_path, monkeypatch,
                                 replies=[10, END], with_embeddings=False)

    tokens = runner.run(AUDIO)

    np.testing.assert_array_equal(tokens[0], [10, END])
    # Raw ids in the model's token input dtype, no embedding lookup.
    for call in models["decoder"].calls:
        assert np.asarray(call[0]).dtype == np.dtype("int32")
    assert [int(np.asarray(c[0])[0, 0]) for c in models["decoder"].calls] == [START, 10]


def test_run_sizes_audio_and_reports_metrics(tmp_path, monkeypatch):
    runner, models = make_runner(tmp_path, monkeypatch, replies=[END])

    assert runner.max_inp_len == ENC_LEN
    runner.run(np.zeros((1, 40), "float32"))
    assert models["encoder"].calls[-1][0].shape == (1, ENC_LEN)  # padded
    runner.run(np.ones(2 * ENC_LEN, "float32"))
    assert models["encoder"].calls[-1][0].shape == (1, ENC_LEN)  # truncated
    assert np.all(np.asarray(models["encoder"].calls[-1][0]) == 1)

    assert runner.last_infer_time > 0
    assert runner.time_to_first_token > 0
    assert runner.last_infer_time >= runner.time_to_first_token
