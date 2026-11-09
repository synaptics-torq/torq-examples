# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Tests for utils.cache.KVCache: zero-init, attach/update (whole-cache
replace and read-only rows), fixed slots, snapshot/restore, and cache
sharing between a decode model and its batched prefill model."""

from types import SimpleNamespace

import numpy as np
import pytest

from utils import cache as cache_module
from utils.cache import KVCache


F32 = "float32"
I64 = "int64"


class _Info:
    def __init__(self, shape, dtype=F32):
        self.shape = list(shape)
        self.dtype = dtype


def info(shape, dtype=F32):
    return _Info(shape, dtype)


class _Array:
    """Device array stand-in; its own buffer for the copies."""

    def __init__(self, a):
        self.a = np.array(a, copy=True)
        self.shape, self.dtype = self.a.shape, np.dtype(self.a.dtype)
        self._buffer_view = SimpleNamespace(get_buffer=lambda: self)

    def to_host(self):
        return self.a

    def __array__(self, dtype=None):
        return self.a


class _CommandBuffer:
    def __init__(self, device):
        pass

    def copy(self, src, dst, src_offset, dst_offset, length):
        dst.a.reshape(-1).view(np.uint8)[dst_offset:dst_offset + length] = (
            src.a.reshape(-1).view(np.uint8)[src_offset:src_offset + length]
        )

    def end(self):
        pass


class _Device:
    def __init__(self):
        self.waits = []
        self.executes = 0
        self.semaphore = SimpleNamespace(wait=lambda v: self.waits.append(v))

    def create_semaphore(self, value):
        return self.semaphore

    def queue_execute(self, cb, waits, signals):
        self.executes += 1


class _Model:
    """Fake utils.runtime.Model: metadata, device staging, recorded calls."""

    def __init__(self, path, inputs, outputs, device=None, result_fn=None):
        self.path = path
        self.inputs_info = inputs
        self.outputs_info = outputs
        self.device = device or _Device()
        self.calls = []
        self._result_fn = result_fn

    def to_device(self, x):
        return _Array(x) if isinstance(x, np.ndarray) else x

    def allocate_device_array(self, x):
        return _Array(x)

    def infer(self, inputs):
        staged = [self.to_device(x) for x in inputs]
        self.calls.append(staged)
        if self._result_fn is not None:
            return self._result_fn(staged)
        return [
            _Array(np.zeros(o.shape, np.dtype(o.dtype))) for o in self.outputs_info
        ]


def _decode_model(seq=4):
    """A single-token decode body: (id, position, [2kv, seq, D] cache)."""
    return _Model(
        "transformer.vmfb",
        [info((1, 1), I64), info((1, 1), I64), info((1, 2, seq, 4))],
        [info((1, 1, 8)), info((1, 2, seq, 4))],
    )


# ── construction ──────────────────────────────────────────────────────────────


def test_zero_init_from_input_metadata():
    model = _decode_model(seq=3)
    cache = KVCache(model)
    assert len(cache.inputs) == 1
    kv = cache.inputs[0]
    assert isinstance(kv, _Array)
    assert kv.a.shape == (1, 2, 3, 4)
    assert kv.a.dtype == np.dtype(F32)
    np.testing.assert_array_equal(kv.a, 0)
    # No rows cache: seq_len is axis 2 of the first rank-4 cache input.
    assert cache.seq_len == 3


def test_missing_metadata_rejected():
    model = _Model("m.vmfb", None, None)
    with pytest.raises(ValueError, match="missing input/output metadata"):
        KVCache(model)


def test_fixed_pattern_must_cover_the_updated_outputs():
    model = _decode_model()
    # One updated output but a pattern with no non-fixed slots to receive it.
    with pytest.raises(ValueError, match="non-fixed slots"):
        KVCache(model, fixed=[True, True])


# ── attach / update ───────────────────────────────────────────────────────────


def test_attach_and_replace_update():
    model = _decode_model()
    cache = KVCache(model)
    layout = cache.attach(model)
    assert layout.rows_axis == (None,)
    new = _Array(np.full((1, 2, 4, 4), 7.0, F32))
    cache.update(layout, [_Array(np.zeros((1, 1, 8))), new], 3)
    assert cache.inputs[0] is new


def test_read_only_row_writes_at_the_step_position(monkeypatch):
    monkeypatch.setattr(cache_module, "HalCommandBuffer", _CommandBuffer)
    # Time-major [1, L, 2kv, D] and head-major [1, 2kv, L, D] row caches,
    # and a whole-cache state (ported from the old row-write test).
    caches = [info([1, 4, 2, 3]), info([1, 2, 4, 3]), info([1, 5])]
    rows = [info([1, 1, 2, 3]), info([1, 2, 1, 3]), info([1, 5])]
    model = _Model("m.vmfb", [info([1, 1]), info([1, 1]), *caches], [info([1]), *rows])
    cache = KVCache(model)
    assert cache.seq_len == 4
    layout = cache.attach(model)
    for pos in range(3):
        out = [
            _Array(np.zeros((1,), F32))
        ] + [_Array(np.full(r.shape, pos + 1, F32)) for r in rows]
        cache.update(layout, out, pos)
    time_major, head_major, state = (kv.a for kv in cache.inputs)
    np.testing.assert_array_equal(time_major[0, :, 0, 0], [1, 2, 3, 0])
    np.testing.assert_array_equal(head_major[0, 1, :, 2], [1, 2, 3, 0])
    np.testing.assert_array_equal(state, np.full((1, 5), 3))
    # One command-buffer submission per row update, chained on the semaphore.
    assert model.device.executes == 3
    assert model.device.waits == [1, 2, 3]


# ── fixed slots ───────────────────────────────────────────────────────────────


def test_fixed_slots_unmoved_by_update_and_reset():
    # Moonshine, one layer: (self_v, self_k, cross_v, cross_k) per layer.
    fixed = [i % 4 >= 2 for i in range(4)]
    model = _Model(
        "m.vmfb",
        [
            info((1, 1, 8)), info((1, 1), I64),
            info((1, 1, 2, 4)), info((1, 1, 2, 4)),
            info((1, 8, 2, 4)), info((1, 8, 2, 4)),
        ],
        [info((1, 1, 16)), info((1, 1, 2, 4)), info((1, 1, 2, 4))],
    )
    cache = KVCache(model, fixed=fixed)
    layout = cache.attach(model)
    assert layout.rows_axis == (None, None)

    cross_v = _Array(np.full((1, 8, 2, 4), 5.0, F32))
    cross_k = _Array(np.full((1, 8, 2, 4), 6.0, F32))
    cache.set_fixed([cross_v, cross_k])
    assert cache.inputs[2] is cross_v and cache.inputs[3] is cross_k

    new_v = _Array(np.full((1, 1, 2, 4), 1.0, F32))
    new_k = _Array(np.full((1, 1, 2, 4), 2.0, F32))
    cache.update(layout, [_Array(np.zeros((1, 1, 16))), new_v, new_k], 0)
    assert cache.inputs[0] is new_v and cache.inputs[1] is new_k
    # The fixed slots are untouched by the update.
    assert cache.inputs[2] is cross_v and cache.inputs[3] is cross_k

    cache.reset()
    np.testing.assert_array_equal(cache.inputs[0].a, 0)
    np.testing.assert_array_equal(cache.inputs[1].a, 0)
    assert cache.inputs[2] is cross_v and cache.inputs[3] is cross_k

    with pytest.raises(ValueError, match="Expected 2 fixed cache tensors"):
        cache.set_fixed([cross_v])


def test_host_io_mode_keeps_the_cache_on_device(monkeypatch):
    # --no-device-io: Model.to_device is a no-op, but caches must stay
    # device-resident (row writes need device buffers) and sharing is by buffer.
    model = _Model(
        "m.vmfb",
        [
            info((1, 1), I64), info((1, 1), I64),
            info((1, 2, 4, 4)), info((1, 8, 2, 4)),
        ],
        [info((1, 1, 16)), info((1, 2, 1, 4))],
    )
    model.to_device = lambda x: x
    monkeypatch.setattr(cache_module, "HalCommandBuffer", _CommandBuffer)
    cache = KVCache(model, fixed=[False, True])
    assert all(isinstance(kv, _Array) for kv in cache.inputs)

    cache.restore(cache.snapshot())
    cache.reset()
    assert all(isinstance(kv, _Array) for kv in cache.inputs)

    cache.set_fixed([np.full((1, 8, 2, 4), 5.0, F32)])  # host array: uploaded
    assert isinstance(cache.inputs[1], _Array)
    device_fixed = _Array(np.zeros((1, 8, 2, 4), F32))
    cache.set_fixed([device_fixed])  # device array: passed through
    assert cache.inputs[1] is device_fixed

    layout = cache.attach(model)
    assert layout.rows_axis == (2,)
    cache.update(
        layout,
        [_Array(np.zeros((1, 1, 16))), _Array(np.full((1, 2, 1, 4), 7.0, F32))],
        3,
    )
    np.testing.assert_array_equal(cache.inputs[0].a[:, :, 3, :], 7.0)
    assert all(isinstance(kv, _Array) for kv in cache.inputs)


# ── snapshot / restore ────────────────────────────────────────────────────────


def test_snapshot_and_restore_round_trip():
    model = _decode_model()
    cache = KVCache(model)
    layout = cache.attach(model)
    one = _Array(np.ones((1, 2, 4, 4), F32))
    cache.update(layout, [_Array(np.zeros((1, 1, 8))), one], 0)
    state = cache.snapshot()
    assert len(state) == 1 and state[0].shape == (1, 2, 4, 4)
    np.testing.assert_array_equal(state[0], 1)

    two = _Array(np.full((1, 2, 4, 4), 2.0, F32))
    cache.update(layout, [_Array(np.zeros((1, 1, 8))), two], 0)
    cache.restore(state)
    assert cache.inputs[0] is not two
    np.testing.assert_array_equal(cache.inputs[0].a, 1)


# ── sharing between models ────────────────────────────────────────────────────


def test_prefill_shares_the_decode_cache():
    decode = _decode_model()
    prefill = _Model(
        "transformer_prefill.vmfb",
        [info((1, 4), I64), info((1, 1), I64), info((1, 2, 4, 4))],
        [info((1, 1, 4)), info((1, 2, 4, 4))],
    )
    cache = KVCache(decode)
    p_layout = cache.attach(prefill)
    d_layout = cache.attach(decode)
    assert p_layout.rows_axis == (None,) and d_layout.rows_axis == (None,)

    # A chunk through the prefill layout extends the same buffers the decode
    # layout will read on the next step: sharing is one object.
    new = _Array(np.full((1, 2, 4, 4), 9.0, F32))
    cache.update(p_layout, [_Array(np.zeros((1, 1, 4))), new], 0)
    assert cache.inputs[0] is new
    # The decode layout reads and updates the very same buffer list.
    cache.update(d_layout, [_Array(np.zeros((1, 1, 8))), _Array(np.zeros((1, 2, 4, 4), F32))], 4)


def test_attach_shape_mismatch_rejected():
    decode = _decode_model()
    prefill = _Model(
        "transformer_prefill.vmfb",
        [info((1, 4), I64), info((1, 1), I64), info((1, 2, 4, 8))],
        [info((1, 1, 4)), info((1, 2, 4, 8))],
    )
    with pytest.raises(ValueError, match="tensor mismatch"):
        KVCache(decode).attach(prefill)


def test_attach_slot_count_mismatch_rejected():
    decode = _decode_model()
    other = _Model(
        "other.vmfb",
        [info((1, 1), I64), info((1, 1), I64), info((1, 2, 4, 4)), info((1, 2, 4, 4))],
        [info((1, 1, 8)), info((1, 2, 4, 4)), info((1, 2, 4, 4))],
    )
    with pytest.raises(ValueError, match="Cannot share KV cache"):
        KVCache(decode).attach(other)


# ── seq_len ───────────────────────────────────────────────────────────────────


def test_seq_len_from_rows_cache_and_from_rank4_cache():
    # Read-only rows cache: the rows axis length wins.
    rows_model = _Model(
        "m.vmfb", [info([1, 1]), info([1, 1]), info([1, 4, 2, 3])],
        [info([1]), info([1, 1, 2, 3])],
    )
    assert KVCache(rows_model).seq_len == 4
    # No rows: axis 2 of the first rank-4 cache input.
    assert KVCache(_decode_model(seq=5)).seq_len == 5
    # No rank-4 cache input at all (conv-state-only caches): unknown.
    conv_model = _Model(
        "m.vmfb", [info([1, 1]), info([1, 1]), info([1, 4, 3])],
        [info([1]), info([1, 4, 3])],
    )
    assert KVCache(conv_model).seq_len is None
