# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Device-resident KV caches shared by models with the same cache layout.

``KVCache`` owns the trailing cache inputs of one or more models (a decode
body plus its batched prefill body). It zero-initialises each slot from the
model's input metadata and hands out a mutable ``inputs`` list, which callers
append after the model's own inputs; updated slots are swapped in place
(:meth:`KVCache.update`), so every model sharing the cache sees the very same
on-device buffers.

Two kinds of slots:

* updated slots — the model returns them as outputs (``[n_outputs:]``); each
  output either replaces its slot's buffer or, for read-only KV cache models,
  holds only the step's new rows, written back at the step position in one
  HAL command-buffer submission;
* fixed slots (``fixed=``) — external state such as encoder-decoder
  cross-attention KV: staged via :meth:`KVCache.set_fixed`, untouched by
  :meth:`KVCache.update` and :meth:`KVCache.reset`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from iree.runtime import HalCommandBuffer

from utils.runtime import Model, to_host


@dataclass(frozen=True)
class Layout:
    """How one model's outputs map onto a shared KVCache (``KVCache.attach``).

    ``rows_axis`` has one entry per updated slot, in slot order: ``None``
    means the output replaces the slot's buffer; an int means the output
    holds only the new rows, written at the step position.
    """

    rows_axis: tuple[int | None, ...]


def _rows_axis(out_shape: Sequence, cache_shape: Sequence) -> int | None:
    """Axis along which *out_shape* is shorter than *cache_shape* (the
    output's new rows), or None when the output is the whole cache."""
    return next((a for a, (o, c) in enumerate(zip(out_shape, cache_shape)) if o < c), None)


class KVCache:
    """The trailing cache inputs of one or more models that share a layout.

    *model* defines the layout: the trailing cache inputs are the last
    ``len(outputs_info) - n_outputs + n_fixed`` inputs, where outputs
    ``[n_outputs:]`` are the updated caches in input order of the non-fixed
    slots. ``fixed[i]`` marks cache-input slot *i* as fixed (encoder-decoder
    cross-attention; moonshine passes ``[i % 4 >= 2 for i in range(4 *
    n_layers)]``). Slots are zero-initialised on device from the input
    metadata.
    """

    def __init__(
        self,
        model: Model,
        *,
        n_outputs: int = 1,
        fixed: Sequence[bool] | None = None,
    ):
        if model.inputs_info is None or model.outputs_info is None:
            raise ValueError(
                f"Model '{model.path}' is missing input/output metadata "
                "required for KV cache management."
            )
        self._model = model
        self._device = model.device
        self._n_outputs = n_outputs

        n_updated = len(model.outputs_info) - n_outputs
        if fixed is None:
            fixed = [False] * n_updated
        else:
            fixed = list(fixed)
        if n_updated < 0 or sum(1 for f in fixed if not f) != n_updated:
            raise ValueError(
                f"Model '{model.path}' updates {n_updated} cache slots, but "
                f"the fixed pattern marks {sum(1 for f in fixed if not f)} "
                "non-fixed slots."
            )
        self._fixed = fixed
        self._n_fixed = sum(fixed)
        self._updated_slots = [i for i, f in enumerate(fixed) if not f]

        in_info = model.inputs_info
        n_slots = len(fixed)
        if len(in_info) < n_slots:
            raise ValueError(
                f"Model '{model.path}' has {len(in_info)} inputs, but the "
                f"cache layout needs {n_slots} trailing cache inputs."
            )
        cache_info = in_info[len(in_info) - n_slots:]
        self._init = [np.zeros(info.shape, dtype=np.dtype(info.dtype)) for info in cache_info]
        # Caches stay device-resident even in host I/O mode (to_device is a
        # no-op there): row writes need device buffers, and sharing is by buffer.
        self.inputs = [model.allocate_device_array(z) for z in self._init]

        rows_axis = self._slot_rows_axis(model)
        self._seq_len = self._query_seq_len(rows_axis)
        if any(axis is not None for axis in rows_axis):
            # Semaphore + counter for the read-only row writes (one submission
            # per update, chained on the previous one).
            self._copy_semaphore = self._device.create_semaphore(0)
            self._copies = 0

    def _query_seq_len(self, rows_axis: Sequence[int | None]) -> int | None:
        """Cache length: the rows axis of a read-only cache, else axis 2 of
        the first rank-4 cache input."""
        seq_len = next(
            (self._init[slot].shape[axis]
             for slot, axis in zip(self._updated_slots, rows_axis) if axis is not None),
            None,
        )
        if seq_len is None:
            seq_len = next((c.shape[2] for c in self._init if len(c.shape) == 4), None)
        return seq_len

    def _slot_rows_axis(self, model: Model) -> tuple[int | None, ...]:
        """Per updated slot, the rows axis of *model*'s cache outputs."""
        return tuple(
            _rows_axis(model.outputs_info[self._n_outputs + j].shape, self._init[slot].shape)
            for j, slot in enumerate(self._updated_slots)
        )

    @property
    def seq_len(self) -> int | None:
        """Occupied-cache length along the sequence axis, when it can be
        determined from the cache inputs."""
        return self._seq_len

    def attach(self, model: Model) -> Layout:
        """Map *model*'s outputs onto this cache; returns a :class:`Layout`.

        A batched prefill model has the same per-layer cache layout as the
        decode model, so both run against one set of on-device cache
        buffers: a prefill chunk extends the context in place and the next
        decode step continues from the very same buffers with no host
        round-trip between them.
        """
        if model.inputs_info is None or model.outputs_info is None:
            raise ValueError(
                f"Model '{model.path}' is missing input/output metadata "
                "required for KV cache management."
            )
        n_slots = len(model.outputs_info) - self._n_outputs + self._n_fixed
        if n_slots != len(self._fixed):
            raise ValueError(
                f"Cannot share KV cache: model has {n_slots} cache tensors, "
                f"owner has {len(self._fixed)}."
            )
        trailing = model.inputs_info[len(model.inputs_info) - n_slots:]
        for mine, theirs in zip(trailing, self._init):
            if tuple(mine.shape) != tuple(theirs.shape) or np.dtype(mine.dtype) != theirs.dtype:
                raise ValueError(
                    "Cannot share KV cache: tensor mismatch "
                    f"({tuple(mine.shape)}/{mine.dtype} vs "
                    f"{tuple(theirs.shape)}/{theirs.dtype})."
                )
        return Layout(rows_axis=self._slot_rows_axis(model))

    def update(self, layout: Layout, outputs: Sequence, position: int) -> None:
        """Fold a model call's *outputs* into the cache using *layout*.

        Replaced slots take the output buffer; rows slots copy their new rows
        at *position* in one command-buffer submission.
        """
        rows = []
        for j, slot in enumerate(self._updated_slots):
            out = outputs[self._n_outputs + j]
            axis = layout.rows_axis[j]
            if axis is None:
                self.inputs[slot] = out
            else:
                rows.append((slot, axis, out))
        if rows:
            self._write_rows(rows, position)

    def _write_rows(self, rows: list[tuple[int, int, object]], position: int) -> None:
        """Copy each rows slot's new rows into it at *position*, in one submission."""
        cb = HalCommandBuffer(self._device)
        for slot, axis, out in rows:
            cache = self.inputs[slot]
            n, length = out.shape[axis], cache.shape[axis]
            inner = int(np.prod(cache.shape[axis + 1:])) * np.dtype(cache.dtype).itemsize
            src, dst = out._buffer_view.get_buffer(), cache._buffer_view.get_buffer()
            # One contiguous block per index of the leading axes (one for a time-major cache).
            for b in range(int(np.prod(cache.shape[:axis]))):
                cb.copy(src, dst, b * n * inner, (b * length + position) * inner, n * inner)
        cb.end()
        self._copies += 1
        self._device.queue_execute(
            cb, [(self._copy_semaphore, self._copies - 1)], [(self._copy_semaphore, self._copies)]
        )
        self._copy_semaphore.wait(self._copies)

    def reset(self) -> None:
        """Zero the updated slots (in place); fixed slots are untouched."""
        for slot in self._updated_slots:
            self.inputs[slot] = self._model.allocate_device_array(self._init[slot])

    def snapshot(self) -> list[np.ndarray]:
        """Host copies of every slot, in slot order."""
        return [to_host(kv).copy() for kv in self.inputs]

    def restore(self, state: Sequence[np.ndarray]) -> None:
        """Restore every slot from a :meth:`snapshot`."""
        for i, arr in enumerate(state):
            self.inputs[i] = self._model.allocate_device_array(arr)

    def set_fixed(self, tensors: Sequence) -> None:
        """Stage the fixed slots from encoder outputs (cross-attention)."""
        if len(tensors) != self._n_fixed:
            raise ValueError(
                f"Expected {self._n_fixed} fixed cache tensors, got {len(tensors)}."
            )
        for slot, c in zip((i for i, f in enumerate(self._fixed) if f), tensors):
            # Encoder outputs may already be device arrays; upload only host arrays.
            if isinstance(c, np.ndarray):
                c = self._model.allocate_device_array(c)
            self.inputs[slot] = c
