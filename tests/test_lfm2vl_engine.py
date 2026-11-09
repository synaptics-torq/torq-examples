# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""LFM2-VL engine: the VL-specific cache fold and image splice, run over a
real ``KVCache``/``Decoder`` with fake models (the engine's constructor,
warm-up and cache wiring are exercised for real)."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HIDDEN = 4
VOCAB = 8
SEQ = 128
N_HEADS = 2
HD = 2
N_CACHES = 15  # the W8 decoder's 15 cache inputs (conv states + 6 KV stacks)

_ENGINE = (
    Path(__file__).resolve().parents[1]
    / "LiquidAI" / "LiquidAI-LFM2-VL-450M" / "src" / "engine.py"
)
_spec = importlib.util.spec_from_file_location("lfm2vl_engine", _ENGINE)
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)
ATTN = engine.ATTN_LAYERS


class _Arr:
    """Device array stand-in."""

    def __init__(self, a):
        self.a = np.array(a, copy=True)
        self.shape, self.dtype = self.a.shape, np.dtype(self.a.dtype)

    def to_host(self):
        return self.a

    def __array__(self, dtype=None):
        return self.a


class _FakeModel:
    """Fake utils.runtime.Model: metadata, device staging, recorded calls."""

    def __init__(self, path, inputs, outputs, result_fn=None):
        self.path = str(path)
        self.inputs_info = [SimpleNamespace(shape=list(s), dtype=d) for s, d in inputs]
        self.outputs_info = [SimpleNamespace(shape=list(s), dtype=d) for s, d in outputs]
        self._specs = outputs
        self.device = SimpleNamespace()
        self.calls = []
        self._result_fn = result_fn

    def allocate_device_array(self, x):
        return _Arr(x)

    def infer(self, inputs):
        self.calls.append(list(inputs))
        if self._result_fn is not None:
            return self._result_fn(list(inputs))
        return [_Arr(np.zeros(s, np.dtype(d))) for s, d in self._specs]


def _cache_specs():
    """The W8 decoder's cache layout: attention KV at ATTN layers, conv elsewhere."""
    specs = []
    for i in range(N_CACHES):
        if i in ATTN:
            specs.append(((1, N_HEADS, SEQ, HD), "float32"))
        else:
            specs.append(((1, 2, 3), "float32"))
    return specs


def make_engine(tmp_path, monkeypatch):
    specs = _cache_specs()
    body = _FakeModel(
        "decoder_nolm_W8.vmfb",
        [((1, 1, HIDDEN), "float32"), ((1, 1), "int64"), *specs],
        [((1, 1, HIDDEN), "float32"), *specs],
    )
    head = _FakeModel(
        "lm_head_W8.vmfb", [((1, 1, HIDDEN), "float32")], [((1, 1, VOCAB), "float32")]
    )
    vision = _FakeModel(
        "vision_encoder_256_W8.vmfb", [((1, 256, 768), "float32")], [((1, 64, HIDDEN), "float32")]
    )
    parts = [
        _FakeModel(
            f"decoder_image_2part_{c}_W8.vmfb",
            [((1, 64, 768), "float32")], [((1, 64, 768), "float32")],
        )
        for c in "AB"
    ]
    models = {
        "decoder_nolm_W8.vmfb": body,
        "lm_head_W8.vmfb": head,
        "vision_encoder_256_W8.vmfb": vision,
        "decoder_image_2part_A_W8.vmfb": parts[0],
        "decoder_image_2part_B_W8.vmfb": parts[1],
    }

    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 7}))
    np.save(tmp_path / "token_embeddings.npy", np.zeros((VOCAB, HIDDEN), np.float32))
    for name in ("lm_head_W8.vmfb", "decoder_image_2part_A_W8.vmfb",
                 "decoder_image_2part_B_W8.vmfb"):
        (tmp_path / name).touch()

    monkeypatch.setattr(
        engine, "Model",
        lambda path, runtime, **kw: models[os.path.basename(str(path))])
    monkeypatch.setattr(engine, "Tokenizer", SimpleNamespace(from_file=lambda p: object()))
    return engine.LFM2VLEngine(str(tmp_path))


def _image_caches():
    """The flat per-layer caches the 2-part image decoder emits: a key/val pair
    (64 image rows) per attention layer, one conv state per other layer."""
    flat = []
    for i in range(N_CACHES):
        if i in ATTN:
            flat.append(np.full((1, N_HEADS // 2, 64, HD), 1.0))  # key
            flat.append(np.full((1, N_HEADS // 2, 64, HD), 2.0))  # val
        else:
            flat.append(np.full((1, 2, 3), 3.0))
    return flat


def test_fold_caches_stacks_key_and_value(tmp_path, monkeypatch):
    eng = make_engine(tmp_path, monkeypatch)
    folded = eng._fold_caches(_image_caches())

    assert len(folded) == N_CACHES
    for i in range(N_CACHES):
        if i in ATTN:
            kv = folded[i]
            assert kv.shape == (1, N_HEADS, SEQ, HD)
            np.testing.assert_array_equal(kv[:, 0, :64], 1.0)  # key -> first half
            np.testing.assert_array_equal(kv[:, 1, :64], 2.0)  # val -> second half
            np.testing.assert_array_equal(kv[:, :, 64:], 0.0)
        else:
            np.testing.assert_array_equal(folded[i], 3.0)       # conv passthrough


def test_fold_caches_rejects_bad_count(tmp_path, monkeypatch):
    eng = make_engine(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="consumed"):
        eng._fold_caches(_image_caches() + [np.zeros((1, 2, 3))])


def test_merge_image_caches_splices_and_preserves_prefix(tmp_path, monkeypatch):
    eng = make_engine(tmp_path, monkeypatch)
    seed = eng._fold_caches(_image_caches())
    s = 4

    # Prefix state: fill positions [0:s] of every KV cache with 9.0.
    state = eng._cache.snapshot()
    for i, (shape, _dt) in enumerate(eng._cache_specs):
        if len(shape) == 4:
            state[i][:, :, :s, :] = 9.0
    eng._cache.restore(state)

    eng._merge_image_caches(seed, s)
    out = eng._cache.snapshot()

    for i, (shape, _dt) in enumerate(eng._cache_specs):
        if len(shape) == 4:
            np.testing.assert_array_equal(out[i][:, :, :s], 9.0)
            np.testing.assert_array_equal(out[i][:, :, s:s + 64], seed[i][:, :, :64])
            np.testing.assert_array_equal(out[i][:, :, s + 64:], 0.0)
        else:
            np.testing.assert_array_equal(out[i], seed[i])  # conv replaced wholesale
