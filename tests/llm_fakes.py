# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Build a real ``utils.llm.LLM`` (or a demo binding) on the host.

The models are fakes that keep the real ``Decoder``/``KVCache`` code paths:
the KV cache holds the token id fed at each position, so tests can read the
cache contents back (``cached_tokens``), and the logits pick the next token
from a script. The tokenizer is character-level, with registered special
tokens matched first.
"""

import json
import os
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np

import utils.llm as llm_module

VOCAB = 4096
CHAR_BASE = 1000  # id of a plain character = CHAR_BASE + ord(char)


class CharTokenizer:
    """Character-level tokenizer; *specials* map whole strings to ids."""

    def __init__(self, specials: dict[str, int], bos_id: int | None = None):
        self.specials = dict(specials)
        self.by_id = {v: k for k, v in self.specials.items()}
        self.bos_id = bos_id  # prepended to every encode when set

    def encode(self, text: str):
        ids = [] if self.bos_id is None else [self.bos_id]
        i = 0
        while i < len(text):
            for s in sorted(self.specials, key=len, reverse=True):
                if text.startswith(s, i):
                    ids.append(self.specials[s])
                    i += len(s)
                    break
            else:
                ids.append(CHAR_BASE + ord(text[i]))
                i += 1
        return SimpleNamespace(ids=ids)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        out = []
        for i in ids:
            if i in self.by_id:
                if not skip_special_tokens:
                    out.append(self.by_id[i])
            elif i >= CHAR_BASE:
                out.append(chr(i - CHAR_BASE))
        return "".join(out)

    def token_to_id(self, s: str):
        return self.specials.get(s)

    def ids(self, text: str) -> list[int]:
        """Ids of *text* without the auto BOS (test convenience)."""
        ids = self.encode(text).ids
        return ids[1:] if self.bos_id is not None else ids


class _Info:
    def __init__(self, shape, dtype):
        self.shape = list(shape)
        self.dtype = dtype


class _Array:
    def __init__(self, a):
        self.a = np.array(a, copy=True)
        self.shape, self.dtype = self.a.shape, self.a.dtype
        self._buffer_view = SimpleNamespace(get_buffer=lambda: self)

    def to_host(self):
        return self.a


class _LazyLogits:
    """Logits that consult the script only when the decoder reads them, so
    steps that skip the logits (warm-up, prefill) do not consume replies."""

    def __init__(self, n: int, next_token: Callable[[], int]):
        self.shape = (1, n, VOCAB)
        self._next_token = next_token

    def to_host(self):
        logits = np.zeros(self.shape, np.float32)
        logits[0, -1, self._next_token()] = 1.0
        return logits


class _Device:
    def create_semaphore(self, value):
        return SimpleNamespace(wait=lambda v: None)

    def queue_execute(self, cb, waits, signals):
        pass


class FakeModel:
    """A decode (n=1) or batched prefill (n>1) body with a fused head.

    The cache input ``[1, 1, seq, 1]`` is returned with the fed token ids
    written at their positions; the logits of the last position are one-hot
    on ``next_fn(cache_tokens)`` (the tokens in the cache so far), evaluated
    only when read.
    """

    def __init__(self, path, n: int, seq: int, next_fn: Callable[[list[int]], int], out_n: int | None = None):
        self.path = path
        self.inputs_info = [_Info((1, n), "int64"), _Info((1, 1), "int64"), _Info((1, 1, seq, 1), "int64")]
        # A fused-head prefill export emits one logits row (the chunk's last
        # position), like the decode model: (1, 1, VOCAB).
        self.outputs_info = [_Info((1, out_n or n, VOCAB), "float32"), _Info((1, 1, seq, 1), "int64")]
        self.device = _Device()
        self.calls: list[tuple[list[int], int]] = []  # (tokens, start position)
        self._next_fn = next_fn

    def to_device(self, x):
        return _Array(x) if isinstance(x, np.ndarray) else x

    def allocate_device_array(self, x):
        return _Array(x)

    def infer(self, inputs):
        # Host buffers arrive as NumPy arrays, the cache as device arrays.
        host = [x.a if isinstance(x, _Array) else np.asarray(x) for x in inputs]
        tokens = [int(t) for t in host[0].reshape(-1)]
        pos = int(host[1].reshape(-1)[0])
        self.calls.append((tokens, pos))
        cache = host[-1].copy()
        cache[0, 0, pos:pos + len(tokens), 0] = tokens
        cached = cache[0, 0, :pos + len(tokens), 0].tolist()
        return [_LazyLogits(len(tokens), lambda: self._next_fn(cached)), _Array(cache)]


def scripted(replies: list[int], then: int) -> Callable[[list[int]], int]:
    """next_fn yielding *replies* in order, then *then* forever."""
    it = iter(replies)
    return lambda cached: next(it, then)


def make_llm(
    tmp_path,
    monkeypatch,
    cls=llm_module.LLM,
    *,
    tokenizer: CharTokenizer,
    config: dict,
    seq: int = 64,
    next_fn: Callable[[list[int]], int] | None = None,
    prefill_size: int | None = None,
    **kwargs,
):
    """Construct *cls* over fake models in *tmp_path*; returns (llm, models).

    ``models["body"]`` / ``models["prefill"]`` are the FakeModels, so tests
    can inspect their ``calls``. ``cached_tokens(llm)`` reads the cache back.
    """
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "transformer.vmfb").write_bytes(b"")
    if prefill_size:
        (tmp_path / "transformer_prefill.vmfb").write_bytes(b"")
    next_fn = next_fn or scripted([], then=config["eos_token_id"] if isinstance(config["eos_token_id"], int) else config["eos_token_id"][0])
    models = {}

    def fake_model(path, runtime, *, function="main", device_outputs=False):
        # Classify by file name: the tmp_path directory carries the test name
        # and can itself contain "prefill".
        name = "prefill" if "prefill" in os.path.basename(str(path)) else "body"
        models[name] = FakeModel(path, prefill_size if name == "prefill" else 1, seq, next_fn,
                                 out_n=1 if name == "prefill" else None)
        return models[name]

    monkeypatch.setattr(llm_module, "Model", fake_model)
    monkeypatch.setattr(llm_module, "Tokenizer", SimpleNamespace(from_file=lambda path: tokenizer))
    kwargs.setdefault("disable_lm_head", True)
    kwargs.setdefault("disable_prefill", not prefill_size)
    llm = cls(str(tmp_path / "transformer.vmfb"), runtime=llm_module.RuntimeConfig(tda="cpu", device_io=False), **kwargs)
    return llm, models


def cached_tokens(llm) -> list[int]:
    """The token ids in the KV cache up to the current turn position."""
    return llm._cache.inputs[0].to_host()[0, 0, :llm._turn_pos, 0].tolist()
