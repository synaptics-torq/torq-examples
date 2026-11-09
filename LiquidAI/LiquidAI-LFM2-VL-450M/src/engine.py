# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""LFM2-VL-450M W8 engine: all-NPU inference with dmabuf single-copy weights.

Board-proven design (SL2610/SL2619, 2 GB boards, ~1.2 GB peak with all models
co-resident):

- every NPU input is a *device array* (`allocate_device_array`) — required when
  the dmabuf allocator is in use; host numpy inputs make the runtime write XRAM
  binding ranges that dmabuf mode excludes from the XRAM window (crash).
- vision encoder + image-prefill parts + decoder + lm_head all stay resident;
  nothing is freed between images or questions.
- the 64 image tokens run once through the 2-part one-shot prefill; their KV
  caches are spliced into the decoder's caches, so the text decoder only ever
  steps text tokens.
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Final, Iterator

import ml_dtypes
import numpy as np
from PIL import Image
from tokenizers import Tokenizer

from utils.cache import KVCache
from utils.decoder import Decoder, generate, sample
from utils.runtime import Model, RuntimeConfig, load_lut

PATCH: Final[int] = 16
MEAN: Final[float] = 0.5
STD: Final[float] = 0.5
# decoder layers with attention (KV) caches; the rest keep conv states only
ATTN_LAYERS: Final[frozenset] = frozenset({2, 5, 8, 10, 12, 14})
_PREFIX: Final[str] = "<|startoftext|>\nuser\n<|image_start|>"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _center_crop_resize_bgr(bgr: np.ndarray, size: int = 256) -> np.ndarray:
    """Any HxWx3 BGR uint8 frame -> size x size RGB float32 [0, 255].

    Center-crop to square then bilinear resize: the static vision encoder needs
    exactly 256 patches (16x16 of a 256x256 image), and any aspect ratio works.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR array, got {bgr.shape}")
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    img = bgr[y0:y0 + s, x0:x0 + s][:, :, ::-1]  # BGR -> RGB
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    return np.asarray(img).astype(np.float32)


def _to_pixel_values(rgb256: np.ndarray) -> np.ndarray:
    """256x256x3 RGB [0,255] -> normalized patchified pixel_values [1, 256, 768] f32."""
    chw = rgb256.transpose(2, 0, 1)
    chw = (chw / 255.0 - MEAN) / STD
    c, h, w = chw.shape
    nph, npw = h // PATCH, w // PATCH
    x = chw.reshape(c, nph, PATCH, npw, PATCH)
    x = x.transpose(1, 3, 2, 4, 0)
    x = x.reshape(nph * npw, PATCH * PATCH * c)
    return x[None, ...]


def _bf16_device(runner: Model, arr: np.ndarray):
    """Host ndarray -> contiguous device array in the runner's IO (bf16) dtype."""
    io_dtype = np.dtype(runner.inputs_info[0].dtype)
    return runner.allocate_device_array(np.ascontiguousarray(arr.astype(io_dtype)))


def _to_host(arr) -> np.ndarray:
    """Device array (or host array) -> host ndarray, bf16 if the payload says so."""
    if hasattr(arr, "to_host"):
        arr = arr.to_host()
    a = np.ascontiguousarray(np.asarray(arr))
    return a.view(ml_dtypes.bfloat16) if a.dtype.kind == "V" else a


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

class LFM2VLEngine:
    """All-NPU LFM2-VL-450M (W8): load once, encode many images, answer many
    questions per image. Thread-unsafe by design (single NPU, single flow)."""

    def __init__(
        self,
        model_dir: str | os.PathLike,
        *,
        decoder: str | os.PathLike | None = None,
        vision: str | os.PathLike | None = None,
        lm_head: str | os.PathLike | None = None,
        image_decoder_prefix: str | os.PathLike | None = None,
        runtime: RuntimeConfig | None = None,
        max_new: int = 128,
        max_seq_len: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 64,
    ):
        self._model_dir = os.fspath(model_dir)
        with open(os.path.join(self._model_dir, "config.json")) as f:
            self._eos_token_id = int(json.load(f).get("eos_token_id", 7))
        self._tokenizer = Tokenizer.from_file(
            os.path.join(self._model_dir, "tokenizer.json"))
        self._embed_lut = load_lut(os.path.join(self._model_dir, "token_embeddings.npy"))

        self._runtime = runtime or RuntimeConfig()
        self._max_new = max_new
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k

        # explicit paths win; otherwise the conventional _W8 names in model_dir
        d = self._model_dir
        self._decoder_path = os.fspath(decoder or os.path.join(d, "decoder_nolm_W8.vmfb"))
        self._vision_path = os.fspath(vision or os.path.join(d, "vision_encoder_256_W8.vmfb"))
        self._lmhead_path = os.fspath(lm_head) if lm_head else os.path.join(d, "lm_head_W8.vmfb")
        prefix = image_decoder_prefix or os.path.join(d, "decoder_image_2part_")
        part_files = sorted(glob.glob(f"{os.fspath(prefix)}*_W8.vmfb"))
        if not part_files:
            raise FileNotFoundError(
                f"no image-prefill vmfbs match {os.fspath(prefix)}*_W8.vmfb")
        self._part_paths = part_files

        # --- load everything (co-resident; W8 + dmabuf fits a 2 GB board) ---
        self._vis = self._mk(self._vision_path, function="embed_images")
        self._parts = [self._mk(p, function="main") for p in self._part_paths]
        self._model = self._mk(self._decoder_path, function="main")
        if os.path.exists(self._lmhead_path):
            self._lmhead = self._mk(self._lmhead_path, function="main")
        else:
            self._lmhead = None
            raise FileNotFoundError(f"lm_head vmfb not found: {self._lmhead_path}")

        # The text decoder runs on the shared core: the body steps against one
        # device-resident KV cache, the split head runs on-device, and the
        # embedding LUT feeds the token input.
        self._cache = KVCache(self._model)
        self._decoder = Decoder(
            self._model, self._cache, head=self._lmhead, embed_lut=self._embed_lut)

        # KV length straight from the decoder's static cache shapes
        msl = max_seq_len
        if msl is None:
            msl = self._decoder.seq_len
        if msl is None:
            raise ValueError("cannot determine max_seq_len; pass it explicitly")
        self._max_seq_len = msl

        self._cache_specs = [(tuple(t.shape), np.dtype(t.dtype)) for t in self._model.inputs_info[2:]]

        self._base_caches: list[np.ndarray] | None = None
        self._base_pos = 0

        # stats (updated by encode_image / infer_stream)
        self._vision_ns = 0
        self._parts_ns = 0
        self._prefix_ns = 0
        self._first_token_ns = 0
        self._last_infer_ns = 0
        self._n_tokens_gen = 0
        self._prefill_tokens = 0

        self._warmup()

    def _mk(self, path: str, function: str) -> Model:
        return Model(path, self._runtime, function=function, device_outputs=True)

    def _warmup(self) -> None:
        """One throwaway step so first-use device staging is off the real timing."""
        self._decoder.step(0, 0)
        self._cache.reset()

    # ---- image ----

    def _fold_caches(self, all_caches: list[np.ndarray]) -> list[np.ndarray]:
        """Fold the parts' per-layer caches (conv0,conv1,key2,val2,...,key14,val14,
        conv15) into the decoder's 16 input-order caches: attention layers combine
        their K+V pair into one stack (key -> heads 0:H/2, value -> heads H/2:H);
        conv layers pass through."""
        seq = self._max_seq_len
        attn_layers = {i for i, (sh, _) in enumerate(self._cache_specs) if len(sh) == 4}
        assert attn_layers == ATTN_LAYERS, f"decoder KV layout changed: {attn_layers}"
        first_kv = next(sh for sh, _ in self._cache_specs if len(sh) == 4)
        H, _, hd = first_kv[1], first_kv[2], first_kv[3]
        h2 = H // 2

        def kv(key: np.ndarray, val: np.ndarray) -> np.ndarray:
            if key.shape != (1, h2, 64, hd) or val.shape != (1, h2, 64, hd):
                raise ValueError(f"image KV shape {key.shape} != expected (1,{h2},64,{hd})")
            c = np.zeros((1, H, seq, hd), ml_dtypes.bfloat16)
            c[:, 0:h2, 0:64, :] = key
            c[:, h2:H, 0:64, :] = val
            return c

        result: list[np.ndarray] = []
        idx = 0
        for layer in range(len(self._cache_specs)):
            if layer in attn_layers:
                result.append(kv(all_caches[idx], all_caches[idx + 1]))
                idx += 2
            else:
                result.append(np.ascontiguousarray(all_caches[idx]))
                idx += 1
        if idx != len(all_caches):
            raise ValueError(
                f"image decoder emitted {len(all_caches)} caches; consumed {idx} "
                "— check the part split / output order")
        if len(result) != len(self._cache_specs):
            raise ValueError("folded cache count does not match the decoder inputs")
        return result

    def _merge_image_caches(self, seed: list[np.ndarray], s: int) -> None:
        """Splice the one-shot image-decoder caches (image at internal seq 0:64)
        into the current caches at seq slice [s:s+64], preserving prefix KV."""
        n_img = 64
        merged = []
        for (shape, _dt), cur, sd in zip(self._cache_specs, self._cache.snapshot(), seed):
            if len(shape) == 3:  # conv state: replaced wholesale
                merged.append(np.ascontiguousarray(sd))
            else:                # KV [1, heads, seq, hd]: prefix + image
                cur[:, :, s:s + n_img, :] = sd[:, :, 0:n_img, :]
                merged.append(np.ascontiguousarray(cur))
        self._cache.restore(merged)

    def encode_image(self, img_bgr: np.ndarray, should_stop=None) -> np.ndarray:
        """Run one image through vision + image-prefill + text prefix.

        Accepts any HxWx3 BGR uint8 array (camera frame, file, screenshot);
        it is center-cropped + resized to 256x256 for the static encoder.
        Returns the 64x1024 image features (f32).
        """
        t0 = time.perf_counter_ns()
        pv = _to_pixel_values(_center_crop_resize_bgr(img_bgr))
        feats = _to_host(self._vis.infer([_bf16_device(self._vis, pv)])[0]).astype(np.float32)
        if feats.shape[0] != 64:
            raise RuntimeError(f"vision encoder returned {feats.shape[0]} tokens, expected 64")
        self._vision_ns = time.perf_counter_ns() - t0

        t0 = time.perf_counter_ns()
        nxt = np.ascontiguousarray(feats)
        all_caches: list[np.ndarray] = []
        for i, part in enumerate(self._parts):
            exp = list(part.inputs_info[0].shape)
            xin = np.ascontiguousarray(nxt).reshape(exp).astype(
                np.dtype(part.inputs_info[0].dtype))
            outs = [_to_host(t) for t in part.infer([part.allocate_device_array(xin)])]
            if i < len(self._parts) - 1:
                nxt = np.ascontiguousarray(outs[-1])  # chaining hidden -> next part
                all_caches.extend(outs[:-1])
            else:
                all_caches.extend(outs)
        seed = self._fold_caches(all_caches)
        self._parts_ns = time.perf_counter_ns() - t0

        t0 = time.perf_counter_ns()
        prefix_ids = self._tokenizer.encode(_PREFIX, add_special_tokens=False).ids
        s = len(prefix_ids)
        if s + 64 + self._max_new + 16 > self._max_seq_len:
            raise ValueError("prefix + image leaves too little KV cache; lower max_new")
        self._cache.reset()
        self._decoder.prefill(prefix_ids, 0, want_logits=False, should_stop=should_stop)
        self._merge_image_caches(seed, s)
        self._base_caches = self._cache.snapshot()
        self._base_pos = s + 64
        self._prefix_ns = time.perf_counter_ns() - t0
        return feats

    # ---- questions ----

    def infer_stream(self, question: str, should_stop=None) -> Iterator[str]:
        """Answer one question about the current image, streaming text chunks.
        Restores a copy of the image-prefill cache; independent of prior questions."""
        if self._base_caches is None:
            raise RuntimeError("call encode_image() before infer()")
        question = (question or "").strip()
        turn = f"<|image_end|>{question}\nassistant\n"
        tids = self._tokenizer.encode(turn, add_special_tokens=False).ids
        if self._base_pos + len(tids) + self._max_new >= self._max_seq_len:
            raise ValueError(f"question too long for the KV cache ({self._max_seq_len})")
        self._prefill_tokens = self._base_pos + len(tids)
        # The per-question working copy is staged before the timing window.
        self._cache.restore(self._base_caches)
        pos = self._base_pos
        self._n_tokens_gen = 0
        self._first_token_ns = 0
        gen: list[int] = []
        t_start = time.perf_counter_ns()
        yield_ns = 0

        def next_token(logits: np.ndarray) -> int:
            return self._decoder.token_id(
                sample(logits, self._temperature, self._top_p, self._top_k))

        try:
            logits = self._decoder.prefill(tids, pos, should_stop=should_stop)
            pos += len(tids)
            first = next_token(logits)
            self._first_token_ns = time.perf_counter_ns() - t_start
            prev_text = ""
            for token in generate(
                self._decoder, first, pos,
                next_token=next_token,
                stop=lambda tok, g: tok == self._eos_token_id,
                max_new=self._max_new,
                should_stop=should_stop,
            ):
                gen.append(token)
                full_text = self._tokenizer.decode([t for t in gen if t != self._eos_token_id])
                y0 = time.perf_counter_ns()
                yield full_text[len(prev_text):]
                yield_ns += time.perf_counter_ns() - y0
                prev_text = full_text
        finally:
            self._n_tokens_gen = max(0, len(gen) - 1)
            self._last_infer_ns = time.perf_counter_ns() - t_start - yield_ns

    def infer(self, question: str, should_stop=None) -> str:
        return "".join(self.infer_stream(question, should_stop=should_stop))

    # ---- stats ----

    @property
    def stats(self) -> dict:
        """Times in milliseconds. ttft_ms = question prefill + first-token step
        (image phases are separate fields; sum them for image->token TTFT).
        decode_ms covers generated tokens only (prefill excluded)."""
        dec_ns = self._last_infer_ns - self._first_token_ns
        dec_ms = dec_ns / 1e6
        n = self._n_tokens_gen
        return {
            "vision_ms": self._vision_ns / 1e6,
            "image_prefill_ms": self._parts_ns / 1e6,
            "prefix_ms": self._prefix_ns / 1e6,
            "ttft_ms": self._first_token_ns / 1e6,
            "decode_ms": dec_ms,
            "decode_ms_per_tok": dec_ms / n if n else 0.0,
            "tok_per_s": n / dec_ms * 1e3 if dec_ms > 0 else 0.0,
            "n_tokens": n,
            "prompt_tokens": self._prefill_tokens,
        }

    def close(self) -> None:
        import gc
        for r in (self._vis, *self._parts, self._decoder, self._cache, self._model, self._lmhead):
            if r is not None:
                del r
        gc.collect()
