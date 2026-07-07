# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""LFM2-VL-450M inference runner: vision encoder (onnxruntime) + decoder (Torq).

The LFM2-VL decoder is the LFM2 hybrid conv + attention LM. The VMFB takes a
``token_embedding`` vector ``[1, 1, 1024]`` as input 0, ``position_ids`` at 1,
and then 16 per-layer cache tensors — either a short-conv state
``[1, 1024, 3]`` or a combined KV state ``[1, 16, 512, 64]`` depending on the
layer type. The caches are threaded **manually** (present -> past, kept on the
torq device) each step: the generic ``ManagedSelfAttnCacheRunner`` mishandles
this model's *mixed* conv/KV cache and yields degenerate output, whereas the
explicit threading here is bit-exact against the fp32 reference. The additions
for the VL model on top of the text decoder are:

  1. a SigLIP vision encoder run on the host CPU via onnxruntime (the encoder
     has dynamic shapes + exotic ops and is not compiled to the chip);
  2. ChatML prompt construction that expands ``<image>`` placeholders to match
     the encoder's per-sub-image token counts;
  3. an embedding-splice that drops the vision feature tokens into the
     ``<image>`` (id 396) positions of the otherwise text-embedded prompt.

Preprocessing reimplements ``Lfm2VlImageProcessorFast`` in numpy/PIL (the HF
fast processor needs torchvision); it follows ``image_processing_lfm2_vl_fast``
exactly: smart-resize within 512x512, tile large images + thumbnail, patchify,
pad, spatial_shapes.
"""

import json
import logging
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

import ml_dtypes
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from torq.runtime import VMFBInferenceRunner

DEFAULT_PROMPT: Final[str] = ""   # no text question: image-only user turn

# LFM2-VL special tokens / preprocessor constants
# (from LiquidAI/LFM2-VL-450M preprocessor_config.json + processor)
IMAGE_TOKEN_ID: Final[int] = 396  # "<image>"
PATCH: Final[int] = 16
DOWNSAMPLE: Final[int] = 2  # connector spatial downsample
MIN_TOKENS: Final[int] = 64
MAX_TOKENS: Final[int] = 256
TILE: Final[int] = 512
MIN_TILES: Final[int] = 2
MAX_TILES: Final[int] = 10
MAX_PIXELS_TOL: Final[float] = 2.0
MEAN: Final[float] = 0.5
STD: Final[float] = 0.5

StopCheck = Callable[[], bool]


class InferenceInterrupted(Exception):
    """Raised when interactive inference is cancelled by the user."""


def _raise_if_stopped(should_stop: StopCheck | None) -> None:
    if should_stop is not None and should_stop():
        raise InferenceInterrupted


# ---------------------------------------------------------------------------
# Image preprocessing (numpy/PIL port of Lfm2VlImageProcessorFast)
# ---------------------------------------------------------------------------

def _round_by_factor(n: float, f: int) -> int:
    return round(n / f) * f


def _smart_resize(h: int, w: int, upscale: bool = True) -> tuple[int, int]:
    total = PATCH * DOWNSAMPLE
    min_px = MIN_TOKENS * PATCH**2 * DOWNSAMPLE**2
    max_px = MAX_TOKENS * PATCH**2 * DOWNSAMPLE**2
    h_bar = max(total, _round_by_factor(h, total))
    w_bar = max(total, _round_by_factor(w, total))
    if h_bar * w_bar > max_px:
        beta = math.sqrt((h * w) / max_px)
        h_bar = max(total, math.floor(h / beta / total) * total)
        w_bar = max(total, math.floor(w / beta / total) * total)
    elif upscale and h_bar * w_bar < min_px:
        # Default: upscale small images up to the MIN_TOKENS (64) floor. With
        # upscale=False (native-res mode) a small image is left at its own
        # resolution (e.g. 128x128 -> 8x8 patches -> 16 image tokens).
        beta = math.sqrt(min_px / (h * w))
        h_bar = math.ceil(h * beta / total) * total
        w_bar = math.ceil(w * beta / total) * total
    return w_bar, h_bar  # (W, H) to match HF


def _is_image_too_large(h: int, w: int) -> bool:
    total = PATCH * DOWNSAMPLE
    h_bar = max(PATCH, _round_by_factor(h, total))
    w_bar = max(PATCH, _round_by_factor(w, total))
    return h_bar * w_bar > MAX_TOKENS * PATCH**2 * DOWNSAMPLE**2 * MAX_PIXELS_TOL


def _target_ratios() -> list[tuple[int, int]]:
    ratios = {
        (w, h)
        for n in range(MIN_TILES, MAX_TILES + 1)
        for w in range(1, n + 1)
        for h in range(1, n + 1)
        if MIN_TILES <= w * h <= MAX_TILES
    }
    return sorted(ratios, key=lambda x: x[0] * x[1])


def _find_closest_aspect_ratio(ar, ratios, w, h):
    best_diff, best = float("inf"), (1, 1)
    area = w * h
    for r in ratios:
        tar = r[0] / r[1]
        diff = abs(ar - tar)
        if diff < best_diff:
            best_diff, best = diff, r
        elif diff == best_diff and area > 0.5 * TILE * TILE * r[0] * r[1]:
            best = r
    return best


def _pil_resize(arr_hwc, out_w, out_h):
    im = Image.fromarray(np.clip(arr_hwc, 0, 255).astype(np.uint8))
    im = im.resize((out_w, out_h), Image.BILINEAR)
    return np.asarray(im).astype(np.float32)


def _patchify(img_chw):
    c, h, w = img_chw.shape
    nph, npw = h // PATCH, w // PATCH
    x = img_chw.reshape(c, nph, PATCH, npw, PATCH)
    x = x.transpose(1, 3, 2, 4, 0)  # [nph, npw, PATCH, PATCH, C]
    x = x.reshape(nph * npw, PATCH * PATCH * c)
    return x, nph, npw


def preprocess(pil_img: Image.Image, do_split: bool = True, native: bool = False):
    """Return (pixel_values[S,768], pixel_mask[S,P], spatial_shapes[S,2], grid).

    native=True processes the image at its own resolution: no upscale to the
    MIN_TOKENS floor and no patch padding (the encoder runs on exactly the real
    patches). It also disables tiling so the result is a single sub-image.
    """
    img = np.asarray(pil_img.convert("RGB")).astype(np.float32)  # HWC [0,255]
    H, W = img.shape[:2]
    new_w, new_h = _smart_resize(H, W, upscale=not native)
    large = _is_image_too_large(H, W)

    sub_images = []  # HWC float [0,255]
    if do_split and large and not native:
        ar = W / H
        gw, gh = _find_closest_aspect_ratio(ar, _target_ratios(), W, H)
        resized = _pil_resize(img, TILE * gw, TILE * gh)
        for r in range(gh):
            for c in range(gw):
                sub_images.append(resized[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE, :])
        sub_images.append(_pil_resize(img, new_w, new_h))  # thumbnail last
        grid = (gh, gw)
    else:
        sub_images.append(_pil_resize(img, new_w, new_h))
        grid = (1, 1)

    if native:
        pad_to = None  # no padding: encoder runs on the real patch count
    else:
        max_thumb = MAX_TOKENS * DOWNSAMPLE**2
        tile_patches = (TILE // PATCH) ** 2 if do_split else 0
        pad_to = max(max_thumb, tile_patches)

    pv, masks, shapes = [], [], []
    for sub in sub_images:
        chw = sub.transpose(2, 0, 1)
        chw = (chw / 255.0 - MEAN) / STD  # -> [-1, 1]
        patches, nph, npw = _patchify(chw)
        n = patches.shape[0]
        cap = n if pad_to is None else pad_to
        mask = np.ones((cap,), dtype=np.int64)
        if n < cap:
            pad = np.zeros((cap - n, patches.shape[1]), dtype=np.float32)
            patches = np.concatenate([patches, pad], axis=0)
            mask[n:] = 0
        pv.append(patches)
        masks.append(mask)
        shapes.append([nph, npw])
    return (
        np.stack(pv).astype(np.float32),
        np.stack(masks).astype(np.int64),
        np.array(shapes, dtype=np.int64),
        grid,
    )


def _tokens_for(shape) -> int:
    return math.ceil(shape[0] / DOWNSAMPLE) * math.ceil(shape[1] / DOWNSAMPLE)


def _image_block(shapes, grid) -> str:
    """Replicate Lfm2VlProcessor.expand_text_with_placeholders: the per-sub-image
    <image> counts equal the vision encoder's per-sub-image output token counts."""
    rows, cols = grid
    out = ["<|image_start|>"]
    if rows > 1 or cols > 1:
        idx = 0
        for r in range(rows):
            for c in range(cols):
                out.append(f"<|img_row_{r + 1}_col_{c + 1}|>")
                out.append("<image>" * _tokens_for(shapes[idx]))
                idx += 1
        out.append("<|img_thumbnail|>")
        out.append("<image>" * _tokens_for(shapes[-1]))  # thumbnail last
    else:
        out.append("<image>" * _tokens_for(shapes[0]))
    out.append("<|image_end|>")
    return "".join(out)


class LiquidVLStatic:
    """LFM2-VL-450M runner: SigLIP vision encoder (ORT) + LFM2 decoder (Torq)."""

    __slots__ = (
        "_logger", "_model", "_lmhead", "_model_dir", "_tokenizer",
        "_temperature", "_top_p", "_top_k",
        "_eos_token_id", "_max_seq_len", "_max_new",
        "_do_split", "_native",
        "_token_embeddings", "_pos_buf", "_emb_buf",
        "_cache_specs", "_caches",
        "_vsess", "_vis_inputs", "_vis_vmfb",
        "_dec_model_path", "_dec_lmhead_path", "_dec_n_threads",
        "_dec_runtime_flags", "_dec_max_seq_len",
        "_img_dec_prefix", "_img_prefill_ns", "_img_load_ns", "_cpu_lmhead", "_lmhead_w",
        "_base_caches", "_base_pos",
        "_n_tokens_gen", "_prefill_tokens", "_last_infer_ns", "_time_to_first_token_ns",
        "_start_time_ns", "_vision_ns",
    )

    def __init__(
        self,
        model_path: str | os.PathLike,
        vision_path: str | os.PathLike,
        max_seq_len: int | None = None,
        n_threads: int | None = None,
        *,
        max_new: int = 64,
        do_split: bool = True,
        native: bool = True,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 64,
        runtime_flags: list[str] | None = None,
        lmhead_path: str | os.PathLike | None = None,
        image_decoder_prefix: str | os.PathLike | None = None,
        cpu_lm_head: bool = False,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        # Compute the (tied) lm_head on the host CPU from token_embeddings instead
        # of loading lm_head.vmfb. Frees one NPU context — needed for the image
        # prefill path, where vision + 3 image parts + decoder already sit at the
        # device-memory edge and a resident lm_head degrades decode compute.
        self._cpu_lmhead = bool(cpu_lm_head)
        # One-shot image-prefill: when set, the 64 image tokens are filled by the
        # 5-part image decoder (`{prefix}{0..4}.vmfb`) in one batched chain instead
        # of 64 per-token decoder calls. The path is image-first (image at seq
        # positions 0..63); see _image_prefill_5part / run_stream.
        self._img_dec_prefix = str(image_decoder_prefix) if image_decoder_prefix else None
        self._img_prefill_ns = 0
        # Model load + free of the image-decoder parts is a one-time/amortized cost
        # (and a memory-juggling artifact: parts don't co-fit), tracked separately so
        # img_prefill_time reflects only the forward passes + prefix prefill.
        self._img_load_ns = 0
        # The image-prefill path can't co-fit an NPU lm_head (vision + 3 image parts
        # + decoder are already at the device-memory edge), so default it to the CPU
        # lm_head unless an explicit lm_head vmfb was passed.
        if self._img_dec_prefix is not None and lmhead_path is None:
            self._cpu_lmhead = True
        self._lmhead_w = None  # fp32 [vocab, hidden] lm_head weight, built on demand
        self._base_caches = None  # saved KV cache after [prefix][image] (per-image)
        self._base_pos = 0        # seq position after the image prefill

        # Stash decoder-load params: when the vision encoder is an NPU vmfb it and
        # the decoder don't fit on the NPU at the same time, so the decoder load is
        # *deferred* until after the image is encoded and the vision vmfb is freed.
        self._dec_model_path = model_path
        self._dec_lmhead_path = lmhead_path
        self._dec_n_threads = n_threads
        self._dec_runtime_flags = runtime_flags
        self._dec_max_seq_len = max_seq_len
        self._model = None
        self._lmhead = None
        self._cache_specs = []
        self._caches = []
        self._max_seq_len = None
        self._pos_buf = None
        self._emb_buf = None

        # --- config / tokenizer / embeddings (decoder-independent) ---
        self._model_dir = Path(model_path).parent
        with open(self._model_dir / "config.json") as f:
            cfg = json.load(f)
        self._eos_token_id = int(cfg.get("eos_token_id", 7))
        self._tokenizer = Tokenizer.from_file(str(self._model_dir / "tokenizer.json"))
        self._max_new = max_new
        self._do_split = do_split
        self._native = native
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._token_embeddings = self._load_embeddings()
        if self._token_embeddings is None:
            raise FileNotFoundError(
                f"token_embeddings.npy not found in {self._model_dir}"
            )

        # --- vision encoder (loaded first so a vmfb encoder can run + be freed
        #     before the decoder is loaded) ---
        # A .vmfb path runs the SigLIP encoder on the Torq NPU (function
        # `embed_images`, static `pixel_values [1,64,768] bf16 -> [16,1024] bf16`).
        # A .onnx path runs it on the CPU via onnxruntime (dynamic patch count).
        self._vsess = None
        self._vis_inputs = set()
        self._vis_vmfb = None
        if str(vision_path).endswith(".vmfb"):
            self._vis_vmfb = VMFBInferenceRunner(
                vision_path, function="embed_images", device_uri="torq",
                n_threads=n_threads, runtime_flags=runtime_flags, device_outputs=True,
            )
            self._logger.info(
                "Loaded vision encoder (NPU vmfb) '%s' — decoder load deferred "
                "until after encode (vision is freed first to fit)", str(vision_path))
        else:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if n_threads is not None:
                so.intra_op_num_threads = n_threads
            self._vsess = ort.InferenceSession(
                str(vision_path), sess_options=so, providers=["CPUExecutionProvider"],
            )
            self._vis_inputs = {i.name for i in self._vsess.get_inputs()}
            self._logger.info("Loaded vision encoder (CPU ORT) '%s'", str(vision_path))
            # CPU vision doesn't use the NPU, so the decoder can load now.
            self._ensure_decoder()

        self._n_tokens_gen = 0
        self._prefill_tokens = 0
        self._last_infer_ns = 0
        self._time_to_first_token_ns = 0
        self._start_time_ns = 0
        self._vision_ns = 0
        self._logger.info("Loaded decoder '%s'", str(model_path))

    # ---- stats ----
    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    @property
    def last_infer_time(self) -> float:
        return self._last_infer_ns / 1e6

    @property
    def time_to_first_token(self) -> float:
        return self._time_to_first_token_ns / 1e6

    @property
    def vision_time(self) -> float:
        return self._vision_ns / 1e6

    @property
    def img_prefill_time(self) -> float:
        """Image -> KV cache forward work: image-decoder part inferences + the 5-token
        prefix prefill. Excludes vmfb load + free (see img_load_time)."""
        return self._img_prefill_ns / 1e6

    @property
    def img_load_time(self) -> float:
        """One-time/amortized cost of loading + freeing the image-decoder part vmfbs
        (excluded from img_prefill_time)."""
        return self._img_load_ns / 1e6

    @property
    def generated_tokens(self) -> int:
        return self._n_tokens_gen

    @property
    def prefill_tokens(self) -> int:
        return self._prefill_tokens

    # ---- setup helpers ----
    def _load_embeddings(self) -> np.ndarray | None:
        paths = list(self._model_dir.glob("token_embeddings.npy"))
        if not paths:
            return None
        arr = np.load(paths[0], mmap_mode="r")  # keep file-backed
        if arr.dtype == np.dtype("V2"):
            arr = arr.view(ml_dtypes.bfloat16)
        return arr

    def _query_model_seq_len(self) -> int | None:
        info = self._model.inputs_info
        if info is None:
            return None
        if len(info) >= 3 and len(info[2].shape) == 2 and isinstance(info[2].shape[1], int):
            return info[2].shape[1]
        for t in info:
            if len(t.shape) == 4 and isinstance(t.shape[2], int):
                return t.shape[2]
        return None

    def _ensure_decoder(self) -> None:
        """Load the decoder (+ lm_head), size buffers/caches, warm up. Idempotent.
        Deferred until after the NPU vision encoder is freed (they don't co-fit)."""
        if self._model is not None:
            return
        self._model = VMFBInferenceRunner(
            self._dec_model_path, function="main", device_uri="torq",
            n_threads=self._dec_n_threads, runtime_flags=self._dec_runtime_flags,
            load_method="preload", load_model_to_mem=True, device_outputs=True,
        )
        if self._dec_lmhead_path is not None:
            self._lmhead = VMFBInferenceRunner(
                self._dec_lmhead_path, function="main", device_uri="torq",
                n_threads=self._dec_n_threads, runtime_flags=self._dec_runtime_flags,
                device_outputs=True,
            )
            self._logger.info("Loaded standalone lm_head '%s'", str(self._dec_lmhead_path))
        # If the decoder outputs the hidden state (it's a *body* like decoder_nolm,
        # out[0] last dim == hidden size) and no lm_head was given, compute the tied
        # lm_head on the CPU — otherwise out[0] would be misread as logits (garbage).
        # A fused decoder (out[0] == vocab logits) needs no lm_head.
        out_info = self._model.outputs_info
        if (self._lmhead is None and not self._cpu_lmhead and out_info
                and out_info[0].shape[-1] == self._token_embeddings.shape[-1]):
            self._cpu_lmhead = True
            self._logger.info(
                "decoder outputs hidden + no --lm-head: using CPU lm_head "
                "(pass --lm-head <lm_head.vmfb> for the NPU lm_head)")
        model_seq_len = self._query_model_seq_len()
        msl = self._dec_max_seq_len
        if msl is None:
            msl = model_seq_len
        elif model_seq_len is not None and msl != model_seq_len:
            self._logger.warning("max_seq_len=%d != model KV dim=%d; using %d",
                                 msl, model_seq_len, model_seq_len)
            msl = model_seq_len
        if msl is None:
            raise ValueError("Cannot determine max_seq_len; pass it explicitly.")
        self._max_seq_len = msl
        in_info = self._model.inputs_info
        self._pos_buf = np.zeros((1, 1), dtype=np.dtype(in_info[1].dtype))
        self._emb_buf = np.zeros(
            (1, 1, self._token_embeddings.shape[-1]), dtype=np.dtype(in_info[0].dtype))
        self._cache_specs = [(tuple(t.shape), np.dtype(t.dtype)) for t in in_info[2:]]
        self._reset_caches()
        # warm up (one-time device staging / first-invoke cost off the real TTFT)
        self._step(np.zeros(self._emb_buf.shape[-1], np.float32), 0, sample=True)
        self._reset_caches()
        self._logger.info("Loaded decoder '%s'", str(self._dec_model_path))

    # ---- vision ----
    def encode_image(self, pil_img: Image.Image):
        """Preprocess + run the vision encoder per sub-image. Returns
        (features[N, hidden], shapes, grid)."""
        if self._vsess is None and self._vis_vmfb is None:
            raise RuntimeError(
                "NPU vision encoder was freed after the first image (it can't be "
                "co-resident with the decoder). The vmfb-vision path is one-shot: "
                "re-run the process per image (use --image), or use the .onnx "
                "(CPU) vision encoder for the interactive loop.")
        pv, mask, shapes, grid = preprocess(
            pil_img, do_split=self._do_split, native=self._native)
        t0 = time.perf_counter_ns()
        feats = []
        for i in range(pv.shape[0]):
            if self._vis_vmfb is not None:
                # NPU encoder: static `embed_images(pixel_values[1,64,768] bf16)`.
                # Only pixel_values (spatial_shapes/mask are baked into the vmfb).
                exp = self._vis_vmfb.inputs_info[0].shape
                if list(pv[i].shape) != list(exp[1:]):
                    hint = ""
                    if self._img_dec_prefix is not None and exp[1] != 256:
                        hint = (" The --image-decoder path needs 64 image tokens "
                                "(256 patches): use vision_encoder_256.vmfb, not the "
                                "16-token vision_encoder.vmfb.")
                    raise ValueError(
                        f"vision vmfb expects pixel_values{list(exp)} but got "
                        f"[1, {pv[i].shape[0]}, {pv[i].shape[1]}] — the image must "
                        f"produce {exp[1]} patches (use a matching resolution/encoder)."
                        + hint
                    )
                out = self._vis_vmfb.infer([pv[i:i + 1].astype(ml_dtypes.bfloat16)])[0]
                h = np.asarray(out.to_host())
                feats.append((h.view(ml_dtypes.bfloat16) if h.dtype.kind == "V" else h)
                             .astype(np.float32))
            else:
                # CPU ORT encoder: run each sub-image at batch=1 (the pos-emb
                # interpolation can't batch tiles with different spatial_shapes).
                feed = {"pixel_values": pv[i:i + 1], "spatial_shapes": shapes[i:i + 1]}
                if "pixel_attention_mask" in self._vis_inputs:
                    feed["pixel_attention_mask"] = mask[i:i + 1]
                feats.append(self._vsess.run(None, feed)[0])
        self._vision_ns = time.perf_counter_ns() - t0
        return np.concatenate(feats, axis=0).astype(np.float32), shapes, grid

    # ---- one-shot image prefill (image decoder, in-process) ----
    # LFM2-VL layer types: attention at layers 2,5,8,10,12,14; conv elsewhere.
    _ATTN_LAYERS = frozenset({2, 5, 8, 10, 12, 14})

    def _image_prefill_chain(self, feats: np.ndarray) -> list[np.ndarray]:
        """Run the 64 image features through the image decoder and return the 16
        per-layer caches in ``decoder_nolm`` input order (host numpy, bf16, image at
        internal seq 0:64; KV stacks key into heads 0:8, value into 8:16).

        Generic over the split. ``--image-decoder`` is either a single full vmfb
        (path ends in ``.vmfb``) or a chain prefix, in which case the runner loads
        the sorted ``{prefix}*.vmfb`` (e.g. ``decoder_image_2part_{A,B}`` or
        ``decoder_image_3part_{0,1,2}``). Each non-final part emits a trailing
        layer-boundary hidden that feeds the next part; together the parts emit the
        22 per-layer caches in layer order (conv0,conv1,key2,val2,...,key14,val14,
        conv15), which are folded into the 16 decoder caches (KV pairs combined).
        Each part's input is reshaped to its own signature (2-part uses [64,1024],
        3-part uses [1,64,1024]). Run **in-process** (a subprocess wedges the device
        for the subsequent decoder); ``del`` frees each part's DDR (NPU is 512 KB
        SRAM). Validated on-board against per-token decoder_nolm prefill (cos>=0.9999).
        """
        import gc
        import glob
        bf16 = ml_dtypes.bfloat16

        prefix = str(self._img_dec_prefix)
        if prefix.endswith(".vmfb"):
            part_files = [prefix]
        else:
            part_files = sorted(glob.glob(f"{prefix}*.vmfb"))
            if not part_files:
                raise FileNotFoundError(
                    f"no image-decoder vmfbs match {prefix}*.vmfb")

        # Time the forward passes only; load + free are tracked separately (they are a
        # one-time/amortized cost and a memory-juggling artifact -- the parts don't
        # co-fit, so each is loaded -> inferred -> freed in turn).
        infer_ns = 0
        load_ns = 0
        nxt = np.asarray(feats)                       # [64,1024]; reshaped per part
        all_caches: list[np.ndarray] = []
        for i, pf in enumerate(part_files):
            tL = time.perf_counter_ns()
            part = VMFBInferenceRunner(
                pf, function="main", device_uri="torq",
                n_threads=self._dec_n_threads, runtime_flags=self._dec_runtime_flags,
                device_outputs=True,
            )
            load_ns += time.perf_counter_ns() - tL        # vmfb load (excluded from prefill)
            exp = list(part.inputs_info[0].shape)     # [64,1024] or [1,64,1024]
            xin = np.ascontiguousarray(nxt).reshape(exp).astype(bf16)
            tI = time.perf_counter_ns()
            res = part.infer([xin])
            outs = [self._to_host_bf16(t) for t in res]
            infer_ns += time.perf_counter_ns() - tI       # forward pass (the prefill work)
            del res
            tF = time.perf_counter_ns()
            del part                              # free the part's DDR (required: NPU
            gc.collect()                          # is 512 KB SRAM, weights in DDR; keeping
            load_ns += time.perf_counter_ns() - tF        # free (excluded from prefill)
            #                                       partA+partB resident OOMs the board)
            if i < len(part_files) - 1:
                nxt = np.asarray(outs[-1])            # chaining hidden -> next part
                all_caches.extend(outs[:-1])
            else:
                all_caches.extend(outs)
        self._img_prefill_ns = infer_ns               # forward only (load+free excluded)
        self._img_load_ns = load_ns

        # Fold the 22 per-layer caches (layer order) into 16 (KV pairs combined).
        def kv(key: np.ndarray, val: np.ndarray) -> np.ndarray:
            c = np.zeros((1, 16, 512, 64), bf16)
            c[:, 0:8, 0:64, :] = key
            c[:, 8:16, 0:64, :] = val
            return c

        result: list[np.ndarray] = []
        idx = 0
        for layer in range(16):
            if layer in self._ATTN_LAYERS:
                result.append(kv(all_caches[idx], all_caches[idx + 1]))
                idx += 2
            else:
                result.append(np.ascontiguousarray(all_caches[idx].astype(bf16)))
                idx += 1
        if idx != len(all_caches):
            raise ValueError(
                f"image decoder emitted {len(all_caches)} caches; expected 22 "
                f"(consumed {idx}) — check the part split / output order")
        return result

    # ---- prompt ----
    def build_prompt_ids(self, prompt_text: str, shapes, grid):
        block = _image_block(shapes.tolist(), grid)
        prompt = (
            f"<|startoftext|><|im_start|>user\n{block}{prompt_text}"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        ids = self._tokenizer.encode(prompt, add_special_tokens=False).ids
        img_pos = [k for k, t in enumerate(ids) if t == IMAGE_TOKEN_ID]
        return ids, img_pos

    # ---- decoder ----
    def _reset_caches(self) -> None:
        """Zero-init the per-layer caches as on-device arrays on the body runner."""
        self._caches = [
            self._model.allocate_device_array(np.zeros(shape, dtype))
            for shape, dtype in self._cache_specs
        ]

    @staticmethod
    def _to_host_bf16(arr) -> np.ndarray:
        a = np.ascontiguousarray(np.asarray(arr.to_host()))
        return a.view(ml_dtypes.bfloat16) if a.dtype.kind == "V" else a

    def _cpu_logits(self, hidden: np.ndarray) -> np.ndarray:
        """Tied lm_head on the host: logits = embed_tokens.weight @ hidden. The fp32
        weight is built once (the bf16->fp32 cast of the 134 MB weight, not the
        matmul, dominates — re-casting it every token would be ~10x slower). The
        host has room: the NPU models live in a separate device carveout."""
        if self._lmhead_w is None:
            self._lmhead_w = np.asarray(self._token_embeddings).astype(np.float32)
        hf = hidden.astype(np.float32, copy=False).ravel()  # [hidden]
        return self._lmhead_w @ hf                            # [vocab]

    def _step(self, emb_vec: np.ndarray, seq_pos: int, *, sample: bool):
        self._emb_buf[0, 0, :] = emb_vec  # cast into the decoder's IO dtype
        self._pos_buf[0, 0] = seq_pos
        out = self._model.infer([self._emb_buf, self._pos_buf, *self._caches])
        # out[0] = logits (full decoder) or hidden_out (body); out[1:] = present
        # caches -> become past for the next step.
        self._caches = list(out[1:])
        if not sample:
            return None  # prefill token: skip lm_head entirely
        if self._cpu_lmhead:
            # body hidden -> host; tied lm_head computed on the CPU (no NPU context)
            hidden = self._to_host_bf16(out[0])[0, -1]
            return self._sample(self._cpu_logits(hidden))
        if self._lmhead is not None:
            # Hand the body's hidden state to the lm_head. The body and lm_head
            # are compiled independently, so their on-device tensor layouts don't
            # match -> normalize through the host (hidden is [1,1,1024] = 2 KB, so
            # this is negligible; the lm_head is preloaded, no per-call faulting).
            hidden = self._to_host_bf16(out[0])
            logits = self._to_host_bf16(self._lmhead.infer([hidden])[0])
        else:
            logits = self._to_host_bf16(out[0])
        return self._sample(logits[0, -1])

    def _sample(self, logits: np.ndarray) -> int:
        logits = logits.astype(np.float32, copy=False)
        if self._temperature <= 0:
            return int(logits.argmax())
        k = min(self._top_k, logits.shape[-1])
        top_k_idx = np.argpartition(logits, -k)[-k:]
        x = logits[top_k_idx]
        x /= self._temperature
        x -= x.max()
        np.exp(x, out=x)
        x /= x.sum()
        order = np.argsort(x)[::-1]
        cdf = np.cumsum(x[order])
        cut = int(np.searchsorted(cdf, self._top_p)) + 1
        keep = order[:cut]
        p = x[keep]
        p /= p.sum()
        return int(np.random.choice(top_k_idx[keep], p=p))

    # ---- prefill ----
    def _prefill_per_token(self, ids, img_pos, feats, should_stop):
        """Standard prefill: embed the whole prompt (vision features spliced into
        the <image> slots) and run it one token at a time. Returns (next_tok, pos)."""
        emb = np.asarray(self._token_embeddings)[np.asarray(ids)].astype(np.float32)
        if img_pos:
            emb[img_pos] = feats
        self._reset_caches()
        for k in range(len(ids) - 1):
            _raise_if_stopped(should_stop)
            self._step(emb[k], k, sample=False)
        _raise_if_stopped(should_stop)
        next_tok = self._step(emb[len(ids) - 1], len(ids) - 1, sample=True)
        return next_tok, len(ids)

    def _prefill_with_image_caches(self, ids, s, feats, seed_caches, should_stop):
        """One-shot image prefill: prefill the prefix tokens ids[:s]
        (<|startoftext|><|im_start|>user\\n<|image_start|>) per-token, splice the
        image decoder's caches into the <image> span [s:s+64], then prefill the
        suffix ids[s+64:] per-token. The last suffix token samples -> first
        generated token. Returns (next_tok, pos). The 64 image tokens never hit the
        decoder (their caches come from the one-shot chain), which is the TTFT win."""
        self._reset_caches()
        # prefix: ids[0:s] at positions 0..s-1
        pre = np.asarray(self._token_embeddings)[np.asarray(ids[:s])].astype(np.float32)
        for k in range(s):
            _raise_if_stopped(should_stop)
            self._step(pre[k], k, sample=False)
        # splice the one-shot image caches into seq slice [s:s+64]
        self._merge_image_caches(seed_caches, s)
        # suffix: ids[s+64:] at positions s+64 ..
        suf = ids[s + 64:]
        semb = np.asarray(self._token_embeddings)[np.asarray(suf)].astype(np.float32)
        for j in range(len(suf) - 1):
            _raise_if_stopped(should_stop)
            self._step(semb[j], s + 64 + j, sample=False)
        _raise_if_stopped(should_stop)
        last = s + 64 + len(suf) - 1
        next_tok = self._step(semb[-1], last, sample=True)
        return next_tok, last + 1

    def _merge_image_caches(self, seed, s) -> None:
        """Splice the one-shot image-decoder caches (image at internal seq 0:64)
        into the current caches at seq slice [s:s+64], preserving the prefix KV at
        [0:s]. Conv states are replaced wholesale (the depthwise window is 3, so by
        the last image token the prefix has slid out). The merged caches are host
        numpy and get uploaded on the next _step."""
        merged = []
        for (shape, _dt), cur, sd in zip(self._cache_specs, self._caches, seed):
            if len(shape) == 3:        # conv [1,1024,3] -> image-decoder conv
                merged.append(np.ascontiguousarray(sd))
            else:                       # KV [1,16,512,64] -> prefix[0:s] + image[s:s+64]
                host = self._to_host_bf16(cur).copy()
                host[:, :, s:s + 64, :] = sd[:, :, 0:64, :]
                merged.append(np.ascontiguousarray(host))
        self._caches = merged

    # ---- conversation (multi-turn over one image; KV-cache copy per question) ----
    def begin_image(self, image_path: str | os.PathLike,
                    should_stop: StopCheck | None = None) -> None:
        """Encode the image once and prime the KV cache with
        ``<|startoftext|><|im_start|>user\\n<|image_start|>[image]``. The resulting
        cache is *saved*; each ``ask()`` restores a copy of it, prefills only the
        question, and generates — so every question is independent and the sequence
        stays short (the image is 64 tokens). Call this once per image, then ask()."""
        self._vision_ns = 0
        self._img_prefill_ns = 0
        self._img_load_ns = 0
        img = Image.open(image_path)
        feats, shapes, grid = self.encode_image(img)
        # NPU vision encoder + decoder don't co-fit: free the vision vmfb now.
        if self._vis_vmfb is not None and self._model is None:
            import gc
            del self._vis_vmfb
            self._vis_vmfb = None
            gc.collect()
            self._logger.info("Freed NPU vision encoder; loading decoder")
        if tuple(grid) != (1, 1):
            raise ValueError("multi-turn / cache-reuse supports a single image; "
                             "run with --native-res (no tiling)")
        if self._img_dec_prefix is not None:
            if feats.shape[0] != 64:
                raise ValueError(
                    f"image-prefill needs exactly 64 image tokens, got "
                    f"{feats.shape[0]} (use a 256-res image + vision_encoder_256.vmfb)")
            seed_caches = self._image_prefill_chain(feats)   # sets _img_prefill_ns
        else:
            seed_caches = None
        self._ensure_decoder()
        n_img = feats.shape[0]
        prefix_ids = self._tokenizer.encode(
            "<|startoftext|><|im_start|>user\n<|image_start|>", add_special_tokens=False).ids
        s = len(prefix_ids)
        if s + n_img + self._max_new + 16 > self._max_seq_len:
            raise ValueError(
                f"image ({n_img} tok) + prefix leaves too little KV cache "
                f"({self._max_seq_len}) for a question + answer")
        t0 = time.perf_counter_ns()
        self._reset_caches()
        pre = np.asarray(self._token_embeddings)[np.asarray(prefix_ids)].astype(np.float32)
        for k in range(s):
            _raise_if_stopped(should_stop)
            self._step(pre[k], k, sample=False)
        if seed_caches is not None:
            self._merge_image_caches(seed_caches, s)            # image at [s:s+64]
        else:
            for j in range(n_img):
                _raise_if_stopped(should_stop)
                self._step(feats[j], s + j, sample=False)       # per-token image embed
        # Save the [prefix][image] cache as host copies; ask() restores a copy.
        self._base_caches = [
            np.ascontiguousarray(self._to_host_bf16(c)) if hasattr(c, "to_host")
            else np.ascontiguousarray(c)
            for c in self._caches
        ]
        self._base_pos = s + n_img
        self._img_prefill_ns += time.perf_counter_ns() - t0     # + prefix/image prefill

    def ask(self, question: str | None = None,
            should_stop: StopCheck | None = None):
        """Generator: answer one question about the current image, streaming text.
        Restores the saved image-prefill cache (copy), prefills the question, then
        decodes. Independent of prior questions. ``begin_image()`` must run first."""
        if self._base_caches is None:
            raise RuntimeError("call begin_image() before ask()")
        question = (question or "").strip()
        self._n_tokens_gen = 0
        self._last_infer_ns = 0
        self._time_to_first_token_ns = 0
        turn = f"<|image_end|>{question}<|im_end|>\n<|im_start|>assistant\n"
        tids = self._tokenizer.encode(turn, add_special_tokens=False).ids
        if self._base_pos + len(tids) + self._max_new >= self._max_seq_len:
            raise ValueError(
                f"question too long for the KV cache ({self._max_seq_len} tokens)")
        self._prefill_tokens = self._base_pos + len(tids)  # prefix + image + question
        temb = np.asarray(self._token_embeddings)[np.asarray(tids)].astype(np.float32)
        # restore a copy of the image-prefill cache (so the saved base is untouched)
        self._caches = [c.copy() for c in self._base_caches]
        pos = self._base_pos
        gen: list[int] = []
        self._start_time_ns = time.perf_counter_ns()
        yield_ns = 0
        try:
            for j in range(len(tids) - 1):
                _raise_if_stopped(should_stop)
                self._step(temb[j], pos, sample=False)
                pos += 1
            _raise_if_stopped(should_stop)
            next_tok = self._step(temb[-1], pos, sample=True)
            pos += 1
            self._time_to_first_token_ns = time.perf_counter_ns() - self._start_time_ns

            prev_text = self._tokenizer.decode([next_tok])
            ys = time.perf_counter_ns(); yield prev_text; yield_ns += time.perf_counter_ns() - ys
            gen = [next_tok]
            while next_tok != self._eos_token_id and len(gen) < self._max_new:
                _raise_if_stopped(should_stop)
                if pos >= self._max_seq_len:
                    self._logger.warning("Max sequence length reached")
                    break
                emb_vec = np.asarray(self._token_embeddings)[next_tok].astype(np.float32)
                next_tok = self._step(emb_vec, pos, sample=True)
                pos += 1
                gen.append(next_tok)
                full_text = self._tokenizer.decode([t for t in gen if t != self._eos_token_id])
                chunk = full_text[len(prev_text):]
                ys = time.perf_counter_ns(); yield chunk; yield_ns += time.perf_counter_ns() - ys
                prev_text = full_text
        finally:
            self._n_tokens_gen = max(0, len(gen) - 1)
            self._last_infer_ns = time.perf_counter_ns() - self._start_time_ns - yield_ns

    # ---- run (one-shot) ----
    def run(self, image_path: str | os.PathLike, prompt_text: str | None = None,
            should_stop: StopCheck | None = None) -> str:
        return "".join(self.run_stream(image_path, prompt_text, should_stop=should_stop))

    def run_stream(self, image_path: str | os.PathLike,
                   prompt_text: str | None = None,
                   should_stop: StopCheck | None = None):
        """One-shot: encode the image and answer a single prompt (begin_image + ask)."""
        prompt_text = prompt_text if prompt_text is not None else DEFAULT_PROMPT
        self.begin_image(image_path, should_stop=should_stop)
        yield from self.ask(prompt_text, should_stop=should_stop)
