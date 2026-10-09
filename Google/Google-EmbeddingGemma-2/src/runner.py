# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""EmbeddingGemma-2 runner: text, image and video embeddings on Torq.

Static graphs (one vmfb each):

    vision_<H>x<W>   pixel_patches [1, N, 768]                 -> image_embeds [1, N/9, 512]
    text_body_s<S>   inputs_embeds [1, S, 512], attention_bias [1, 1, 1, S]
                                                               -> last_hidden_state [1, S, 768]

Host side: tokenizer, token-embedding LUT (mmap'd from eMMC), image/video
preprocessing, writing image/video soft tokens into their placeholder slots,
mean pooling over real tokens, optional Matryoshka truncation, L2 normalisation.

Backends:
  * ``torq`` — ``*.vmfb`` on the NPU via ``torq.runtime.VMFBInferenceRunner``.
  * ``ort``  — the fp32 static ``*.onnx`` graphs on the CPU via onnxruntime
    (identical host code; used as the accuracy/performance baseline).
"""

import json
import logging
import time
from pathlib import Path

import ml_dtypes
import numpy as np
from PIL import Image

try:
    from . import preprocess as pp
except ImportError:  # run as a script from src/
    import preprocess as pp

logger = logging.getLogger("EmbeddingGemma2.runner")


class _Component:
    """One static graph on one backend, loaded on first use."""

    def __init__(self, path: Path, backend: str, ort_threads: int = 0):
        self.path, self.backend, self.ort_threads = path, backend, ort_threads
        self._impl = None
        self.last_ms = 0.0

    def _load(self):
        if self.backend == "torq":
            from torq.runtime import VMFBInferenceRunner

            self._impl = VMFBInferenceRunner(str(self.path))
        else:
            import onnxruntime as ort

            so = ort.SessionOptions()
            if self.ort_threads:
                so.intra_op_num_threads = self.ort_threads
            self._impl = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        logger.info("loaded %s (%s)", self.path.name, self.backend)

    def warmup(self):
        """Load the network and run it once on zeros, which maps it into the NPU."""
        if self._impl is None:
            self._load()
        if self.backend == "torq":
            shapes = [tuple(t.shape) for t in self._impl.inputs_info]
        else:
            shapes = [tuple(i.shape) for i in self._impl.get_inputs()]
        self(*[np.zeros(sh, dtype=np.float32) for sh in shapes])

    def __call__(self, *inputs: np.ndarray) -> np.ndarray:
        if self._impl is None:
            self._load()
        t0 = time.perf_counter()
        if self.backend == "torq":
            outs = self._impl.infer([np.ascontiguousarray(x.astype(ml_dtypes.bfloat16)) for x in inputs])
            out = np.asarray(outs[0]).astype(np.float32)
            if any(np.any(x) for x in inputs) and (not np.any(out) or not np.all(np.isfinite(out))):
                # a wedged NPU can return instantly with an empty output and no error
                raise RuntimeError(f"{self.path.name}: NPU returned an all-zero or non-finite output; "
                                   "the NPU is likely wedged (reboot the board)")
        else:
            names = [i.name for i in self._impl.get_inputs()]
            out = self._impl.run(None, {n: x.astype(np.float32) for n, x in zip(names, inputs)})[0]
        self.last_ms = (time.perf_counter() - t0) * 1e3
        return out


class EmbeddingGemma2:
    # The SL2619 NPU maps at most ~1.5 GB of network XRAM at once (measured: 1.48 GB fits,
    # 1.58 GB fails to map), and releasing a network is not reliable on the current driver,
    # so the resident set is chosen up front from the requested modalities.
    NPU_BUDGET_MB = 1450

    def __init__(self, model_dir: str | Path, backend: str = "torq", ort_threads: int = 0,
                 image_size: tuple[int, int] | None = None,
                 modalities: tuple[str, ...] = ("text", "image", "video"),
                 npu_budget_mb: float | None = None, warmup: bool | None = None,
                 audio_frames: int | None = None):
        self.model_dir = Path(model_dir)
        if backend not in ("torq", "ort"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        manifest_path = self.model_dir / "embeddinggemma2_manifest.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        self.seq_lens = sorted(self.manifest.get("seq_lens", [128, 512]))
        self.image_size = tuple(image_size or self.manifest.get("image_sizes", [[384, 384]])[0])
        self.max_frames = self.manifest.get("video", {}).get("max_frames", 7)
        audio_cfg = self.manifest.get("audio") or {}
        self.audio_windows = sorted(audio_cfg.get("windows", [audio_cfg.get("num_frames", 1120)]))
        # one audio encoder per session: 1120 frames = 11.2 s (280 tokens), 560, 280, 140
        self.audio_frames = audio_frames or self.audio_windows[-1]
        if self.audio_frames not in self.audio_windows:
            raise ValueError(f"no audio encoder for {self.audio_frames} frames (have {self.audio_windows})")

        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        lut = np.load(self.model_dir / "token_embeddings.npy", mmap_mode="r")
        self.lut = lut.view(ml_dtypes.bfloat16) if lut.dtype == np.dtype("V2") else lut

        ext = ".vmfb" if backend == "torq" else ".onnx"
        h, w = self.image_size
        self.vision = _Component(self.model_dir / f"vision_{h}x{w}{ext}", backend, ort_threads)
        if backend == "torq":
            self.seq_lens = self._fit_buckets(set(modalities), npu_budget_mb or self.NPU_BUDGET_MB)
        self.text = {s: _Component(self.model_dir / f"text_body_s{s}{ext}", backend, ort_threads)
                     for s in self.seq_lens}
        self.audio = _Component(self.model_dir / f"audio_{self.audio_frames}{ext}", backend, ort_threads)
        mel_path = self.model_dir / "mel_filters.npy"
        self.mel_filters = np.load(mel_path) if mel_path.exists() else None
        self.mel_window = np.load(self.model_dir / "mel_window.npy") if mel_path.exists() else None
        self.timings: dict[str, float] = {}
        if warmup if warmup is not None else backend == "torq":
            self._warmup(set(modalities))

    def _warmup(self, modalities: set[str]):
        """Start every network this session needs, largest XRAM span first. The NPU maps
        each network into one contiguous address range, so mapping small networks first
        can fragment the space and leave no room for the large ones."""
        spans = self.manifest.get("xram_span_mb", {})
        comps = [(f"text_body_s{s}", self.text[s]) for s in self.seq_lens]
        if modalities & {"image", "video"}:
            comps.append((self.vision.path.stem, self.vision))
        if "audio" in modalities:
            comps.append((self.audio.path.stem, self.audio))
        t0 = time.perf_counter()
        for name, comp in sorted(comps, key=lambda c: -spans.get(c[0], 0)):
            comp.warmup()
        logger.info("warm-up of %d networks: %.1f s", len(comps), time.perf_counter() - t0)

    def _fit_buckets(self, modalities: set[str], budget_mb: float) -> list[int]:
        """Text buckets to keep resident: all of them unless the requested encoders plus
        the buckets exceed the NPU mapping budget, in which case the shortest buckets go
        (their inputs then run in the next larger bucket)."""
        spans = self.manifest.get("xram_span_mb", {})
        h, w = self.image_size
        encoders = []
        if modalities & {"image", "video"}:
            encoders.append(f"vision_{h}x{w}")
        if "audio" in modalities:
            encoders.append(f"audio_{self.audio_frames}")
        buckets = list(self.seq_lens)

        def total():
            return sum(spans.get(n, 0) for n in encoders) + sum(spans.get(f"text_body_s{s}", 0) for s in buckets)

        while len(buckets) > 1 and total() > budget_mb:
            dropped = buckets.pop(0)
            logger.warning("NPU mapping budget %.0f MB: not loading text bucket S=%d", budget_mb, dropped)
        if total() > budget_mb:
            logger.warning("requested networks need %.0f MB of NPU mappings (budget %.0f MB)", total(), budget_mb)
        return buckets

    # ---------------- building blocks ----------------

    def _bucket(self, n_tokens: int) -> int:
        for s in self.seq_lens:
            if n_tokens <= s:
                return s
        raise ValueError(f"{n_tokens} tokens exceed the largest static sequence length {self.seq_lens[-1]}")

    def vision_tokens(self, image: Image.Image) -> np.ndarray:
        """Soft tokens [n, 512] for one image or video frame."""
        t0 = time.perf_counter()
        v = pp.patchify(image, self.image_size)
        self.timings["preprocess_ms"] = self.timings.get("preprocess_ms", 0.0) + (time.perf_counter() - t0) * 1e3
        out = self.vision(v.patches[None])
        self.timings["vision_ms"] = self.timings.get("vision_ms", 0.0) + self.vision.last_ms
        return out.reshape(-1, out.shape[-1])

    def embed_ids(self, ids: list[int], media: np.ndarray | None = None, dim: int = 768) -> np.ndarray:
        s = self._bucket(len(ids))
        embeds, mask = pp.build_inputs_embeds(ids, self.lut, s, media)
        hidden = self.text[s](embeds, pp.attention_bias(mask))
        self.timings[f"text_s{s}_ms"] = self.text[s].last_ms
        return pp.pool_and_normalize(hidden, mask, dim)

    # ---------------- public API ----------------

    def encode_text(self, text: str, prompt_name: str | None = None, dim: int = 768) -> np.ndarray:
        self.timings = {}
        return self.embed_ids(pp.text_ids(self.tokenizer, text, prompt_name), dim=dim)

    def encode_image(self, image: str | Path | Image.Image, dim: int = 768) -> np.ndarray:
        self.timings = {}
        img = Image.open(image) if not isinstance(image, Image.Image) else image
        feats = self.vision_tokens(img)
        return self.embed_ids(pp.image_ids(feats.shape[0]), feats, dim)

    def encode_frames(self, frames: list[Image.Image], dim: int = 768) -> np.ndarray:
        self.timings = {}
        feats = [self.vision_tokens(f) for f in frames[: self.max_frames]]
        return self.embed_ids(pp.video_ids(len(feats), feats[0].shape[0]), np.concatenate(feats), dim)

    def encode_audio(self, audio: "str | Path | np.ndarray", dim: int = 768) -> np.ndarray:
        """Embed 16 kHz mono audio (a WAV path or a float waveform). Clips longer than the
        session's static window (``audio_frames`` x 10 ms) are truncated."""
        self.timings = {}
        t0 = time.perf_counter()
        wave = pp.load_audio(str(audio)) if not isinstance(audio, np.ndarray) else audio
        mel, mask = pp.audio_features(wave, self.mel_filters, self.mel_window, num_frames=self.audio_frames)
        n = pp.audio_num_tokens(mask)
        self.timings["preprocess_ms"] = (time.perf_counter() - t0) * 1e3
        tokens = self.audio(mel[None])
        self.timings["audio_ms"] = self.audio.last_ms
        tokens = tokens.reshape(-1, tokens.shape[-1])[:n]
        return self.embed_ids(pp.audio_ids(n), tokens, dim)

    def encode_video(self, path: str | Path, fps: float = 1.0, dim: int = 768) -> np.ndarray:
        t0 = time.perf_counter()
        frames = pp.sample_video_frames(str(path), fps=fps, max_frames=self.max_frames)
        decode_ms = (time.perf_counter() - t0) * 1e3
        emb = self.encode_frames(frames, dim)
        self.timings["decode_ms"] = decode_ms
        self.timings["frames"] = len(frames)
        return emb
