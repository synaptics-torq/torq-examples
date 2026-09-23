# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Public API for the LFM2-VL-450M W8 demo.

Import it, load the models once, then encode images and ask questions:

    import sys
    sys.path.insert(0, "<repo>/LiquidAI/LiquidAI-LFM2-VL-450M-INT8/src")
    from lfm2vl import load_models

    vl = load_models("/path/to/models")        # dmabuf, all models co-resident,
                                                # NPU clock set for inference
    vl.encode_image(img_bgr)                    # any HxWx3 BGR np.ndarray
    answer = vl.infer("What is this?")          # -> str
    for chunk in vl.infer_stream("Detail it"):  # streaming text
        print(chunk, end="", flush=True)
    vl.close()

``encode_image`` takes a raw numpy array on purpose (center-cropped + resized to
256x256 for the static encoder), so any image source — file, screenshot, or a
camera frame from your own capture code — plugs in directly.
"""

from __future__ import annotations

import os
from typing import Iterator

import numpy as np

try:
    from .engine import LFM2VLEngine  # package-style import (if on sys.path as a package)
except ImportError:
    from engine import LFM2VLEngine  # script import: python src/infer.py

__all__ = ["LFM2VL", "load_models"]


class LFM2VL:
    """All-NPU LFM2-VL-450M (W8) handle: encode images, ask questions.

    Time attributes after each call (ms): ``vision_time``, ``img_prefill_time``,
    ``prefix_time`` (from the last ``encode_image``) and ``time_to_first_token``,
    ``last_infer_time``, ``generated_tokens``, ``prefill_tokens`` (from the last
    ``infer``/``infer_stream``).
    """

    def __init__(self, model_dir: str | os.PathLike, **engine_kwargs):
        try:
            from utils.runtime import setup_npu_for_inference
            setup_npu_for_inference()
        except Exception:
            pass  # devfreq controls unavailable: run at the board default
        try:
            self._engine = LFM2VLEngine(model_dir, **engine_kwargs)
        except Exception:
            self.close()
            raise

    # -- image --
    def encode_image(self, img_bgr: np.ndarray, should_stop=None) -> np.ndarray:
        """Encode a fresh image (HxWx3 BGR uint8); resets the conversation.

        Center-cropped + resized to 256x256 for the static encoder.
        Returns the 64x1024 image features (f32).
        """
        feats = self._engine.encode_image(img_bgr, should_stop=should_stop)
        self.vision_time = self._engine.stats["vision_ms"]
        self.img_prefill_time = self._engine.stats["image_prefill_ms"]
        self.prefix_time = self._engine.stats["prefix_ms"]
        return feats

    # -- language --
    def infer(self, question: str, should_stop=None) -> str:
        """Answer one question about the current image (blocking, full text)."""
        return self._engine.infer(question, should_stop=should_stop)

    def infer_stream(self, question: str, should_stop=None) -> Iterator[str]:
        """Answer one question, streaming text chunks."""
        yield from self._engine.infer_stream(question, should_stop=should_stop)

    # -- stats / lifecycle --
    @property
    def stats(self) -> dict:
        """Full timing dict (see ``engine.LFM2VLEngine.stats``)."""
        return self._engine.stats

    def close(self) -> None:
        """Release the NPU contexts and restore the NPU clock."""
        engine = getattr(self, "_engine", None)
        if engine is not None:
            engine.close()
        try:
            from utils.runtime import cleanup_npu_after_inference
            cleanup_npu_after_inference()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_models(model_dir: str | os.PathLike, **kwargs) -> LFM2VL:
    """Load the LFM2-VL W8 model set (call once per session).

    ``model_dir`` holds: ``decoder_nolm.vmfb``, ``vision_encoder_256.vmfb``,
    ``lm_head.vmfb``, ``decoder_image_2part_*.vmfb`` (A, B), ``config.json``,
    ``tokenizer.json``, ``token_embeddings.npy``.

    Extra kwargs go to the engine (``tda``, ``max_new``, ``temperature``,
    ``top_p``, ``top_k``, ``max_seq_len``, ``n_threads``). The NPU clock is set
    for inference (1 GHz where supported) and restored on ``close()``.
    """
    return LFM2VL(model_dir, **kwargs)
