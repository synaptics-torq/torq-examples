# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import logging
import os
from pathlib import Path
from time import perf_counter_ns
from typing import Final

import numpy as np

from torq.runtime import VMFBInferenceRunner

from utils.inference import ManagedEncDecCacheRunner

_START_TOKEN_ID: Final[int] = 1
_END_TOKEN_ID: Final[int] = 2
_DEFAULT_INPUT_FREQ: Final[int] = 16000


class MoonshineRunner:
    """Moonshine speech-to-text inference runner.

    Requires ``encoder.vmfb`` and ``decoder.vmfb`` in the model directory.
    ``decoder_token_embeddings.npy`` enables embedding-lookup input to the
    decoder.
    """

    __slots__ = (
        "_logger",
        "_debug_logging",
        "_model_dir",
        "_encoder",
        "_decoder",
        "_token_embeddings",
        "_max_inp_len",
        "_input_freq",
        "_device_io",
        "_n_tokens_gen",
        "_last_infer_ns",
        "_time_to_first_token_ns",
    )

    def __init__(
        self,
        model_dir: str | os.PathLike,
        *,
        input_freq: int = _DEFAULT_INPUT_FREQ,
        n_threads: int | None = None,
        runtime_flags: list[str] | None = None,
        device_io: bool = False,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._debug_logging: bool = self._logger.isEnabledFor(logging.DEBUG)
        model_dir = Path(model_dir)
        self._model_dir = model_dir
        self._device_io = device_io

        encoder_path = model_dir / "encoder.vmfb"
        decoder_path = model_dir / "decoder.vmfb"
        for req in (encoder_path, decoder_path):
            if not req.is_file():
                raise ValueError(f"Missing required model '{req.name}' in {model_dir}")

        rk: dict = dict(n_threads=n_threads, runtime_flags=runtime_flags)

        self._encoder = VMFBInferenceRunner(
            encoder_path,
            device_outputs=device_io,
            **rk,
        )
        self._decoder = ManagedEncDecCacheRunner(
            decoder_path,
            input_cache_start_idx=2,  # [token_emb, current_len, *cache]
            cache_start_idx=1,  # [logits, *self_cache]
            device_io=device_io,
            **rk,
        )

        self._token_embeddings = self._load_embeddings(model_dir)

        enc_info = self._encoder.inputs_info
        if enc_info:
            self._max_inp_len: int | None = int(enc_info[0].shape[-1])
        else:
            self._max_inp_len = None
            self._logger.warning(
                "Cannot determine max input length from encoder metadata; "
                "audio will not be padded/truncated."
            )

        self.input_freq = input_freq
        self._n_tokens_gen: int = 0
        self._last_infer_ns: int = 0
        self._time_to_first_token_ns: int = 0

        self._logger.info("Loaded Moonshine models from '%s'", model_dir)
        if self._debug_logging:
            self._logger.warning("DEBUG logging enabled: inference time metrics may be inflated")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def last_infer_time(self) -> float:
        """Total inference time of the last ``run()`` call, in milliseconds."""
        return self._last_infer_ns / 1e6

    @property
    def time_to_first_token(self) -> float:
        """Encoder + first decoder step time, in milliseconds."""
        return self._time_to_first_token_ns / 1e6

    @property
    def generated_tokens(self) -> int:
        return self._n_tokens_gen

    @property
    def max_inp_len(self) -> int | None:
        return self._max_inp_len

    @property
    def input_freq(self) -> int:
        return self._input_freq

    @input_freq.setter
    def input_freq(self, value: int) -> None:
        if value <= 0:
            raise ValueError(f"input_freq must be positive, got {value}")
        self._input_freq = int(value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_embeddings(model_dir: Path) -> np.ndarray | None:
        paths = list(model_dir.glob("*token_embeddings.npy"))
        if not paths:
            return None
        arr = np.load(paths[0])
        if arr.dtype == np.dtype("V2"):
            import ml_dtypes

            arr = arr.view(ml_dtypes.bfloat16)
        return arr

    def _size_input(self, audio: np.ndarray) -> np.ndarray:
        """Pad or truncate audio to ``max_inp_len``."""
        if self._max_inp_len is None:
            return audio
        audio = np.ravel(audio)
        if len(audio) > self._max_inp_len:
            self._logger.debug(
                "Truncating input from %d to %d", len(audio), self._max_inp_len
            )
            audio = audio[: self._max_inp_len]
        elif len(audio) < self._max_inp_len:
            self._logger.debug(
                "Padding input from %d to %d", len(audio), self._max_inp_len
            )
            audio = np.pad(audio, (0, self._max_inp_len - len(audio)))
        return audio.reshape(1, -1)

    def _get_token_input(self, token_id: int) -> np.ndarray:
        """Embedding lookup or raw token ID for decoder input."""
        if self._token_embeddings is not None:
            return np.expand_dims(self._token_embeddings[token_id], axis=(0, 1))
        return np.array([[token_id]], dtype=np.int32)

    def _run_encoder(self, audio: np.ndarray) -> list[np.ndarray]:
        enc_info = self._encoder.inputs_info
        if enc_info:
            audio = audio.astype(np.dtype(enc_info[0].dtype), copy=False)
        if self._device_io:
            audio = self._encoder.allocate_device_array(audio)
        enc_out = self._encoder.infer([audio])
        self._logger.debug(
            "Infer '%s': %.3f ms",
            str(self._encoder.model_path), self._encoder.infer_time_ms
        )
        return enc_out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        audio: np.ndarray,
        *,
        max_tokens: int | None = None,
    ) -> np.ndarray:
        """Transcribe audio to a token array.

        Args:
            audio: Raw audio waveform, shape ``(1, N)`` or ``(N,)``.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            Token IDs as ``np.ndarray`` with shape ``(1, T)``.
        """
        self._n_tokens_gen = 0
        audio = self._size_input(audio)

        if max_tokens is None:
            max_tokens = max(1, int((audio.shape[-1] / self._input_freq) * 6))

        start_ns = perf_counter_ns()

        # 1. Encode → cross-attention KV caches
        encoder_out = self._run_encoder(audio)

        # 2. Initialize decoder caches
        dec_info = self._decoder.inputs_info
        cross_cache = list(encoder_out)
        if dec_info:
            # Cast cross-cache to the dtype the decoder expects for encoder KV
            cross_dtype = np.dtype(dec_info[self._decoder._input_cache_start + 2].dtype)
            cross_cache = [
                c if hasattr(c, "to_host") else c.astype(cross_dtype, copy=False)
                for c in cross_cache
            ]
        self._decoder.reset_kv()
        self._decoder.set_cross_cache(cross_cache)

        # 3. Autoregressive decoding (all steps use the unified decoder)
        next_token = _START_TOKEN_ID
        tokens = []

        for i in range(max_tokens):
            token_emb = self._get_token_input(next_token)
            current_len = np.array([[i]], dtype=np.int64)
            if dec_info:
                token_emb = token_emb.astype(
                    np.dtype(dec_info[0].dtype), copy=False
                )
                current_len = current_len.astype(
                    np.dtype(dec_info[1].dtype), copy=False
                )
            [logits] = self._decoder.infer([token_emb, current_len])
            self._logger.debug(
                "Infer '%s': %.3f ms",
                str(self._decoder.model_path), self._decoder.infer_time_ms
            )

            if i == 0:
                self._time_to_first_token_ns = perf_counter_ns() - start_ns

            next_token = int(np.asarray(logits)[0, -1].argmax())
            self._n_tokens_gen += 1
            tokens.append(next_token)

            if next_token == _END_TOKEN_ID:
                break

        self._last_infer_ns = perf_counter_ns() - start_ns
        return np.array([tokens])
