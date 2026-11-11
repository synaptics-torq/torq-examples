# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import logging
import os
from pathlib import Path
from time import perf_counter_ns
from typing import Final

import numpy as np

from utils.cache import KVCache
from utils.decoder import Decoder, generate, sample
from utils.runtime import Model, RuntimeConfig, load_lut

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
        "_cache",
        "_cross_dtype",
        "_max_inp_len",
        "_input_freq",
        "_n_tokens_gen",
        "_last_infer_ns",
        "_time_to_first_token_ns",
    )

    def __init__(
        self,
        model_dir: str | os.PathLike,
        *,
        input_freq: int = _DEFAULT_INPUT_FREQ,
        runtime: RuntimeConfig,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._debug_logging: bool = self._logger.isEnabledFor(logging.DEBUG)
        model_dir = Path(model_dir)
        self._model_dir = model_dir

        encoder_path = model_dir / "encoder.vmfb"
        decoder_path = model_dir / "decoder.vmfb"
        for req in (encoder_path, decoder_path):
            if not req.is_file():
                raise ValueError(f"Missing required model '{req.name}' in {model_dir}")

        self._encoder = Model(encoder_path, runtime)

        # The decoder body takes [token, current_len, per layer:
        # self_v, self_k, cross_v, cross_k] and returns
        # [logits, per layer: self_v, self_k]; the cross slots are fixed
        # encoder state, staged via the cache's set_fixed.
        body = Model(decoder_path, runtime, device_outputs=True)
        in_info, out_info = body.inputs_info, body.outputs_info
        if in_info is None or out_info is None:
            raise ValueError(
                f"Model '{decoder_path}' is missing input/output metadata "
                "required for KV cache management."
            )
        n_layers = (len(out_info) - 1) // 2
        if len(in_info) != 2 + 4 * n_layers:
            raise ValueError(
                f"Unexpected decoder input layout in '{decoder_path}': "
                f"expected 2 user inputs plus 4 cache inputs per layer "
                f"({2 + 4 * n_layers} for {n_layers} layers), got {len(in_info)}."
            )
        self._cache = KVCache(body, fixed=[i % 4 >= 2 for i in range(4 * n_layers)])
        # Cross inputs are the 3rd/4th cache input of the first layer.
        self._cross_dtype = np.dtype(in_info[4].dtype)
        emb_paths = list(model_dir.glob("*token_embeddings.npy"))
        self._decoder = Decoder(
            body, self._cache, embed_lut=load_lut(emb_paths[0]) if emb_paths else None
        )

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

    def _run_encoder(self, audio: np.ndarray) -> list[np.ndarray]:
        enc_info = self._encoder.inputs_info
        if enc_info:
            audio = audio.astype(np.dtype(enc_info[0].dtype), copy=False)
        enc_out = self._encoder.infer([audio])
        self._logger.debug(
            "Infer '%s': %.3f ms",
            str(self._encoder.path), self._encoder.infer_time_ms
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

        # 1. Encode → cross-attention KV. Host outputs are cast to the fixed
        #    slots' dtype; device outputs pass through.
        cross = [
            c if not isinstance(c, np.ndarray) else c.astype(self._cross_dtype, copy=False)
            for c in self._run_encoder(audio)
        ]

        # 2. Reset the self caches and stage the fixed cross slots.
        self._cache.reset()
        self._cache.set_fixed(cross)

        if max_tokens < 1:
            self._last_infer_ns = perf_counter_ns() - start_ns
            return np.array([[]])

        # 3. First token (START at position 0), then the shared decode loop.
        first = sample(self._decoder.step(_START_TOKEN_ID, 0))
        self._time_to_first_token_ns = perf_counter_ns() - start_ns

        tokens = list(
            generate(
                self._decoder,
                first,
                1,
                next_token=sample,
                stop=lambda token, _gen: token == _END_TOKEN_ID,
                max_new=max_tokens,
            )
        )
        self._n_tokens_gen = len(tokens)

        self._last_infer_ns = perf_counter_ns() - start_ns
        return np.array([tokens])
