# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import os
from typing import Final

from utils.llm import DecoderOnlyLLMRunner, InferenceInterrupted

__all__ = ["LiquidStatic", "InferenceInterrupted"]

DEFAULT_SYS_PROMPT: Final[str] = (
    "You are a helpful AI assistant. "
    "Answer in 1-2 sentences. No lists, no bullet points, no repetition."
)


class LiquidStatic(DecoderOnlyLLMRunner):
    """LFM2.5 (LiquidAI) inference runner for static-shape Torq VMFBs.

    LFM2.5 is a hybrid model — each layer is either a conv block (sliding
    ``past_conv.N`` state ``[1, 1024, 3]``) or an attention block (combined
    ``past_key_values.X.key_value`` cache ``[1, 16, 256, 64]``). The shared
    :class:`~utils.llm.DecoderOnlyLLMRunner` cache manager is agnostic to what
    each cached tensor represents: it zero-inits every per-layer cache from the
    model's input-shape metadata and shuttles each output back to its matching
    input, so the mixed conv/KV caches thread correctly without special-casing.

    The VMFB takes two non-cache inputs — ``token_embedding`` ``[1, 1, 1024]``
    (a CPU-side embedding-LUT lookup; the VMFB has no 65 K embedding table) and
    ``position_ids`` ``[1, 1]`` — followed by the per-layer cache inputs. With
    ``lm_head_path`` the decoder is split into a body (hidden output) and a
    standalone lm_head, so prefill skips the ``[1024, 65536]`` lm_head (lower
    TTFT, no decode-throughput cost).
    """

    __slots__ = ("_instruct_model", "_sys_prompt", "_nl_token_id", "_double_nl_token_id")

    def __init__(
        self,
        model_path: str | os.PathLike,
        max_seq_len: int | None = None,
        max_prompt_tokens: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = False,
        *,
        cache_keep_n: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 64,
        runtime_flags: list[str] | None = None,
        device_io: bool = False,
        sys_prompt: str | None = None,
        lm_head_path: str | os.PathLike | None = None,
        disable_lm_head: bool = False,
    ):
        self._instruct_model = instruct_model
        self._sys_prompt = (sys_prompt or DEFAULT_SYS_PROMPT) if instruct_model else None
        # LFM2.5's caches are a mix of rank-4 KV windows and rank-3 conv states.
        # The shared sliding-window shift (shift_kv) slices a KV seq axis and is
        # not safe for the conv caches, so it stays off: generation stops at the
        # 256-token KV limit. cache_keep_n is accepted for CLI compatibility but
        # not applied (matches the pre-refactor behaviour).
        super().__init__(
            model_path,
            max_seq_len=max_seq_len,
            max_prompt_tokens=max_prompt_tokens,
            n_threads=n_threads,
            cache_keep_n=None,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            runtime_flags=runtime_flags,
            device_io=device_io,
            lm_head_path=lm_head_path,
            disable_lm_head=disable_lm_head,
        )

    @property
    def is_instruct_model(self) -> bool:
        return self._instruct_model

    def _query_model_seq_len(self) -> int | None:
        """Sequence length is the KV-cache window (axis 2 of a rank-4
        ``past_key_values.*`` input). Overrides the base, which reads
        ``inputs_info[2]`` — for LFM2.5 that slot is a conv cache
        ``[1, 1024, 3]`` (axis 2 == 3), not the 256-token KV window."""
        info = self._model.inputs_info
        if info is None:
            return None
        for t in info:
            if len(t.shape) == 4 and isinstance(t.shape[2], int):
                return t.shape[2]
        return None

    def _on_model_config_loaded(self, cfg: dict) -> None:
        self._nl_token_id = self._tokenizer.encode("\n").ids[-1]
        self._double_nl_token_id = self._tokenizer.encode("\n\n").ids[-1]

    def tokenize(self, text: str, role: str | None = None) -> list[int]:
        if not self._instruct_model or role is None:
            return self._tokenizer.encode(text).ids
        # LFM2.5 ChatML format: <|im_start|>role\ntext<|im_end|>\n
        # BOS (<|startoftext|>) is added once at warmup start; strip any
        # auto-prepended BOS here.
        if role == "assistant":
            ids = self._tokenizer.encode("<|im_start|>assistant\n").ids
        else:
            ids = self._tokenizer.encode(
                "<|im_start|>" + role + "\n" + text + "<|im_end|>\n"
            ).ids
        if ids and ids[0] == self._bos_token_id:
            ids = ids[1:]
        return ids

    def _build_prompt_tokens(self, user_input: str) -> list[int]:
        tokens = self.tokenize(user_input, "user")
        if self._instruct_model:
            tokens += self.tokenize("", "assistant")
        return tokens

    def _build_warmup_tokens(self) -> list[int]:
        if not self._instruct_model:
            return []
        # LFM2.5 format: <|startoftext|><|im_start|>system\n{sys}<|im_end|>\n
        return [self._bos_token_id] + self.tokenize(self._sys_prompt or "", "system")

    def _should_stop(self, token: int, gen: list[int]) -> bool:
        if token == self._eos_token_id:
            return True
        if not self._instruct_model and len(gen) > 2:
            if token == self._double_nl_token_id:
                return True
            return all(t == self._nl_token_id for t in gen[-2:])
        return False
