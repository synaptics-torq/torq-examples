# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import os
from typing import Final

from utils.llm import DecoderOnlyLLMRunner, InferenceInterrupted

__all__ = ["QwenStatic", "InferenceInterrupted"]

DEFAULT_SYS_PROMPT: Final[str] = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
    "Answer in 1-2 sentences. No lists, no bullet points, no repetition."
)


class QwenStatic(DecoderOnlyLLMRunner):
    """Qwen2.5 inference runner for static-shape Torq VMFBs.

    The ``torq-export-model qwen --split-lm-head`` export takes
    ``token_embedding`` ``[1, 1, 896]`` (a CPU-side lookup into
    ``token_embeddings.npy``) and ``position_ids`` ``[1, 1]``, followed by one
    combined ``past_key_values.N.key_value`` cache ``[1, 4, 256, 64]`` per
    layer. The transformer outputs the hidden state, and the sibling
    ``lm_head.vmfb`` turns it into logits over the trimmed vocabulary, which
    ``token_id_lut.npy`` maps back to tokenizer IDs. All of this is handled by
    :class:`~utils.llm.DecoderOnlyLLMRunner`; this class only adds Qwen's
    ChatML template and stop tokens.
    """

    __slots__ = (
        "_instruct_model",
        "_sys_prompt",
        "_nl_token_id",
        "_double_nl_token_id",
        "_im_end_id",
        "_endoftext_id",
    )

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
        max_gen_tokens: int | None = None,
        runtime_flags: list[str] | None = None,
        device_io: bool = False,
        sys_prompt: str | None = None,
        lm_head_path: str | os.PathLike | None = None,
        disable_lm_head: bool = False,
        prefill_model_path: str | os.PathLike | None = None,
        disable_prefill: bool = False,
    ):
        self._instruct_model = instruct_model
        self._sys_prompt = (sys_prompt or DEFAULT_SYS_PROMPT) if instruct_model else None
        super().__init__(
            model_path,
            max_seq_len=max_seq_len,
            max_prompt_tokens=max_prompt_tokens,
            n_threads=n_threads,
            cache_keep_n=cache_keep_n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_gen_tokens=max_gen_tokens,
            runtime_flags=runtime_flags,
            device_io=device_io,
            lm_head_path=lm_head_path,
            disable_lm_head=disable_lm_head,
            prefill_model_path=prefill_model_path,
            disable_prefill=disable_prefill,
        )

    @property
    def is_instruct_model(self) -> bool:
        return self._instruct_model

    def _on_model_config_loaded(self, cfg: dict) -> None:
        self._nl_token_id = self._tokenizer.encode("\n").ids[-1]
        self._double_nl_token_id = self._tokenizer.encode("\n\n").ids[-1]
        self._im_end_id = self._tokenizer.token_to_id("<|im_end|>")
        self._endoftext_id = self._tokenizer.token_to_id("<|endoftext|>")

    def tokenize(self, text: str, role: str | None = None) -> list[int]:
        if not self._instruct_model or role is None:
            return self._tokenizer.encode(text).ids
        # Qwen ChatML format: <|im_start|>role\ntext<|im_end|>\n
        # Qwen uses no BOS token (config bos_token_id is <|endoftext|>).
        if role == "assistant":
            return self._tokenizer.encode("<|im_start|>assistant\n").ids
        return self._tokenizer.encode(
            "<|im_start|>" + role + "\n" + text + "<|im_end|>\n"
        ).ids

    def _build_prompt_tokens(self, user_input: str) -> list[int]:
        tokens = self.tokenize(user_input, "user")
        if self._instruct_model:
            tokens += self.tokenize("", "assistant")
        return tokens

    def _build_warmup_tokens(self) -> list[int]:
        if not self._instruct_model:
            return []
        return self.tokenize(self._sys_prompt or "", "system")

    def _should_stop(self, token: int, gen: list[int]) -> bool:
        if token in (self._eos_token_id, self._im_end_id, self._endoftext_id):
            return True
        if not self._instruct_model and len(gen) > 2:
            if token == self._double_nl_token_id:
                return True
            return all(t == self._nl_token_id for t in gen[-2:])
        return False
