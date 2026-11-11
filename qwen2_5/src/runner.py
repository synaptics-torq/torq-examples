# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import os
from typing import Final

from utils.llm import CHATML, LLM
from utils.runtime import RuntimeConfig

DEFAULT_SYS_PROMPT: Final[str] = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
    "Answer in 1-2 sentences. No lists, no bullet points, no repetition."
)


class QwenStatic(LLM):
    """Qwen2.5 chat on the shared :class:`~utils.llm.LLM` with the ChatML template.

    The ``torq-export-model qwen --split-lm-head`` export takes the token
    embedding (a CPU-side lookup into ``token_embeddings.npy``) and
    ``position_ids``, followed by one combined ``past_key_values.N.key_value``
    cache per layer. The sibling ``lm_head.vmfb.trim`` turns the hidden state
    into logits over the trimmed vocabulary, which ``token_id_lut.npy`` maps
    back to tokenizer IDs. ``LLM`` handles all of this.
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        max_seq_len: int | None = None,
        *,
        runtime: RuntimeConfig,
        instruct_model: bool = False,
        sys_prompt: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model_path, runtime, template=CHATML, chat=instruct_model,
            sys_prompt=sys_prompt or DEFAULT_SYS_PROMPT, max_seq_len=max_seq_len, **kwargs,
        )
