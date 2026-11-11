# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import os
from dataclasses import replace
from typing import Final

from utils.llm import CHATML, LLM
from utils.runtime import RuntimeConfig

DEFAULT_SYS_PROMPT: Final[str] = (
    "You are a helpful AI assistant. "
    "Answer in 1-2 sentences. No lists, no bullet points, no repetition."
)

# ChatML turns; the warm-up opens with <|startoftext|> (BOS), which the
# tokenizer also prepends to every encode, and only the config EOS stops.
LFM2_CHATML = replace(CHATML, bos_prefix=True, strip_bos=True, stop_tokens=())


class LiquidStatic(LLM):
    """LFM2.5 (LiquidAI) chat on the shared :class:`~utils.llm.LLM`.

    LFM2.5 is a hybrid model: each layer is either a conv block (a sliding
    ``past_conv.N`` state) or an attention block (a ``past_key_values.N``
    cache). The KV cache manager is agnostic to what each cached tensor
    represents, so the mixed caches need no special handling, and the newer
    exports' fixed all-ones ``attention_mask`` input is built from the model
    metadata. The sliding KV window works too: rolling replays the kept
    tokens, which rebuilds the conv states along with the KV caches.
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
            model_path, runtime, template=LFM2_CHATML, chat=instruct_model,
            sys_prompt=sys_prompt or DEFAULT_SYS_PROMPT, max_seq_len=max_seq_len, **kwargs,
        )
