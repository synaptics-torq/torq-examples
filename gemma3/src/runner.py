# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import os
from typing import Final

from utils.llm import GEMMA, LLM
from utils.runtime import RuntimeConfig

DEFAULT_SYS_PROMPT: Final[str] = (
    "You are a helpful AI assistant named Gemma. "
    "Answer in 1-2 sentences. No lists, no bullet points, no repetition."
)


class Gemma3Static(LLM):
    """Gemma 3 chat on the shared :class:`~utils.llm.LLM` with the Gemma chat template."""

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
            model_path, runtime, template=GEMMA, chat=instruct_model,
            sys_prompt=sys_prompt or DEFAULT_SYS_PROMPT, max_seq_len=max_seq_len, **kwargs,
        )
