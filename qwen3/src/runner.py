# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import os
from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import Final

from utils.llm import CHATML, LLM
from utils.runtime import RuntimeConfig

__all__ = ["Qwen3Static", "hide_thinking"]

DEFAULT_SYS_PROMPT: Final[str] = (
    "You are a helpful assistant. "
    "Answer in 1-2 sentences. No lists, no bullet points, no repetition."
)

# With thinking off, the answer starts with an empty think block, as
# apply_chat_template(..., enable_thinking=False) does.
CHATML_NO_THINKING = replace(
    CHATML, generation_prompt=CHATML.generation_prompt + "<think>\n\n</think>\n\n"
)


class Qwen3Static(LLM):
    """Qwen3 chat on the shared :class:`~utils.llm.LLM` with the ChatML template.

    With ``thinking=True`` (the default) the model first writes its reasoning
    in a ``<think>...</think>`` block and then the answer (see
    :func:`hide_thinking`). With ``thinking=False`` it answers at once.
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        max_seq_len: int | None = None,
        *,
        runtime: RuntimeConfig,
        instruct_model: bool = False,
        sys_prompt: str | None = None,
        thinking: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_path, runtime, template=CHATML if thinking else CHATML_NO_THINKING,
            chat=instruct_model, sys_prompt=sys_prompt or DEFAULT_SYS_PROMPT,
            max_seq_len=max_seq_len, **kwargs,
        )


def hide_thinking(chunks: Iterable[str]) -> Iterator[str]:
    """Yield only the answer of a thinking-mode response.

    The text up to and including ``</think>`` is held back. If the response
    ends before ``</think>`` (the context is full while the model thinks), one
    note is yielded instead of an empty answer.
    """
    buf = ""
    answering = False
    started = False  # the blank lines after </think> are not part of the answer
    for chunk in chunks:
        if not answering:
            buf += chunk
            end = buf.find("</think>")
            if end < 0:
                continue
            answering = True
            chunk = buf[end + len("</think>"):]
        if not started:
            chunk = chunk.lstrip()
            started = bool(chunk)
        if chunk:
            yield chunk
    if not answering:
        yield "[No answer: the context was full before the model finished thinking.]"
