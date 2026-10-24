# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""FunctionGemma protocol on the shared Gemma 3 model runtime."""

from typing import Iterator

from gemma3.src.runner import Gemma3Static
from functiongemma.src.function_tools import (
    DEFAULT_TOOLS,
    START_FUNCTION_RESPONSE,
    format_developer_block,
)
from utils.inference import StopCheck


class FunctionGemmaStatic(Gemma3Static):
    __slots__ = ("_tools", "_function_response_next")

    def __init__(self, *args, tools: list[dict] | None = None, **kwargs):
        self._tools = DEFAULT_TOOLS if tools is None else tools
        self._function_response_next = False
        super().__init__(*args, instruct_model=True, **kwargs)

    def _on_model_config_loaded(self, cfg: dict) -> None:
        super()._on_model_config_loaded(cfg)
        stop = self._tokenizer.token_to_id(START_FUNCTION_RESPONSE)
        markers = (
            "<start_function_call>", "<end_function_call>",
            "<end_function_response>", "<escape>",
        )
        if stop is None or stop not in self._eos_token_ids or any(
            self._tokenizer.token_to_id(marker) is None for marker in markers
        ):
            raise ValueError(
                "FunctionGemma export required: function tokens must exist and "
                f"{START_FUNCTION_RESPONSE} must be a configured stop token"
            )

    @property
    def awaiting_function_response(self) -> bool:
        return self._pending_token is not None and self._pending_token == self._tokenizer.token_to_id(START_FUNCTION_RESPONSE)

    def _decode_generated_tokens(self, tokens: list[int]) -> str:
        # Call delimiters and <escape> are special tokens too.
        return self._tokenizer.decode(tokens, skip_special_tokens=False)

    def _build_warmup_tokens(self) -> list[int]:
        tokens = [self._bos_token_id] + self.tokenize(
            format_developer_block(self._tools), "developer"
        )
        if isinstance(self._max_prompt_tokens, int) and len(tokens) > self._max_prompt_tokens:
            raise ValueError("Developer tool block exceeds --max-inp-len")
        if len(tokens) >= self._max_seq_len:
            raise ValueError("Developer tool block leaves no room in the KV cache")
        return tokens

    def _build_prompt_tokens(self, user_input: str) -> list[int]:
        if self._function_response_next:
            ids = self.tokenize(user_input)
            return ids[1:] if ids and ids[0] == self._bos_token_id else ids
        tokens = super()._build_prompt_tokens(user_input)
        if self._pending_token == self._end_of_turn_id:
            newline = self.tokenize("\n")
            if newline and newline[0] == self._bos_token_id:
                newline = newline[1:]
            # A continued model turn ends with <end_of_turn>\n. A fresh
            # conversation after capacity rollover already has its prefix.
            if self._turn_pos + 1 + len(newline) + len(tokens) <= self._max_seq_len:
                tokens = newline + tokens
        return tokens

    def _apply_prompt_limit(self, tokens: list[int]) -> list[int]:
        limit = self._max_user_tokens
        if limit is None:
            limit = self._max_prompt_tokens
        if isinstance(limit, int) and len(tokens) > limit:
            raise ValueError("FunctionGemma prompt exceeds --max-inp-len")
        pending = int(self._pending_token is not None)
        start = self._turn_pos if self._function_response_next else self._warmup_len
        if start + pending + len(tokens) > self._max_seq_len:
            raise ValueError("FunctionGemma prompt exceeds available KV cache capacity")
        return tokens

    def _roll_cache(self, keep: int, should_stop: StopCheck | None) -> None:
        # A few trailing tokens cannot preserve a tool call or its arguments.
        raise ValueError("FunctionGemma KV cache full; start a new conversation")

    def run_function_response(
        self, response_text: str, should_stop: StopCheck | None = None
    ) -> Iterator[str]:
        """Continue a stopped call, preserving the pending response-start token."""
        if not self.awaiting_function_response:
            raise ValueError("Function response requires a pending function call")
        self._function_response_next = True
        try:
            yield from self.run_stream(response_text, should_stop=should_stop)
        finally:
            self._function_response_next = False
