# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import os
from typing import Final

from utils.llm import (
    DecoderOnlyLLMRunner,
)

DEFAULT_SYS_PROMPT: Final[str] = (
    "You are a helpful AI assistant named Gemma. "
    "Answer in 1-2 sentences. No lists, no bullet points, no repetition."
)


class Gemma3Static(DecoderOnlyLLMRunner):
    __slots__ = (
        "_sys_prompt",
        "_n_layers",
        "_n_kv_heads",
        "_head_dim",
        "_instruct_model",
        "_nl_token_id",
        "_double_nl_token_id",
        "_end_of_turn_id",
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
        runtime_flags: list[str] | None = None,
        device_io: bool = False,
        sys_prompt: str | None = None,
        lm_head_path: str | os.PathLike | None = None,
        disable_lm_head: bool = False,
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
            runtime_flags=runtime_flags,
            device_io=device_io,
            lm_head_path=lm_head_path,
            disable_lm_head=disable_lm_head,
        )

    @property
    def is_instruct_model(self) -> bool:
        return self._instruct_model

    def _on_model_config_loaded(self, cfg: dict) -> None:
        self._n_layers = cfg["num_hidden_layers"]
        self._n_kv_heads = cfg["num_key_value_heads"]
        self._head_dim = cfg["head_dim"]
        self._nl_token_id = self._tokenizer.encode("\n").ids[-1]
        self._double_nl_token_id = self._tokenizer.encode("\n\n").ids[-1]
        self._end_of_turn_id = self._tokenizer.token_to_id("<end_of_turn>")

    def tokenize(self, text: str, role: str | None = None) -> list[int]:
        if not self._instruct_model or role is None:
            return self._tokenizer.encode(text).ids
        # Gemma 3 chat format: <start_of_turn>role\ntext<end_of_turn>\n
        # BOS is added once at warmup start; strip auto-prepended BOS here.
        if role == "model":
            ids = self._tokenizer.encode("<start_of_turn>model\n").ids
        else:
            ids = self._tokenizer.encode(
                "<start_of_turn>" + role + "\n" + text + "<end_of_turn>\n"
            ).ids
        if ids and ids[0] == self._bos_token_id:
            ids = ids[1:]
        return ids

    def _build_prompt_tokens(self, user_input: str) -> list[int]:
        tokens = self.tokenize(user_input, "user")
        if self._instruct_model:
            tokens += self.tokenize("", "model")
        return tokens

    def _build_warmup_tokens(self) -> list[int]:
        if not self._instruct_model:
            return []
        return [self._bos_token_id] + self.tokenize(self._sys_prompt or "", "system")

    def _should_stop(self, token: int, gen: list[int]) -> bool:
        if token == self._eos_token_id:
            return True
        if self._end_of_turn_id is not None and token == self._end_of_turn_id:
            return True
        if not self._instruct_model and len(gen) > 2:
            if token == self._double_nl_token_id:
                return True
            return all(t == self._nl_token_id for t in gen[-2:])
        return False


def format_answer(
    answer: str,
    infer_time: float,
    ttft: float,
    stats: list[str] | None = None,
    agent_name: str = "Agent",
) -> str:
    GREEN: Final[str] = "\033[32m"
    RESET: Final[str] = "\033[0m"
    metrics = [f"{infer_time:.3f} ms", f"TTFT: {ttft:.3f} ms"]
    metrics.extend(str(s) for s in (stats or []))
    return GREEN + f"{agent_name}: {answer}" + RESET + f" ({', '.join(metrics)})"
