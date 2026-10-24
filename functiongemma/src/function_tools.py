# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""FunctionGemma function-calling support.

FunctionGemma checkpoint chat template wraps the conversation in a
developer tool-declaration block and uses dedicated function turns:

* the model emits ``<start_function_call>call:name{arg:value}``
  ``<end_function_call>`` followed by ``<start_function_response>``, which is
  a stop token for the model,
* the host appends ``response:name{...}<end_function_response>`` with no
  turn header, and the model continues from there.

These helpers render and parse that format as plain strings, so the demo
keeps its tokenizers-only dependency.
"""

import logging
import re
from typing import Any, Callable, Final

logger = logging.getLogger("function_tools")

FUNCTIONGEMMA_DEV_INSTRUCTION: Final[str] = (
    "You are a model that can do function calling with the following functions"
)

START_FUNCTION_RESPONSE: Final[str] = "<start_function_response>"


def _format_argument(value: Any) -> str:
    """Render a value the way the checkpoint template does."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f"<escape>{value}<escape>"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        return "{" + ",".join(
            f"{key}:{_format_argument(val)}" for key, val in sorted(value.items())
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_argument(val) for val in value) + "]"
    return str(value)


def format_function_declaration(tool: dict) -> str:
    """Render one tool's ``<start_function_declaration>`` body.

    Supports object parameters with scalar-typed properties, as published for
    FunctionGemma; other parameter types are not rendered.
    """
    fn = tool["function"]
    parts = [
        f"declaration:{fn['name']}{{description:<escape>{fn.get('description', '')}<escape>"
    ]
    params = fn.get("parameters") or {}
    if params:
        inner = []
        props = params.get("properties") or {}
        if props:
            inner.append(
                "properties:{"
                + ",".join(
                    f"{name}:{{description:<escape>{spec.get('description', '')}<escape>,"
                    f"type:<escape>{str(spec.get('type', '')).upper()}<escape>}}"
                    for name, spec in sorted(props.items())
                )
                + "}"
            )
        if params.get("required"):
            inner.append(
                "required:["
                + ",".join(f"<escape>{item}<escape>" for item in params["required"])
                + "]"
            )
        if params.get("type"):
            inner.append(f"type:<escape>{str(params['type']).upper()}<escape>")
        parts.append(",parameters:{" + ",".join(inner) + "}")
    parts.append("}")
    return "".join(parts)


def format_developer_block(tools: list[dict]) -> str:
    """Render the developer block body (instruction + tool declarations).

    The result is one chat message; wrap it in the ``developer`` turn with
    the runner's ``tokenize``.
    """
    declarations = "".join(
        f"<start_function_declaration>{format_function_declaration(tool)}"
        f"<end_function_declaration>"
        for tool in tools
    )
    return FUNCTIONGEMMA_DEV_INSTRUCTION + declarations


def format_function_response(name: str, value: Any) -> str:
    """Render the host-appended function response (no turn header)."""
    if isinstance(value, dict):
        inner = ",".join(f"{key}:{_format_argument(val)}" for key, val in sorted(value.items()))
    else:
        inner = f"value:{_format_argument(value)}"
    return f"response:{name}{{{inner}}}<end_function_response>"


# ── Parsing model-generated function calls ───────────────────────────────────


def _parse_value(text: str, pos: int) -> tuple[Any, int]:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text):
        raise ValueError("Missing argument value")
    ch = text[pos]
    if ch == "{":
        return _parse_mapping(text, pos)
    if ch == "[":
        return _parse_sequence(text, pos)
    if text.startswith("<escape>", pos):
        end = text.index("<escape>", pos + 8)
        return text[pos + 8 : end], end + 8
    end = pos
    while end < len(text) and text[end] not in ",}]:":
        end += 1
    token = text[pos:end].strip()
    if token == "true":
        return True, end
    if token == "false":
        return False, end
    try:
        return int(token), end
    except ValueError:
        pass
    try:
        return float(token), end
    except ValueError:
        pass
    if token in ("null", "None"):
        return None, end
    raise ValueError(f"Invalid argument value: {token!r}")


def _parse_mapping(text: str, pos: int) -> tuple[dict, int]:
    result: dict = {}
    pos += 1  # {
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos < len(text) and text[pos] == "}":
        return result, pos + 1
    while pos < len(text):
        sep = text.index(":", pos)
        key = text[pos:sep].strip()
        value, pos = _parse_value(text, sep + 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key) or key in result:
            raise ValueError(f"Invalid or duplicate argument name: {key!r}")
        result[key] = value
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        if text[pos] == ",":
            pos += 1
            continue
        if text[pos] == "}":
            return result, pos + 1
        raise ValueError(f"Malformed function call arguments near {text[pos : pos + 20]!r}")
    raise ValueError("Unterminated argument mapping")


def _parse_sequence(text: str, pos: int) -> tuple[list, int]:
    items: list = []
    pos += 1  # [
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos < len(text) and text[pos] == "]":
        return items, pos + 1
    while pos < len(text):
        value, pos = _parse_value(text, pos)
        items.append(value)
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        if text[pos] == ",":
            pos += 1
            continue
        if text[pos] == "]":
            return items, pos + 1
        raise ValueError(f"Malformed function call arguments near {text[pos : pos + 20]!r}")
    raise ValueError("Unterminated argument sequence")


_START_CALL = "<start_function_call>"
_END_CALL = "<end_function_call>"


def parse_function_call(text: str) -> tuple[str, dict] | None:
    """Parse one complete call; reject malformed or multiple calls."""
    if text.count(_START_CALL) != 1 or text.count(_END_CALL) != 1:
        return None
    start = text.index(_START_CALL) + len(_START_CALL)
    end = text.index(_END_CALL)
    body = text[start:end]
    if not body.startswith("call:"):
        return None
    name, sep, obj = body[len("call:"):].partition("{")
    if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        return None
    try:
        args, pos = _parse_mapping("{" + obj, 0)
        if pos == len(obj) + 1:
            return name, args
    except (ValueError, RecursionError) as exc:
        logger.warning("Unparsable function call %r: %s", body, exc)
    return None


# ── Built-in demo functions ──────────────────────────────────────────────────

#: Offline temperature dataset; the board has no network access.
_TEMPERATURES: Final[dict[str, int]] = {
    "london": 18,
    "paris": 21,
    "berlin": 19,
    "tokyo": 24,
    "new york": 22,
    "sydney": 15,
    "mumbai": 33,
    "cairo": 29,
    "reykjavik": 11,
    "auckland": 17,
}


def get_temperature(city: str) -> int:
    """Get the temperature in a city."""
    try:
        return _TEMPERATURES[city.lower()]
    except KeyError:
        raise ValueError(f"unknown city '{city}'") from None


#: Tool schemas advertised to the model in the developer block.
DEFAULT_TOOLS: Final[list[dict]] = [
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Get the temperature in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name."},
                },
                "required": ["city"],
            },
        },
    }
]

#: Functions the demo can execute, keyed by the schema's function name.
FUNCTIONS: Final[dict[str, Callable[..., Any]]] = {
    "get_temperature": get_temperature,
}
