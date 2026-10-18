"""Wire-format helpers shared by the chat providers."""

from __future__ import annotations

import json
from typing import Any

from localmelo.melo.schema import ToolCall, new_tool_call_id

# Key used to keep tool-call arguments that are not a JSON object. The
# call is still executed through the normal executor path, which reports
# the unexpected argument back to the model instead of the whole run
# crashing on a parse error.
RAW_ARGUMENTS_KEY = "_raw_arguments"


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Decode a tool call's ``function.arguments`` into a dict.

    The OpenAI spec sends a JSON-encoded string; Ollama's native API
    sends an object. ``None`` and empty strings mean "no arguments".
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {RAW_ARGUMENTS_KEY: raw}
        if isinstance(value, dict):
            return value
        return {RAW_ARGUMENTS_KEY: raw}
    return {RAW_ARGUMENTS_KEY: raw}


def parse_tool_calls(raw: Any) -> list[ToolCall]:
    """Parse every entry of a response's ``tool_calls`` array.

    Backend ids are kept. A missing, non-string or repeated id is replaced
    by a generated one so every call can be answered by exactly one
    ``role="tool"`` message. An entry without a function name is kept with
    an empty name so ``Checker.post_plan`` rejects the response, rather
    than the call silently disappearing.
    """
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    seen_ids: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        fn = item.get("function")
        if not isinstance(fn, dict):
            fn = {}
        name = fn.get("name")
        call_id = item.get("id")
        if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
            call_id = new_tool_call_id()
        seen_ids.add(call_id)
        calls.append(
            ToolCall(
                tool_name=name if isinstance(name, str) else "",
                arguments=parse_tool_arguments(fn.get("arguments")),
                id=call_id,
            )
        )
    return calls
