from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.providers.llm._protocol import (
    parse_tool_calls,
    prepare_extra_body,
)


def _tool_def_to_openai(td: ToolDef) -> dict[str, Any]:
    """Convert internal ToolDef to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": td.name,
            "description": td.description,
            "parameters": td.parameters,
        },
    }


def _tool_call_to_openai(tc: ToolCall) -> dict[str, Any]:
    """Serialize a ToolCall as an OpenAI ``tool_calls[]`` entry."""
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.tool_name,
            # The spec carries arguments as a JSON-encoded string.
            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
        },
    }


def _message_to_openai(msg: Message) -> dict[str, Any]:
    """Serialize a Message as an OpenAI chat message.

    Only spec fields are sent: ``thinking``, ``usage``, ``timestamp`` and
    ``tool_result`` stay local. An assistant message with tool calls keeps
    its (possibly empty) text and lists every call; a tool result names
    the call it answers through ``tool_call_id``.
    """
    if msg.role == "assistant" and msg.tool_calls:
        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [_tool_call_to_openai(tc) for tc in msg.tool_calls],
        }
    if msg.role == "tool":
        out: dict[str, Any] = {"role": "tool"}
        if msg.tool_call_id:
            out["tool_call_id"] = msg.tool_call_id
        out["content"] = msg.content
        return out
    return {"role": msg.role, "content": msg.content}


def _reasoning_text(msg: dict[str, Any]) -> str:
    """Return the reasoning text a server split out of the answer, if any.

    vLLM-style reasoning parsers (and oMLX) use ``reasoning_content``;
    newer servers use ``reasoning``.
    """
    for key in ("reasoning_content", "reasoning"):
        value = msg.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _coerce_token_count(value: Any) -> int:
    """Safely coerce a token count to a non-negative int.

    Handles None, missing, string-encoded ints, floats, and negatives.
    Returns 0 for anything unparseable or negative.
    """
    if value is None:
        return 0
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return max(n, 0)


def _normalize_usage(raw_usage: Any) -> dict[str, int] | None:
    """Normalize a raw OpenAI-style usage dict into a clean ``dict[str, int]``.

    Rules:
    - If *raw_usage* is not a dict, returns ``None``.
    - Each token field is coerced via :func:`_coerce_token_count`.
    - If ``total_tokens`` is missing or zero but prompt/completion are
      present, it is back-filled as their sum.
    """
    if not isinstance(raw_usage, dict):
        return None
    prompt = _coerce_token_count(raw_usage.get("prompt_tokens"))
    completion = _coerce_token_count(raw_usage.get("completion_tokens"))
    total = _coerce_token_count(raw_usage.get("total_tokens"))
    if total == 0 and (prompt or completion):
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


class OpenAICompatLLM(BaseLLMProvider):
    """OpenAI-compatible chat/completion provider.

    Works with any backend that exposes ``POST /chat/completions``
    in the OpenAI format:

    - **Local**: MLC LLM, Ollama, vLLM, LMStudio, llama.cpp server
    - **Cloud**: OpenAI, Groq, Together AI, Fireworks, DeepSeek, Mistral

    Parameters
    ----------
    base_url : str
        Server base URL **including** the ``/v1`` prefix,
        e.g. ``"http://localhost:11434/v1"`` or ``"https://api.openai.com/v1"``.
    model : str
        Model identifier sent in the ``"model"`` field.
    api_key : str | None
        Bearer token for authenticated endpoints.
    timeout : float
        HTTP timeout in seconds (default 120).
    extra_body : Mapping[str, Any] | None
        Extra top-level fields merged into every request body, e.g.
        ``{"chat_template_kwargs": {"enable_thinking": False}}``. Fields
        the provider owns (``model``, ``messages``, ``tools``,
        ``tool_choice``, ``stream``) are rejected with ``ValueError``.
    transport : httpx.AsyncBaseTransport | None
        Optional HTTP transport, e.g. ``httpx.MockTransport`` in tests.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        *,
        extra_body: Mapping[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.extra_body = prepare_extra_body(extra_body)
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers=headers,
            transport=transport,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> Message:
        payload: dict[str, Any] = {
            **self.extra_body,
            "model": self.model,
            "messages": [_message_to_openai(m) for m in messages],
        }
        if tools:
            payload["tools"] = [_tool_def_to_openai(t) for t in tools]
            payload["tool_choice"] = "auto"

        resp = await self._http.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]

        return Message(
            role="assistant",
            content=msg.get("content") or "",
            tool_calls=parse_tool_calls(msg.get("tool_calls")),
            usage=_normalize_usage(data.get("usage")),
            thinking=_reasoning_text(msg),
        )

    async def close(self) -> None:
        await self._http.aclose()
