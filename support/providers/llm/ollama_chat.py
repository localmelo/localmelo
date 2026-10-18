"""Ollama native ``/api/chat`` provider with ``think: true`` support."""

from __future__ import annotations

from typing import Any

import httpx

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.providers.llm._protocol import parse_tool_calls


def _tool_def_to_openai(td: ToolDef) -> dict[str, Any]:
    """Convert internal ToolDef to OpenAI function-calling format.

    Ollama accepts the same tool schema as OpenAI.
    """
    return {
        "type": "function",
        "function": {
            "name": td.name,
            "description": td.description,
            "parameters": td.parameters,
        },
    }


def _coerce_token_count(value: Any) -> int:
    """Safely coerce a token count to a non-negative int."""
    if value is None:
        return 0
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return max(n, 0)


def _normalize_usage(data: dict[str, Any]) -> dict[str, int] | None:
    """Normalize Ollama native usage fields to standard dict.

    Ollama returns ``prompt_eval_count`` and ``eval_count`` instead of
    the OpenAI-style ``prompt_tokens`` / ``completion_tokens``.
    """
    prompt = _coerce_token_count(data.get("prompt_eval_count"))
    completion = _coerce_token_count(data.get("eval_count"))
    total = prompt + completion
    if not total:
        return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _inline_thinking_prefix(thinking: str) -> str:
    """The ``<think>`` block :class:`OllamaNativeChat` prepends to content."""
    return f"<think>\n{thinking}\n</think>\n\n"


def _message_to_ollama(msg: Message, tool_names: dict[str, str]) -> dict[str, Any]:
    """Serialize a Message for Ollama's native ``/api/chat``.

    Assistant turns are sent back the way Ollama returned them: thinking
    in its own field (the inline ``<think>`` copy this provider adds to
    ``content`` is removed again) and tool calls with object arguments.
    Tool results carry ``tool_name``, which is how the native API ties a
    result to its call. *tool_names* maps tool call ids seen earlier in
    the conversation to their tool names.
    """
    if msg.role == "assistant":
        content = msg.content
        if msg.thinking:
            prefix = _inline_thinking_prefix(msg.thinking)
            if content.startswith(prefix):
                content = content[len(prefix) :]
        out: dict[str, Any] = {"role": "assistant", "content": content}
        if msg.thinking:
            out["thinking"] = msg.thinking
        if msg.tool_calls:
            out["tool_calls"] = [
                {"function": {"name": tc.tool_name, "arguments": tc.arguments}}
                for tc in msg.tool_calls
            ]
        return out
    if msg.role == "tool":
        out = {"role": "tool", "content": msg.content}
        if msg.tool_result is not None and msg.tool_result.tool_name:
            name = msg.tool_result.tool_name
        else:
            name = tool_names.get(msg.tool_call_id, "")
        if name:
            out["tool_name"] = name
        return out
    return {"role": msg.role, "content": msg.content}


def _messages_to_ollama(messages: list[Message]) -> list[dict[str, Any]]:
    tool_names: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for msg in messages:
        for tc in msg.tool_calls:
            tool_names[tc.id] = tc.tool_name
        out.append(_message_to_ollama(msg, tool_names))
    return out


class OllamaNativeChat(BaseLLMProvider):
    """Ollama native ``/api/chat`` provider.

    Uses the Ollama-specific chat endpoint (NOT the OpenAI-compat layer)
    so that extended-thinking (``think: true``) is supported natively.

    Parameters
    ----------
    base_url : str
        Raw Ollama server URL, e.g. ``"http://localhost:11434"``.
        Must **not** include ``/v1``.
    model : str
        Model identifier, e.g. ``"qwen3:8b"``.
    timeout : float
        HTTP timeout in seconds (default 300).
    transport : httpx.AsyncBaseTransport | None
        Optional HTTP transport, e.g. ``httpx.MockTransport`` in tests.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 300.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> Message:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_to_ollama(messages),
            "stream": False,
            "think": True,
        }
        if tools:
            payload["tools"] = [_tool_def_to_openai(t) for t in tools]

        resp = await self._http.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        msg = data.get("message", {})
        thinking = msg.get("thinking") or ""
        content = msg.get("content") or ""

        # Prepend <think> block into content for compatibility with
        # consumers that expect inline thinking tags.
        if thinking:
            combined_content = _inline_thinking_prefix(thinking) + content
        else:
            combined_content = content

        return Message(
            role="assistant",
            content=combined_content,
            # Every call is kept; ids are generated when Ollama sends none.
            tool_calls=parse_tool_calls(msg.get("tool_calls")),
            usage=_normalize_usage(data),
            thinking=thinking,
        )

    async def close(self) -> None:
        await self._http.aclose()
