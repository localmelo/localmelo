"""Ollama native ``/api/chat`` provider with ``think: true`` support."""

from __future__ import annotations

import json
from typing import Any

import httpx

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef


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
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> Message:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
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
            combined_content = f"<think>\n{thinking}\n</think>\n\n{content}"
        else:
            combined_content = content

        # Parse tool calls (Ollama uses the same format as OpenAI).
        tool_call = None
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            tool_call = ToolCall(tool_name=fn["name"], arguments=args)

        return Message(
            role="assistant",
            content=combined_content,
            tool_call=tool_call,
            usage=_normalize_usage(data),
            thinking=thinking,
        )

    async def close(self) -> None:
        await self._http.aclose()
