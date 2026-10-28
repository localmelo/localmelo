"""Ollama native ``/api/chat`` provider with ``think: true`` support."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef, describe_images
from localmelo.support.providers._http import (
    attempt_headers,
    malformed_body,
    post_json,
)
from localmelo.support.providers.llm._protocol import (
    parse_tool_calls,
    prepare_extra_body,
    raw_tool_arguments,
)


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


def _tool_call_to_ollama(tc: ToolCall) -> dict[str, Any]:
    """Serialize a ToolCall for an assistant turn of the native API.

    The native API takes arguments as an object, so arguments that were
    not a JSON object (see :func:`raw_tool_arguments`) are sent as ``{}``.
    """
    arguments = {} if raw_tool_arguments(tc.arguments) is not None else tc.arguments
    return {"function": {"name": tc.tool_name, "arguments": arguments}}


def _message_to_ollama(msg: Message, tool_names: dict[str, str]) -> dict[str, Any]:
    """Serialize a Message for Ollama's native ``/api/chat``.

    Assistant turns are sent back the way Ollama returned them: thinking
    in its own field (the inline ``<think>`` copy this provider adds to
    ``content`` is removed again) and tool calls with object arguments
    (see :func:`_tool_call_to_ollama`). Tool results carry ``tool_name``,
    which is how the native API ties a result to its call. *tool_names*
    maps tool call ids seen earlier in the conversation to their tool
    names.
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
            out["tool_calls"] = [_tool_call_to_ollama(tc) for tc in msg.tool_calls]
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
    """Serialize a conversation for ``/api/chat``.

    Images use the native ``images`` field (raw base64). Like the OpenAI
    provider, images of a tool result go into a user message after the
    run of tool results, labelled with the tool call they belong to.

    Only images with data are sent. Reference-only images (see
    :meth:`ImagePart.as_reference`) are not rendered at all: the label
    describes only the attached images, and a message whose images are
    all references gets no ``images`` field and no extra user message.
    Whoever writes a message's text names its reference-only images
    there; the Agent does, with :func:`describe_images` in the tool
    message.
    """
    tool_names: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    trailing: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role != "tool" and trailing:
            out.extend(trailing)
            trailing = []
        for tc in msg.tool_calls:
            tool_names[tc.id] = tc.tool_name
        entry = _message_to_ollama(msg, tool_names)
        attached = [img for img in msg.images if img.data]
        images = [img.data for img in attached]
        if images and msg.role == "tool":
            label = (
                f"Output of tool call {msg.tool_call_id}: {describe_images(attached)}"
            )
            trailing.append({"role": "user", "content": label, "images": images})
        elif images:
            entry["images"] = images
        out.append(entry)
    out.extend(trailing)
    return out


class OllamaNativeChat(BaseLLMProvider):
    """Ollama native ``/api/chat`` provider.

    Uses the Ollama-specific chat endpoint (NOT the OpenAI-compat layer)
    so that extended-thinking (``think: true``) is supported natively.

    A failed request raises
    :class:`~localmelo.melo.contracts.providers.ProviderError`. Every
    request sends ``LLM_REQUEST_ATTEMPT`` in the ``X-Localmelo-Attempt``
    header, and ``LLM_REQUEST_KIND``, when it is set, in the
    ``X-Localmelo-Call-Kind`` header. After a reply,
    ``last_reasoning_key`` is ``thinking`` when it has thinking, else
    ``""``, and ``last_usage`` is its usage (token counts only).

    Parameters
    ----------
    base_url : str
        Raw Ollama server URL, e.g. ``"http://localhost:11434"``.
        Must **not** include ``/v1``.
    model : str
        Model identifier, e.g. ``"qwen3:8b"``.
    timeout : float
        HTTP timeout in seconds (default 300).
    extra_body : Mapping[str, Any] | None
        Extra top-level fields merged into every ``/api/chat`` body, e.g.
        ``{"think": False}`` or ``{"options": {"temperature": 0}}``.
        ``model``, ``messages``, ``tools`` and ``stream`` cannot be set.
    transport : httpx.AsyncBaseTransport | None
        Optional HTTP transport, e.g. ``httpx.MockTransport`` in tests.
    requires_reasoning_replay : bool
        Whether the server rejects requests that omit a reply's reasoning.
        This provider sends ``thinking`` back anyway when it is present;
        the agent's ``drop`` mode strips it first, and is refused for a
        provider that sets this. ``False`` by default.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 300.0,
        *,
        extra_body: Mapping[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        requires_reasoning_replay: bool = False,
    ) -> None:
        self.model = model
        self.extra_body = prepare_extra_body(extra_body)
        # The native wire field is fixed; it has no "auto" or other key.
        self.reasoning_field = "thinking"
        self.requires_reasoning_replay = requires_reasoning_replay
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
            "think": True,  # extra_body may turn this off
            **self.extra_body,
            "model": self.model,
            "messages": _messages_to_ollama(messages),
            "stream": False,
        }
        if tools:
            payload["tools"] = [_tool_def_to_openai(t) for t in tools]

        path = "/api/chat"
        resp, data = await post_json(
            self._http, path, payload, headers=attempt_headers()
        )
        msg = data.get("message", {}) if isinstance(data, dict) else None
        if not isinstance(msg, dict):
            raise malformed_body(resp, path, "message is not an object")
        thinking = msg.get("thinking") or ""
        content = msg.get("content") or ""

        # Prepend <think> block into content for compatibility with
        # consumers that expect inline thinking tags.
        if thinking:
            combined_content = _inline_thinking_prefix(thinking) + content
        else:
            combined_content = content

        # Set right before the return, with no await in between.
        self.last_reasoning_key = "thinking" if thinking else ""
        self.last_usage = _normalize_usage(data)
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
