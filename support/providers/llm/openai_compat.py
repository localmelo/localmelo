from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from localmelo.melo.contracts.providers import LLM_REQUEST_KIND, BaseLLMProvider
from localmelo.melo.schema import ImagePart, Message, ToolCall, ToolDef, describe_images
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
    """Serialize a ToolCall as an OpenAI ``tool_calls[]`` entry.

    Arguments that were not a JSON object (see :func:`raw_tool_arguments`)
    are sent as ``{}``. Servers that check the calls in a request's history
    need an object there: SGLang rejects the whole request otherwise, and
    vLLM replaces such arguments with ``{}`` itself.
    """
    arguments = {} if raw_tool_arguments(tc.arguments) is not None else tc.arguments
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.tool_name,
            # The spec carries arguments as a JSON-encoded string.
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _image_parts(images: Sequence[ImagePart]) -> list[dict[str, Any]]:
    """``image_url`` content parts (base64 data URLs) for images with data."""
    return [
        {"type": "image_url", "image_url": {"url": img.data_url}}
        for img in images
        if img.data
    ]


def _message_to_openai(
    msg: Message, *, reasoning_key: str | None = None
) -> dict[str, Any]:
    """Serialize a Message as an OpenAI chat message.

    Only spec fields are sent: ``usage``, ``timestamp`` and ``tool_result``
    stay local. ``thinking`` stays local unless *reasoning_key* is set, in
    which case an assistant message with non-empty thinking adds it under
    exactly that one key (see :func:`_messages_to_openai`). An assistant
    message with tool calls keeps its (possibly empty) text and lists every
    call; a tool result names the call it answers through ``tool_call_id``.
    A user message with images that have data becomes a content array of
    ``text`` and ``image_url`` parts. Images on other roles are not sent
    here (see :func:`_messages_to_openai` for tool results and for images
    without data; the spec allows images only in user content).
    """
    if msg.role == "user" and msg.images:
        parts = _image_parts(msg.images)
        if parts:
            text = [{"type": "text", "text": msg.content}] if msg.content else []
            return {"role": "user", "content": text + parts}
    if msg.role == "assistant" and msg.tool_calls:
        out: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [_tool_call_to_openai(tc) for tc in msg.tool_calls],
        }
        if reasoning_key and msg.thinking:
            out[reasoning_key] = msg.thinking
        return out
    if msg.role == "tool":
        result: dict[str, Any] = {"role": "tool"}
        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        result["content"] = msg.content
        return result
    plain: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if reasoning_key and msg.role == "assistant" and msg.thinking:
        plain[reasoning_key] = msg.thinking
    return plain


def _tool_images_message(msg: Message, tool_names: dict[str, str]) -> dict[str, Any]:
    """The user message that carries the images of one tool result.

    Its label names the tool call and describes only the images it
    attaches, the ones with data.
    """
    attached = [img for img in msg.images if img.data]
    name = tool_names.get(msg.tool_call_id, "")
    if not name and msg.tool_result is not None:
        name = msg.tool_result.tool_name
    label = f"Output of tool call {msg.tool_call_id}"
    if name:
        label += f" ({name})"
    label += f": {describe_images(attached)}"
    return {
        "role": "user",
        "content": [{"type": "text", "text": label}, *_image_parts(attached)],
    }


def _messages_to_openai(
    messages: Sequence[Message], *, reasoning_key: str | None = None
) -> list[dict[str, Any]]:
    """Serialize a conversation, moving tool-result images into user turns.

    OpenAI tool messages are text-only, and oMLX 0.3.9 drops image parts
    from ``role="tool"`` messages (it places image tokens on user turns
    only, the same way it converts images inside Anthropic tool_result
    blocks). So the images of each tool result travel in their own
    ``role="user"`` message, placed right after the run of tool results
    and labelled with the tool call they belong to.

    Only images with data are sent. Reference-only images (see
    :meth:`ImagePart.as_reference`) are not rendered at all: the label
    describes only the attached images, and a message whose images are
    all references is sent as text alone, with no extra user message.
    Whoever writes a message's text names its reference-only images
    there; the Agent does, with :func:`describe_images` in the tool
    message.
    """
    out: list[dict[str, Any]] = []
    trailing: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}
    for msg in messages:
        if msg.role != "tool" and trailing:
            out.extend(trailing)
            trailing = []
        for tc in msg.tool_calls:
            tool_names[tc.id] = tc.tool_name
        out.append(_message_to_openai(msg, reasoning_key=reasoning_key))
        if msg.role == "tool" and _image_parts(msg.images):
            trailing.append(_tool_images_message(msg, tool_names))
    out.extend(trailing)
    return out


def _choice_message(data: Any) -> dict[str, Any] | None:
    """``choices[0].message`` of a chat completion, or ``None`` if it has none."""
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    return message if isinstance(message, dict) else None


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


def _reasoning_key(msg: dict[str, Any]) -> str:
    """The key that :func:`_reasoning_text` takes the reasoning from, or ``""``."""
    for key in ("reasoning_content", "reasoning"):
        value = msg.get(key)
        if isinstance(value, str) and value:
            return key
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


# Usage fields beyond the three of _normalize_usage, each kept only when the
# server sends it: name -> (the usage object that holds it, its key there).
_USAGE_DETAILS = {
    "cached_tokens": ("prompt_tokens_details", "cached_tokens"),
    "reasoning_tokens": ("completion_tokens_details", "reasoning_tokens"),
}


def _usage_details(raw_usage: Any) -> dict[str, int] | None:
    """The whole usage of a reply, for ``last_usage``.

    ``None`` when *raw_usage* is not a dict, else the three keys of
    :func:`_normalize_usage`, plus ``cached_tokens`` from
    ``prompt_tokens_details.cached_tokens`` and ``reasoning_tokens`` from
    ``completion_tokens_details.reasoning_tokens``, each only when the
    server sent it, coerced with :func:`_coerce_token_count`.
    """
    usage = _normalize_usage(raw_usage)
    if usage is None:
        return None
    for name, (group, key) in _USAGE_DETAILS.items():
        details = raw_usage.get(group)
        if isinstance(details, dict) and details.get(key) is not None:
            usage[name] = _coerce_token_count(details[key])
    return usage


class OpenAICompatLLM(BaseLLMProvider):
    """OpenAI-compatible chat/completion provider.

    Works with any backend that exposes ``POST /chat/completions``
    in the OpenAI format:

    - **Local**: MLC LLM, Ollama, vLLM, LMStudio, llama.cpp server
    - **Cloud**: OpenAI, Groq, Together AI, Fireworks, DeepSeek, Mistral

    A failed request raises
    :class:`~localmelo.melo.contracts.providers.ProviderError`. Every
    request sends ``LLM_REQUEST_ATTEMPT`` in the ``X-Localmelo-Attempt``
    header, and ``LLM_REQUEST_KIND``, when it is set, in the
    ``X-Localmelo-Call-Kind`` header. After a reply,
    ``last_reasoning_key`` is ``reasoning_content`` or ``reasoning``, the
    key its reasoning came from, or ``""``, and ``last_usage`` is the
    reply's usage with ``cached_tokens`` and ``reasoning_tokens`` from the
    usage details when the server sent them.

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
    extra_body_by_kind : Mapping[str, Mapping[str, Any]] | None
        Extra fields for one call kind, merged over ``extra_body`` for that
        request only (R5's ``--retrieval-extra-body`` sets
        ``{"retrieve": ...}``). Each entry is prepared like ``extra_body``.
    transport : httpx.AsyncBaseTransport | None
        Optional HTTP transport, e.g. ``httpx.MockTransport`` in tests.
    reasoning_field : str | None
        The request key this provider sends an assistant turn's thinking
        back under: ``None`` (the default, never), ``"auto"`` (the key the
        last reply's reasoning came from), ``"reasoning_content"`` or
        ``"reasoning"``. Any other value raises ``ValueError``.
    requires_reasoning_replay : bool
        Whether the server rejects requests that omit a reply's reasoning.
        The agent then evicts whole units instead of cutting or skipping
        reasoning. ``False`` by default.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        *,
        extra_body: Mapping[str, Any] | None = None,
        extra_body_by_kind: Mapping[str, Mapping[str, Any]] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        reasoning_field: str | None = None,
        requires_reasoning_replay: bool = False,
        **kwargs: Any,
    ) -> None:
        if reasoning_field not in (None, "auto", "reasoning_content", "reasoning"):
            raise ValueError(
                "reasoning_field must be None, 'auto', 'reasoning_content' or "
                f"'reasoning', got {reasoning_field!r}"
            )
        self.model = model
        self.extra_body = prepare_extra_body(extra_body)
        self.extra_body_by_kind = {
            kind: prepare_extra_body(body)
            for kind, body in (extra_body_by_kind or {}).items()
        }
        self.reasoning_field = reasoning_field
        self.requires_reasoning_replay = requires_reasoning_replay
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
        reasoning_key = self.reasoning_field
        if reasoning_key == "auto":
            reasoning_key = self.last_reasoning_key or None
        by_kind = self.extra_body_by_kind.get(LLM_REQUEST_KIND.get(), {})
        payload: dict[str, Any] = {
            **self.extra_body,
            **by_kind,
            "model": self.model,
            "messages": _messages_to_openai(messages, reasoning_key=reasoning_key),
        }
        if tools:
            payload["tools"] = [_tool_def_to_openai(t) for t in tools]
            payload["tool_choice"] = "auto"

        path = "/chat/completions"
        resp, data = await post_json(
            self._http, path, payload, headers=attempt_headers()
        )
        msg = _choice_message(data)
        if msg is None:
            raise malformed_body(resp, path, "no choices[0].message object")

        # Set right before the return, with no await in between.
        self.last_reasoning_key = _reasoning_key(msg)
        self.last_usage = _usage_details(data.get("usage"))
        return Message(
            role="assistant",
            content=msg.get("content") or "",
            tool_calls=parse_tool_calls(msg.get("tool_calls")),
            usage=_normalize_usage(data.get("usage")),
            thinking=_reasoning_text(msg),
        )

    async def close(self) -> None:
        await self._http.aclose()
