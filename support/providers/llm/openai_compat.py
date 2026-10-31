from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from datetime import datetime
from typing import Any

import httpx

from localmelo.melo.contracts.providers import (
    LLM_REQUEST_KIND,
    BaseLLMProvider,
    ProviderError,
    StreamFragment,
    thinking_disabled,
)
from localmelo.melo.schema import ImagePart, Message, ToolCall, ToolDef, describe_images
from localmelo.support.providers._http import (
    RETRYABLE_STATUS_CODES,
    _context_overflow,
    attempt_headers,
    malformed_body,
    post_json,
    unreachable,
)
from localmelo.support.providers.llm._protocol import (
    parse_tool_calls,
    prepare_extra_body,
    raw_tool_arguments,
)
from localmelo.support.providers.llm._sse import sse_data
from localmelo.support.providers.llm._vision import server_takes_images

# The most bytes of a response that is one JSON body (not a stream) that
# stream_chat reads, for a server that ignores ``stream``.
_MAX_WHOLE_BODY = 16 * 1024 * 1024

_UTF8_BOM = b"\xef\xbb\xbf"


_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION = re.compile(r"(?:\d+(?:\.\d+)?(?:ms|h|m|s))+|\d+(?:\.\d+)?")
_UNIT_S = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "": 1.0}


def _reset_at(value: str, now: float) -> float | None:
    """A reset header as a time: a duration (``6m0s``, ``20ms``, ``30``) or RFC 3339."""
    if _DURATION.fullmatch(value):
        parts = _DURATION_PART.findall(value) or [(value, "")]
        return now + sum(float(n) * _UNIT_S[unit] for n, unit in parts)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def rate_limits(
    headers: Mapping[str, str], now: float | None = None
) -> tuple[int, float | None] | None:
    """The tightest rate-limit window a response reports, or None.

    ``(percent used, reset time)`` from ``x-ratelimit-{limit,remaining,reset}-W``
    or ``anthropic-ratelimit-W-{limit,remaining,reset}``: the window with the
    largest share used, its reset as epoch seconds (None when not given).
    """
    now = time.time() if now is None else now
    windows: dict[tuple[str, str], dict[str, str]] = {}
    for raw, value in headers.items():
        name = raw.lower()
        for prefix in ("x-ratelimit-", "anthropic-ratelimit-"):
            if name.startswith(prefix):
                rest = name[len(prefix) :]
                if prefix[0] == "x":
                    field, _, window = rest.partition("-")
                else:
                    window, _, field = rest.rpartition("-")
                if window and field in ("limit", "remaining", "reset"):
                    windows.setdefault((prefix, window), {})[field] = value.strip()
    best: tuple[int, float | None] | None = None
    for fields in windows.values():
        try:
            limit, remaining = int(fields["limit"]), int(fields["remaining"])
        except (KeyError, ValueError):
            continue
        if limit <= 0:
            continue
        used = min(100, max(0, round(100 * (limit - remaining) / limit)))
        if best is None or used > best[0]:
            best = (used, _reset_at(fields.get("reset", ""), now))
    return best


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


async def _head_chunks(chunks: AsyncIterator[bytes]) -> list[bytes]:
    """Read chunks until one shows how the body starts, and return them all.

    Blank chunks, and a byte order mark (or the part of one that has come),
    show nothing yet. The caller puts them back in front of the rest of the
    body (:func:`_prepend`).
    """
    head: list[bytes] = []
    async for chunk in chunks:
        head.append(chunk)
        seen = b"".join(head)
        if not _UTF8_BOM.startswith(seen) and seen.removeprefix(_UTF8_BOM).strip():
            break
    return head


def _is_object(head: list[bytes]) -> bool:
    """Whether a body that starts with *head* is one JSON object, not events.

    A server-sent-events body never starts with a brace: its lines begin
    with a field name, a colon (a comment) or nothing at all.
    """
    return b"".join(head).removeprefix(_UTF8_BOM).lstrip().startswith(b"{")


async def _prepend(
    head: list[bytes], rest: AsyncIterator[bytes]
) -> AsyncIterator[bytes]:
    """The chunks of *head*, then those of *rest*."""
    for chunk in head:
        yield chunk
    async for chunk in rest:
        yield chunk


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
    usage details when the server sent them. ``last_limits`` is the
    tightest rate limit the last response's headers reported
    (:func:`rate_limits`), for the terminal's status bar.

    :meth:`stream_chat` copes with a server that does not stream: it reads a
    response that is one JSON body as the completion :meth:`chat` would
    return, and asks again, once and without streaming, when a stream ends
    before it delivered anything. Either way the reply is marked
    ``whole=True`` and ``sends_whole_replies`` becomes true: this provider
    asks that server for whole replies from then on.

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
        self.thinking_off = thinking_disabled(
            {**self.extra_body, **self.extra_body_by_kind.get("plan", {})}
        )
        # Set once the server has answered a request for a stream with a whole
        # reply (see stream_chat); later stream_chat calls then do not ask it
        # to stream.
        self.sends_whole_replies = False
        self.last_limits: tuple[int, float | None] | None = None
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers=headers,
            transport=transport,
            follow_redirects=False,
        )

    def _payload(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        """The request body; ``stream`` comes right after ``messages``."""
        reasoning_key = self.reasoning_field
        if reasoning_key == "auto":
            reasoning_key = self.last_reasoning_key or None
        payload: dict[str, Any] = {
            **self.extra_body,
            **self.extra_body_by_kind.get(LLM_REQUEST_KIND.get(), {}),
            "model": self.model,
            "messages": _messages_to_openai(messages, reasoning_key=reasoning_key),
        }
        if stream:
            payload["stream"] = True
        if tools:
            payload["tools"] = [_tool_def_to_openai(t) for t in tools]
            payload["tool_choice"] = "auto"
        return payload

    def _completion(
        self, resp: httpx.Response, path: str, data: Any
    ) -> tuple[Message, str]:
        """The reply in a chat completion body, and its finish reason.

        A body without a finish reason gets ``tool_calls`` when the reply
        calls tools and ``stop`` otherwise.
        """
        msg = _choice_message(data)
        if msg is None:
            raise malformed_body(resp, path, "no choices[0].message object")

        # Set right before the return, with no await in between.
        self.last_reasoning_key = _reasoning_key(msg)
        self.last_usage = _usage_details(data.get("usage"))
        reply = Message(
            role="assistant",
            content=msg.get("content") or "",
            tool_calls=parse_tool_calls(msg.get("tool_calls")),
            usage=_normalize_usage(data.get("usage")),
            thinking=_reasoning_text(msg),
        )
        finish = data["choices"][0].get("finish_reason")
        if not isinstance(finish, str) or not finish:
            finish = "tool_calls" if reply.tool_calls else "stop"
        return reply, finish

    async def _complete(
        self, messages: list[Message], tools: list[ToolDef] | None
    ) -> tuple[Message, str]:
        """Send one request that does not ask for a stream: its reply and finish reason."""
        path = "/chat/completions"
        resp, data = await post_json(
            self._http, path, self._payload(messages, tools), headers=attempt_headers()
        )
        self.last_limits = rate_limits(resp.headers)
        return self._completion(resp, path, data)

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> Message:
        reply, _ = await self._complete(messages, tools)
        return reply

    async def _read_body(
        self,
        response: httpx.Response,
        head: list[bytes],
        chunks: AsyncIterator[bytes],
    ) -> tuple[Message, str]:
        """Read a response that is one JSON body: its reply and finish reason.

        *head* holds the chunks already read and *chunks* the rest. The body
        is read as :meth:`chat` reads it, with a bound on its size.
        """
        body = bytearray()
        async for chunk in _prepend(head, chunks):
            body.extend(chunk)
            if len(body) > _MAX_WHOLE_BODY:
                raise ProviderError("Response body exceeds size limit")
        # What an error message quotes, as a response that was read.
        seen = httpx.Response(response.status_code, content=bytes(body[:65_536]))
        path = "/chat/completions"
        try:
            data = json.loads(body)
        except ValueError as exc:
            raise malformed_body(seen, path, "not JSON") from exc
        return self._completion(seen, path, data)

    async def stream_chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> AsyncGenerator[StreamFragment, None]:
        """Stream one OpenAI-compatible response; never retry after a delta.

        A server that does not stream still answers. A response that is one
        JSON body is read as the completion :meth:`chat` would return. A
        stream that ends before it delivered anything is asked for again,
        once, without streaming. Either way the reply comes marked
        ``whole=True``, and the server is not asked to stream again. A stream
        that breaks after part of a reply reached the caller keeps its error.
        """
        if self.sends_whole_replies:
            reply, finish = await self._complete(messages, tools)
            yield StreamFragment(
                kind="reply", reply=reply, finish_reason=finish, whole=True
            )
            return
        payload = self._payload(messages, tools, stream=True)
        content: list[str] = []
        thinking: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        usage: dict[str, int] | None = None
        raw_usage: Any = None
        reason_key = ""
        text_chars = 0
        tool_chars = 0
        finish = ""
        done = False
        emitted = False
        opened = False
        whole: tuple[Message, str] | None = None
        try:
            async with self._http.stream(
                "POST", "/chat/completions", json=payload, headers=attempt_headers()
            ) as response:
                if not response.is_success:
                    error_body = bytearray()
                    async for part in response.aiter_bytes(chunk_size=8192):
                        error_body.extend(part)
                        if len(error_body) >= 65_536:
                            break
                    error_response = httpx.Response(
                        response.status_code, content=bytes(error_body[:65_536])
                    )
                    raise ProviderError(
                        f"POST /chat/completions failed with HTTP {response.status_code}",
                        status_code=response.status_code,
                        retryable=response.status_code in RETRYABLE_STATUS_CODES,
                        context_overflow=_context_overflow(error_response),
                    )
                opened = True
                self.last_limits = rate_limits(response.headers)
                chunks = response.aiter_bytes()
                head = await _head_chunks(chunks)
                if _is_object(head):
                    # One JSON body, not events: a server that ignores
                    # ``stream``. Nothing is left for the loop below to read.
                    whole = await self._read_body(response, head, chunks)
                    head = []
                async for frame in sse_data(_prepend(head, chunks)):
                    if frame == "[DONE]":
                        done = True
                        break
                    try:
                        obj = json.loads(frame, parse_constant=_reject_nonfinite)
                    except ValueError as exc:
                        raise ProviderError("Malformed SSE JSON") from exc
                    if not isinstance(obj, dict) or obj.get("error") is not None:
                        raise ProviderError("Malformed SSE response or server error")
                    if obj.get("usage") is not None:
                        raw_usage = obj["usage"]
                        usage = _normalize_usage(raw_usage)
                        if usage is None:
                            raise ProviderError("Malformed SSE usage")
                        emitted = True
                        yield StreamFragment(kind="usage", usage=usage)
                    choices = obj.get(
                        "choices", [] if obj.get("usage") is not None else None
                    )
                    if not isinstance(choices, list):
                        raise ProviderError("Malformed SSE choices")
                    for choice in choices:
                        if not isinstance(choice, dict):
                            raise ProviderError("Malformed SSE choice")
                        if choice.get("index", 0) != 0:
                            continue
                        delta = choice.get("delta", {})
                        if not isinstance(delta, dict):
                            raise ProviderError("Malformed SSE delta")
                        if finish and delta:
                            raise ProviderError("SSE delta after finish")
                        for field, kind, parts in (
                            ("content", "content", content),
                            ("reasoning_content", "reasoning", thinking),
                            ("reasoning", "reasoning", thinking),
                        ):
                            value = delta.get(field)
                            if value is not None:
                                if not isinstance(value, str):
                                    raise ProviderError("Malformed SSE text")
                                if value:
                                    text_chars += len(value)
                                    if text_chars > 1_048_576:
                                        raise ProviderError(
                                            "SSE text exceeds size limit"
                                        )
                                    parts.append(value)
                                    if kind == "reasoning":
                                        reason_key = field
                                    emitted = True
                                    yield StreamFragment(kind=kind, text=value)  # type: ignore[arg-type]
                        fragments = delta.get("tool_calls")
                        if fragments is not None:
                            if not isinstance(fragments, list):
                                raise ProviderError("Malformed SSE tool fragments")
                            for fragment in fragments:
                                if not isinstance(fragment, dict):
                                    raise ProviderError("Malformed SSE tool fragment")
                                if fragment.get("type") not in (None, "function"):
                                    raise ProviderError("Unsupported SSE tool type")
                                index = fragment.get("index")
                                if (
                                    isinstance(index, bool)
                                    or not isinstance(index, int)
                                    or not 0 <= index < 256
                                ):
                                    raise ProviderError("Malformed SSE tool index")
                                call = calls.setdefault(
                                    index, {"id": "", "name": "", "arguments": ""}
                                )
                                function = fragment.get("function") or {}
                                if not isinstance(function, dict):
                                    raise ProviderError("Malformed SSE tool function")
                                for key, value in (
                                    ("id", fragment.get("id")),
                                    ("name", function.get("name")),
                                    ("arguments", function.get("arguments")),
                                ):
                                    if value is not None:
                                        if not isinstance(value, str):
                                            raise ProviderError(
                                                "Malformed SSE tool field"
                                            )
                                        tool_chars += len(value)
                                        if tool_chars > 1_048_576:
                                            raise ProviderError(
                                                "SSE tool fields exceed size limit"
                                            )
                                        if (
                                            key not in ("id", "name")
                                            or value != call[key]
                                        ):
                                            call[key] += value
                                emitted = True
                                yield StreamFragment(kind="tool", tool_index=index)
                        final = choice.get("finish_reason")
                        if final is not None:
                            if not isinstance(final, str) or finish:
                                raise ProviderError("Malformed SSE finish reason")
                            finish = final
        except httpx.ConnectError as exc:
            raise unreachable(self._http, exc, retryable=not emitted) from exc
        except httpx.RemoteProtocolError as exc:
            if emitted or not opened:
                raise ProviderError(
                    f"Streaming request failed: {type(exc).__name__}",
                    retryable=not emitted,
                ) from exc
            # The server cut the body before anything reached the caller: a
            # stream that ended, handled below.
        except httpx.TransportError as exc:
            raise ProviderError(
                f"Streaming request failed: {type(exc).__name__}", retryable=not emitted
            ) from exc
        except (httpx.RequestError, httpx.StreamError) as exc:
            raise ProviderError(
                f"Streaming request failed: {type(exc).__name__}"
            ) from exc
        if whole is None and (not done or not finish):
            if emitted:
                raise ProviderError("Streaming response ended before completion")
            # Nothing reached the caller, so the request can be sent again: ask
            # for the whole reply. This raises its own error if it fails.
            whole = await self._complete(messages, tools)
        if whole is not None:
            self.sends_whole_replies = True
            yield StreamFragment(
                kind="reply", reply=whole[0], finish_reason=whole[1], whole=True
            )
            return
        if bool(calls) != (finish == "tool_calls"):
            raise ProviderError("Tool calls were not completed")
        raw_calls = []
        for index in sorted(calls):
            call = calls[index]
            try:
                args = (
                    json.loads(call["arguments"], parse_constant=_reject_nonfinite)
                    if call["arguments"].strip()
                    else {}
                )
            except ValueError as exc:
                raise ProviderError("Malformed completed tool arguments") from exc
            if not isinstance(args, dict) or not call["name"]:
                raise ProviderError("Malformed completed tool call")
            raw_calls.append(
                {
                    "id": call["id"],
                    "function": {"name": call["name"], "arguments": args},
                }
            )
        reply = Message(
            role="assistant",
            content="".join(content),
            thinking="".join(thinking),
            tool_calls=parse_tool_calls(raw_calls),
            usage=usage,
        )
        self.last_reasoning_key = reason_key
        self.last_usage = _usage_details(raw_usage)
        yield StreamFragment(kind="reply", reply=reply, finish_reason=finish)

    async def takes_images(self) -> bool | None:
        """Whether the server says the model reads images (oMLX, Ollama); else None."""
        root = str(self._http.base_url).rstrip("/").removesuffix("/v1")
        return await server_takes_images(
            self._http, self.model, status="/models/status", show=f"{root}/api/show"
        )

    async def close(self) -> None:
        await self._http.aclose()
