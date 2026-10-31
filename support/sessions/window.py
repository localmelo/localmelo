"""The window codec: a working-memory window as versioned, redacted JSON.

A session's ``windows`` row keeps the window an Agent held when a turn
ended, so that the ``window`` resume arm can put it back (see
:meth:`SessionRunner.resume`). :func:`encode_window` writes it,
:func:`decode_window` reads it; the decoded messages go through
``Agent.load_conversation``, which keeps only valid units.

What the JSON keeps of each message, and what it leaves out:

- every field a request is built from: role, content, thinking, tool calls,
  the tool call id, the tool result's name, error and duration, image
  references, the timestamp and the usage;
- not ``tool_result.output``, which decodes as ``""``: a request reads a
  tool message's ``content``, which holds the text the model saw, and a
  ``tool_result`` only for its tool name. The output stays in
  ``history.db`` and in the ``tool.finished`` event;
- images as references only (MIME type, SHA-256 and size), as working
  memory keeps tool images already.

An assistant message's ``content`` and ``thinking`` pass
:func:`~localmelo.melo.checker.redaction.redact_secrets`, because the live
window keeps them as the model wrote them; tool-call arguments pass
:func:`~localmelo.melo.contracts.events.json_safe_payload`. A version bump
keeps the decoders of older versions.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.contracts.events import json_safe_payload
from localmelo.melo.schema import (
    ImagePart,
    Message,
    ToolCall,
    ToolResult,
    split_leading_thinking,
    split_thinking,
)

WINDOW_CODEC_VERSION: Final = 1

# Characters of encoded JSON that a window checkpoint may have; a longer one
# is not stored (the older row stays), and a checkpoint is never truncated.
WINDOW_MAX_CHARS: Final = 2_000_000

# A text holds a <think> block by split_thinking's rule exactly when it has
# a closing tag, opens with an unclosed one after leading whitespace, or
# starts with a closed <thinking> block.
_CLOSING_TAG = re.compile(r"</think>", re.IGNORECASE)
_OPENING_TAG = re.compile(r"\s*<think>", re.IGNORECASE)


def has_think_block(text: str) -> bool:
    """Whether :func:`~localmelo.melo.schema.split_thinking` finds a block in *text*."""
    return bool(
        _CLOSING_TAG.search(text)
        or _OPENING_TAG.match(text)
        or split_leading_thinking(text) is not None
    )


def without_think_block(text: str) -> str:
    """*text* without its ``<think>`` block; a text without one stays as it is."""
    return split_thinking(text)[1] if has_think_block(text) else text


def _expired(message: Message, thinking_cutoff: float | None) -> bool:
    """Whether *message* is an assistant message older than the cutoff."""
    return (
        thinking_cutoff is not None
        and message.role == "assistant"
        and message.timestamp < thinking_cutoff
    )


def encode_window(
    messages: Sequence[Message], *, thinking_cutoff: float | None = None
) -> str:
    """*messages* as the codec's JSON, ``{"v": 1, "messages": [...]}``.

    With *thinking_cutoff*, every assistant message whose timestamp is
    older than it is written with ``thinking`` ``""`` and its content
    without its ``<think>`` block; the message itself stays, so the window
    keeps its units.
    """
    encoded = [_encode_message(msg, _expired(msg, thinking_cutoff)) for msg in messages]
    return json.dumps(
        {"v": WINDOW_CODEC_VERSION, "messages": encoded},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def window_thinking_since(
    messages: Sequence[Message], *, thinking_cutoff: float | None = None
) -> float | None:
    """The oldest timestamp of an assistant message that keeps thinking.

    That is a message the encoding with *thinking_cutoff* leaves with
    ``thinking`` or a ``<think>`` block in its content; ``None`` when there
    is none. The store keeps it as ``windows.thinking_since``, so retention
    knows when a stored window next holds thinking that has expired.
    """
    since: float | None = None
    for msg in messages:
        if msg.role != "assistant" or _expired(msg, thinking_cutoff):
            continue
        if msg.thinking or has_think_block(msg.content):
            stamp = _finite(msg.timestamp)
            since = stamp if since is None else min(since, stamp)
    return since


def decode_window(text: str) -> list[Message] | None:
    """The messages of an encoded window; ``None`` when it cannot be read.

    ``None`` comes for another version, malformed JSON or a message of the
    wrong shape; this function never raises.
    """
    try:
        data = json.loads(text)
        if not isinstance(data, dict) or data.get("v") != WINDOW_CODEC_VERSION:
            return None
        items = data["messages"]
        if not isinstance(items, list):
            return None
        return [_decode_message(item) for item in items]
    except Exception:
        return None


# ── Encoding ──


def _finite(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _image_reference(image: ImagePart) -> dict[str, Any]:
    return {
        "mime_type": image.mime_type,
        "sha256": image.sha256,
        "size_bytes": image.size_bytes,
    }


def _encode_message(msg: Message, expired: bool) -> dict[str, Any]:
    content, thinking = msg.content, msg.thinking
    if msg.role == "assistant":
        if expired:
            content, thinking = without_think_block(content), ""
        content = redact_secrets(content)[0]
        thinking = redact_secrets(thinking)[0]
    result = msg.tool_result
    usage = None
    if msg.usage is not None:
        usage = {
            str(name): value
            for name, value in msg.usage.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
    return {
        "role": msg.role,
        "content": content,
        "thinking": thinking,
        "tool_calls": [
            {
                "id": call.id,
                "tool_name": call.tool_name,
                "arguments": json_safe_payload(call.arguments),
            }
            for call in msg.tool_calls
        ],
        "tool_call_id": msg.tool_call_id,
        "tool_result": (
            None
            if result is None
            else {
                "tool_name": result.tool_name,
                "error": result.error,
                "duration_ms": _finite(result.duration_ms),
                "images": [_image_reference(image) for image in result.images],
            }
        ),
        "images": [_image_reference(image) for image in msg.images],
        "timestamp": _finite(msg.timestamp),
        "usage": usage,
    }


# ── Decoding ──


def _text(item: Mapping[str, Any], key: str) -> str:
    value = item.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{key} is not a string")
    return value


def _decode_image(item: object) -> ImagePart:
    if not isinstance(item, dict):
        raise ValueError("an image is not an object")
    size = item.get("size_bytes", 0)
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError("size_bytes is not an integer")
    return ImagePart(
        mime_type=_text(item, "mime_type"),
        sha256=_text(item, "sha256"),
        size_bytes=size,
    )


def _decode_message(item: object) -> Message:
    if not isinstance(item, dict):
        raise ValueError("a message is not an object")
    calls = []
    for call in item.get("tool_calls") or []:
        if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
            raise ValueError("a tool call is not an object with arguments")
        calls.append(
            ToolCall(
                tool_name=_text(call, "tool_name"),
                arguments=call["arguments"],
                id=_text(call, "id"),
            )
        )
    result = None
    raw_result = item.get("tool_result")
    if raw_result is not None:
        if not isinstance(raw_result, dict):
            raise ValueError("tool_result is not an object")
        result = ToolResult(
            tool_name=_text(raw_result, "tool_name"),
            output="",
            error=_text(raw_result, "error"),
            duration_ms=_finite(raw_result.get("duration_ms")),
            images=[_decode_image(image) for image in raw_result.get("images") or []],
        )
    usage = item.get("usage")
    if usage is not None:
        if not isinstance(usage, dict):
            raise ValueError("usage is not an object")
        usage = {
            str(name): value
            for name, value in usage.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
    return Message(
        role=_text(item, "role"),
        content=_text(item, "content"),
        thinking=_text(item, "thinking"),
        tool_calls=calls,
        tool_call_id=_text(item, "tool_call_id"),
        tool_result=result,
        images=[_decode_image(image) for image in item.get("images") or []],
        timestamp=_finite(item.get("timestamp")),
        usage=usage,
    )
