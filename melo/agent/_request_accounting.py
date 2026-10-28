"""Best-effort, versioned request accounting; this is not a tokenizer."""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass
from typing import Any

from localmelo.melo.schema import ImagePart, Message, ToolDef, decode_base64

PROXY_VERSION = "utf8-bytes-v2"
# Allowance is an estimate, not a guarantee: vision tokenization is model-specific.
IMAGE_TOKENS = 16384


def _image_dimensions(raw: bytes) -> tuple[int, int] | None:
    """Read recognized image headers only; never decompress image pixels."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 33:
        header = raw[12:29]
        if (
            raw[8:12] == b"\x00\x00\x00\r"
            and header[:4] == b"IHDR"
            and zlib.crc32(header) == int.from_bytes(raw[29:33], "big")
        ):
            return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    elif raw[:6] in (b"GIF87a", b"GIF89a") and len(raw) >= 13:
        return int.from_bytes(raw[6:8], "little"), int.from_bytes(raw[8:10], "little")
    elif raw.startswith(b"\xff\xd8"):
        pos = 2
        while pos + 4 <= len(raw):
            if raw[pos] != 0xFF:
                return None
            while pos < len(raw) and raw[pos] == 0xFF:
                pos += 1
            if pos >= len(raw):
                return None
            marker = raw[pos]
            pos += 1
            if marker in (0xD8, 0xD9, 0xDA, 0x00):
                return None
            if marker == 0x01 or 0xD0 <= marker <= 0xD7:
                continue
            if pos + 2 > len(raw):
                return None
            size = int.from_bytes(raw[pos : pos + 2], "big")
            if size < 2 or pos + size > len(raw):
                return None
            if marker in (
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            ):
                if size < 8:
                    return None
                return int.from_bytes(raw[pos + 5 : pos + 7], "big"), int.from_bytes(
                    raw[pos + 3 : pos + 5], "big"
                )
            pos += size
    elif len(raw) >= 30 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        if int.from_bytes(raw[4:8], "little") + 8 > len(raw):
            return None
        if raw[12:16] == b"VP8X" and raw[16:20] == b"\x0a\x00\x00\x00":
            return 1 + int.from_bytes(raw[24:27], "little"), 1 + int.from_bytes(
                raw[27:30], "little"
            )
        if raw[12:16] == b"VP8L" and raw[20] == 0x2F:
            bits = int.from_bytes(raw[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if raw[12:16] == b"VP8 " and raw[23:26] == b"\x9d\x01\x2a":
            return (
                int.from_bytes(raw[26:28], "little") & 0x3FFF,
                int.from_bytes(raw[28:30], "little") & 0x3FFF,
            )
    return None


def _image_allowance(image: ImagePart) -> int:
    # Bound decoding work even if called outside the checker's 20 MiB policy.
    if not isinstance(image.data, str) or len(image.data) > 28 * 1024 * 1024:
        return IMAGE_TOKENS
    raw = decode_base64(image.data)
    dimensions = _image_dimensions(raw) if raw else None
    if dimensions is None or min(dimensions) <= 0:
        return IMAGE_TOKENS
    width, height = dimensions
    return min(IMAGE_TOKENS, ((width + 27) // 28) * ((height + 27) // 28))


@dataclass(frozen=True)
class ContextBudget:
    context_window: int
    output_reserve: int
    safety_margin: int

    def __post_init__(self) -> None:
        values = (self.context_window, self.output_reserve, self.safety_margin)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("context budget values must be positive integers")
        if self.input_limit <= 0:
            raise ValueError("context window must exceed reserve and margin")

    @property
    def input_limit(self) -> int:
        return self.context_window - self.output_reserve - self.safety_margin


def request_proxy(
    messages: list[Message], tools: list[ToolDef] | None, *, replay: bool
) -> tuple[int, int]:
    """Estimate assembled message and tool content, excluding image payloads.

    This provider-independent representation includes protocol structure but is
    not the provider's exact wire serialization or model's chat template.
    """
    entries: list[dict[str, Any]] = []
    image_allowance = 0
    for message in messages:
        entry: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_call_id:
            entry["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            entry["tool_calls"] = [
                {"id": call.id, "name": call.tool_name, "arguments": call.arguments}
                for call in message.tool_calls
            ]
        if replay and message.role == "assistant" and message.thinking:
            entry["thinking"] = message.thinking
        if message.role in ("user", "tool"):
            image_allowance += sum(
                _image_allowance(image) for image in message.images if image.data
            )
        entries.append(entry)
    schemas = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tools or ()
    ]
    wire = json.dumps(
        {"messages": entries, "tools": schemas}, ensure_ascii=False, default=str
    )
    return (
        len(wire.encode("utf-8", errors="surrogatepass")) + 2
    ) // 3 + image_allowance, image_allowance


class RequestAccounting:
    """Per-run calibration and telemetry; no prompt content is retained."""

    def __init__(self, budget: ContextBudget) -> None:
        self.budget = budget
        self.ratios: dict[tuple[str, str], float] = {}
        self.records: list[dict[str, Any]] = []

    def start(
        self,
        model: str,
        kind: str,
        messages: list[Message],
        tools: list[ToolDef] | None,
        *,
        replay: bool,
        attempt: int,
    ) -> tuple[dict[str, Any], int]:
        proxy, image_allowance = request_proxy(messages, tools, replay=replay)
        ratio = self.ratios.get((model, kind))
        estimate = math.ceil(proxy * max(1.0, ratio)) if ratio is not None else proxy
        record = {
            "kind": kind,
            "model": model,
            "attempt": attempt,
            "proxy_version": PROXY_VERSION,
            "source": "calibrated" if ratio is not None and ratio > 1 else "proxy",
            "estimate": estimate,
            "measured_prompt_tokens": None,
            "input_limit": self.budget.input_limit,
            "over_limit": estimate > self.budget.input_limit,
            "image_allowance": image_allowance,
        }
        self.records.append(record)
        return record, proxy

    def finish(self, record: dict[str, Any], proxy: int, usage: Any) -> None:
        measured = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if type(measured) is int and measured > 0:
            record["measured_prompt_tokens"] = measured
            self.ratios[(record["model"], record["kind"])] = measured / proxy
        else:
            self.ratios.pop((record["model"], record["kind"]), None)
