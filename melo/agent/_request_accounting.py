"""Best-effort, versioned request accounting; this is not a tokenizer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from localmelo.melo.schema import Message, ToolDef

PROXY_VERSION = "utf8-bytes-v1"
# A fixed allowance, not a guarantee: actual vision tokenization is model-specific.
IMAGE_TOKENS = 16384


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
    image_count = 0
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
            image_count += sum(bool(image.data) for image in message.images)
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
    image_allowance = image_count * IMAGE_TOKENS
    return (len(wire.encode("utf-8")) + 2) // 3 + image_allowance, image_allowance


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
