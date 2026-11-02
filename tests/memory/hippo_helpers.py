"""Fakes for the memory-item tests: a chat model that answers the encoder with canned
JSON, and small builders. Nothing here calls a model server."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.memory import hippo_encode
from localmelo.support.memory.hippo_encode import Turn

Reply = Callable[[str], str]


class StubEncoder(BaseLLMProvider):
    """A chat model that answers each encoder request with ``reply(prompt)``.

    ``prompts`` holds the user message of every request (a retry counts), ``bodies``
    the request body fields the request carried for its kind (``extra_body_by_kind``).
    With *usage*, each answer reports that many tokens as a server would.
    """

    def __init__(
        self,
        reply: Reply | str = '{"items": [], "ignore": []}',
        *,
        model: str = "stub-encoder",
        usage: tuple[int, int] | None = None,
    ) -> None:
        self.model = model
        self.reply = reply if callable(reply) else (lambda _prompt, text=reply: text)
        self.usage = usage
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.bodies: list[dict[str, Any]] = []
        self.last_usage: dict[str, int] | None = None

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        assert tools is None
        self.systems.append(str(messages[0].content))
        prompt = str(messages[-1].content)
        self.prompts.append(prompt)
        self.bodies.append(dict(self.extra_body_by_kind.get("retrieve", {})))
        content = self.reply(prompt)
        if content == "RAISE":
            raise RuntimeError("the chat server is down")
        if self.usage is not None:
            self.last_usage = {
                "prompt_tokens": self.usage[0],
                "completion_tokens": self.usage[1],
            }
        return Message("assistant", content)

    async def close(self) -> None:
        return None


def says(
    items: list[dict[str, Any]] | None = None, ignore: list[int] | None = None
) -> str:
    """The JSON a well-behaved encoder replies with."""
    return json.dumps({"items": items or [], "ignore": ignore or []})


def item(
    title: str,
    text: str,
    kind: str = "semantic",
    sources: list[int] | None = None,
) -> dict[str, Any]:
    return {"title": title, "text": text, "kind": kind, "sources": sources or [0]}


def at(day: str, clock: str = "14:02") -> float:
    """Local time of ``YYYY-MM-DD`` at *clock*, as the prompt writes it."""
    return datetime.fromisoformat(f"{day}T{clock}:00").timestamp()


def turn(
    n: int,
    typed: str,
    answer: str = "",
    when: float | None = None,
) -> Turn:
    """The exchange numbered *n* (its store turn is *n* + 1), on 2023-05-20 plus *n*
    minutes unless *when* says."""
    start = when if when is not None else at("2023-05-20", "14:02") + 60 * n
    return Turn(n + 1, start, start + 5, typed, answer)


def numbers_in(prompt: str) -> list[int]:
    """The exchange numbers a prompt shows, in order."""
    return [
        int(line[1 : line.index("]")])
        for line in prompt.split("\n")
        if line.startswith("[") and "] 20" in line
    ]


def window_sizes(turns: list[Turn], limit: int) -> list[int]:
    return [len(w.positions) for w in hippo_encode.pack(turns, limit)]
