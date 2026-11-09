"""Fakes for the memory-item tests: a chat model that answers the encoder with canned
JSON, and small builders. Nothing here calls a model server."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.memory import hippo_encode
from localmelo.support.memory.hippo_encode import Turn
from localmelo.support.memory.index import ConversationIndex

Reply = Callable[[str], str]


class StubEncoder(BaseLLMProvider):
    """A chat model that answers each encoder request with ``reply(prompt)``.

    ``prompts`` holds the user message of every request (a retry counts), ``bodies``
    the request body fields the request carried for its kind (``extra_body_by_kind``).
    With *usage*, each answer reports that many (prompt, completion) tokens as a
    server would; *usage* may also be a function of the prompt that gives them, or
    ``None`` for an answer that reports none.
    """

    def __init__(
        self,
        reply: Reply | str = '{"items": [], "ignore": []}',
        *,
        model: str = "stub-encoder",
        usage: tuple[int, int] | Callable[[str], tuple[int, int] | None] | None = None,
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
        tokens = self.usage(prompt) if callable(self.usage) else self.usage
        if tokens is not None:
            self.last_usage = {
                "prompt_tokens": tokens[0],
                "completion_tokens": tokens[1],
            }
        elif callable(self.usage):
            self.last_usage = None
        return Message("assistant", content)

    async def close(self) -> None:
        return None


class Held(StubEncoder):
    """A stub chat model whose judge requests wait until ``release`` is set, and say
    so (``asked``) when one arrives. With ``release`` never set, a request is one
    that never answers. Encoder requests are answered at once."""

    def __init__(self, reply: Reply | str = '{"items": [], "ignore": []}') -> None:
        super().__init__(reply)
        # Events, shared with the copy the encoder makes of the provider.
        self.asked = asyncio.Event()
        self.release = asyncio.Event()

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        if is_judge(str(messages[-1].content)):
            self.asked.set()
            await self.release.wait()
        return await super().chat(messages, tools)


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


def is_judge(prompt: str) -> bool:
    """Whether a request is the judge's, a JSON object, not an encoder window (which
    starts with the chat title)."""
    return prompt.startswith("{")


def verdict(kind: str = "other", item: int | None = None) -> str:
    """The JSON a well-behaved judge replies with (``other`` names no item)."""
    return json.dumps({"verdict": kind, "item": None if kind == "other" else item or 1})


def judging(
    encode: Reply | str, judge: Callable[[dict[str, Any]], str] | None = None
) -> Reply:
    """A stub reply that answers the judge's requests with ``judge(request)`` (the
    request's JSON; ``other`` when none is given) and the encoder's with *encode*."""
    asked = encode if callable(encode) else (lambda _prompt, text=encode: text)

    def reply(prompt: str) -> str:
        if not is_judge(prompt):
            return asked(prompt)
        return judge(json.loads(prompt)) if judge else verdict()

    return reply


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


async def relate(
    index: ConversationIndex,
    *,
    replace: tuple[int, int] | None = None,
    link: tuple[int, int] | None = None,
    conflict: bool = False,
) -> None:
    """Change how items already in *index* relate, in one transaction (it rolls back
    and raises when a step fails), as ``add_item`` does for a new item's.

    *replace* is ``(old, new)``: *old* is marked replaced by *new*, and any conflict
    it was in ends. *link* is ``(first, second)``: the two are linked both ways as
    related, or, with *conflict*, in conflict.
    """
    conn = await index._ready()
    try:
        if replace is not None:
            await index._replace(conn, *replace)
        if link is not None:
            await index._link(conn, *link, conflict=conflict)
        await conn.commit()
    except BaseException:
        await conn.rollback()
        raise


def numbers_in(prompt: str) -> list[int]:
    """The exchange numbers a prompt shows, in order."""
    return [
        int(line[1 : line.index("]")])
        for line in prompt.split("\n")
        if line.startswith("[") and "] 20" in line
    ]


def window_sizes(turns: list[Turn], limit: int) -> list[int]:
    return [len(w.positions) for w in hippo_encode.pack(turns, limit)]


# ── Windows that run out of room, and long exchanges (#428) ──

CAP = hippo_encode.MAX_OUTPUT_TOKENS
# What a reply looks like when the output cap cuts it off in the middle of an item.
CUT_OFF = (
    '{"items": [{"title": "A", "text": "The user said a", "kind": "profile", '
    '"sources": [0]}, {"title": "B", "te'
)


def labels(prompt: str) -> list[int]:
    """The exchange numbers a prompt shows, a part's ``(continued)`` label too."""
    return [int(n) for n in re.findall(r"^\[(\d+)\] ", prompt, re.MULTILINE)]


def by_label(prompt: str) -> str:
    """One item for each exchange label the prompt shows; a part after the first of
    an exchange says so in its text."""
    kind = "continued" if "(continued)" in prompt else "first"
    return says(
        [
            item(f"Q{n}", f"Question {n}, {kind} part.", "episodic", [n])
            for n in labels(prompt)
        ]
    )


def runs_out_over(chars: int, *, report: bool = True) -> StubEncoder:
    """A chat model whose reply to a prompt of more than *chars* characters is cut
    off at the output cap (with the token count of a server when *report*), and
    that answers a shorter one with :func:`by_label`."""

    def reply(prompt: str) -> str:
        return CUT_OFF if len(prompt) > chars else by_label(prompt)

    def usage(prompt: str) -> tuple[int, int] | None:
        if not report:
            return None
        return (len(prompt) // 4, CAP if len(prompt) > chars else 150)

    return StubEncoder(reply, usage=usage)


def padded(count: int, size: int = 500) -> list[Turn]:
    """*count* exchanges of about *size* characters each."""
    return [turn(n, f"Question {n}?", "x" * size) for n in range(count)]


def paragraphs(count: int, size: int) -> str:
    """*count* paragraphs of *size* characters, a blank line between."""
    return "\n\n".join(chr(97 + n % 26) * size for n in range(count))


def body(prompt: str) -> str:
    """The chat text of a prompt (after its title line)."""
    return prompt.split("\n\n", 1)[1]


def prompt_of_window(window: hippo_encode.Window) -> str:
    return hippo_encode.prompt_of("T", window)
