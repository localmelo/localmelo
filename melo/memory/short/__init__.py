from __future__ import annotations

import dataclasses
import itertools
from collections import deque
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from localmelo.melo.memory.short.note_store import NoteStore
from localmelo.melo.schema import EMPTY_REPLY_LABEL, EMPTY_REPLY_NUDGE, Message

if TYPE_CHECKING:
    from localmelo.melo.schema import ReflectionEntry


def _opens_tool_turn(msg: Message) -> bool:
    return msg.role == "assistant" and bool(msg.tool_calls)


def _is_empty_reply_label(msg: Message, empty_prefix: bool = False) -> bool:
    """Whether *msg* is the agent's stand-in for an empty planning reply.

    With *empty_prefix*, a label that carries the reply's note after a
    line break (the compress modes) still counts, by prefix; ``legacy``
    keeps the exact match.
    """
    if msg.role != "assistant" or msg.tool_calls:
        return False
    if msg.content == EMPTY_REPLY_LABEL:
        return True
    return empty_prefix and msg.content.startswith(EMPTY_REPLY_LABEL + "\n")


def _joins(unit: Sequence[Message], msg: Message, empty_prefix: bool = False) -> bool:
    """Whether *msg*, directly after *unit*, belongs to that unit."""
    if _opens_tool_turn(unit[0]):
        return msg.role == "tool"
    return (
        len(unit) == 1
        and _is_empty_reply_label(unit[0], empty_prefix)
        and msg.role == "user"
        and msg.content == EMPTY_REPLY_NUDGE
    )


def group_turns(
    messages: Iterable[Message], *, empty_prefix: bool = False
) -> list[list[Message]]:
    """Split *messages* into units that must stay together.

    An assistant message with tool calls opens a unit that also owns the
    ``role="tool"`` messages directly after it. The agent's stand-in for
    an empty reply (``EMPTY_REPLY_LABEL``, an assistant message) opens a
    unit that also owns the nudge directly after it (``EMPTY_REPLY_NUDGE``,
    a user message). Every other message is a unit of its own. Trimming
    whole units never separates a tool call from its results, which
    OpenAI-compatible servers reject, nor a nudge from its label: left at
    the start of the window, the nudge would follow the task query, which
    the agent puts back in front of the window, and some chat templates
    reject two user messages in a row. *empty_prefix* matches a label by
    prefix, for the compress modes' label that carries a note (see
    :func:`_is_empty_reply_label`).
    """
    units: list[list[Message]] = []
    for msg in messages:
        if units and _joins(units[-1], msg, empty_prefix):
            units[-1].append(msg)
        else:
            units.append([msg])
    return units


def tail_turns(
    messages: Sequence[Message], max_messages: int, *, empty_prefix: bool = False
) -> list[Message]:
    """Return the newest whole units of *messages* that fit *max_messages*.

    The newest unit is always included, even when it alone is larger than
    the budget, so the latest tool call is never cut off from its results.
    *empty_prefix* is passed to :func:`group_turns`.
    """
    kept: list[list[Message]] = []
    count = 0
    for unit in reversed(group_turns(messages, empty_prefix=empty_prefix)):
        if kept and count + len(unit) > max_messages:
            break
        kept.append(unit)
        count += len(unit)
    return [msg for unit in reversed(kept) for msg in unit]


class WorkingMemory:
    """Conversation window + the single latest reflection entry.

    The conversation buffer keeps at most ``max_len`` messages, evicting
    the oldest first like the old ShortTerm deque. Eviction works on
    whole units (see :func:`group_turns`): an assistant tool-call message
    and its tool results are dropped together, never split, and so are
    an empty reply's label and the nudge after it. The newest unit is
    kept even if it alone exceeds ``max_len``.

    Only one reflection is retained at a time — a new reflection replaces
    the previous one. Treating reflections as compressed *current state*
    rather than a log keeps the planning prompt bounded across attempts.
    """

    def __init__(self, max_len: int, *, empty_prefix: bool = False) -> None:
        if max_len < 0:
            raise ValueError("max_len must be non-negative")
        self._max_len = max_len
        self._empty_prefix = empty_prefix
        self._buf: deque[Message] = deque()
        self._reflection: ReflectionEntry | None = None
        # The run's reasoning notes and drafts (R4), RAM only, bounded.
        self.notes = NoteStore()

    @property
    def empty_prefix(self) -> bool:
        """Whether an empty reply's label is matched by prefix (compress modes)."""
        return self._empty_prefix

    # ── Conversation buffer ──

    def append(self, msg: Message) -> None:
        self._buf.append(msg)
        self._evict()

    def append_turn(self, assistant: Message, results: Sequence[Message]) -> None:
        """Append an assistant message and what answers it as one unit.

        That is a tool-call message and its results, or an empty reply's
        label and its nudge (see :func:`group_turns`).
        """
        self._buf.append(assistant)
        self._buf.extend(results)
        self._evict()

    def get_window(self) -> list[Message]:
        return list(self._buf)

    def clear(self) -> None:
        """Clear conversation buffer only. The latest reflection survives."""
        self._buf.clear()

    def drop_oldest_units(self, count: int) -> int:
        """Drop the *count* oldest units of the buffer (see :func:`group_turns`).

        Units go whole, so a tool call never loses its results, and the
        newest unit always stays. Returns the number of messages dropped.
        """
        dropped = 0
        for _ in range(count):
            if not self._buf:
                break
            size = self._leading_unit_size()
            if size >= len(self._buf):
                break  # only the newest unit is left
            for _ in range(size):
                self._buf.popleft()
            dropped += size
        return dropped

    def strip_thinking(self, count: int, *, only: set[int] | None = None) -> int:
        """Clear the thinking of the *count* oldest units of the buffer.

        Only an assistant message's ``thinking`` is cleared; its content,
        calls and results stay, so the unit is not dropped. Returns the
        number of messages changed. The newest unit always stays, as in
        :meth:`drop_oldest_units`: a window with one unit changes nothing.
        With *only*, a set of ``id()`` values, only those messages change;
        the fit-to-cap loop uses it so it never clears reasoning the
        request does not carry (an earlier task's, or a turn over the cap).
        """
        changed = 0
        index = 0
        for _ in range(count):
            if index >= len(self._buf):
                break
            size = self._unit_size_at(index)
            if size >= len(self._buf) - index:
                break  # only the newest unit is left
            for offset in range(index, index + size):
                msg = self._buf[offset]
                if msg.role != "assistant" or not msg.thinking:
                    continue
                if only is not None and id(msg) not in only:
                    continue
                self._buf[offset] = dataclasses.replace(msg, thinking="")
                changed += 1
            index += size
        return changed

    def _unit_size_at(self, start: int) -> int:
        """Messages in the unit that starts at *start* (see :func:`group_turns`)."""
        unit = [self._buf[start]]
        for msg in itertools.islice(self._buf, start + 1, None):
            if not _joins(unit, msg, self._empty_prefix):
                break
            unit.append(msg)
        return len(unit)

    def _evict(self) -> None:
        while len(self._buf) > self._max_len:
            size = self._leading_unit_size()
            if size >= len(self._buf):
                return  # only the newest unit is left
            for _ in range(size):
                self._buf.popleft()

    def _leading_unit_size(self) -> int:
        """Messages in the oldest unit of the buffer (see :func:`group_turns`)."""
        unit = [self._buf[0]]
        for msg in itertools.islice(self._buf, 1, None):
            if not _joins(unit, msg, self._empty_prefix):
                break
            unit.append(msg)
        return len(unit)

    # ── Reflection entry (single-latest) ──

    def add_reflection(self, entry: ReflectionEntry) -> None:
        """Replace the current reflection with *entry*.

        Reflections are compressed state, not a log — the planner sees at
        most one prior reflection per attempt.
        """
        self._reflection = entry

    def get_reflections(self) -> list[ReflectionEntry]:
        """Return the latest reflection in a list (``[]`` or ``[entry]``)."""
        return [self._reflection] if self._reflection is not None else []

    def clear_reflections(self) -> None:
        self._reflection = None

    def clear_all(self) -> None:
        """Clear both conversation buffer and the latest reflection."""
        self._buf.clear()
        self._reflection = None


# Backward-compatibility alias
ShortTerm = WorkingMemory
