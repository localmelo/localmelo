from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from localmelo.melo.schema import Message

if TYPE_CHECKING:
    from localmelo.melo.schema import ReflectionEntry


class WorkingMemory:
    """Conversation window + the single latest reflection entry.

    The conversation buffer (deque) behaves identically to the old ShortTerm.
    Only one reflection is retained at a time — a new reflection replaces
    the previous one. Treating reflections as compressed *current state*
    rather than a log keeps the planning prompt bounded across attempts.
    """

    def __init__(self, max_len: int) -> None:
        self._buf: deque[Message] = deque(maxlen=max_len)
        self._reflection: ReflectionEntry | None = None

    # ── Conversation buffer (backward-compatible) ──

    def append(self, msg: Message) -> None:
        self._buf.append(msg)

    def get_window(self) -> list[Message]:
        return list(self._buf)

    def clear(self) -> None:
        """Clear conversation buffer only. The latest reflection survives."""
        self._buf.clear()

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
