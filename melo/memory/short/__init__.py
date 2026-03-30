from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from localmelo.melo.schema import Message

if TYPE_CHECKING:
    from localmelo.melo.schema import ReflectionEntry


class WorkingMemory:
    """Conversation window + structured reflection entries.

    The conversation buffer (deque) behaves identically to the old ShortTerm.
    Reflection entries persist across attempts within one task; they are
    cleared only when promoted to long-term memory at task termination.
    """

    def __init__(self, max_len: int) -> None:
        self._buf: deque[Message] = deque(maxlen=max_len)
        self._reflections: list[ReflectionEntry] = []

    # ── Conversation buffer (backward-compatible) ──

    def append(self, msg: Message) -> None:
        self._buf.append(msg)

    def get_window(self) -> list[Message]:
        return list(self._buf)

    def clear(self) -> None:
        """Clear conversation buffer only. Reflections survive."""
        self._buf.clear()

    # ── Reflection entries ──

    def add_reflection(self, entry: ReflectionEntry) -> None:
        self._reflections.append(entry)

    def get_reflections(self) -> list[ReflectionEntry]:
        return list(self._reflections)

    def clear_reflections(self) -> None:
        self._reflections.clear()

    def clear_all(self) -> None:
        """Clear both conversation buffer and reflections."""
        self._buf.clear()
        self._reflections.clear()


# Backward-compatibility alias
ShortTerm = WorkingMemory
