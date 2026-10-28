"""Read a conversation's cross-window change mark without a bounded turn window."""

from __future__ import annotations

from dataclasses import dataclass

from localmelo.support.sessions.store import SessionStore


@dataclass(frozen=True)
class SessionMark:
    foreign_newest: int
    undone: int


async def session_mark(
    store: SessionStore, session_id: str, *, holder: str
) -> SessionMark:
    """Ignore this process's finished turns, but notice every undo."""
    return SessionMark(*await store.change_counters(session_id, holder=holder))
