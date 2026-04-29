"""Shared aiosqlite helpers for the persistent memory backends."""

from __future__ import annotations

import aiosqlite

_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA busy_timeout=5000",
    "PRAGMA foreign_keys=ON",
    "PRAGMA synchronous=NORMAL",
)


async def apply_pragmas(conn: aiosqlite.Connection) -> None:
    """Apply the standard PRAGMA set on a freshly opened connection."""
    for stmt in _PRAGMAS:
        await conn.execute(stmt)
