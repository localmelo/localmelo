"""Task-owned tool execution hook; infrastructure may attach a durable fence."""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable

TOOL_ENTRY: contextvars.ContextVar[Callable[[], Awaitable[None]] | None] = (
    contextvars.ContextVar("TOOL_ENTRY", default=None)
)

# Set only while invoking a tool whose entry callback has committed.
_ENTRY_MARKED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_ENTRY_MARKED", default=False
)
