"""Task-owned tool execution hook; infrastructure may attach a durable record.

``TOOL_ENTRY`` holds, for a task's run, what the executor enters around a
call that can change something: given the tool's name and whether the
call's effect stays where melo can look at it afterwards (``local``, decided
by the executor before the call runs), an async context that records the
call durably before it starts and closes that record when the call returns,
however it returns. A call it cannot record does not run.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

TOOL_ENTRY: contextvars.ContextVar[
    Callable[[str, bool], AbstractAsyncContextManager[None]] | None
] = contextvars.ContextVar("TOOL_ENTRY", default=None)

# Set only while invoking a tool whose entry record has committed.
_ENTRY_MARKED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_ENTRY_MARKED", default=False
)
