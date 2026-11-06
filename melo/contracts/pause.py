"""Holding a run while the person does what only they can do (a page they took over).

A tool that hands something to the person (melo's browser hands them a page to
sign in on) holds the run with :class:`RunPause`, which the agent sets in
:data:`PAUSE` around each tool call of a run:

* :meth:`RunPause.hold` puts the run on hold: the run reports ``run.paused``,
  and no model call and no tool of the run starts (each waits in
  :meth:`RunPause.wait`, the gate before every send to the model and every
  tool start; a tool already running goes on) until :meth:`RunPause.release`,
  which reports ``run.resumed`` with how it ended.
  Holding again while held only says what changed (a new ``run.paused``). A
  hold records who set it (*by*): a release that names another holder does
  nothing, so one holder never ends another's hold.
* :meth:`RunPause.suspended` is for a tool call that waits for the person
  itself: inside it neither the call's time limit (:data:`TOOL_DEADLINE`, which
  the executor sets around a tool's body) nor the run's own counts. Nothing
  counts down while the person has the run.
* A run that ends while held releases the hold as ``stopped`` (the person's
  Stop) or ``ended`` (it ended another way) and calls the holder's *on_end*
  with that, so the holder can take back what it handed over.

Stop is never held back: a cancellation goes through a hold as it goes through
any await.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from contextvars import ContextVar
from typing import Any, Literal, Protocol

# How a hold ended: the person said so (``done``), closed what they were given
# (``closed``), what they had went away by itself (``gone``: a page closed its own
# window) or with what showed it (``lost``: the browser stopped), the run was
# stopped (``stopped``) or it ended another way (``ended``).
Resumed = Literal["done", "closed", "gone", "lost", "stopped", "ended"]


class RunPause(Protocol):
    """One run's hold (see the module docstring)."""

    @property
    def running(self) -> bool:
        """Whether the run is still going: a hold of a run that ended does nothing."""
        ...

    @property
    def held(self) -> bool:
        """Whether the run is on hold now."""
        ...

    def hold(
        self,
        what: str,
        detail: Mapping[str, Any] | None = None,
        on_end: Callable[[Resumed], None] | None = None,
        *,
        by: object | None = None,
    ) -> None:
        """Put the run on hold for *what* (``browser``), with *detail* for the surfaces; *by* is the holder."""
        ...

    def release(self, how: Resumed, *, by: object | None = None) -> None:
        """End the hold; nothing happens when the run is not held, or when *by* is not its holder."""
        ...

    async def wait(self) -> None:
        """The gate before a model call: wait while the run is held, no time limit counting."""
        ...

    def suspended(self) -> AbstractAsyncContextManager[None]:
        """Inside a tool call: no time limit counts while the body waits."""
        ...


# Set by the agent around each tool call of a run; None outside one.
PAUSE: ContextVar[RunPause | None] = ContextVar("localmelo_pause", default=None)
# Set by the executor around a tool's body: the call's time limit, which a hold stops.
TOOL_DEADLINE: ContextVar[asyncio.Timeout | None] = ContextVar(
    "localmelo_tool_deadline", default=None
)
