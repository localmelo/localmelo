"""The run's hold (:class:`~localmelo.melo.contracts.pause.RunPause`): the person has the run.

One per run. Every request of the run to the model waits in :meth:`Hold.wait`
first (the chat's request path calls it); a tool calls :meth:`Hold.hold` and
:meth:`Hold.release` (see the contract). While the run waits, or a tool waits inside
:meth:`Hold.suspended`, its time limits are moved on by the time it waited: an
``asyncio.Timeout`` is set to no deadline and given back what it had left.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from localmelo.melo.contracts.pause import TOOL_DEADLINE, Resumed

Paused = Callable[[str, dict[str, Any]], None]
ResumedSink = Callable[[str, str, float], None]


@dataclass
class _Held:
    what: str
    detail: dict[str, Any]
    since: float
    on_end: Callable[[Resumed], None] | None = field(default=None)
    by: object | None = None


class Hold:
    """A run's hold; *paused* and *resumed* report ``run.paused`` and ``run.resumed``."""

    def __init__(
        self,
        paused: Paused,
        resumed: ResumedSink,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._paused = paused
        self._resumed = resumed
        self._clock = clock
        self._held: _Held | None = None
        self._free = asyncio.Event()
        self._free.set()
        self._running = True
        # The run's own time limit (run_task's timeout_s), stopped while held.
        self.deadline: asyncio.Timeout | None = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def held(self) -> bool:
        return self._held is not None

    def hold(
        self,
        what: str,
        detail: Mapping[str, Any] | None = None,
        on_end: Callable[[Resumed], None] | None = None,
        *,
        by: object | None = None,
    ) -> None:
        if not self._running:
            return
        said = dict(detail or {})
        held = self._held
        if held is None:
            self._held = _Held(what, said, self._clock(), on_end, by)
            self._free.clear()
        elif said == held.detail and what == held.what:
            return
        else:
            held.what, held.detail = what, said
            held.on_end = on_end or held.on_end
            held.by = by if by is not None else held.by
        self._paused(what, said)

    def release(self, how: Resumed, *, by: object | None = None) -> None:
        held = self._held
        if held is None or (by is not None and held.by is not by):
            return
        self._held = None
        self._free.set()
        self._resumed(held.what, how, max(0.0, self._clock() - held.since))

    async def wait(self) -> None:
        """The gate before a model call: wait while the run is held, with no time limit counting."""
        if self._held is None:
            return
        async with self.suspended():
            await self._free.wait()

    @contextlib.asynccontextmanager
    async def suspended(self) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        stopped: list[tuple[asyncio.Timeout, float]] = []
        for timer in (self.deadline, TOOL_DEADLINE.get()):
            when = timer.when() if timer is not None else None
            if timer is None or when is None:
                continue  # none set, or one an outer hold stopped already
            with contextlib.suppress(RuntimeError):  # one not running any more
                timer.reschedule(None)
                stopped.append((timer, when - loop.time()))
        try:
            yield
        finally:
            now = loop.time()
            for timer, left in stopped:
                with contextlib.suppress(RuntimeError):
                    timer.reschedule(now + max(left, 0.0))

    def end(self, how: Resumed) -> None:
        """The run is over: a hold still on is released as *how*, and its holder told."""
        self._running = False
        held = self._held
        if held is None:
            return
        self.release(how)
        if held.on_end is not None:
            with contextlib.suppress(
                Exception
            ):  # a holder's failure never stops the run's end
                held.on_end(how)
