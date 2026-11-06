"""When the computer is idle: the gates the steps that work on their own share.

:func:`idle_reason` says whether the computer is idle. :class:`~.idle_sleep.IdleSleep`
(tidies the notes) and :class:`~.idle_encode.IdleEncode` (writes memory items from the
chats) both ask it before they touch the memory, and never work otherwise. It is idle
when all of this holds:

- the shared GPU lock file is absent (read only, never written);
- no turn started or ended, and no session was changed, in the last :data:`IDLE_S` (20
  minutes) in the gateway's chats, and no turn runs now (a terminal chat that runs on its own,
  and workers outside the gateway, are not seen);
- no worker task is queued, running, held or parked;
- the computer is on AC power (an unreadable or unnamed power source counts as no).

:class:`IdleStep` is what the steps share besides: a loop that looks about every
:data:`POLL_S` seconds, and the one lock they take turns with, so two of them never work
on the memory at once.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from localmelo.support.gateway.machine_monitor import (
    parse_battery,
    read_lock_file,
    read_pmset,
)
from localmelo.support.sessions.store import SessionStoreError

if TYPE_CHECKING:
    from . import Console

logger = logging.getLogger(__name__)

POLL_S: Final = 60.0
IDLE_S: Final = 20 * 60.0
# The newest sessions looked at for a running turn.
SESSIONS_SEEN: Final = 100
# A task in one of these states is running or will run: the computer is not idle.
BUSY_TASKS: Final = ("queued", "running", "held", "parked")


@dataclass(frozen=True)
class IdleOptions:
    """What a test may replace: the clock, the beats, and the readers of the lock and the power.

    *lock_present* answers whether the GPU lock file exists (the ``[scheduler] gpu_lock_file``
    when None); *pmset* gives the text of ``pmset -g batt`` (the real command when None).
    """

    clock: Callable[[], float] = time.time
    poll_s: float = POLL_S
    idle_s: float = IDLE_S
    lock_present: Callable[[], bool] | None = None
    pmset: Callable[[], str] | None = None


async def idle_reason(console: Console, options: IdleOptions, now: float) -> str:
    """Why the computer is not idle at *now* (the first gate that says no, cheapest
    first): ``lock``, ``active``, ``tasks`` or ``battery``; ``""`` when it is idle."""
    if _lock_present(console, options):
        return "lock"
    if not await _away(console, options, now):
        return "active"
    if not await _no_tasks(console):
        return "tasks"
    if not await _on_ac(options):
        return "battery"
    return ""


def _lock_present(console: Console, options: IdleOptions) -> bool:
    """Whether the shared GPU lock file exists, whoever holds it (read only)."""
    if options.lock_present is not None:
        return options.lock_present()
    scheduler = console.cfg.scheduler
    if not scheduler.gpu_lock_file:
        return False
    found = read_lock_file(
        scheduler.gpu_lock_file,
        now_wall=options.clock(),
        self_holder=scheduler.lock_self,
    )
    return found.present


async def _away(console: Console, options: IdleOptions, now: float) -> bool:
    """Whether no chat has had a turn, or a change, for :attr:`~IdleOptions.idle_s`, and none runs."""
    try:
        sessions = await console.store.list_sessions(
            limit=SESSIONS_SEEN, kinds=("chat", "front")
        )
    except SessionStoreError:
        return False
    # A turn's start and end both set the session's time; the newest session is first.
    return not any(s.busy or now - s.updated_at < options.idle_s for s in sessions)


async def _no_tasks(console: Console) -> bool:
    try:
        busy = await console.workers.tasks.list_tasks(BUSY_TASKS)
    except SessionStoreError:
        return False
    return not busy


async def _on_ac(options: IdleOptions) -> bool:
    read = options.pmset or read_pmset
    try:
        text = await asyncio.to_thread(read)
    except Exception:
        return False
    return parse_battery(text).on_ac is True


class IdleStep:
    """A step that works beside the gateway when the computer is idle.

    :meth:`start` begins a loop that calls :meth:`tick` every ``options.poll_s``
    seconds. A subclass says what one look does in :meth:`tick` and holds :attr:`busy`
    while it works. *busy* is the lock the steps take turns with: pass one step's to
    the next, so they share it.
    """

    # What the log calls the step when a look fails.
    label = "idle step"

    def __init__(
        self,
        console: Console,
        options: IdleOptions,
        busy: asyncio.Lock | None = None,
    ) -> None:
        self.console = console
        self.options = options
        # Why the last look did not work (``""`` after a run); for tests and the log.
        self.blocked = ""
        self.busy = asyncio.Lock() if busy is None else busy
        self._task: asyncio.Task[None] | None = None

    @property
    def clock(self) -> Callable[[], float]:
        return self.options.clock

    # ── Lifecycle ──

    def start(self) -> None:
        """Begin looking; calling it again does nothing."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def aclose(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.options.poll_s)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("The %s failed", self.label)

    async def tick(self) -> Any:
        """One look: work when the computer is idle."""
        raise NotImplementedError
