"""Sleep on its own: tidy the person's notes while the computer is idle, and say so in the main chat.

``[memory] sleep`` (``off`` by default) says how :meth:`ChatMemory.sleep
<localmelo.support.memory.ChatMemory.sleep>` tidies the notes. :class:`IdleSleep` runs beside
the gateway and, about every :data:`POLL_S` seconds, runs one sleep when all of this holds, and
never touches the notes otherwise:

- the setting is not ``off`` and memory across chats is on (with ``off`` the loop reads nothing);
- the last idle run began at least :data:`SPACING_S` (6 hours) ago; its time is kept in the
  memory index's ``meta`` table (``idle_sleep_at``), so a restart does not run it early;
- the shared GPU lock file is absent (read only, never written);
- no chat or main-chat turn started or ended, and no session was changed, in the last
  :data:`IDLE_S` (20 minutes), and no turn runs now;
- no worker task is queued, running, held or parked;
- the computer is on AC power (an unreadable or unnamed power source counts as no).

A run that changed at least one note adds one ``sleep`` echo to the main chat, "While you were
away, melo tidied 3 notes", with what changed (at most :data:`SHOWN` lines) and Undo, which
puts the whole run back (:meth:`Front._act_sleep
<localmelo.support.gateway.console.front.Front._act_sleep>`). A run that changed nothing
leaves a log line only. :meth:`ChatMemory.sleep` stops by itself when the person comes back
during a run; what it changed by then still counts.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from localmelo.support.gateway.i18n import said, t_in
from localmelo.support.gateway.machine_monitor import (
    parse_battery,
    read_lock_file,
    read_pmset,
)
from localmelo.support.sessions.store import SessionStoreError

from .errors import ApiError
from .routes_memory import identity_changed

if TYPE_CHECKING:
    from localmelo.support.memory.sleep import Change, SleepReport

    from . import Console

logger = logging.getLogger(__name__)

POLL_S: Final = 60.0
IDLE_S: Final = 20 * 60.0
SPACING_S: Final = 6 * 3600.0
META_KEY: Final = "idle_sleep_at"
# The newest sessions looked at for a running turn.
SESSIONS_SEEN: Final = 100
# What the row lists of a run, and how much of each note.
SHOWN: Final = 5
TEXT_CHARS: Final = 120
# A task in one of these states is running or will run: the computer is not idle.
BUSY_TASKS: Final = ("queued", "running", "held", "parked")
# The word the row shows for each kind of change (the catalog's ``mainChat.echo.sleepOp.*``).
CHANGE_WORDS: Final = {
    "O1": "merged",
    "O2": "updated",
    "O3": "added",
    "O4": "reworded",
    "ADD": "added",
    "UPDATE": "updated",
    "RETIRE": "retired",
}


@dataclass(frozen=True)
class IdleSleepOptions:
    """What a test may replace: the clock, the beats, and the readers of the lock and the power.

    *lock_present* answers whether the GPU lock file exists (the ``[scheduler] gpu_lock_file``
    when None); *pmset* gives the text of ``pmset -g batt`` (the real command when None).
    """

    clock: Callable[[], float] = time.time
    poll_s: float = POLL_S
    idle_s: float = IDLE_S
    spacing_s: float = SPACING_S
    lock_present: Callable[[], bool] | None = None
    pmset: Callable[[], str] | None = None


def _cut(text: str) -> str:
    """One note's words on one line, cut at :data:`TEXT_CHARS`."""
    words = " ".join(text.split())
    return words if len(words) <= TEXT_CHARS else words[: TEXT_CHARS - 1].rstrip() + "…"


def _item(change: Change) -> dict[str, str]:
    return {"change": CHANGE_WORDS.get(change.op, "changed"), "text": _cut(change.text)}


class IdleSleep:
    """Runs sleep when the computer is idle, and tells the main chat."""

    def __init__(
        self, console: Console, options: IdleSleepOptions | None = None
    ) -> None:
        self.console = console
        self.options = options or IdleSleepOptions()
        self.clock = self.options.clock
        # Why the last look did not run a sleep (``""`` after a run); for tests and the log.
        self.blocked = ""
        self._task: asyncio.Task[None] | None = None
        self._busy = asyncio.Lock()

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
                logger.exception("The idle sleep failed")

    # ── One look ──

    async def tick(self) -> SleepReport | None:
        """Run one sleep when the computer is idle; its report, or ``None`` when it did not run."""
        async with self._busy:
            self.blocked = await self._reason()
            if self.blocked:
                return None
            return await self._run()

    async def _reason(self) -> str:
        """Why no sleep runs now (the first gate that says no, cheapest first), or ``""``."""
        console, now = self.console, self.clock()
        cfg = console.cfg
        if cfg.memory.sleep == "off" or not console.memory.enabled:
            return "off"
        if console.store_state() != "ok" or console.main_id is None:
            return "store"
        last = await console.memory.stores()[1].meta(META_KEY)
        if 0 <= now - last < self.options.spacing_s:
            return "spacing"
        if self._lock_present():
            return "lock"
        if not await self._away(now):
            return "active"
        if not await self._no_tasks():
            return "tasks"
        if not await self._on_ac():
            return "battery"
        return ""

    def _lock_present(self) -> bool:
        """Whether the shared GPU lock file exists, whoever holds it (read only)."""
        if self.options.lock_present is not None:
            return self.options.lock_present()
        scheduler = self.console.cfg.scheduler
        if not scheduler.gpu_lock_file:
            return False
        found = read_lock_file(
            scheduler.gpu_lock_file,
            now_wall=self.clock(),
            self_holder=scheduler.lock_self,
        )
        return found.present

    async def _away(self, now: float) -> bool:
        """Whether no chat has had a turn, or a change, for :attr:`~IdleSleepOptions.idle_s`, and none runs."""
        try:
            sessions = await self.console.store.list_sessions(
                limit=SESSIONS_SEEN, kinds=("chat", "front")
            )
        except SessionStoreError:
            return False
        # A turn's start and end both set the session's time; the newest session is first.
        return not any(
            s.busy or now - s.updated_at < self.options.idle_s for s in sessions
        )

    async def _no_tasks(self) -> bool:
        try:
            busy = await self.console.workers.tasks.list_tasks(BUSY_TASKS)
        except SessionStoreError:
            return False
        return not busy

    async def _on_ac(self) -> bool:
        read = self.options.pmset or read_pmset
        try:
            text = await asyncio.to_thread(read)
        except Exception:
            return False
        return parse_battery(text).on_ac is True

    # ── One run ──

    async def _run(self) -> SleepReport | None:
        console = self.console
        memory = console.memory
        # The time is kept before the run, so a run that fails is not tried again for hours.
        await memory.stores()[1].set_meta(META_KEY, self.clock())
        pinned = await memory.pinned_quotes()
        try:
            report = await memory.sleep(None, chats=console.store)
        except Exception:
            logger.exception("The idle sleep could not tidy the notes")
            return None
        await identity_changed(console, pinned)
        logger.info(
            "Idle sleep %s: %d changes%s",
            report.run_id,
            len(report.changes),
            f", stopped: {report.aborted}" if report.aborted else "",
        )
        if report.changes:
            await self._tell(report)
        return report

    async def _tell(self, report: SleepReport) -> None:
        """The main chat's one line: how many notes, what changed, and Undo."""
        count = len(report.changes)
        data: dict[str, Any] = {
            "echo": "sleep",
            "run_id": report.run_id,
            "count": count,
            "variant": report.variant,
            "line": t_in("en", "front.echo.slept", n=count),
            "said": said("front.echo.slept", n=count),
            "items": [_item(change) for change in report.changes[:SHOWN]],
        }
        if count > SHOWN:
            data["more"] = count - SHOWN
        try:
            await self.console.front.echo(data)
        except (ApiError, SessionStoreError):
            logger.warning("The main chat could not say what the idle sleep tidied")
