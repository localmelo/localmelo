"""Sleep on its own: tidy the person's notes while the computer is idle, and say so in the main chat.

``[memory] sleep`` (``off`` by default) says how :meth:`ChatMemory.sleep
<localmelo.support.memory.ChatMemory.sleep>` tidies the notes. :class:`IdleSleep` runs beside
the gateway and, about every :data:`~.idle_gates.POLL_S` seconds, runs one sleep when all of this holds, and
never touches the notes otherwise:

- the setting is not ``off`` and memory across chats is on (with ``off`` the loop reads nothing);
- the last idle run began at least :data:`SPACING_S` (6 hours) ago; its time is kept in the
  memory index's ``meta`` table (``idle_sleep_at``), so a restart does not run it early. A
  clock set back does not skip the spacing: a stored time up to :data:`CLOCK_BACK_S` (a day)
  ahead of the clock still blocks, and nothing writes the time while a run is blocked; a time
  further ahead than that is taken as a clock that was wrong, and the next run replaces it;
- the computer is idle (:func:`~.idle_gates.idle_reason`: the shared GPU lock file is
  absent, no chat had a turn or a change in the last 20 minutes and none runs, no worker
  task is queued, running, held or parked, and it is on AC power).

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from localmelo.support.gateway.i18n import said, t_in
from localmelo.support.sessions.store import SessionStoreError

from .errors import ApiError
from .idle_gates import IdleOptions, IdleStep, idle_reason
from .routes_memory import identity_changed

if TYPE_CHECKING:
    from localmelo.support.memory.sleep import Change, SleepReport

    from . import Console

logger = logging.getLogger(__name__)

SPACING_S: Final = 6 * 3600.0
# A stored run time this far ahead of the clock means the clock was set back: the spacing holds.
CLOCK_BACK_S: Final = 24 * 3600.0
META_KEY: Final = "idle_sleep_at"
# What the row lists of a run, and how much of each note.
SHOWN: Final = 5
TEXT_CHARS: Final = 120
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
class IdleSleepOptions(IdleOptions):
    """What a test may replace: the idle gates' options (the clock, the beats, and the
    readers of the lock and the power) and the spacing between two runs."""

    spacing_s: float = SPACING_S


def _cut(text: str) -> str:
    """One note's words on one line, cut at :data:`TEXT_CHARS`."""
    words = " ".join(text.split())
    return words if len(words) <= TEXT_CHARS else words[: TEXT_CHARS - 1].rstrip() + "…"


def _item(change: Change) -> dict[str, str]:
    return {"change": CHANGE_WORDS.get(change.op, "changed"), "text": _cut(change.text)}


class IdleSleep(IdleStep):
    """Runs sleep when the computer is idle, and tells the main chat."""

    label = "idle sleep"
    options: IdleSleepOptions

    def __init__(
        self,
        console: Console,
        options: IdleSleepOptions | None = None,
        busy: asyncio.Lock | None = None,
    ) -> None:
        super().__init__(console, options or IdleSleepOptions(), busy)
        self._running = False

    @property
    def running(self) -> bool:
        """Whether a sleep is tidying the notes now (they are locked until it ends)."""
        return self._running

    # ── One look ──

    async def tick(self) -> SleepReport | None:
        """Run one sleep when the computer is idle; its report, or ``None`` when it did not run."""
        async with self.busy:
            # Taken before the first gate: a turn that starts while the gates
            # are read still stops the run.
            began = self.clock()
            self.blocked = await self._reason(began)
            if self.blocked:
                return None
            return await self._run(began)

    async def _reason(self, now: float) -> str:
        """Why no sleep runs now (the first gate that says no, cheapest first), or ``""``."""
        console = self.console
        cfg = console.cfg
        if cfg.memory.sleep == "off" or not console.memory.enabled:
            return "off"
        if console.store_state() != "ok" or console.main_id is None:
            return "store"
        last = await console.memory.stores()[1].meta(META_KEY)
        if -CLOCK_BACK_S <= now - last < self.options.spacing_s:
            return "spacing"
        return await idle_reason(console, self.options, now)

    # ── One run ──

    async def _run(self, began: float) -> SleepReport | None:
        console = self.console
        memory = console.memory
        # The time is kept before the run, so a run that fails is not tried again for hours.
        await memory.stores()[1].set_meta(META_KEY, self.clock())
        pinned = await memory.pinned_quotes()
        self._running = True
        try:
            report = await memory.sleep(None, chats=console.store, began=began)
        except Exception:
            logger.exception("The idle sleep could not tidy the notes")
            return None
        finally:
            self._running = False
        await identity_changed(console, pinned)
        logger.info(
            "Idle sleep %s: %d changes%s",
            report.run_id,
            len(report.changes),
            f" ({report.aborted})" if report.aborted else "",
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
