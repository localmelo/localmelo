"""Encode chats into memory items on their own, while the computer is idle (#428).

With ``[memory] recall_source = "items"`` recall reads memory items and the turns no item
holds yet (:meth:`ChatMemory.waiting_turns
<localmelo.support.memory.ChatMemory.waiting_turns>`). Items come from the chat model,
which reads a chat's turns window by window (:meth:`ChatMemory.compress_chat
<localmelo.support.memory.ChatMemory.compress_chat>`). :class:`IdleEncode` runs beside
the gateway and, about every :data:`~.idle_gates.POLL_S` seconds, encodes while all of
this holds, and reads nothing otherwise:

- ``recall_source`` is ``items`` and memory across chats is on (with ``turns``, the
  default, the loop reads nothing);
- the chat model is a local one (the chat backend is in ``LOCAL_BACKENDS``): a hosted
  model costs money and would send every chat out again, so those chats stay turns;
- the computer is idle (:func:`~.idle_gates.idle_reason`), and a chat model is set up.

There is no spacing: a run goes on while a backlog exists, one window at a time, and
asks the gates again before each window, so it stops when the person comes back (the
window in flight ends first). The oldest chat goes first, by its first waiting turn, so
a newer statement can replace an older one. A window that fails stops the run: it is
tried again at the next look, never in a tight loop; after :data:`MAX_FAILURES`
failures in a row a chat is skipped until the gateway restarts, and its turns stay
recalled as turns. The run shares :attr:`~.idle_gates.IdleStep.busy` with the sleep, so
the two take turns. Nothing is written to the encoder's cache file, and no text of a
chat or an item is logged.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from localmelo.support.backends.openai_family import LOCAL_BACKENDS

from .idle_gates import IdleOptions, IdleStep, idle_reason

if TYPE_CHECKING:
    from localmelo.support.sessions.store import SessionInfo, TurnInfo

    from . import Console

logger = logging.getLogger(__name__)

# Failures in a row after which a chat is skipped until the gateway restarts.
MAX_FAILURES: Final = 3
# Every chat is looked at, as a reindex does.
CHATS_SEEN: Final = 100_000


@dataclass
class EncodeRun:
    """What one run did. *stopped* says why it ended: ``empty`` (nothing waits),
    ``failed`` (a window failed) or the gate that said no (``off``, ``hosted``,
    ``store``, ``lock``, ``active``, ``tasks``, ``battery``, ``model``)."""

    chats: set[str] = field(default_factory=set)
    windows: int = 0
    items: int = 0
    refused: int = 0
    failures: int = 0
    stopped: str = ""


class IdleEncode(IdleStep):
    """Writes memory items from the chats while the computer is idle."""

    label = "idle encoder"

    def __init__(
        self,
        console: Console,
        options: IdleOptions | None = None,
        busy: asyncio.Lock | None = None,
    ) -> None:
        super().__init__(console, options or IdleOptions(), busy)
        # Failures in a row for each chat; kept until the gateway restarts.
        self.failures: dict[str, int] = {}

    # ── One look ──

    async def tick(self) -> EncodeRun | None:
        """Encode while the computer is idle and chats wait; what it did, or ``None``
        when a gate said no at the first look."""
        async with self.busy:
            self.blocked = await self._reason()
            if self.blocked:
                return None
            return await self._run()

    async def _reason(self) -> str:
        """Why nothing is encoded now (the first gate that says no, cheapest first), or ``""``."""
        console = self.console
        cfg = console.cfg
        if cfg.memory.recall_source != "items" or not console.memory.enabled:
            return "off"
        if cfg.chat_backend not in LOCAL_BACKENDS:
            return "hosted"
        if console.store_state() != "ok" or console.main_id is None:
            return "store"
        reason = await idle_reason(console, self.options, self.clock())
        if not reason and console.memory.chat_provider() is None:
            return "model"
        return reason

    # ── One run ──

    async def _run(self) -> EncodeRun:
        run = EncodeRun()
        while not run.stopped:
            run.stopped = await self._window(run) or await self._reason()
        if run.windows or run.failures:
            logger.info(
                "Idle encode: %d chats, %d windows, %d items stored, %d refused, "
                "%d failures, stopped: %s",
                len(run.chats),
                run.windows,
                run.items,
                run.refused,
                run.failures,
                run.stopped,
            )
        return run

    async def _window(self, run: EncodeRun) -> str:
        """Encode one window of the oldest chat that has one. ``""`` when it was
        encoded, ``failed`` when it was not, and ``empty`` when no chat waits."""
        memory = self.console.memory
        for session, waiting in await self._backlog():
            try:
                out = await memory.encode_window(
                    self.console.store, session.id, session.title, waiting
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "The idle encoder could not read a window", exc_info=True
                )
                return self._failed(run, session.id)
            if out is None:  # nothing in this chat is worth reading: the next one
                continue
            if out.encoding.failed_from == 0:  # the first window failed: no mark
                return self._failed(run, session.id)
            self.failures.pop(session.id, None)
            run.chats.add(session.id)
            run.windows += 1
            run.items += sum(1 for _, route in out.routed if route.item is not None)
            run.refused += out.refused
            return ""
        return "empty"

    def _failed(self, run: EncodeRun, session_id: str) -> str:
        """Count one failure for the chat; the run ends, and the next look tries again."""
        self.failures[session_id] = self.failures.get(session_id, 0) + 1
        run.failures += 1
        return "failed"

    async def _backlog(self) -> list[tuple[SessionInfo, list[TurnInfo]]]:
        """The chats that have turns to encode with those turns, the chat whose first
        waiting turn is the oldest first. A chat skipped for its failures is not here.
        """
        console = self.console
        found: list[tuple[SessionInfo, list[TurnInfo]]] = []
        for session in await console.store.list_sessions(
            limit=CHATS_SEEN, kinds=("chat", "front")
        ):
            if self.failures.get(session.id, 0) >= MAX_FAILURES:
                continue
            waiting = await console.memory.waiting_turns(console.store, session.id)
            if waiting:
                found.append((session, waiting))
        found.sort(key=lambda each: each[1][0].started_at)
        return found
