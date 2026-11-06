"""The browser card: what a chat's browser says in the conversation.

When melo browses in a chat, the chat shows one card for the run, like a reply: the live page, what melo is doing
and who has the page. The card is an ``attention.browser`` event that belongs to no turn, stored in the chat that
owns it and delivered and replayed like its other cards (:mod:`.cards`): no second event store, no second stream.
Its data is the card's identity, ``card_key`` (the chat and the run) with ``owner_chat_id``, ``source_session_id``,
``source_turn``, ``run_id`` and ``browser_session_id`` (the browser tools' :attr:`generation
<localmelo.support.browser.provider.BrowserToolProvider.generation>`: a tab id means something only with it), and
what it says: ``state`` (``working``, ``waiting`` for the person's answer, ``you`` while they have the page,
``done``, ``stopped``, with ``by`` set to ``you`` for a Stop), ``action`` (the kind of step melo is on, which the
page words in the reader's language) and ``site`` (a bare host).

Nothing page content reaches it: no picture, typed value, full address or query string is stored, only the words
the program wrote (:meth:`BrowserToolProvider.card
<localmelo.support.browser.provider.BrowserToolProvider.card>`). The browser tools tell a change through
:meth:`~localmelo.support.browser.provider.BrowserToolProvider.watch` from inside their own step, so the change is
only noted there; a task of the loop writes it, one write for a burst, and only when the card differs from the one
last written. The run it belongs to is the one the chat's events say is running: a card begins with the first
browser step of a run, ends with its turn, and a run that never browses has none.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.sessions.store import StoredEvent

if TYPE_CHECKING:
    from . import Console

logger = logging.getLogger(__name__)

BROWSER_EVENT: Final = "attention.browser"
# The states of a card whose run is using the browser (the others are its end).
ACTIVE: Final = frozenset({"working", "waiting", "you"})
# The newest rows a page of a chat looks through for its browser cards (each row holds its whole card).
BROWSER_ROWS: Final = 400


def _retire(source: _Source) -> None:
    """The run of *source* is over: every stream pinned to it is told."""
    for ends in list(source.pins):
        ends.set()
    source.pins.clear()


@dataclass(eq=False)
class _Source:
    """One open chat's browser tools and the run they work in."""

    session_id: str
    provider: weakref.ReferenceType[BrowserToolProvider]
    generation: str
    run: str = ""
    turn: int = 0
    # The card last written for the run (None before its first), and how the run ended once it did.
    card: dict[str, Any] | None = None
    ended: tuple[str, str] | None = None
    dirty: bool = False
    task: asyncio.Task[None] | None = field(default=None, repr=False)
    # What is pinned to the run (see :meth:`BrowserCards.pin`); a stream that is gone is dropped from it by itself.
    pins: weakref.WeakSet[asyncio.Event] = field(
        default_factory=weakref.WeakSet, repr=False
    )


class BrowserCards:
    """Projects each open chat's browser into its ``attention.browser`` card. See the module docstring."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._sources: dict[str, _Source] = {}
        self._listening = False

    def listen(self) -> None:
        """Hear the store's events, for the run a chat is in and the end of its turn (once)."""
        if not self._listening:
            self._listening = True
            self._console.store.add_listener(self._heard)

    def attach(self, session_id: str, agent: object) -> None:
        """Follow the browser tools of *session_id*'s new Agent (a rebuilt Agent replaces the old ones)."""
        provider = next(
            (
                p
                for p in getattr(agent, "tool_providers", ())
                if isinstance(p, BrowserToolProvider)
            ),
            None,
        )
        old = self._sources.pop(session_id, None)
        if old is not None:
            _retire(old)
        if provider is None:
            return
        source = _Source(session_id, weakref.ref(provider), provider.generation)
        self._sources[session_id] = source
        provider.watch(lambda: self._poke(source))

    def is_current(self, session_id: str, run: str) -> bool:
        """Whether *run* is the run that session *session_id*'s browser works for now."""
        source = self._sources.get(session_id)
        return source is not None and source.run == run and source.ended is None

    def pin(self, session_id: str, run: str) -> asyncio.Event:
        """An event that is set once *run* is no longer the run of *session_id*'s browser, and is set already if it is not.

        A stream a card opened is for the card's run: the run's end, the next run of the chat and a rebuilt Agent
        set it, so the stream ends with its run and never carries the pictures of the next one.
        """
        ends = asyncio.Event()
        source = self._sources.get(session_id)
        if source is None or source.run != run or source.ended is not None:
            ends.set()
        else:
            source.pins.add(ends)
        return ends

    async def close_orphans(self, rows: list[StoredEvent]) -> list[StoredEvent]:
        """*rows* (each the newest of its card) with a card left active by a run that is gone closed as stopped.

        A gateway that stopped in the middle of a run never wrote its card's end; the card would read as working
        for ever. A run this gateway does not follow any more is over: its card is written as stopped, so the page
        and a replay agree.
        """
        shown = []
        for row in rows:
            data = row.event.get("data")
            if (
                isinstance(data, dict)
                and data.get("state") in ACTIVE
                and not self._running(row.session_id, str(data.get("run_id", "")))
            ):
                row = await self._console.store.append_session_event(
                    row.session_id,
                    BROWSER_EVENT,
                    {**data, "state": "stopped", "action": ""},
                )
            shown.append(row)
        return shown

    def _running(self, session_id: str, run: str) -> bool:
        source = self._sources.get(session_id)
        return source is not None and source.run == run

    # ── What the browser and the store tell ──

    def _poke(self, source: _Source) -> None:
        """Something the card says may have changed: note it, and write it soon (never blocks)."""
        if self._sources.get(source.session_id) is not source or not source.run:
            return
        if source.ended is None:
            self._schedule(source)

    def _heard(self, stored: StoredEvent) -> None:
        source = self._sources.get(stored.session_id)
        if source is None:
            return
        kind = stored.event.get("type")
        if kind == "run.started":
            _retire(source)
            source.run = str(stored.event.get("run_id") or "")
            source.turn = stored.turn or 0
            source.card, source.ended = None, None
        elif kind == "turn.finished" and source.run:
            data = stored.event.get("data")
            data = data if isinstance(data, dict) else {}
            if source.card is None:
                source.run = ""  # the run never browsed
                _retire(source)
                return
            done = data.get("status") == "completed"
            by = "you" if data.get("cause") == "user_stop" else ""
            source.ended = ("done" if done else "stopped", by)
            self._schedule(source)

    def _schedule(self, source: _Source) -> None:
        source.dirty = True
        if source.task is not None and not source.task.done():
            return
        with contextlib.suppress(RuntimeError):  # no running loop: nothing to write to
            source.task = asyncio.get_running_loop().create_task(self._flush(source))

    async def _flush(self, source: _Source) -> None:
        """Write the card as it is now, once for every change noted since (a burst is one write)."""
        while source.dirty:
            source.dirty = False
            card = self._card(source)
            if card is None or card == source.card:
                continue
            source.card = card
            try:
                await self._console.store.append_session_event(
                    source.session_id, BROWSER_EVENT, card
                )
            except Exception as exc:  # a card is best effort, as the others are
                logger.debug("No browser card: %s", exc)
        if source.ended is not None:
            source.run = ""
            _retire(
                source
            )  # after the card says the run is over: the card and the stream's end arrive in that order

    def _card(self, source: _Source) -> dict[str, Any] | None:
        """The card for the run now, or None when there is nothing to say yet."""
        found = source.card
        if source.ended is not None:
            if found is None:
                return None
            state, by = source.ended
            return {**found, "state": state, "action": "", **({"by": by} if by else {})}
        provider = source.provider()
        facts = provider.card() if provider is not None else None
        if facts is None:
            return None
        if found is None and not (facts.busy or facts.state != "working"):
            return None  # a page left open is no browsing: a step, or a person who has it, begins the card
        return {
            "card_key": f"{source.session_id}:{source.run}",
            "owner_chat_id": source.session_id,
            "source_session_id": source.session_id,
            "source_turn": source.turn,
            "run_id": source.run,
            "browser_session_id": source.generation,
            "state": facts.state,
            "action": facts.action,
            "site": facts.site,
        }


__all__ = ["BROWSER_EVENT", "BROWSER_ROWS", "BrowserCards"]
