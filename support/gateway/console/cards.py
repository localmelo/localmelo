"""Cards in the main chat: what a side chat or a task came back with.

When a side chat's run ends, a task finishes or fails, or either waits for an
approval, the main chat gets a card: an ``attention.card`` event that belongs
to no turn. A card is an event, never a turn: no model reads it, and the main
chat's window never holds it. Its data is ``state`` (``done``, ``failed`` or
``waiting``), the ``source_id`` and ``source_kind`` (``chat`` or ``task``) it
opens, the source's ``title`` and one plain ``line``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

from localmelo.support.sessions.display import activity_text, status_text
from localmelo.support.sessions.store import StoredEvent

from .activity import result_line

if TYPE_CHECKING:
    from . import Console

logger = logging.getLogger(__name__)

CARD_EVENT: Final = "attention.card"
# Newest cards a page of the main chat carries at most.
CARD_PAGE: Final = 50
_TASK_ENDS: Final = {"done": "done", "failed": "failed", "expired": "failed"}
# Events heard and not yet flushed, at most: the newest are kept.
_HEARD_MAX: Final = 500
# The data keys a card is built from.
_READ: Final = ("turn", "status", "cause", "request_id", "state")


def card_json(event: StoredEvent) -> dict[str, Any]:
    """One card as the page reads it: its event id, time and data."""
    data = event.event.get("data")
    return {
        "id": event.id,
        "ts": event.event.get("ts"),
        **(data if isinstance(data, dict) else {}),
    }


class MainChatCards:
    """Hears the store's events; turns those that come back into cards.

    The listener only notes an event; :meth:`flush` writes the cards, from a
    route that reads the main chat or the list, the idle sweep, and the
    console's close. So no write runs behind a request's back, and a waiting
    card whose approval was answered meanwhile is never written. ``at`` in a
    card's data is when its source event happened.
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._main_id: str | None = None
        self._heard_events: list[StoredEvent] = []
        # The newest card, for the list's unread dot.
        self.last: dict[str, Any] | None = None

    async def start(self, main_id: str) -> None:
        """Begin listening for *main_id*, reading its newest card once."""
        if self._main_id is None:
            self._console.store.add_listener(self._heard)
        self._main_id = main_id
        found = await self._console.store.events_of_type(main_id, CARD_EVENT, limit=1)
        self.last = card_json(found[-1]) if found else None

    def _heard(self, stored: StoredEvent) -> None:
        main = self._main_id
        kind = stored.event.get("type")
        if main is None:
            return
        if stored.session_id == main:
            if kind == CARD_EVENT:
                self.last = card_json(stored)
            return
        if kind in ("turn.finished", "approval.requested", "sched.finished"):
            # Only what a card reads: an approval's arguments can be large.
            data = stored.event.get("data") or {}
            light = {key: data.get(key) for key in _READ if key in data}
            event = {"type": kind, "ts": stored.event.get("ts"), "data": light}
            self._heard_events.append(
                StoredEvent(stored.id, stored.session_id, stored.turn, event)
            )
            del self._heard_events[:-_HEARD_MAX]

    async def flush(self) -> None:
        """Write a card for each event heard since the last flush."""
        main = self._main_id
        if main is None:
            return
        heard, self._heard_events = self._heard_events, []
        for stored in heard:
            # A card is best effort: a source deleted meanwhile, a store that
            # closed or a task that is gone leaves the main chat without it.
            try:
                card = await self._build(stored)
                if card is not None:
                    at = stored.event.get("ts")
                    await self._console.store.append_session_event(
                        main, CARD_EVENT, {**card, "at": at}
                    )
            except Exception as exc:
                logger.debug("No card in the main chat: %s", exc)

    async def _build(self, stored: StoredEvent) -> dict[str, Any] | None:
        store = self._console.store
        info = await store.get_session(stored.session_id)
        if info is None or info.kind not in ("chat", "task"):
            return None
        kind = stored.event.get("type")
        data = stored.event.get("data") or {}
        card = {"source_id": info.id, "source_kind": info.kind, "title": info.title}
        if kind == "approval.requested":
            asked = data.get("request_id")
            if not any(
                request.request_id == asked
                for request in self._console.broker.pending(info.id)
            ):
                return None  # answered or withdrawn before anyone saw it
            return {**card, "state": "waiting", "line": _waiting_line()}
        if kind == "turn.finished":
            # A side chat's own run; a task's turns end in its sched event.
            if info.kind != "chat" or data.get("cause") != "run":
                return None
            turn = int(data.get("turn") or 0)
            state = "done" if data.get("status") == "completed" else "failed"
        elif info.kind == "task" and data.get("state") in _TASK_ENDS:
            turn, state = 0, _TASK_ENDS[str(data.get("state"))]
        else:
            return None
        turns = await store.list_turns(
            info.id, limit=1, before=turn + 1 if turn else None
        )
        last = turns[-1] if turns else None
        if state == "done":
            line = result_line("", last.answer) if last is not None else ""
        elif last is not None and last.status != "completed":
            line = status_text(last.status, last.cause).title
        else:
            record = await self._console.workers.tasks.get_task(info.id)
            line = record.wait_reason or "Stopped before it finished"
        return {**card, "state": state, "line": line}


def _waiting_line() -> str:
    return activity_text(state="waiting").text


__all__ = ["CARD_EVENT", "CARD_PAGE", "MainChatCards", "card_json"]
