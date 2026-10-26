"""Stream a session's stored events to subscribers, without gaps or repeats.

:class:`SessionHub` serves the gateway's event streams. It hears every
commit of the gateway's store through a listener, because the gateway is
the only process that writes its file, and replays from the store what a
subscriber has not seen yet: after a reconnect (``Last-Event-ID``), and
after its bounded queue overflowed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Final

from localmelo.support.sessions.store import SessionStore, StoredEvent

# What a subscriber's queue holds besides events: the hub closed, or the
# queue overflowed and was emptied.
_CLOSED: Final = "closed"
_OVERFLOW: Final = "overflow"
_PAGE: Final = 1000


class _Subscriber:
    def __init__(self, queue_size: int) -> None:
        self.queue: asyncio.Queue[StoredEvent | str] = asyncio.Queue(maxsize=queue_size)

    def offer(self, item: StoredEvent | str) -> None:
        """Queue *item*; a full queue is emptied and marked instead."""
        if self.queue.full():
            while not self.queue.empty():
                self.queue.get_nowait()
            if item != _CLOSED:
                item = _OVERFLOW
        self.queue.put_nowait(item)


class SessionHub:
    """Live and replayed events of the gateway's sessions.

    *store* must be the gateway's store: the hub sees only this process's
    commits. Each subscriber has a queue of *queue_size* events; one that
    overflows is emptied, and the subscriber re-reads the store from its
    last event before it goes on live, so a slow client neither misses nor
    repeats an event while memory stays bounded.
    """

    def __init__(self, store: SessionStore, *, queue_size: int = 1000) -> None:
        if store.writer != "gateway":
            raise ValueError(
                "SessionHub needs the gateway's store: it sees only this process's commits"
            )
        if (
            isinstance(queue_size, bool)
            or not isinstance(queue_size, int)
            or queue_size < 1
        ):
            raise ValueError(f"queue_size must be an int >= 1, got {queue_size!r}")
        self._store = store
        self._queue_size = queue_size
        self._subscribers: dict[str, set[_Subscriber]] = {}
        self._closed = False
        store.add_listener(self._heard)

    def _heard(self, event: StoredEvent) -> None:
        if self._closed or event.id is None:
            return
        for subscriber in list(self._subscribers.get(event.session_id, ())):
            subscriber.offer(event)

    async def subscribe(
        self, session_id: str, *, after_id: int = 0
    ) -> AsyncGenerator[StoredEvent, None]:
        """The session's events with ids above *after_id*: stored ones, then live.

        The subscriber is registered before the replay, so an event stored
        during it is either replayed or queued, and is yielded once.
        Closing the iterator unregisters it; :meth:`aclose` ends it.
        """
        if self._closed:
            return
        subscriber = _Subscriber(self._queue_size)
        self._subscribers.setdefault(session_id, set()).add(subscriber)
        last = after_id
        try:
            while True:
                # Replay what the store holds past the last id yielded.
                while True:
                    page = await self._store.list_events(
                        session_id, after_id=last, limit=_PAGE
                    )
                    for event in page:
                        assert event.id is not None
                        last = event.id
                        yield event
                    if len(page) < _PAGE:
                        break
                # Then live, until the queue overflows or the hub closes.
                while True:
                    item = await subscriber.queue.get()
                    if item == _CLOSED:
                        return
                    if item == _OVERFLOW:
                        break  # re-read from the last id
                    assert isinstance(item, StoredEvent) and item.id is not None
                    if item.id <= last:
                        continue  # replayed already
                    last = item.id
                    yield item
        finally:
            subscribers = self._subscribers.get(session_id)
            if subscribers is not None:
                subscribers.discard(subscriber)
                if not subscribers:
                    del self._subscribers[session_id]

    async def aclose(self) -> None:
        """End every subscription and stop listening; idempotent."""
        if self._closed:
            return
        self._closed = True
        self._store.remove_listener(self._heard)
        for subscribers in self._subscribers.values():
            for subscriber in subscribers:
                subscriber.offer(_CLOSED)
