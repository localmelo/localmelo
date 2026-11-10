"""Stream a session's stored events to subscribers, without gaps or repeats.

:class:`SessionHub` serves the gateway's event streams. It hears every
commit of the gateway's store through a listener, because the gateway is
the only process that writes its file, and replays from the store what a
subscriber has not seen yet: after a reconnect (``Last-Event-ID``), and
after its bounded queue overflowed.

It also carries the live text of a reply as it is written (:mod:`.live`) to
the subscribers that show a session to a person, and nobody else: those
frames are not stored and have no id. A subscriber that joins, or re-reads
after an overflow, gets the text so far once, then the rest.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Final

from localmelo.support.sessions.live import live_frame
from localmelo.support.sessions.store import SessionStore, StoredEvent

# What a subscriber's queue holds besides events: the hub closed, or the
# queue overflowed and was emptied.
_CLOSED: Final = "closed"
_OVERFLOW: Final = "overflow"
_PAGE: Final = 1000


def _live_key(data: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    """The request a live frame's text belongs to: its turn, step and attempt."""
    return data.get("turn"), data.get("step"), data.get("attempt")


class _Subscriber:
    def __init__(self, queue_size: int, viewer: bool = False) -> None:
        self.queue: asyncio.Queue[StoredEvent | str] = asyncio.Queue(maxsize=queue_size)
        # A surface that shows the session to a person (#465); only it gets
        # the live text of a reply.
        self.viewer = viewer
        # The request whose live text this subscriber has, and how much of it.
        self.live_key: tuple[Any, Any, Any] | None = None
        self.live_sent = 0

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
        # Each session's reply being written: its last live frame's key data
        # and the text so far of that request (``at`` 0).
        self._live: dict[str, dict[str, Any]] = {}
        self._closed = False
        store.add_listener(self._heard)

    def _heard(self, event: StoredEvent) -> None:
        if self._closed or event.id is None:
            return
        self._settled(event)
        for subscriber in list(self._subscribers.get(event.session_id, ())):
            subscriber.offer(event)

    def _settled(self, event: StoredEvent) -> None:
        """Forget the live text that *event* makes part of the record.

        A step's stored ``assistant.message`` is the finished copy of what it
        streamed (or of a later step's, which a fast tool lets start first),
        and ``turn.finished`` ends the turn's.
        """
        live = self._live.get(event.session_id)
        if live is None or live.get("turn") != event.turn:
            return
        kind, data = event.event.get("type"), event.event.get("data")
        step = data.get("step") if isinstance(data, Mapping) else None
        if kind == "turn.finished" or (
            kind == "assistant.message"
            and isinstance(step, int)
            and step >= int(live.get("step") or 0)
        ):
            del self._live[event.session_id]

    def live(self, frame: StoredEvent) -> None:
        """Send one live frame of a reply (:func:`.live.live_frame`) to the viewers.

        The hub keeps the text so far of the request it belongs to, for a
        viewer that joins later.
        """
        data = frame.event.get("data")
        if self._closed or frame.id is not None or not isinstance(data, Mapping):
            return
        at, text = data.get("at"), data.get("text")
        if not isinstance(at, int) or not isinstance(text, str):
            return
        live = self._live.get(frame.session_id)
        if live is None or _live_key(live) != _live_key(data):
            if at != 0:
                return  # the start of this request's text was never heard
            live = {**data, "text": ""}
        live["text"] = live["text"][:at] + text
        self._live[frame.session_id] = live
        for subscriber in list(self._subscribers.get(frame.session_id, ())):
            if subscriber.viewer:
                subscriber.offer(frame)

    def _so_far(self, session_id: str, subscriber: _Subscriber) -> StoredEvent | None:
        """The text so far of *session_id*'s reply, as one frame from ``at`` 0."""
        live = self._live.get(session_id)
        if live is None or not subscriber.viewer:
            return None
        subscriber.live_key, subscriber.live_sent = _live_key(live), len(live["text"])
        return live_frame(session_id, live["turn"], {**live, "at": 0})

    def _live_part(
        self, frame: StoredEvent, subscriber: _Subscriber
    ) -> StoredEvent | None:
        """*frame* as *subscriber* needs it: without what it has, or None.

        Each request's frames reach a subscriber in one run from ``at`` 0, with
        no repeat: a part it has is dropped, and a gap is filled with the text
        so far.
        """
        data = frame.event["data"]
        at, text = data["at"], data["text"]
        if _live_key(data) != subscriber.live_key:
            if at != 0:
                return self._so_far(frame.session_id, subscriber)
            subscriber.live_key, subscriber.live_sent = _live_key(data), len(text)
            return frame
        have = subscriber.live_sent
        if at > have:
            return self._so_far(frame.session_id, subscriber)
        if at + len(text) <= have:
            return None
        subscriber.live_sent = at + len(text)
        part = {**data, "at": have, "text": text[have - at :]}
        return live_frame(frame.session_id, data["turn"], part)

    def viewers(self, session_id: str) -> int:
        """How many subscribers of *session_id* show it to a person now (#465)."""
        return sum(
            1
            for subscriber in self._subscribers.get(session_id, ())
            if subscriber.viewer
        )

    async def subscribe(
        self, session_id: str, *, after_id: int = 0, viewer: bool = False
    ) -> AsyncGenerator[StoredEvent, None]:
        """The session's events with ids above *after_id*: stored ones, then live.

        The subscriber is registered before the replay, so an event stored
        during it is either replayed or queued, and is yielded once. A
        *viewer* also gets the live frames of a reply (``id`` ``None``), the
        text so far first.
        Closing the iterator unregisters it; :meth:`aclose` ends it.
        *viewer*: the subscriber shows the session to a person (the console,
        the app, the terminal), which :meth:`viewers` counts.
        """
        if self._closed:
            return
        subscriber = _Subscriber(self._queue_size, viewer)
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
                # A viewer gets the reply's text so far once, after the record.
                so_far = self._so_far(session_id, subscriber)
                if so_far is not None:
                    yield so_far
                # Then live, until the queue overflows or the hub closes.
                while True:
                    item = await subscriber.queue.get()
                    if item == _CLOSED:
                        return
                    if item == _OVERFLOW:
                        break  # re-read from the last id
                    assert isinstance(item, StoredEvent)
                    if item.id is None:
                        part = self._live_part(item, subscriber)
                        if part is not None:
                            yield part
                        continue
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
        self._live.clear()
        self._store.remove_listener(self._heard)
        for subscribers in self._subscribers.values():
            for subscriber in subscribers:
                subscriber.offer(_CLOSED)
