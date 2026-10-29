"""Server-Sent Events for a session's stored and live events.

One frame per event: ``id: <event id>`` and ``data: <the event JSON>``, on
one line, then a blank line. The stream announces ``retry: 2000`` once,
sends ``: keepalive`` every :data:`SSE_KEEPALIVE_S` seconds, and ends when
the client goes away, the gateway shuts down, or (with ``follow=false``)
the stored events run out. A worker session lives in a file another process
writes, which the hub cannot see: its stream reads the store about every
:data:`SSE_POLL_S` seconds instead, and ends when the run is no longer live.

The generator keeps one pending ``__anext__`` task across keepalive ticks
and never cancels it on a timeout: cancelling it would finalize the hub's
generator and end the stream. :class:`EventStreamResponse` does the
clean-up in a shielded ``finally``, so a disconnect does not depend on
garbage collection.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Final

import anyio
from starlette.responses import StreamingResponse

from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore, StoredEvent

# Seconds between ``: keepalive`` comments.
SSE_KEEPALIVE_S: Final = 15.0
# Seconds between reads of a store the hub cannot see.
SSE_POLL_S: Final = 1.0
# The reconnect delay a client is told once, in milliseconds.
SSE_RETRY_MS: Final = 2000
# Events per replay page.
SSE_PAGE: Final = 1000
_RETRY_FRAME: Final = f"retry: {SSE_RETRY_MS}\n\n"
_KEEPALIVE_FRAME: Final = ": keepalive\n\n"

logger = logging.getLogger(__name__)


def event_data(event: StoredEvent) -> dict[str, Any]:
    """The JSON object one SSE frame carries, and the one a turn lists."""
    return {
        "id": event.id,
        "session_id": event.session_id,
        "turn": event.turn,
        **event.event,
    }


def frame(event: StoredEvent) -> str:
    """One SSE frame: its id, one data line, and a blank line."""
    data = json.dumps(
        event_data(event),
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"id: {event.id}\ndata: {data}\n\n"


class EventStream:
    """The body of an SSE response: a replay, then live events.

    *follow* ``False`` reads the stored events after *after_id* and ends;
    ``True`` subscribes to the hub and waits, sending keepalives and ending
    when *shutdown* is set. With *live*, a following stream reads the store
    every *poll_s* seconds instead and ends once ``await live()`` is false
    and no event is left. :meth:`aclose` is safe to call twice and is what
    :class:`EventStreamResponse` calls on the way out.
    """

    def __init__(
        self,
        store: SessionStore,
        hub: SessionHub,
        shutdown: asyncio.Event,
        session_id: str,
        *,
        after_id: int = 0,
        follow: bool = True,
        high_water: int | None = None,
        on_close: Callable[[], None] | None = None,
        live: Callable[[], Awaitable[bool]] | None = None,
        poll_s: float | None = None,
    ) -> None:
        self._store = store
        self._hub = hub
        self._shutdown = shutdown
        self._session_id = session_id
        self._after_id = after_id
        self._follow = follow
        self._high_water = high_water
        self._on_close = on_close
        self._live = live
        self._poll_s = SSE_POLL_S if poll_s is None else poll_s
        self._quiet_since = time.monotonic()
        self._iterator: AsyncIterator[StoredEvent] | None = (
            hub.subscribe(session_id, after_id=after_id)
            if follow and live is None
            else None
        )
        self._pending: asyncio.Task[StoredEvent] | None = None
        self._shutdown_wait: asyncio.Task[bool] | None = None
        self._buffer: list[StoredEvent] = []
        self._last = after_id
        self._started = False
        self._closed = False

    def __aiter__(self) -> EventStream:
        return self

    async def __anext__(self) -> str:
        if self._closed:
            raise StopAsyncIteration
        if not self._started:
            self._started = True
            return _RETRY_FRAME
        if self._follow and self._live is not None:
            return await self._poll_next(self._live)
        if self._follow:
            return await self._follow_next()
        return await self._replay_next()

    async def _next_event(self) -> StoredEvent:
        assert self._iterator is not None
        return await anext(self._iterator)

    async def _follow_next(self) -> str:
        if self._pending is None:
            self._pending = asyncio.ensure_future(self._next_event())
        if self._shutdown_wait is None:
            self._shutdown_wait = asyncio.ensure_future(self._shutdown.wait())
        done, _ = await asyncio.wait(
            {self._pending, self._shutdown_wait},
            timeout=SSE_KEEPALIVE_S,
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Shutdown wins over a ready event: a session with a steady stream
        # must not keep the gateway waiting for it to catch up.
        if self._shutdown_wait in done or self._shutdown.is_set():
            raise StopAsyncIteration
        if self._pending in done:
            task, self._pending = self._pending, None
            try:
                event = task.result()
            except StopAsyncIteration:
                raise StopAsyncIteration from None
            self._last = event.id if event.id is not None else self._last
            return frame(event)
        # A deleted session ends the stream at the next keepalive; a
        # reconnect then gets the route's 404.
        try:
            exists = await self._store.get_session(self._session_id) is not None
        except Exception:
            # A store error is not a deletion: keep the connection alive and
            # try again at the next tick. A reconnect gets the 503 if the
            # store is still unavailable then.
            logger.warning(
                "Could not check session %s during a keepalive",
                self._session_id,
                exc_info=True,
            )
            return _KEEPALIVE_FRAME
        if not exists:
            raise StopAsyncIteration
        return _KEEPALIVE_FRAME

    async def _poll_next(self, live: Callable[[], Awaitable[bool]]) -> str:
        while not self._buffer:
            if self._shutdown.is_set():
                raise StopAsyncIteration
            try:
                # Liveness first: an event written before the run ended is
                # then still read below, and one written after is not missed.
                running = await live()
                page = await self._store.list_events(
                    self._session_id, after_id=self._last, limit=SSE_PAGE
                )
            except Exception:
                # A store error is not the end of the run: try again.
                logger.warning(
                    "Could not poll session %s", self._session_id, exc_info=True
                )
                running, page = True, []
            if page:
                self._buffer = list(page)
                break
            if not running:
                raise StopAsyncIteration
            if time.monotonic() - self._quiet_since >= SSE_KEEPALIVE_S:
                self._quiet_since = time.monotonic()
                return _KEEPALIVE_FRAME
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._shutdown.wait(), self._poll_s)
        event = self._buffer.pop(0)
        self._last = event.id if event.id is not None else self._last
        self._quiet_since = time.monotonic()
        return frame(event)

    async def _replay_next(self) -> str:
        while not self._buffer:
            if self._shutdown.is_set():
                raise StopAsyncIteration
            if self._high_water is not None and self._last >= self._high_water:
                raise StopAsyncIteration
            page = await self._store.list_events(
                self._session_id, after_id=self._last, limit=SSE_PAGE
            )
            if self._high_water is not None:
                # One-shot catch-up stops at the id the request read, so a
                # turn that starts while pages arrive cannot add frames.
                page = [
                    event
                    for event in page
                    if event.id is not None and event.id <= self._high_water
                ]
            if not page:
                raise StopAsyncIteration
            self._buffer = list(page)
        event = self._buffer.pop(0)
        self._last = event.id if event.id is not None else self._last
        return frame(event)

    async def aclose(self) -> None:
        """Cancel the pending read and unregister the subscription."""
        if self._closed:
            return
        self._closed = True
        pending, self._pending = self._pending, None
        if pending is not None and not pending.done():
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending
        shutdown_wait, self._shutdown_wait = self._shutdown_wait, None
        if shutdown_wait is not None and not shutdown_wait.done():
            shutdown_wait.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await shutdown_wait
        iterator, self._iterator = self._iterator, None
        if iterator is not None:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
        if self._on_close is not None:
            self._on_close()
            self._on_close = None


class EventStreamResponse(StreamingResponse):
    """A ``StreamingResponse`` whose clean-up runs even on a disconnect.

    Starlette cancels the body task on a disconnect; the shielded
    ``finally`` still cancels the pending hub read and unregisters it.
    """

    def __init__(self, stream: EventStream, **kwargs: Any) -> None:
        super().__init__(stream, **kwargs)
        self._stream = stream

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self._stream.aclose()
