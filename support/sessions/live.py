"""A gateway turn's reply as it is written: the turn's live text.

:class:`LiveReply` is the core's stream display
(:class:`~localmelo.melo.contracts.streaming.StreamDisplay`) for one turn the
gateway runs. What the core releases to it, already filtered and redacted,
goes out as ``reply.live`` frames through *publish* (the hub's
:meth:`~localmelo.support.sessions.hub.SessionHub.live`): the first piece of a
request at once, then at most one frame every :data:`LIVE_FRAME_S`. A frame is
for the screen only: it is never stored, it has no event id, and the turn's
stored events stay its record (the finished reply is its ``assistant.message``
and ``run.finished``, as without a stream).

A frame's ``data``: ``turn`` and ``turn_id``; ``step``, the planning step;
``attempt``, 1 for the step's first request and more for a retry, whose text
starts over; ``at``, where ``text`` starts in that request's text so far
(``0`` starts it, or gives all of it again to a listener that joins late).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from typing import Any, Final, cast

from localmelo.melo.contracts.events import EVENT_SCHEMA_VERSION
from localmelo.melo.schema import ReasoningTags
from localmelo.support.sessions.store import StoredEvent, TurnInfo

# The type of a live frame; the core and the store have no event of this type.
LIVE_EVENT: Final = "reply.live"
# Seconds between two frames of one request: a fast model does not flood a page.
LIVE_FRAME_S: Final = 0.04

Publish = Callable[[StoredEvent], None]
_FRAMINGS: Final = frozenset({"explicit", "closing_only", "none"})


def declared_framing(tags: Mapping[str, str], model: str) -> ReasoningTags | None:
    """*model*'s framing as ``[cli] reasoning_tags`` declares it, else ``None``.

    ``None`` keeps the core's conservative rule: a reply's text is held until
    it ends, unless the server sends the model's reasoning in a field of its own.
    """
    hint = tags.get(model)
    return cast(ReasoningTags, hint) if hint in _FRAMINGS else None


def live_frame(session_id: str, turn: int, data: Mapping[str, Any]) -> StoredEvent:
    """A ``reply.live`` frame: an event of the turn that is not stored (no id)."""
    return StoredEvent(
        None,
        session_id,
        turn,
        {"v": EVENT_SCHEMA_VERSION, "type": LIVE_EVENT, "data": dict(data)},
    )


class LiveReply:
    """The stream display of one gateway turn (see the module docstring).

    *reasoning_tags* is the model's declared framing (``[cli] reasoning_tags``),
    or ``None`` for the core's conservative rule. :attr:`first_word_s` is the
    seconds from the turn's start to the first piece with a visible character,
    ``None`` while none was shown. Call :meth:`close` once the run is over:
    the display shows nothing after it, also to work that copied the run's
    context (a task its turn started).
    """

    def __init__(
        self,
        turn: TurnInfo,
        publish: Publish,
        *,
        reasoning_tags: ReasoningTags | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.reasoning_tags = reasoning_tags
        self.first_word_s: float | None = None
        self._turn = turn
        self._publish = publish
        self._clock = clock
        self._step = 0
        self._attempts: dict[int, int] = {}
        self._sent = 0  # characters of the open request already published
        self._pending = ""
        self._timer: asyncio.TimerHandle | None = None
        self._closed = False

    def step(self, number: int) -> None:
        if self._closed:
            return
        self._flush()
        self._step = number

    def opened(self) -> None:
        if self._closed:
            return
        self._flush()
        self._attempts[self._step] = self._attempts.get(self._step, 0) + 1
        self._sent = 0

    def held(self) -> None:
        """Text the core does not release yet: nothing to show."""

    def content(self, delta: str) -> None:
        if not delta or self._closed:
            return
        self._pending += delta
        if self._sent == 0:
            self._flush()  # a request's first words show at once
        elif self._timer is None:
            self._timer = asyncio.get_running_loop().call_later(
                LIVE_FRAME_S, self._flush
            )

    def closed(self) -> None:
        self._flush()

    def close(self) -> None:
        """The run is over: what is pending goes out, no timer stays, and nothing more shows."""
        self._flush()
        self._closed = True

    def _flush(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if not self._pending:
            return
        text, self._pending = self._pending, ""
        at, self._sent = self._sent, self._sent + len(text)
        if self.first_word_s is None and text.strip():
            started = self._turn.started_at
            self.first_word_s = round(max(0.0, self._clock() - started), 3)
        info = self._turn
        self._publish(
            live_frame(
                info.session_id,
                info.turn,
                {
                    "turn": info.turn,
                    "turn_id": info.id,
                    "step": self._step,
                    "attempt": self._attempts.get(self._step, 1),
                    "at": at,
                    "text": text,
                },
            )
        )
