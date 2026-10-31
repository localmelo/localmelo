"""Cards: what a side chat or a task came back with.

When a side chat's run ends, a task finishes or fails, or either waits for an
approval, a chat gets a card: an ``attention.card`` event that belongs to no
turn. A side chat's own run comes back to the main chat. A task comes back to
the chat that started it, the main chat or a side chat, and when a side
chat finishes work (its own answer, or one of its tasks) the main chat's
next turn also reads one line about it (:meth:`Front.side_done
<localmelo.support.gateway.console.front.Front.side_done>`). A card is an
event, never a turn: no model reads it, and a chat's window never holds it. Its data is ``state`` (``done``, ``failed``,
``stopped``, ``waiting`` or ``answered``), the ``source_id`` and ``source_kind`` (``chat``
or ``task``) it opens, the source's short ``title`` and its plain ``line``: up
to two lines of the answer on a done card, the reason on a failed one, what is
left on a stopped one (a task that ran out of steps, which can continue), what
is asked on a waiting one.

A waiting card carries its ``request_id``. When that approval is answered, a
card with state ``answered``, the same ``request_id`` and the answer in words as
its ``line`` follows it: the page shows the one card with the answer in place
of its Review. An answer is no news, so it never counts as the newest card.

A card is stored in English. One whose line is the console's own words also
keeps them as ``said`` (a catalog key and its parameters, see
:func:`~localmelo.support.gateway.i18n.said`), and is read in the language
the console page asks for.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Final

from localmelo.melo.contracts.review import TITLES
from localmelo.support.gateway.i18n import (
    current,
    said,
    said_in,
    said_text,
    t_in,
    using,
)
from localmelo.support.sessions.display import (
    TITLE_WIDTH,
    clean_output,
    request_said,
    short_title,
    status_text,
    title_width,
)
from localmelo.support.sessions.store import SessionInfo, StoredEvent

from ..auto_blocks import blocked_steps
from ..chore_store import REMINDER_PREFIX, REMINDER_REASON
from ..finishing import NOT_CONTINUED, is_out_of_steps, not_committed
from ..task_store import TaskStoreError
from .activity import answer_lines, answer_word, left_line

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
_READ: Final = (
    "turn",
    "status",
    "cause",
    "request_id",
    "state",
    "tool_name",
    "outcome",
    "origin",
    "remember",
)
# What a waiting card needs of an approval's arguments: where, never what.
_WHERE: Final = ("path", "url")
# Cards of the main chat looked through for the one an answer follows.
_LOOK_BACK: Final = 200
_ERROR_CHARS: Final = 100
# How an approval's answer reads on a card (the catalog's ``cards.answer.*``);
# a grant for a task is for the task.
_ANSWERS: Final = {
    "allowed once": "allowedOnce",
    "allowed for this chat": "allowedForChat",
    "allowed automatically": "allowedAutomatically",
    "denied": "denied",
    "no answer": "noAnswer",
    "withdrawn": "withdrawn",
}


def _words(key: str, **params: str) -> dict[str, Any]:
    """A card's line in English, with its words kept for the reader's language."""
    return {"line": t_in("en", key, **params), "said": said(key, **params)}


def card_json(event: StoredEvent) -> dict[str, Any]:
    """One card as the page reads it: its event id, time and data, its line in the reader's language."""
    data = event.event.get("data")
    card = {
        "id": event.id,
        "ts": event.event.get("ts"),
        **(data if isinstance(data, dict) else {}),
    }
    words = card.get("said")
    if current() != "en" and isinstance(words, dict):
        card["line"] = said_text(words)
    return card


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
        # The approvals with a waiting card and no answer yet, by the chat that
        # holds the card; read once per flush.
        self._unanswered: dict[str, set[str]] | None = None
        # The newest card, for the list's unread dot.
        self._last: StoredEvent | None = None
        self._flushing = asyncio.Lock()

    @property
    def last(self) -> dict[str, Any] | None:
        """The newest card, its line in the language of the request that reads it."""
        return None if self._last is None else card_json(self._last)

    async def start(self, main_id: str) -> None:
        """Begin listening for *main_id*, reading its newest card once."""
        if self._main_id is None:
            self._console.store.add_listener(self._heard)
        self._main_id = main_id
        found = await self._console.store.events_of_type(main_id, CARD_EVENT, limit=20)
        news = [card for card in found if _is_news(card.event.get("data"))]
        self._last = news[-1] if news else None

    def _heard(self, stored: StoredEvent) -> None:
        main = self._main_id
        kind = stored.event.get("type")
        if main is None:
            return
        if stored.session_id == main:
            if kind == CARD_EVENT and _is_news(stored.event.get("data")):
                self._last = stored
            return
        if kind == "tool.finished":
            # A step auto mode blocked in a task: its chat shows it, with Allow once (#369).
            approval = (stored.event.get("data") or {}).get("approval")
            if not (
                isinstance(approval, dict)
                and approval.get("origin") == "auto"
                and approval.get("allowed") is False
            ):
                return
            event = {
                "type": kind,
                "ts": stored.event.get("ts"),
                "data": {"request_id": approval.get("request_id")},
            }
            self._heard_events.append(
                StoredEvent(stored.id, stored.session_id, stored.turn, event)
            )
            del self._heard_events[:-_HEARD_MAX]
        elif kind in (
            "turn.finished",
            "approval.requested",
            "approval.resolved",
            "sched.finished",
        ):
            # Only what a card reads: an approval's arguments can be large.
            data = stored.event.get("data") or {}
            light = {key: data.get(key) for key in _READ if key in data}
            arguments = data.get("arguments")
            if isinstance(arguments, dict):
                where = {
                    key: arguments[key]
                    for key in _WHERE
                    if isinstance(arguments.get(key), str)
                }
                light["arguments"] = {k: v[:500] for k, v in where.items()}
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
        # One flush at a time, so a caller that finds nothing left to write
        # returns only when the writing another one began is done.
        async with self._flushing:
            await self._write_heard(main)

    async def _write_heard(self, main: str) -> None:
        heard, self._heard_events = self._heard_events, []
        self._unanswered = None
        # What is stored reads the same to every reader: English, whoever asked.
        with using("en"):
            await self._write(main, heard)

    @contextlib.asynccontextmanager
    async def holding(self) -> AsyncIterator[None]:
        """Write what is heard, then write no card until the block ends.

        What is heard meanwhile waits, and is written after it to whoever owns
        its task by then: a chat that changes hands inside the block (a deleted
        side chat's tasks) is not written to.
        """
        main = self._main_id
        async with self._flushing:
            if main is not None:
                await self._write_heard(main)
            yield

    async def _write(self, main: str, heard: list[StoredEvent]) -> None:
        for stored in heard:
            # A card is best effort: a source deleted meanwhile, a store that
            # closed or a task that is gone leaves the main chat without it.
            try:
                built = await self._build(main, stored)
                if built is not None:
                    owner, card = built
                    at = stored.event.get("ts")
                    await self._console.store.append_session_event(
                        owner, CARD_EVENT, {**card, "at": at}
                    )
                    await self._side_done(main, owner, card, at)
            except Exception as exc:
                logger.debug("No card: %s", exc)

    async def _side_done(
        self, main: str, owner: str, card: dict[str, Any], at: object
    ) -> None:
        """A side chat's finished run, or its finished task, is news for the main chat."""
        chat = card["source_id"] if card["source_kind"] == "chat" else owner
        if card.get("state") != "done" or card.get("reminder") or chat == main:
            return
        info = await self._console.store.get_session(chat)
        if info is not None:
            when = float(at) if isinstance(at, int | float) else time.time()
            await self._console.front.side_done(
                chat, card_title(info.title), card["line"], when, card["source_kind"]
            )

    async def _owner(self, main: str, info: SessionInfo) -> str:
        """The chat a card goes to: for a task the one that started it, else the main chat."""
        if info.kind != "task":
            return main
        try:
            front = (await self._console.workers.tasks.get_task(info.id)).front_id
        except TaskStoreError:
            return main
        return await self._console.effective_owner(front) or main

    async def _build(
        self, main: str, stored: StoredEvent
    ) -> tuple[str, dict[str, Any]] | None:
        store = self._console.store
        info = await store.get_session(stored.session_id)
        if info is None or info.kind not in ("chat", "task"):
            return None
        owner = await self._owner(main, info)
        card = await self._card(owner, info, stored)
        return None if card is None else (owner, card)

    async def _card(
        self, owner: str, info: SessionInfo, stored: StoredEvent
    ) -> dict[str, Any] | None:
        store = self._console.store
        kind = stored.event.get("type")
        data = stored.event.get("data") or {}
        card: dict[str, Any] = {
            "source_id": info.id,
            "source_kind": info.kind,
            "title": card_title(info.title),
        }
        if kind == "approval.requested":
            asked = data.get("request_id")
            if not any(
                request.request_id == asked
                for request in self._console.broker.pending(info.id)
            ):
                return None  # answered or withdrawn before anyone saw it
            return {
                **card,
                "state": "waiting",
                "request_id": asked,
                **_waiting_words(data),
            }
        if kind == "approval.resolved":
            return await self._answer_card(owner, card, data)
        if kind == "tool.finished":
            return await self._blocked_card(info, card, str(data.get("request_id")))
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
        if last is not None and last.origin:
            return None  # a side chat answering a task's question is no news (#369)
        ended: dict[str, Any]
        note = ""
        if state == "failed" and info.kind == "task":
            note = (await self._console.workers.tasks.get_task(info.id)).wait_reason
            if is_out_of_steps(state, note):
                state = "stopped"  # out of steps: it can continue, so it did not fail
        if state == "done":
            ended = {"line": answer_lines(last.answer) if last is not None else ""}
        elif state == "stopped":
            answer = last.answer if last is not None else ""
            line = left_line(answer) or answer_lines(answer)
            ended = {"line": line} if line else _words("cards.outOfSteps")
        elif note.startswith(NOT_CONTINUED):
            # A continued run that never began: its last turn is an older one.
            ended = {"line": note}
        elif last is not None and last.status != "completed":
            ended = {"line": _reason(last.status, last.cause, last.result)}
        else:
            record = await self._console.workers.tasks.get_task(info.id)
            if record.wait_reason == "invalid_workspace":
                problem = await self._console.workers.folder_problem(info.id)
                ended = _words("cards.folderRefused", problem=problem)
            elif record.wait_reason:
                ended = {"line": record.wait_reason}
            else:
                ended = _words("cards.stoppedEarly")
        if info.kind == "task" and last is not None:
            # A task whose commit is all that is left says so, with the command.
            left = await not_committed(last.answer, info.workspace)
            ended = {"line": left} if left else ended
        words = await self._reminder_words(info.id) if info.kind == "task" else None
        if words is not None:
            # A reminder's card shows the person's words even when its run failed.
            card["reminder"] = True
            if state == "failed":
                ended = _words("cards.reminderFailed", words=words)
        return {**card, "state": state, **ended}

    async def _blocked_card(
        self, info: SessionInfo, card: dict[str, Any], request_id: str
    ) -> dict[str, Any] | None:
        """A task's step auto mode blocked: the block as the task's page has it."""
        if info.kind != "task":
            return None  # a chat's own block is a card in its own turn
        found = await blocked_steps(self._console.store, info.id)
        block = next((b for b in found if b["request_id"] == request_id), None)
        if block is None:
            return None
        review = block["review"] or {}
        words = said("cards.blocked", what=said(f"auto.words.{_first(review)}"))
        return {
            **card,
            "state": "blocked",
            "line": said_in("en", words),
            "said": words,
            **{k: block[k] for k in ("request_id", "step_key", "name", "preview")},
            "review": review,
            "can_allow": block["can_allow"],
        }

    async def _reminder_words(self, task_id: str) -> str | None:
        """The person's words of a reminder's task, or ``None`` for any other task."""
        try:
            record = await self._console.workers.tasks.get_task(task_id)
        except TaskStoreError:
            return None
        if record.reason != REMINDER_REASON:
            return None
        first = str(record.spec.get("objective", "")).split("\n\n")[0]
        return first.removeprefix(REMINDER_PREFIX)

    async def _answer_card(
        self, owner: str, card: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """The card that follows a waiting one with its answer, or ``None``.

        There is none when no waiting card of this approval was written (it
        was answered before anything read the chat that holds it), or when it
        was followed already.
        """
        asked = data.get("request_id")
        if self._unanswered is None:
            self._unanswered = {}
        if owner not in self._unanswered:
            self._unanswered[owner] = await self._read_unanswered(owner)
        if not asked or asked not in self._unanswered[owner]:
            return None
        self._unanswered[owner].discard(asked)
        name = _ANSWERS.get(answer_word(data), "answered")
        if card["source_kind"] == "task" and name == "allowedForChat":
            name = "allowedForTask"
        return {
            **card,
            "state": "answered",
            "request_id": asked,
            **_words(f"cards.answer.{name}"),
        }

    async def waiting_in(self, owner: str) -> list[dict[str, Any]]:
        """The newest waiting cards of chat *owner* that no answer follows, as stored."""
        found = await self._console.store.events_of_type(
            owner, CARD_EVENT, limit=_LOOK_BACK
        )
        waiting: dict[str, dict[str, Any]] = {}
        for event in found:
            item = event.event.get("data") or {}
            asked = item.get("request_id")
            if not asked:
                continue
            if item.get("state") == "waiting":
                waiting[asked] = item
            elif item.get("state") == "answered":
                waiting.pop(asked, None)
        return list(waiting.values())

    async def _read_unanswered(self, owner: str) -> set[str]:
        """The request ids of the newest waiting cards of *owner* that no answer follows."""
        return {str(card["request_id"]) for card in await self.waiting_in(owner)}


def _first(review: dict[str, Any]) -> str:
    """The review's first category, or ``beyond_request`` when it named none."""
    found = review.get("categories")
    first = found[0] if isinstance(found, list) and found else ""
    return str(first) if first in TITLES else "beyond_request"


def _is_news(data: object) -> bool:
    """False for a card that only says how an approval was answered."""
    return not (isinstance(data, dict) and data.get("state") == "answered")


def card_title(title: str) -> str:
    """A card's title: a title that is short stays, a long one is cut at a word."""
    title = " ".join(title.split())
    return title if title_width(title) <= TITLE_WIDTH else short_title(title)


def _reason(status: str, cause: str, result: dict[str, Any] | None) -> str:
    """Why a run did not finish: its status in words, then its error's first line."""
    words = status_text(status, cause).title
    error = result.get("error") if isinstance(result, dict) else None
    first = next(
        (row for row in clean_output(str(error or "")).splitlines() if row.strip()), ""
    )
    first = " ".join(first.split())
    if not first:
        return words
    if len(first) > _ERROR_CHARS:
        first = first[: _ERROR_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return f"{words}: {first}"


def _waiting_words(data: dict[str, Any]) -> dict[str, Any]:
    """What the approval is for, in plain words: ``Wants to run a command``, with its words kept."""
    asked = request_said(str(data.get("tool_name") or ""), data.get("arguments"))
    key = str(asked["key"])
    if key.startswith("steps.ask."):
        verb = key.removeprefix("steps.ask.")
        words = said(f"cards.wants.{verb}", **asked["params"])
    else:
        words = said("cards.wants.use", what=said("steps.aTool"))
    return {"line": said_in("en", words), "said": words}


__all__ = ["CARD_EVENT", "CARD_PAGE", "MainChatCards", "card_json", "card_title"]
