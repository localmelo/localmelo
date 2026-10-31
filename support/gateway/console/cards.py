"""Cards in the main chat: what a side chat or a task came back with.

When a side chat's run ends, a task finishes or fails, or either waits for an
approval, the main chat gets a card: an ``attention.card`` event that belongs
to no turn. A card is an event, never a turn: no model reads it, and the main
chat's window never holds it. Its data is ``state`` (``done``, ``failed``,
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

import logging
from typing import TYPE_CHECKING, Any, Final

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
from localmelo.support.sessions.store import StoredEvent

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
        # The approvals with a waiting card and no answer yet; read once per flush.
        self._unanswered: set[str] | None = None
        # The newest card, for the list's unread dot.
        self._last: StoredEvent | None = None

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
        if kind in (
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
        heard, self._heard_events = self._heard_events, []
        self._unanswered = None
        # What is stored reads the same to every reader: English, whoever asked.
        with using("en"):
            await self._write(main, heard)

    async def _write(self, main: str, heard: list[StoredEvent]) -> None:
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
            return await self._answer_card(card, data)
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
        self, card: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """The card that follows a waiting one with its answer, or ``None``.

        There is none when no waiting card of this approval was written (it
        was answered before anything read the main chat), or when it was
        followed already.
        """
        asked = data.get("request_id")
        if self._unanswered is None:
            self._unanswered = await self._read_unanswered()
        if not asked or asked not in self._unanswered:
            return None
        self._unanswered.discard(asked)
        name = _ANSWERS.get(answer_word(data), "answered")
        if card["source_kind"] == "task" and name == "allowedForChat":
            name = "allowedForTask"
        return {
            **card,
            "state": "answered",
            "request_id": asked,
            **_words(f"cards.answer.{name}"),
        }

    async def _read_unanswered(self) -> set[str]:
        """The request ids of the newest waiting cards that no answer follows."""
        if self._main_id is None:
            return set()
        found = await self._console.store.events_of_type(
            self._main_id, CARD_EVENT, limit=_LOOK_BACK
        )
        waiting: set[str] = set()
        for event in found:
            item = event.event.get("data") or {}
            asked = item.get("request_id")
            if not asked:
                continue
            if item.get("state") == "waiting":
                waiting.add(asked)
            elif item.get("state") == "answered":
                waiting.discard(asked)
        return waiting


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
