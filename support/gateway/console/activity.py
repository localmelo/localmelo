"""``GET /api/activity``: what ran today, for the panel beside a chat.

One list from four sources: the console's chats and the scheduler's tasks
(the gateway's file), and the worker runs and terminal chats (the CLI file,
read-only, a terminal chat redacted). Each session is one row, its newest
turn of the day: what it is doing now while it runs or waits, else how it
ended and a one-line result. Rows that run or wait come first, then the rest
newest first, at most :data:`MAX_ROWS`; ``more`` says some were left out. A
source that cannot be read adds nothing and is named in ``unavailable``.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Final

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from localmelo.support.sessions.display import (
    activity_text,
    split_thinking,
    status_text,
)
from localmelo.support.sessions.store import DayTurn, SessionStoreError

from .errors import ApiError
from .running import TASK_OUTCOMES, TITLE_CHARS, _turn_outcome, item_activity

if TYPE_CHECKING:
    from . import Console

MAX_ROWS: Final = 100
RESULT_CHARS: Final = 140
# How far back `since` may reach: a week.
MAX_SPAN_S: Final = 7 * 86400
LIVE: Final = ("running", "waiting", "held", "queued", "paused")
# A task's state as a row says it; a finished one as `status.recent` does.
TASK_STATES: Final = {
    "running": "running",
    "held": "held",
    "queued": "queued",
    "waiting": "queued",
    "parked": "paused",
    "interrupted": "paused",
    **TASK_OUTCOMES,
}
UNTITLED: Final = {
    "chat": "Chat",
    "task": "Task",
    "run": "Worker run",
    "terminal": "Terminal chat",
}
_MARKUP: Final = re.compile(r"^[\s#>*_`-]+")
_SENTENCE: Final = re.compile(r"(?<=[.!?])\s|(?<=[。！？])")


def result_line(note: str, answer: str) -> str:
    """The digest when there is one, else the answer's first sentence: one line, cut."""
    text = " ".join((note or split_thinking(answer or "")[1]).split())
    text = _MARKUP.sub("", text)
    first = _SENTENCE.split(text, maxsplit=1)[0]
    if len(first) <= RESULT_CHARS:
        return first
    cut = first[: RESULT_CHARS - 1].rsplit(" ", 1)[0] or first[: RESULT_CHARS - 1]
    return cut.rstrip(",;: ") + "…"


def local_midnight(now: float) -> float:
    """The start of *now*'s day on this computer's clock."""
    day = time.localtime(now)
    return time.mktime((day.tm_year, day.tm_mon, day.tm_mday, 0, 0, 0, 0, 0, -1))


def _row(
    kind: str,
    owner: str,
    title: str,
    state: str,
    at: float,
    **more: Any,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "id": owner,
        "title": (title or UNTITLED[kind])[:TITLE_CHARS],
        "state": state,
        "activity": "",
        "icon": more.pop("icon", state),
        "result": "",
        "at": at,
        **more,
    }


def _ended(turn: DayTurn) -> tuple[str, str, str]:
    """(state, icon, result) of a turn that ended: done reads by its last tool's kind."""
    if turn.status == "running":  # its process is gone, so it never finished
        return "stopped", "stopped", ""
    state = _turn_outcome(turn.status, turn.cause)
    result = result_line(turn.note, turn.answer)
    if state != "done":
        return state, state, result or status_text(turn.status, turn.cause).title
    icon = activity_text(turn.tool).icon if turn.tool else "done"
    return state, icon if icon not in ("tool", "think") else "done", result


async def _live(row: dict[str, Any], console: Console) -> None:
    """Fill a running row's state and words from what it is doing now."""
    text, _ = await item_activity(console, row["kind"], row["id"])
    if text.icon in ("wait", "held"):
        row["state"] = "waiting" if text.icon == "wait" else "held"
    row.update(activity=text.text, icon=text.icon)


async def _session_rows(
    console: Console, turns: Iterable[DayTurn], kind_of: dict[str, str]
) -> list[dict[str, Any]]:
    rows = []
    for turn in turns:
        kind = kind_of[turn.kind]
        if turn.status == "running" and turn.live:
            row = _row(kind, turn.session_id, turn.title, "running", turn.started_at)
            await _live(row, console)
        else:
            state, icon, result = _ended(turn)
            at = turn.finished_at or turn.started_at
            row = _row(kind, turn.session_id, turn.title, state, at, icon=icon)
            row["result"] = result
        row["turns"] = turn.turns
        rows.append(row)
    return rows


async def _task_rows(
    console: Console, since: float, sessions: dict[str, DayTurn]
) -> list[dict[str, Any]]:
    rows = []
    for record in await console.workers.tasks.tasks_since(since):
        state = TASK_STATES.get(record.state, "queued")
        at = record.finished_at or record.updated_at
        row = _row("task", record.task_id, record.title, state, at)
        row["level"] = record.level_eff
        if state == "running":
            await _live(row, console)
        elif state in LIVE:
            text = activity_text(state=state)
            row.update(activity=text.text, icon=text.icon)
        else:
            turn = sessions.get(record.task_id)
            ended = _ended(turn) if turn is not None else (state, state, "")
            result = ended[2] or (record.wait_reason if state == "failed" else "")
            row.update(icon=ended[1] if state == "done" else state, result=result)
        rows.append(row)
    return rows


async def day_activity(console: Console, since: float) -> dict[str, Any]:
    """The ``GET /api/activity`` body for the day that starts at *since*."""
    rows: list[dict[str, Any]] = []
    unavailable: list[str] = []
    errors = (SessionStoreError, OSError, AttributeError)
    if console.store_state() == "ok":
        sessions: dict[str, DayTurn] = {}
        try:
            found = await console.store.day_turns(
                since, ("chat", "task"), limit=2 * MAX_ROWS + 1
            )
            sessions = {turn.session_id: turn for turn in found if turn.kind == "task"}
            chats = [turn for turn in found if turn.kind == "chat"]
            rows += await _session_rows(console, chats, {"chat": "chat"})
        except errors:
            unavailable.append("chats")
        try:
            rows += await _task_rows(console, since, sessions)
        except errors:
            unavailable.append("tasks")
    else:
        unavailable += ["chats", "tasks"]
    try:
        found = await console.worker_history.day_turns(since, limit=MAX_ROWS + 1)
        rows += await _session_rows(
            console, found, {"worker": "run", "chat": "terminal"}
        )
    except errors:
        unavailable.append("runs")
    rows.sort(key=lambda row: (row["state"] in LIVE, row["at"]), reverse=True)
    return {
        "since": since,
        "rows": rows[:MAX_ROWS],
        "more": len(rows) > MAX_ROWS,
        "unavailable": unavailable,
    }


def add_routes(app: FastAPI, console: Console) -> None:
    @app.get("/api/activity")
    async def activity(since: float | None = None) -> JSONResponse:
        now = time.time()
        if since is None:
            since = local_midnight(now)
        elif not math.isfinite(since) or since > now:
            raise ApiError(400, "invalid_request", "since must be a past time.")
        return JSONResponse(
            content=await day_activity(console, max(since, now - MAX_SPAN_S)),
            headers={"Cache-Control": "no-store"},
        )


__all__ = ["MAX_ROWS", "add_routes", "day_activity", "local_midnight", "result_line"]
