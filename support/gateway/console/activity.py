"""``GET /api/activity``: what ran today, for the panel beside a chat.

One list from four sources: the console's chats and the scheduler's tasks
(the gateway's file), and the worker runs and terminal chats (the CLI file,
read-only, a terminal chat redacted). Each session is one row, its newest
turn of the day: what it is doing now while it runs or waits, else how it
ended and a one-line result. Rows that run or wait come first, then the rest
newest first, at most :data:`MAX_ROWS`; ``more`` says some were left out. A
source that cannot be read adds nothing and is named in ``unavailable``.

With ``session_id`` the list follows that chat: a side chat's own row and the
tasks it started, with the main chat (or no ``session_id``) listing all of
the day. Each row also carries the approvals its session asked for today, in
plain words with how they were answered, and a task's row its ``folder``.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Final

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from localmelo.support.sessions.display import (
    activity_text,
    home_label,
    request_words,
    split_thinking,
    status_text,
)
from localmelo.support.sessions.store import (
    DayTurn,
    SessionStoreError,
    StoredEvent,
)

from .errors import ApiError
from .running import TASK_OUTCOMES, TITLE_CHARS, _turn_outcome, item_activity

if TYPE_CHECKING:
    from . import Console

MAX_ROWS: Final = 100
RESULT_CHARS: Final = 140
# The approvals a row lists: its newest.
MAX_APPROVALS: Final = 20
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
# A task that failed because it could not use its folder (it never runs in
# another one), as its row and its card say it.
FOLDER_REFUSED: Final = "Could not use its folder"
UNTITLED: Final = {
    "chat": "Chat",
    "task": "Task",
    "run": "Worker run",
    "terminal": "Terminal chat",
}
_MARKUP: Final = re.compile(r"^[\s#>*_`-]+")
_SENTENCE: Final = re.compile(r"(?<=[.!?])\s|(?<=[。！？])")
# What the run writes in place of an answer: `[interrupt: …]`, `[no answer: …]`.
_MARKER: Final = re.compile(r"^\s*\[(interrupt|no answer|answer not kept)\b[^\]]*\]")
_MARKER_WORDS: Final = {
    "interrupt": "Stopped before answering",
    "no answer": "Ended without an answer",
    "answer not kept": "",
}
_FENCE: Final = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_CODE_LINE: Final = re.compile(
    r"^\s*(?:def |class |import |from \S+ import |#include|#!|//|function |const |"
    r"let |var |public |package |<\?php|<!doctype|<html|[{}\[\]]\s*$)"
    r"|^(?: {4}|\t)|[;{}]\s*$",
    re.IGNORECASE,
)
# A file or folder path: `~/a/b`, `/Users/me/c.png`, `./out/d`, `C:\Users\e`.
_PATH: Final = re.compile(
    r"(?<![\w/.:~])(?:~(?:/[^\s\"'`<>|*?]+)+|(?:\.{1,2})?(?:/[^\s/\"'`<>|*?]+){2,}"
    r"|[A-Za-z]:\\[^\s\"'`<>|*?]+)"
)
_PICTURE_SUFFIXES: Final = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".heic")
_PATH_END: Final = ".,;:!?)"
MADE_PICTURE: Final = "Made a picture"
WROTE_CODE: Final = "Wrote some code"


def _prose(text: str) -> str:
    """*text* without its code: fenced blocks go, and code lines when most lines are code."""
    lines = [line for line in _FENCE.sub("\n", text).splitlines() if line.strip()]
    code = [line for line in lines if _CODE_LINE.search(line)]
    if len(lines) >= 3 and 2 * len(code) >= len(lines):
        lines = [line for line in lines if line not in code]
    return " ".join(lines)


def _named(found: re.Match[str]) -> str:
    """A path as its file's name, the punctuation after it kept."""
    path = found.group().rstrip(_PATH_END)
    return (
        path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        + found.group()[len(path) :]
    )


def _no_paths(text: str) -> str:
    """*text* with a name for each path; just ``Made a picture`` when it only says a picture's path."""
    found = _PATH.search(text)
    if found is None:
        return text
    label = re.sub(r"[\W_]+", " ", _PATH.sub(" ", text)).split()
    if (
        found.group().rstrip(_PATH_END).lower().endswith(_PICTURE_SUFFIXES)
        and len(label) <= 4
    ):
        return MADE_PICTURE
    return _PATH.sub(_named, text)


def result_line(note: str, answer: str) -> str:
    """The digest when there is one, else the answer's first sentence: one line, cut.

    It is plain words: a run's own marker (``[interrupt: …]``) reads as what
    happened, a path reads as ``Made a picture`` or the file's name, and
    code gives way to the first sentence around it (``Wrote some code`` when
    there is none).
    """
    raw = note or split_thinking(answer or "")[1]
    marker = _MARKER.match(raw)
    if marker:
        return _MARKER_WORDS[marker.group(1)]
    text = " ".join(_prose(raw).split())
    if raw.strip() and not text:
        return WROTE_CODE
    text = _no_paths(_MARKUP.sub("", text))
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
        "approvals": [],
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
        row["_chat"] = turn.session_id if kind == "chat" else ""
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
        row["_chat"] = record.front_id or ""
        session = await console.store.get_session(record.task_id)
        row["folder"] = (
            home_label(session.workspace) if session and session.workspace else ""
        )
        if state == "running":
            await _live(row, console)
        elif state in LIVE:
            text = activity_text(state=state)
            row.update(activity=text.text, icon=text.icon)
        else:
            turn = sessions.get(record.task_id)
            ended = _ended(turn) if turn is not None else (state, state, "")
            result = ended[2] or (record.wait_reason if state == "failed" else "")
            if record.wait_reason == "invalid_workspace" and state == "failed":
                result = FOLDER_REFUSED
            row.update(icon=ended[1] if state == "done" else state, result=result)
        rows.append(row)
    return rows


def _answer(data: Mapping[str, Any]) -> str:
    """How an ``approval.resolved`` ended, in the words a row lists."""
    outcome, origin = data.get("outcome"), data.get("origin")
    if outcome == "withdrawn":
        return "withdrawn"
    if origin == "timeout":
        return "no answer"
    if outcome != "allowed":
        return "denied"
    if origin == "policy":
        return "allowed automatically"
    if origin == "session" or data.get("remember"):
        return "allowed for this chat"
    return "allowed once"


def approvals_of(events: Iterable[StoredEvent]) -> dict[str, list[dict[str, Any]]]:
    """Each session's approvals from its ``approval.*`` events, oldest first.

    ``{"id", "what", "answer", "at"}``: ``what`` is the call in plain words,
    ``answer`` one of ``allowed once``, ``allowed for this chat``, ``allowed
    automatically``, ``denied``, ``no answer``, ``withdrawn`` or ``waiting``,
    ``at`` when it was asked. An answer without its request adds nothing.
    """
    asked: dict[tuple[str, str], dict[str, Any]] = {}
    for stored in events:
        event = stored.event
        data = event.get("data")
        data = data if isinstance(data, Mapping) else {}
        key = (stored.session_id, str(data.get("request_id", "")))
        if event.get("type") == "approval.requested":
            root = data.get("workspace_root")
            ts = event.get("ts")
            asked[key] = {
                "id": key[1],
                "what": request_words(
                    str(data.get("tool_name") or ""),
                    data.get("arguments"),
                    root=root if isinstance(root, str) else None,
                ),
                "answer": "waiting",
                "at": float(ts) if isinstance(ts, int | float) else 0.0,
            }
        elif key in asked:
            asked[key]["answer"] = _answer(data)
    found: dict[str, list[dict[str, Any]]] = {}
    for (session_id, _), item in asked.items():
        found.setdefault(session_id, []).append(item)
    return {sid: items[-MAX_APPROVALS:] for sid, items in found.items()}


async def _add_approvals(
    console: Console, rows: list[dict[str, Any]], since: float
) -> bool:
    """Put each row's approvals on it; False when a source could not be read."""
    sources = (
        (console.store if console.store_state() == "ok" else None, ("chat", "task")),
        (console.worker_history, ("run", "terminal")),
    )
    complete = True
    for store, kinds in sources:
        mine = [row for row in rows if row["kind"] in kinds]
        if store is None or not mine:
            continue
        try:
            found = approvals_of(
                await store.approval_events({row["id"] for row in mine}, since)
            )
        except (SessionStoreError, OSError, AttributeError):
            complete = False
            continue
        for row in mine:
            row["approvals"] = found.get(row["id"], [])
    return complete


async def _scope(console: Console, session_id: str | None) -> str | None:
    """The chat whose rows to keep, or ``None`` for the whole day.

    The main chat, which holds the whole day, and no chat at all are ``None``;
    any other session, an unknown one included, is a side chat.
    """
    if not session_id:
        return None
    try:
        info = await console.store.get_session(session_id)
    except (SessionStoreError, OSError, AttributeError):
        info = None
    if info is not None and (info.kind == "front" or info.meta.get("main") is True):
        return None
    return session_id


async def day_activity(
    console: Console, since: float, session_id: str | None = None
) -> dict[str, Any]:
    """The ``GET /api/activity`` body for the day that starts at *since*.

    With *session_id* of a side chat, only that chat's row and the tasks it
    started; otherwise the whole day.
    """
    rows: list[dict[str, Any]] = []
    unavailable: list[str] = []
    errors = (SessionStoreError, OSError, AttributeError)
    if console.store_state() == "ok":
        sessions: dict[str, DayTurn] = {}
        try:
            found = await console.store.day_turns(
                since, ("chat", "front", "task"), limit=2 * MAX_ROWS + 1
            )
            sessions = {turn.session_id: turn for turn in found if turn.kind == "task"}
            chats = [turn for turn in found if turn.kind != "task"]
            rows += await _session_rows(
                console, chats, {"chat": "chat", "front": "chat"}
            )
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
    chat = await _scope(console, session_id)
    if chat is not None:
        rows = [row for row in rows if chat in (row["id"], row.get("_chat"))]
    for row in rows:
        row.pop("_chat", None)
    rows.sort(key=lambda row: (row["state"] in LIVE, row["at"]), reverse=True)
    if not await _add_approvals(console, rows[:MAX_ROWS], since):
        unavailable.append("approvals")
    return {
        "since": since,
        "scope": "all" if chat is None else "chat",
        "rows": rows[:MAX_ROWS],
        "more": len(rows) > MAX_ROWS,
        "unavailable": unavailable,
    }


def add_routes(app: FastAPI, console: Console) -> None:
    @app.get("/api/activity")
    async def activity(
        since: float | None = None,
        session_id: str | None = Query(default=None, max_length=128),
    ) -> JSONResponse:
        now = time.time()
        if since is None:
            since = local_midnight(now)
        elif not math.isfinite(since) or since > now:
            raise ApiError(400, "invalid_request", "since must be a past time.")
        return JSONResponse(
            content=await day_activity(
                console, max(since, now - MAX_SPAN_S), session_id
            ),
            headers={"Cache-Control": "no-store"},
        )


__all__ = [
    "MAX_ROWS",
    "add_routes",
    "approvals_of",
    "day_activity",
    "local_midnight",
    "result_line",
]
