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
plain words with how they were answered, and a task's row its ``folder``. In
the whole day's list a task a side chat started also carries ``from``, that
chat's title.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Final

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from localmelo.support.gateway.i18n import said_text, t, t_in
from localmelo.support.sessions.display import (
    activity_text,
    drop_leading_ack,
    home_label,
    request_said,
    split_thinking,
    status_text,
)
from localmelo.support.sessions.store import (
    DayTurn,
    SessionStoreError,
    StoredEvent,
)

from ..finishing import NOT_CONTINUED, is_out_of_steps
from ..front_rules import HANDOFF_TEXTS
from ..task_endings import can_finish, report_text, shown
from .errors import ApiError
from .followups import FOLLOWED_ECHO, QUIET_DIDS, followed_words
from .running import (
    TASK_OUTCOMES,
    TITLE_CHARS,
    WAITS_FOR_YOU,
    _turn_outcome,
    item_activity,
    spoken,
)

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
# another one), as its row says it and, in English, its card.
FOLDER_REFUSED: Final = t_in("en", "activity.folderRefused")
_MARKUP: Final = re.compile(r"^[\s#>*_`-]+")
# Inline Markdown a line drops, keeping the words: links and images, strong and
# emphasis marks paired around words (not ``x**2``), code marks (the code itself
# stays as it is). Struck text goes with its marks.
_CODE_SPAN: Final = re.compile(r"`([^`\n]+)`")
_KEPT_CODE: Final = re.compile(r"\x00(\d+)\x00")
_LINK: Final = re.compile(r"!?\[([^\]\n]*)\]\([^)\s]*\)")
_STRUCK: Final = re.compile(r"[ \t]*~~(?=\S)(.+?)(?<=\S)~~")
_STRONG: Final = re.compile(
    r"(?<![A-Za-z0-9])(\*\*|__)(?=\S)(.+?)(?<=\S)\1(?![A-Za-z0-9])"
)
_STAR: Final = re.compile(
    r"(?<![A-Za-z0-9*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![A-Za-z0-9*])"
)
_UNDER: Final = re.compile(r"(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])")
_STRAY: Final = re.compile(r"(?<![A-Za-z0-9])(?:\*\*|~~)|(?:\*\*|~~)(?![A-Za-z0-9])|`")
# A short lead-in that only says how the reply is laid out: "In short:", "分两层说：".
_LEAD_IN: Final = re.compile(
    r"^(?:in short|short answer|in brief|tl;?dr|to sum up|in two parts|two parts|"
    r"two things|here(?:'s| is) (?:what happened|the short version|the answer)|"
    r"分两层说|分两点说|分两部分说|简单说|简单来说|简而言之|一句话|总结一下|总之|先说结论|"
    r"结论|简短回答|简答)\s*[:：]\s*",
    re.IGNORECASE,
)
# The main chat's echo rows (`front.ECHO_EVENT`): what the chat did besides answering.
_ECHO: Final = "attention.echo"
_SENTENCE: Final = re.compile(r"(?<=[.!?])\s|(?<=[。！？])")
# What the run writes in place of an answer: `[interrupt: …]`, `[no answer: …]`.
_MARKER: Final = re.compile(r"^\s*\[(interrupt|no answer|answer not kept)\b[^\]]*\]")
_MARKER_WORDS: Final = {
    "interrupt": "activity.marker.interrupt",
    "no answer": "activity.marker.noAnswer",
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
# A card carries this many lines of an answer, each cut at this many characters.
CARD_LINES: Final = 2
LINE_CHARS: Final = 120


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
        return t("activity.madePicture")
    return _PATH.sub(_named, text)


def plain_text(text: str) -> str:
    """*text* without inline Markdown, and without a lead-in that only announces its layout.

    ``分两层说: **我(主聊天)自己**:没有联网工具。`` reads ``我(主聊天)自己:没有联网工具。``;
    code keeps its words (``Edited `__init__.py`.`` reads ``Edited __init__.py.``),
    struck text goes, and ``x**2`` stays.
    """
    code: list[str] = []

    def keep(found: re.Match[str]) -> str:
        code.append(found.group(1))
        return f"\x00{len(code) - 1}\x00"

    text = _CODE_SPAN.sub(keep, text.replace("\x00", ""))
    text = _LINK.sub(r"\1", text)
    text = _STRUCK.sub("", text)
    text = _STRONG.sub(r"\2", text)
    text = _UNDER.sub(r"\1", _STAR.sub(r"\1", text))
    text = _STRAY.sub("", text)
    text = _KEPT_CODE.sub(lambda found: code[int(found.group(1))], text)
    rest = _LEAD_IN.sub("", text.strip(), count=1)
    return rest if rest.strip() else text.strip()


def _marker_words(marker: str) -> str:
    """What a run's own marker (``[interrupt: …]``) says happened."""
    key = _MARKER_WORDS[marker]
    return t(key) if key else ""


def result_line(note: str, answer: str) -> str:
    """The digest when there is one, else the answer's first sentence: one line, cut.

    It is plain words: a run's own marker (``[interrupt: …]``) reads as what
    happened, a path reads as ``Made a picture`` or the file's name, and
    code gives way to the first sentence around it (``Wrote some code`` when
    there is none). A ``Done.`` that opens the answer gives way to what the
    run says after it, as it does on the chat's card.
    """
    raw = note or split_thinking(report_text(answer))[1]
    marker = _MARKER.match(raw)
    if marker:
        return _marker_words(marker.group(1))
    kept = drop_leading_ack([line for line in raw.splitlines() if line.strip()])
    if kept is not None:
        raw = "\n".join(kept)
    text = " ".join(_prose(raw).split())
    if raw.strip() and not text:
        return t("activity.wroteCode")
    text = _no_paths(plain_text(_MARKUP.sub("", text)))
    first = _SENTENCE.split(text, maxsplit=1)[0]
    if len(first) <= RESULT_CHARS:
        return first
    cut = first[: RESULT_CHARS - 1].rsplit(" ", 1)[0] or first[: RESULT_CHARS - 1]
    return cut.rstrip(",;: ") + "…"


def _cut(text: str, chars: int) -> str:
    """*text* in at most *chars* characters, cut at a word with ``…``."""
    if len(text) <= chars:
        return text
    cut = text[: chars - 1].rsplit(" ", 1)[0] or text[: chars - 1]
    return cut.rstrip(",;: ") + "…"


def answer_lines(answer: str, count: int = CARD_LINES) -> str:
    """The first *count* lines of an answer, in plain words, one per line.

    What a card in the main chat shows of a finished task: what changed, the
    tests, the commit. An opening ``Done.`` is left out when more follows, code
    and markup are not shown, a path reads as its file's name, and each line
    is cut at :data:`LINE_CHARS`. An answer of one paragraph gives its first
    sentences. A run's own marker reads as what happened (``Stopped before
    answering``), an answer that is only code ``Wrote some code``, and an
    answer that only says it is done, ``Done.`` itself.
    """
    raw = split_thinking(report_text(answer))[1]
    marker = _MARKER.match(raw)
    if marker:
        return _marker_words(marker.group(1))
    lines = [line for line in _FENCE.sub("\n", raw).splitlines() if line.strip()]
    code = [line for line in lines if _CODE_LINE.search(line)]
    if len(lines) >= 3 and 2 * len(code) >= len(lines):
        lines = [line for line in lines if line not in code]
    rows = [
        " ".join(_MARKUP.sub("", line).replace("`", "").replace("**", "").split())
        for line in lines
    ]
    rows = [row for row in rows if row]
    kept = drop_leading_ack(rows)
    if kept is None:
        return rows[0] if rows else (t("activity.wroteCode") if raw.strip() else "")
    if len(kept) == 1:
        kept = [part for part in _SENTENCE.split(kept[0]) if part.strip()]
    return "\n".join(_cut(_no_paths(row), LINE_CHARS) for row in kept[:count])


# A sentence of a stop answer that says what is left undone.
_LEFT: Final = re.compile(
    r"\b(?:remaining|remains?|left to|still (?:need|needs|has|have) to|not (?:yet )?"
    r"(?:done|committed|tested|run)|untested|uncommitted|next step|to-?do|except for|"
    r"what['’]?s left|what is left)\b|\bleft\s*[:：]"
    r"|剩下|还剩|还需要?|尚未|未(?:完成|提交|测试|运行)|下一步|待完成|还没",
    re.IGNORECASE,
)


# A label the stop answer puts before what is left; the task page names the row itself.
_LEFT_LABEL: Final = re.compile(
    r"^\s*(?:remaining|what['’]?s left|what is left|left|still to do|"
    r"剩下|还剩下?|剩余|待完成)\s*[:：]\s*",
    re.IGNORECASE,
)


def left_line(answer: str, *, bare: bool = False) -> str:
    """The sentence of a stop answer that says what is left to do, or ``""``.

    ``Done except for the final commit: git add … && git commit …`` gives
    that sentence, cut as a card's lines are; an answer that names nothing
    left gives ``""`` and the caller shows the answer's own lines. A card and
    a row show the sentence as the answer wrote it; with *bare* a label that
    opens it (``Remaining: run the tests``, ``还剩下：运行测试``) is dropped,
    for the task page, whose row is called "What is left" already.
    """
    raw = _FENCE.sub("\n", split_thinking(report_text(answer))[1])
    for line in raw.splitlines():
        row = " ".join(_MARKUP.sub("", line).replace("`", "").replace("**", "").split())
        for part in _SENTENCE.split(row):
            if _LEFT.search(part):
                text = _LEFT_LABEL.sub("", part) if bare else part
                if text.strip():
                    return _cut(_no_paths(text), LINE_CHARS)
    return ""


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
        "title": (title or t(f"activity.untitled.{kind}"))[:TITLE_CHARS],
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
    # Whether the model is answering, as health's items say it (the portrait thinks).
    row.update(activity=spoken(text), icon=text.icon, answering=text.answering)


async def chat_did(console: Console, turn: DayTurn) -> str:
    """What a chat did in its newest turn besides answering, in plain words, or ``""``.

    The tasks it started or offered (its echo rows since the turn began; a
    task offered and then started reads once, as started), what its follow-up
    on a task did (#411: "Continued the task: …", "Asked you: …") and the
    notes it kept (its ``remember`` calls that succeeded), joined with `` · ``.
    """
    try:
        echoes = await console.store.events_of_type(turn.session_id, _ECHO, limit=50)
        finished = await console.store.events_of_type(
            turn.session_id, "tool.finished", limit=200
        )
    except (SessionStoreError, OSError, AttributeError):
        return ""
    tasks: dict[str, tuple[str, str]] = {}
    followed: list[str] = []
    for stored in echoes:
        ts, data = stored.event.get("ts"), stored.event.get("data")
        if not isinstance(ts, int | float) or ts < turn.started_at:
            continue
        data = data if isinstance(data, Mapping) else {}
        kind = data.get("echo")
        if kind == FOLLOWED_ECHO:
            if data.get("did") in QUIET_DIDS:
                continue  # skipped or stopped: the follow-up did nothing (#431)
            title = " ".join(str(data.get("title") or "").split())
            key, params = followed_words(
                str(data.get("did") or ""),
                _cut(title, TITLE_CHARS) or t("activity.untitled.task"),
                _cut(str(data.get("question") or ""), RESULT_CHARS),
            )
            followed.append(t(key, **params))
            continue
        if kind not in ("started", "suggest"):
            continue
        key = str(data.get("task_id") or data.get("title") or stored.id)
        if kind == "started" or key not in tasks:
            tasks[key] = (str(kind), " ".join(str(data.get("title") or "").split()))
    words = [
        t(
            "activity.did.started" if kind == "started" else "activity.did.offered",
            title=_cut(title, TITLE_CHARS) or t("activity.untitled.task"),
        )
        for kind, title in tasks.values()
    ]
    words += followed
    notes = sum(
        1
        for stored in finished
        if stored.turn == turn.turn
        and isinstance(stored.event.get("data"), Mapping)
        and stored.event["data"].get("name") == "remember"
        and stored.event["data"].get("status") == "success"
    )
    if notes:
        words.append(
            t("activity.did.note")
            if notes == 1
            else t("activity.did.notes", count=str(notes))
        )
    return _cut(" · ".join(words), RESULT_CHARS)


async def _session_rows(
    console: Console, turns: Iterable[DayTurn], kind_of: dict[str, str]
) -> list[dict[str, Any]]:
    rows = []
    for turn in turns:
        kind = kind_of[turn.kind]
        # The main chat is named in the reader's language, whatever was stored.
        title = t("chat.main") if turn.kind == "front" else turn.title
        if turn.status == "running" and turn.live:
            row = _row(kind, turn.session_id, title, "running", turn.started_at)
            await _live(row, console)
        else:
            state, icon, result = _ended(turn)
            at = turn.finished_at or turn.started_at
            row = _row(kind, turn.session_id, title, state, at, icon=icon)
            # A console chat says what it did, then what its reply says; a reply
            # that is only the hand-off sentence repeats what it did.
            did = await chat_did(console, turn) if kind == "chat" else ""
            if did and turn.answer.strip() in HANDOFF_TEXTS:
                result = ""
            said = " · ".join(part for part in (did, result) if part)
            row["result"] = _cut(said, RESULT_CHARS) if did else result
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
        # A run that used all its steps is stopped, not failed, and can continue.
        stopped = is_out_of_steps(record.state, record.wait_reason)
        state = "stopped" if stopped else state
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
            # A task held on an approval waits for the person; any other hold is
            # the scheduler's, and the gate says why.
            asks = record.state == "held" and record.wait_reason in WAITS_FOR_YOU
            reason = "Approval" if asks else console.gate.held(record.task_id)
            text = activity_text(state=state, reason=reason)
            row.update(activity=spoken(text), icon=text.icon)
        else:
            turn = sessions.get(record.task_id)
            ended = _ended(turn) if turn is not None else (state, state, "")
            result = ended[2] or (record.wait_reason if state == "failed" else "")
            if stopped:
                left = left_line(turn.answer) if turn is not None else ""
                result = left or ended[2] or t("cards.outOfSteps")
                row["can_continue"] = True
            elif record.wait_reason == "invalid_workspace" and state == "failed":
                result = t("activity.folderRefused")
            elif state == "failed" and record.wait_reason.startswith(NOT_CONTINUED):
                result = record.wait_reason
            row.update(icon=ended[1] if state == "done" else state, result=result)
            # How it ended, decided by code; a draft finishes with one tap.
            row["ending"] = shown(record)
            if can_finish(row["ending"]):
                row["can_continue"] = True
        rows.append(row)
    return rows


def answer_word(data: Mapping[str, Any]) -> str:
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
                "what": said_text(
                    request_said(
                        str(data.get("tool_name") or ""),
                        data.get("arguments"),
                        root=root if isinstance(root, str) else None,
                    )
                ),
                "answer": "waiting",
                "at": float(ts) if isinstance(ts, int | float) else 0.0,
            }
        elif key in asked:
            asked[key]["answer"] = answer_word(data)
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


async def _name_owners(console: Console, rows: list[dict[str, Any]]) -> None:
    """Put ``from`` (the side chat's title) on each task row of a side chat, for the whole day's list."""
    titles = {row["id"]: row["title"] for row in rows if row["kind"] == "chat"}
    for row in rows:
        owner = row.get("_chat")
        if row["kind"] != "task" or not owner or owner == console.main_id:
            continue
        if owner not in titles:
            try:
                info = await console.store.get_session(owner)
            except (SessionStoreError, OSError, AttributeError):
                info = None
            titles[owner] = info.title if info is not None else ""
        if titles[owner]:
            row["from"] = titles[owner]


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
    else:
        await _name_owners(console, rows)
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
            raise ApiError(400, "invalid_request", t("errors.sincePast"))
        return JSONResponse(
            content=await day_activity(
                console, max(since, now - MAX_SPAN_S), session_id
            ),
            headers={"Cache-Control": "no-store"},
        )


__all__ = [
    "MAX_ROWS",
    "add_routes",
    "answer_lines",
    "chat_did",
    "plain_text",
    "answer_word",
    "approvals_of",
    "day_activity",
    "local_midnight",
    "result_line",
]
