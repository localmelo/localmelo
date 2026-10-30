"""What runs now: the ``status`` object of ``GET /api/health``.

``running`` counts the chats the console runs, the tasks the scheduler runs,
and the worker runs and terminal chats that hold a live lease in the CLI file.
``tasks`` is the scheduler's queue, as the status row shows it. ``items`` lists
what runs, newest kinds first, for the header's list: it is capped, and the
counts are not. ``waiting`` lists the chats and tasks that wait for the
person's answer to a tool approval, for the macOS app's menu bar. ``recent``
says how the newest finished chat turn, task, run or terminal turn ended and
how many seconds ago (``null`` when none did), for the app's desktop pet.
Each item says what it is doing in plain words (``activity``, with its kind
``icon``: :func:`~localmelo.support.sessions.display.activity_text`), and
``focus`` names the one item a glance shows: one that waits for the person
first, else the one whose newest event is the latest (``null`` when nothing
runs). A part whose store cannot be read counts as nothing running and
nothing recent: health is served while a store is unavailable.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from localmelo.support.sessions.display import (
    ActivityText,
    activity_text,
    event_activity,
    status_text,
)
from localmelo.support.sessions.store import SessionStoreError
from localmelo.support.usage_snapshot import TaskCounts

if TYPE_CHECKING:
    from . import Console

COUNTED = ("running", "queued", "held")
# Rows of the header's list; the counts above it still say how many run.
MAX_ITEMS = 20
TITLE_CHARS = 200
# The kind a worker file's session has in `status.items` and `status.recent`.
WORKER_KINDS = {"worker": "run", "chat": "terminal"}
# How a task ended, as `status.recent` says it: the person's stop and an expiry are no failure.
# The newest events read to say what an item is doing.
EVENTS_READ = 40
TASK_OUTCOMES = {
    "done": "done",
    "failed": "failed",
    "cancelled": "stopped",
    "expired": "stopped",
}


async def live_tasks(console: Console) -> list[Any] | None:
    """The records of tasks that are running, queued or held; ``None`` if unreadable."""
    try:
        records = await console.workers.tasks.list_tasks(COUNTED)
    except (SessionStoreError, AttributeError):
        return None
    return [record for record in records if record.state in COUNTED]


def task_counts(live: list[Any] | None) -> TaskCounts | None:
    """The queue the status row and health report, from :func:`live_tasks`."""
    if live is None:
        return None
    counts = {name: sum(record.state == name for record in live) for name in COUNTED}
    top = min((record.level_eff for record in live), default=None)
    return TaskCounts(counts["running"], counts["queued"], counts["held"], top)


async def _chat_items(console: Console) -> list[dict[str, str]]:
    items = []
    for entry in console.sessions.entries():
        if not entry.runner.busy:
            continue
        title = ""
        try:
            info = await console.store.get_session(entry.session_id)
            title = info.title if info is not None else ""
        except SessionStoreError:
            pass
        items.append({"kind": "chat", "id": entry.session_id, "title": title})
    return items


def _waiting(console: Console, live: list[Any] | None) -> list[dict[str, str]]:
    """The chats and tasks with an approval request that waits for an answer.

    The broker keys a chat's requests by its session id and a task's by its task
    id, so one pass over both says which: the chats first, then the tasks.
    """
    owners = [("chat", entry.session_id) for entry in console.sessions.entries()]
    owners += [("task", record.task_id) for record in live or ()]
    return [
        {"kind": kind, "id": owner}
        for kind, owner in owners
        if console.broker.pending(owner)
    ][:MAX_ITEMS]


def _turn_outcome(status: str, cause: str) -> str:
    """``done`` for a run that completed, ``stopped`` for the person's stop, else ``failed``.

    It follows the words the console shows for the end (``ok``, ``stopped``,
    and the ``warn`` and ``error`` that ask something of the person).
    """
    severity = status_text(status, cause).severity
    return {"ok": "done", "stopped": "stopped"}.get(severity, "failed")


async def recent_outcome(console: Console) -> dict[str, Any] | None:
    """How the newest finished chat turn, task, run or terminal turn ended, or ``None``.

    ``{"kind", "id", "outcome", "ended_s_ago"}``: ``kind`` as in ``items``,
    ``outcome`` ``done``, ``failed`` or ``stopped``, the age in seconds to
    a tenth. A turn that a shutdown, a crash, a takeover, a close or a park
    ended is no outcome (the store's ``OUTCOME_CAUSES``).
    """
    found: list[tuple[float, str, str, str]] = []  # (finished_at, kind, id, outcome)
    if console.store_state() == "ok":
        try:
            chat = await console.store.latest_finished_turn(("chat",))
            if chat is not None:
                found.append(
                    (
                        chat.finished_at,
                        "chat",
                        chat.session_id,
                        _turn_outcome(chat.status, chat.cause),
                    )
                )
        except SessionStoreError:
            pass
        try:
            task = await console.workers.tasks.latest_finished()
            if task is not None and task[1] in TASK_OUTCOMES:
                found.append((task[2], "task", task[0], TASK_OUTCOMES[task[1]]))
        except (SessionStoreError, AttributeError):
            pass
    try:
        turn = await console.worker_history.latest_finished_turn()
        if turn is not None:
            found.append(
                (
                    turn.finished_at,
                    WORKER_KINDS[turn.kind],
                    turn.session_id,
                    _turn_outcome(turn.status, turn.cause),
                )
            )
    except (SessionStoreError, OSError):
        pass
    if not found:
        return None
    finished_at, kind, owner, outcome = max(found)
    return {
        "kind": kind,
        "id": owner,
        "outcome": outcome,
        "ended_s_ago": round(max(0.0, time.time() - finished_at), 1),
    }


async def item_activity(
    console: Console, kind: str, owner: str
) -> tuple[ActivityText, float]:
    """What a running chat, task, run or terminal chat is doing, and when it last changed.

    A chat or task that waits for an approval or for the scheduler says so;
    anything else reads from its newest events. An unreadable store reads
    ``Thinking``, changed at 0.
    """
    store = console.worker_history if kind in ("run", "terminal") else console.store
    try:
        events = await store.latest_events(owner, limit=EVENTS_READ)
    except (SessionStoreError, OSError, AttributeError):
        events = []
    changed = float(events[0].event.get("ts") or 0) if events else 0.0
    if kind in ("chat", "task") and console.broker.pending(owner):
        return activity_text(state="waiting"), changed
    if kind in ("chat", "task") and console.gate.held(owner):
        return activity_text(state="held"), changed
    found = event_activity(stored.event for stored in events)
    return found or activity_text(state="thinking"), changed


async def _with_activity(
    console: Console, items: list[dict[str, Any]], waiting: list[dict[str, str]]
) -> dict[str, Any] | None:
    """Add ``activity`` and ``icon`` to each item; the item ``focus`` names."""
    changed: dict[str, float] = {}
    for item in items:
        text, changed[item["id"]] = await item_activity(
            console, item["kind"], item["id"]
        )
        item.update(activity=text.text, icon=text.icon)
    if not items:
        return None
    asking = {entry["id"] for entry in waiting}
    first = next((item for item in items if item["id"] in asking), None)
    return dict(first or max(items, key=lambda item: changed[item["id"]]))


async def running_status(console: Console) -> dict[str, Any]:
    """The ``status`` object: counts by kind, the queue, the running, waiting and recent items."""
    chats: list[dict[str, str]] = []
    tasks: list[dict[str, str]] = []
    live: list[Any] | None = None
    queue = TaskCounts(0, 0, 0)
    if console.store_state() == "ok":
        chats = await _chat_items(console)
        live = await live_tasks(console)
        queue = task_counts(live) or queue
        tasks = [
            {"kind": "task", "id": record.task_id, "title": record.title}
            for record in live or ()
            if record.state == "running"
        ]
    runs: list[dict[str, str]] = []
    terminal: list[dict[str, str]] = []
    try:
        found = await console.worker_history.running()
    except (SessionStoreError, OSError):
        found = []
    for kind, sid, title in found:
        item = {
            "kind": "run" if kind == "worker" else "terminal",
            "id": sid,
            "title": title,
        }
        (runs if kind == "worker" else terminal).append(item)
    items: list[dict[str, Any]] = [
        {**item, "title": item["title"][:TITLE_CHARS]}
        for item in (*chats, *tasks, *runs, *terminal)
    ][:MAX_ITEMS]
    waiting = _waiting(console, live)
    focus = await _with_activity(console, items, waiting)
    return {
        "running": {
            "chats": len(chats),
            "tasks": len(tasks),
            "runs": len(runs),
            "terminal": len(terminal),
        },
        "tasks": {
            "running": queue.running,
            "queued": queue.queued,
            "held": queue.held,
            "top_level": queue.top_level,
        },
        "items": items,
        "waiting": waiting,
        "recent": await recent_outcome(console),
        "focus": focus,
    }
