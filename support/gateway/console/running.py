"""What runs now: the ``status`` object of ``GET /api/health``.

``running`` counts the chats the console runs, the tasks the scheduler runs,
and the worker runs and terminal chats that hold a live lease in the CLI file.
``tasks`` is the scheduler's queue, as the status row shows it. ``items`` lists
what runs, newest kinds first, for the header's list: it is capped, and the
counts are not. A part whose store cannot be read counts as nothing running:
health is served while a store is unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from localmelo.support.sessions.store import SessionStoreError
from localmelo.support.usage_snapshot import TaskCounts

if TYPE_CHECKING:
    from . import Console

COUNTED = ("running", "queued", "held")
# Rows of the header's list; the counts above it still say how many run.
MAX_ITEMS = 20
TITLE_CHARS = 200


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


async def running_status(console: Console) -> dict[str, Any]:
    """The ``status`` object: counts by kind, the task queue and the running items."""
    chats: list[dict[str, str]] = []
    tasks: list[dict[str, str]] = []
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
    items = [
        {**item, "title": item["title"][:TITLE_CHARS]}
        for item in (*chats, *tasks, *runs, *terminal)
    ]
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
        "items": items[:MAX_ITEMS],
    }
