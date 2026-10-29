"""``GET /api/sessions/{id}/status``: the terminal's status numbers for one session.

The session is a chat or task of the gateway's file or a worker run of the CLI
file. The route folds the session's stored events with the same
:class:`UsageSnapshot` the terminal feeds, so the console row and the
terminal's ``/status`` and status bar agree. The fold is cached per session
at the last event id it has read: a poll with no new event reads one id, and
a poll after new events reads only those.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from localmelo.support.sessions.store import SessionStore, SessionStoreError
from localmelo.support.usage_snapshot import TaskCounts, UsageSnapshot

from .probe import LOCAL_BACKENDS
from .routes_sessions import _read_session, _store_call

if TYPE_CHECKING:
    from . import Console

# Sessions whose fold is kept; the least recently read one is dropped first.
_CACHED_FOLDS = 64
_PAGE = 1000
_COUNTED = ("running", "queued", "held")


async def _task_counts(console: Console) -> TaskCounts | None:
    """The scheduler's queue, or ``None`` when the task store cannot be read."""
    try:
        records = await console.workers.tasks.list_tasks()
    except (SessionStoreError, AttributeError):
        return None
    live = [record for record in records if record.state in _COUNTED]
    counts = {name: sum(record.state == name for record in live) for name in _COUNTED}
    top = min((record.level_eff for record in live), default=None)
    return TaskCounts(counts["running"], counts["queued"], counts["held"], top)


def add_routes(app: FastAPI, console: Console) -> None:
    folds: OrderedDict[str, tuple[int, UsageSnapshot]] = OrderedDict()
    lock = asyncio.Lock()

    async def fold(session_id: str, store: SessionStore) -> UsageSnapshot:
        scope = None if store is console.worker_history else session_id
        async with lock:
            latest = await _store_call(
                console, store.latest_event_id(session_id), scope
            )
            last, snap = folds.pop(session_id, (0, UsageSnapshot()))
            if latest < last:
                last, snap = 0, UsageSnapshot()
            while last < latest:
                page = await _store_call(
                    console,
                    store.list_events(session_id, after_id=last, limit=_PAGE),
                    scope,
                )
                for stored in page:
                    snap.observe(stored.event)
                if not page or page[-1].id is None:
                    break
                last = page[-1].id
            folds[session_id] = (last, snap)
            while len(folds) > _CACHED_FOLDS:
                folds.popitem(last=False)
            return snap

    @app.get("/api/sessions/{session_id}/status")
    async def session_status(session_id: str) -> JSONResponse:
        console.require_store()
        # A chat or task of the gateway's file, or a worker run of the CLI file.
        _, store = await _read_session(console, session_id)
        snap = await fold(session_id, store)
        tasks = await _task_counts(console)
        local = console.cfg.chat_backend in LOCAL_BACKENDS
        return JSONResponse(
            content={
                "session_id": session_id,
                **snap.to_dict(local=local, now=time.time(), tasks=tasks),
            },
            headers={"Cache-Control": "no-store"},
        )


__all__ = ["add_routes"]
