"""API v2 memory across chats: the notes melo keeps, and the switch that turns them on.

The notes are what the user said about themselves (see
:mod:`localmelo.support.memory`). ``GET /api/memory/records`` lists them newest
first (``?session_id=`` keeps one chat's), each with where it came from: a note
outlives a deleted chat, and says so. ``PATCH /api/memory/records/{id}`` rewords
one, ``DELETE /api/memory/records/{id}`` removes it and its vector and blanks the
text of its decision rows (with ``?undo=true``, a turn's "Undo", the note it
replaced is current again), and ``DELETE /api/memory/records?all=true`` forgets
every note and the index of every chat. ``PUT /api/memory`` switches
``[memory] across_chats`` and ``POST /api/memory/reindex`` rebuilds the index
from the chats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Query
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr

from localmelo.melo.memory.long import StoredEntry
from localmelo.support import config
from localmelo.support.memory.recall import chat_label, day, origin

from .errors import ApiError

if TYPE_CHECKING:
    from . import Console


class EditBody(BaseModel):
    """Request body of ``PATCH /api/memory/records/{id}``."""

    model_config = ConfigDict(extra="forbid")

    text: StrictStr


class SwitchBody(BaseModel):
    """Request body of ``PUT /api/memory``."""

    model_config = ConfigDict(extra="forbid")

    on: StrictBool


async def _chats(console: Console, ids: set[str]) -> dict[str, Any]:
    """The chats that still exist among *ids* (session id to its info)."""
    found: dict[str, Any] = {}
    if console.store_state() != "ok":
        return found
    for session_id in ids:
        info = await console.store.get_session(session_id)
        if info is not None:
            found[session_id] = info
    return found


def _row(
    entry: StoredEntry, console: Console, chats: dict[str, Any] | None
) -> dict[str, Any]:
    """A note as the page and the clients read it."""
    prov = entry.provenance
    surface = console.memory.surface
    label = str(prov.get("chat") or chat_label("chat", "", surface))
    gone = (
        chats is not None and bool(entry.session_id) and entry.session_id not in chats
    )
    if chats and entry.session_id in chats:
        info = chats[entry.session_id]
        label = chat_label(info.kind, info.title, surface)
    return {
        "id": entry.id,
        "text": entry.text,
        "topic": str(prov.get("topic") or ""),
        "quote": str(prov.get("quote") or ""),
        "source": f"you said it · {label}, {day(entry.created_at)}",
        "from": origin(label, gone, entry.created_at),
        "chat_gone": gone,
        "edited": bool(prov.get("edited")),
        "session_id": entry.session_id,
        "created_at": entry.created_at,
    }


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the memory routes on *app*."""

    @app.get("/api/memory/records")
    async def memory_records(session_id: str = Query("")) -> dict[str, Any]:
        memory = console.memory
        entries = [
            entry
            for entry in await memory.records()
            if not session_id or entry.session_id == session_id
        ]
        chats = (
            await _chats(console, {e.session_id for e in entries}) if entries else {}
        )
        known = console.store_state() == "ok"
        status = await memory.status()
        return {
            "on": memory.enabled,
            "records": [_row(e, console, chats if known else None) for e in entries],
            "documents": status["documents"],
            "embedding_model": status["embedding_model"],
            "last_error": status["last_error"],
        }

    @app.patch("/api/memory/records/{record_id}")
    async def memory_record_edit(record_id: int, body: EditBody) -> dict[str, Any]:
        try:
            entry = await console.memory.edit_record(record_id, body.text)
        except ValueError as exc:
            raise ApiError(400, "invalid_request", str(exc)) from exc
        if entry is None:
            raise ApiError(404, "not_found", "That note is not in melo's memory.")
        chats = await _chats(console, {entry.session_id})
        return _row(entry, console, chats if console.store_state() == "ok" else None)

    @app.delete("/api/memory/records/{record_id}")
    async def memory_record_delete(
        record_id: int, undo: bool = Query(False)
    ) -> Response:
        if not await console.memory.delete_record(record_id, undo=undo):
            raise ApiError(404, "not_found", "That note is not in melo's memory.")
        return Response(status_code=204)

    @app.delete("/api/memory/records")
    async def memory_forget_all(
        everything: bool = Query(False, alias="all")
    ) -> dict[str, int]:
        if not everything:
            raise ApiError(
                400, "invalid_request", "Forgetting everything needs ?all=true."
            )
        notes, turns = await console.memory.forget_all()
        return {"notes": notes, "turns": turns}

    @app.put("/api/memory")
    async def memory_switch(body: SwitchBody) -> dict[str, bool]:
        saved = config.load()
        saved.memory.across_chats = body.on
        config.save(saved)
        console.cfg.memory.across_chats = body.on
        await console.sessions.settings_changed()
        return {"on": body.on}

    @app.post("/api/memory/reindex")
    async def memory_reindex() -> dict[str, int]:
        chats = console.store if console.store_state() == "ok" else None
        notes, turns = await console.memory.reindex(chats)
        return {"notes": notes, "turns": turns}


__all__: list[str] = []
