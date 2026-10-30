"""API v2 memory across chats: ``GET /api/memory/records``, ``DELETE /api/memory/records/{id}``.

The records are what the user said about themselves (see
:mod:`localmelo.support.memory`). The list is newest first; a delete removes
the record and its vector and blanks the text of its decision rows, and with
``?undo=true`` (a turn's "Undo") the record it replaced is current again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Query
from fastapi.responses import Response

from localmelo.support.memory.recall import chat_label, day

from .errors import ApiError

if TYPE_CHECKING:
    from . import Console


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the memory routes on *app*."""

    @app.get("/api/memory/records")
    async def memory_records() -> dict[str, Any]:
        memory = console.memory
        rows = []
        for entry in await memory.records():
            prov = entry.provenance
            chat = str(prov.get("chat") or chat_label("chat", "", memory.surface))
            rows.append(
                {
                    "id": entry.id,
                    "text": entry.text,
                    "topic": str(prov.get("topic") or ""),
                    "quote": str(prov.get("quote") or ""),
                    "source": f"you said it · {chat}, {day(entry.created_at)}",
                    "session_id": entry.session_id,
                    "created_at": entry.created_at,
                }
            )
        return {"on": memory.enabled, "records": rows}

    @app.delete("/api/memory/records/{record_id}")
    async def memory_record_delete(
        record_id: int, undo: bool = Query(False)
    ) -> Response:
        if not await console.memory.delete_record(record_id, undo=undo):
            raise ApiError(404, "not_found", "That note is not in melo's memory.")
        return Response(status_code=204)


__all__: list[str] = []
