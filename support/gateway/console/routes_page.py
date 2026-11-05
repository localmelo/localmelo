"""API v2 header writes: the page's S2 controls on the one shared gate.

``POST /api/sessions/{id}/level`` sets a chat's P1-P4 and ``POST
/api/presence`` is the typing heartbeat. Both act on the console's one
:class:`ChatGate` — the same gate the workspace page's chats use — and
neither opens the session: a level is a scheduler fact, not an Agent's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import Field

from .bodies import Body
from .errors import ApiError

if TYPE_CHECKING:
    from . import Console


class LevelRequest(Body):
    # strict: a bool or a numeric string is not a level the gate would take.
    level: int = Field(strict=True, ge=1, le=4)


class PresenceRequest(Body):
    session_id: str


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the level and presence routes on *app*."""

    @app.post("/api/sessions/{session_id}/level")
    async def set_level(session_id: str, body: LevelRequest) -> JSONResponse:
        console.require_store()
        await console.require_session(session_id)
        try:
            level = console.gate.set_level(session_id, body.level)
        except ValueError as exc:  # Pydantic bounds it; the gate is the authority.
            raise ApiError(400, "invalid_request", str(exc)) from exc
        return JSONResponse(content={"session_id": session_id, "level": level})

    @app.post("/api/presence")
    async def presence(body: PresenceRequest) -> JSONResponse:
        console.require_store()
        await console.require_session(body.session_id)
        console.gate.presence(body.session_id)
        return JSONResponse(content={"ok": True})


__all__: list[str] = []
