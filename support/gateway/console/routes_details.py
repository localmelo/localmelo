"""API v2 details: a session's tools, grants, memory and runs.

These are the drawer's three tabs. `tools` opens the session, because the
list is the Agent's and a provider's tools appear only once it started;
`memory` never opens it, so a paused session shows the conversation the
gateway kept; `runs` reads the turns it has, with each `RunResult` but its
answer text. The two `DELETE` routes take back a session grant, which lives
in the registry's `RememberingPolicy` and never in the store.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from localmelo.melo.schema import SHORT_TERM_MAX, Message, ToolInfo
from localmelo.support.sessions import display
from localmelo.support.sessions.store import SessionInfo, SessionStoreError, TurnInfo

from .errors import ApiError
from .registry import OpenSession
from .shapes import memory_json, run_json, tool_json

if TYPE_CHECKING:
    from . import Console

# A memory row's text stops here; the page expands the message itself.
MEMORY_TEXT_CHARS = 2_000
# Runs per page: the drawer lists a session's history, not its turns.
RUN_LIMIT = 20
_RUNS_MAX = 50
# SQLite event ids are signed 64-bit integers.
_TURN_MAX = 2**63 - 1


# ── Helpers ──


async def _store_call(
    console: Console, coro: Any, session_id: str | None = None
) -> Any:
    """Await one store call, mapping a store failure as 2.2 says."""
    try:
        return await coro
    except SessionStoreError as exc:
        raise await console.store_api_error(exc, session_id) from exc


async def _sessions_call(
    console: Console, coro: Any, session_id: str | None = None
) -> Any:
    """Await a registry call, mapping the failures it can raise."""
    try:
        return await coro
    except SessionStoreError as exc:
        raise await console.store_api_error(exc, session_id) from exc


async def _require_session(console: Console, session_id: str) -> SessionInfo:
    info: SessionInfo | None = await _store_call(
        console, console.store.get_session(session_id), session_id
    )
    if info is None:
        raise ApiError(404, "not_found", "Session not found.")
    return info


async def _open_session(console: Console, session_id: str) -> OpenSession:
    """The session opened, under its own lock, as the turn routes do."""
    async with console.sessions.lock(session_id):
        entry: OpenSession = await _sessions_call(
            console, console.sessions.open(session_id), session_id
        )
    return entry


def _state(console: Console, session_id: str) -> str:
    entry = console.sessions.get(session_id)
    if entry is None:
        return "not_loaded"
    return "open" if entry.agent is not None else "paused"


def _grants(console: Console, session_id: str) -> list[str]:
    entry = console.sessions.get(session_id)
    return [] if entry is None else sorted(entry.policy.allowed_tools)


def _mcp_label(console: Console) -> str:
    path = getattr(console.cfg, "mcp_config", "") or ""
    if not path:
        return ""
    return display.home_label(os.path.expanduser(path))


# ── Routes ──


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the tools, grants, memory and runs routes."""

    @app.get("/api/sessions/{session_id}/tools")
    async def session_tools(session_id: str) -> JSONResponse:
        console.require_store()
        await _require_session(console, session_id)
        entry = await _open_session(console, session_id)
        assert entry.agent is not None
        granted = entry.policy.allowed_tools
        tools: list[ToolInfo] = entry.agent.list_tools()
        return JSONResponse(
            content={
                "tools": [tool_json(tool, tool.name in granted) for tool in tools],
                "mcp_config_label": _mcp_label(console),
                "mcp_error": entry.mcp_error,
            }
        )

    @app.delete("/api/sessions/{session_id}/grants/{tool}")
    async def forget_grant(session_id: str, tool: str) -> JSONResponse:
        console.require_store()
        await _require_session(console, session_id)
        entry = console.sessions.get(session_id)
        if entry is None or tool not in entry.policy.allowed_tools:
            raise ApiError(404, "not_found", "This tool has no session grant.")
        console.sessions.forget_grant(session_id, tool)
        return JSONResponse(content={"grants": _grants(console, session_id)})

    @app.delete("/api/sessions/{session_id}/grants")
    async def forget_grants(session_id: str) -> JSONResponse:
        console.require_store()
        await _require_session(console, session_id)
        console.sessions.forget_grant(session_id)
        return JSONResponse(content={"grants": []})

    @app.get("/api/sessions/{session_id}/memory")
    async def session_memory(session_id: str) -> JSONResponse:
        console.require_store()
        await _require_session(console, session_id)
        entry = console.sessions.get(session_id)
        messages: list[Message] = []
        if entry is not None:
            if entry.agent is not None:
                messages = entry.agent.conversation()
            elif entry.kept is not None:
                messages = entry.kept
        backend = getattr(console.cfg, "embedding_backend", "") or "none"
        return JSONResponse(
            content={
                "agent": _state(console, session_id),
                "working": [
                    memory_json(message, limit=MEMORY_TEXT_CHARS)
                    for message in messages
                ],
                "working_max": SHORT_TERM_MAX,
                "long_term": {
                    "enabled": backend != "none",
                    "backend": backend,
                },
            }
        )

    @app.get("/api/sessions/{session_id}/runs")
    async def session_runs(
        session_id: str,
        before: int | None = Query(None, ge=1, le=_TURN_MAX),
        limit: int = Query(RUN_LIMIT, ge=1, le=_RUNS_MAX),
    ) -> JSONResponse:
        console.require_store()
        await _require_session(console, session_id)
        turns: list[TurnInfo] = await _store_call(
            console,
            console.store.list_turns(session_id, limit=limit, before=before),
            session_id,
        )
        # A running turn has no RunResult yet; it is the transcript's row.
        runs = [run_json(turn) for turn in turns if turn.status != "running"]
        has_earlier = False
        if turns:
            older = await _store_call(
                console,
                console.store.list_turns(session_id, limit=1, before=turns[0].turn),
                session_id,
            )
            has_earlier = bool(older)
        return JSONResponse(content={"runs": runs, "has_earlier": has_earlier})


__all__ = ["MEMORY_TEXT_CHARS", "add_routes"]
