"""API v2 details: a session's tools, grants, memory and runs.

These are the drawer's three tabs. `tools` opens the session, because the
list is the Agent's and a provider's tools appear only once it started;
`memory` never opens it, so a paused session shows the conversation the
gateway kept, every text redacted; `runs` reads the turns it has, with each
`RunResult` but its answer text. Two `DELETE` routes take back a session grant, which lives
in the registry's `RememberingPolicy` and never in the store; a third takes back one approval
the chat recorded for its tasks (#369), which lives in memory too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response

from localmelo.melo.schema import Message, ToolInfo
from localmelo.support.gateway.i18n import t
from localmelo.support.sessions.store import SessionInfo, SessionStoreError, TurnInfo

from .errors import ApiError
from .registry import OpenSession
from .routes_sessions import agent_state
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
        raise ApiError(404, "not_found", t("errors.sessionNotFound"))
    return info


async def _open_session(console: Console, session_id: str) -> OpenSession:
    """The session opened, under its own lock, as the turn routes do."""
    async with console.sessions.lock(session_id):
        entry: OpenSession = await _sessions_call(
            console, console.sessions.open(session_id), session_id
        )
    return entry


# MCP in the console is deferred (U3 decision 5B): no MCP tools load, and
# the Tools tab says so rather than showing a configured path that does
# nothing.
MCP_DEFERRED = "MCP is not built in the console yet"


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
        recorded = console.approvals.entries(session_id)
        return JSONResponse(
            content={
                "tools": [tool_json(tool, tool.name in granted) for tool in tools],
                # The steps the person allowed here, which this chat's tasks may run again.
                "approvals": [approval.to_json() for approval in recorded],
                "mcp_config_label": "",
                "mcp_error": MCP_DEFERRED,
            }
        )

    @app.delete("/api/sessions/{session_id}/grants/{tool}")
    async def forget_grant(session_id: str, tool: str) -> Response:
        console.require_store()
        await _require_session(console, session_id)
        entry = console.sessions.get(session_id)
        if entry is None or tool not in entry.policy.allowed_tools:
            raise ApiError(404, "not_found", t("errors.noGrant"))
        console.sessions.forget_grant(session_id, tool)
        return Response(status_code=204)

    @app.delete("/api/sessions/{session_id}/grants")
    async def forget_grants(session_id: str) -> Response:
        console.require_store()
        await _require_session(console, session_id)
        console.sessions.forget_grant(session_id)
        return Response(status_code=204)

    @app.delete("/api/sessions/{session_id}/approvals/{entry_id}")
    async def revoke_approval(session_id: str, entry_id: str) -> Response:
        """Take back one recorded approval: the step asks again; what ran stays run."""
        console.require_store()
        await _require_session(console, session_id)
        if not console.approvals.revoke(session_id, entry_id):
            raise ApiError(404, "not_found", t("errors.noApproval"))
        return Response(status_code=204)

    @app.get("/api/sessions/{session_id}/memory")
    async def session_memory(session_id: str) -> JSONResponse:
        console.require_store()
        await _require_session(console, session_id)
        entry = console.sessions.get(session_id)
        messages: list[Message] = []
        # Open sessions show their agent's cap; closed ones, the setting's cap.
        window_max = console.cfg.agent.window_for(console.cfg.chat_backend)
        if entry is not None:
            if entry.agent is not None:
                messages = entry.agent.conversation()
                window_max = entry.agent.options.window_messages
            elif entry.kept is not None:
                messages = entry.kept
        backend = getattr(console.cfg, "embedding_backend", "") or "none"
        return JSONResponse(
            content={
                "agent": agent_state(console, session_id),
                "working": [
                    memory_json(message, limit=MEMORY_TEXT_CHARS)
                    for message in messages
                ],
                "working_max": window_max,
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
