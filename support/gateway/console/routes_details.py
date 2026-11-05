"""API v2 details: a session's tools, grants, memory and runs.

These are the drawer's three tabs. `tools` opens the session, because the
list is the Agent's and a provider's tools appear only once it started;
`memory` never opens it, so a paused session shows the conversation the
gateway kept, every text redacted; `runs` reads the turns it has, with each
`RunResult` but its answer text. Two `DELETE` routes take back a grant that holds in the chat
(the grants live in the store's table, see `routes_grants`); a third takes back one approval
the chat recorded for its tasks (#369), which lives in memory only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response

from localmelo.melo.schema import Message, ToolInfo
from localmelo.support.gateway.i18n import t
from localmelo.support.sessions import display
from localmelo.support.sessions.store import TurnInfo

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


async def _open_session(console: Console, session_id: str) -> OpenSession:
    """The session opened, under its own lock, as the turn routes do."""
    async with console.sessions.lock(session_id):
        return await console.sessions.open(session_id)


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
        await console.require_session(session_id)
        entry = await _open_session(console, session_id)
        assert entry.agent is not None
        granted = entry.policy.allowed_tools
        tools: list[ToolInfo] = entry.agent.list_tools()
        recorded = console.approvals.entries(session_id)
        return JSONResponse(
            content={
                "tools": [tool_json(tool, tool.name in granted) for tool in tools],
                # The folders outside the workspace this chat may read in (#405).
                "folders": [
                    {"path": folder, "label": display.home_label(folder)}
                    for folder in entry.policy.allowed_folders
                ],
                # The steps the person allowed here, which this chat's tasks may run again.
                "approvals": [approval.to_json() for approval in recorded],
                "mcp_config_label": "",
                "mcp_error": MCP_DEFERRED,
            }
        )

    # A site's grant is named after its address (``browser-act:https://…``,
    # #465): the name may hold slashes.
    @app.delete("/api/sessions/{session_id}/grants/{tool:path}")
    async def forget_grant(session_id: str, tool: str) -> JSONResponse:
        """Take back the grant of *tool* that holds in the chat, open or not.

        A grant from the main chat holds everywhere, so taking it back here
        takes it back everywhere: ``everywhere`` says so. A write that fails
        is 503 (the checks stopped using it already).
        """
        console.require_store()
        await console.require_session(session_id)
        if tool not in console.sessions.grants(session_id):
            raise ApiError(404, "not_found", t("errors.noGrant"))
        gone = await console.sessions.forget_grant(session_id, tool)
        return JSONResponse(
            content={
                "revoked": [{"key": g.key, "scope": g.scope} for g in gone],
                "everywhere": any(g.scope == "global" for g in gone),
            }
        )

    @app.delete("/api/sessions/{session_id}/grants")
    async def forget_grants(session_id: str) -> JSONResponse:
        """Take back every grant the chat gave (the main chat's: every global one).

        A side chat's clear leaves the main chat's grants, which still hold
        here: ``still_everywhere`` counts them.
        """
        console.require_store()
        await console.require_session(session_id)
        gone = await console.sessions.forget_grant(session_id)
        still = {
            g.key
            for g in console.sessions.book.visible(session_id)
            if g.scope == "global"
        }
        return JSONResponse(
            content={"revoked": len(gone), "still_everywhere": len(still)}
        )

    @app.delete("/api/sessions/{session_id}/folders")
    async def forget_folder(session_id: str, path: str = "") -> JSONResponse:
        """Take back the chat's grant of one folder outside the workspace (#405).

        The folder is the ``path`` query parameter; without one, or for a
        folder the chat was not granted, the answer is 404. The answer is a
        named revoke's: a folder the main chat allowed goes everywhere, and
        ``everywhere`` says so.
        """
        console.require_store()
        await console.require_session(session_id)
        gone = await console.sessions.forget_folder(session_id, path) if path else []
        if not gone:
            raise ApiError(404, "not_found", t("errors.noGrant"))
        return JSONResponse(
            content={
                "revoked": [{"key": g.key, "scope": g.scope} for g in gone],
                "everywhere": any(g.scope == "global" for g in gone),
            }
        )

    @app.delete("/api/sessions/{session_id}/approvals/{entry_id}")
    async def revoke_approval(session_id: str, entry_id: str) -> Response:
        """Take back one recorded approval: the step asks again; what ran stays run."""
        console.require_store()
        await console.require_session(session_id)
        if not console.approvals.revoke(session_id, entry_id):
            raise ApiError(404, "not_found", t("errors.noApproval"))
        return Response(status_code=204)

    @app.get("/api/sessions/{session_id}/memory")
    async def session_memory(session_id: str) -> JSONResponse:
        console.require_store()
        await console.require_session(session_id)
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
        await console.require_session(session_id)
        turns: list[TurnInfo] = await console.store.list_turns(
            session_id, limit=limit, before=before
        )
        # A running turn has no RunResult yet; it is the transcript's row.
        runs = [run_json(turn) for turn in turns if turn.status != "running"]
        has_earlier = False
        if turns:
            older = await console.store.list_turns(
                session_id, limit=1, before=turns[0].turn
            )
            has_earlier = bool(older)
        return JSONResponse(content={"runs": runs, "has_earlier": has_earlier})


__all__ = ["MEMORY_TEXT_CHARS", "add_routes"]
