"""API v2 reads: sessions, turns, events, blobs and submissions.

Every route here is a read. A turn starts in PR 3, and the write routes of
settings in PR 4. An unknown session is a 404 on every path below
``/api/sessions/{id}``; a store that could not open is a 503
``store_unavailable``; health is the one route that serves anyway.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Header, Query
from fastapi.responses import JSONResponse, Response

from localmelo.support.sessions.store import (
    SESSION_KINDS,
    SessionInfo,
    SessionStoreError,
    StoredEvent,
    TurnInfo,
)
from localmelo.support.sessions.submissions import check_client_request_id

from .errors import ApiError
from .shapes import approval_json, session_json, submission_json, turn_json
from .sse import EventStream, EventStreamResponse

if TYPE_CHECKING:
    from . import Console

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
# The image types a blob may be shown inline as; anything else, SVG
# included, is an opaque download.
_INLINE_IMAGES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_TURN_LIMIT = 50
_SESSION_LIMIT = 200
# SQLite event ids are signed 64-bit integers.
_EVENT_ID_MAX = 2**63 - 1


# ── Helpers ──


async def _store_call(
    console: Console, coro: Any, session_id: str | None = None
) -> Any:
    """Await one store call, mapping a store failure as 2.2 says."""
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


async def _turn_events(
    console: Console, session_id: str, turn: int
) -> list[StoredEvent]:
    """Every stored event of one turn, in id order, in pages."""
    events: list[StoredEvent] = []
    after = 0
    while True:
        page = await _store_call(
            console,
            console.store.list_events(
                session_id, after_id=after, limit=1000, turns=(turn, turn)
            ),
            session_id,
        )
        events.extend(page)
        if len(page) < 1000:
            return events
        after = page[-1].id or after


def _completion_tokens(turn: TurnInfo) -> int:
    """A finished turn's output tokens, as its ``RunResult`` reports them."""
    result = turn.result
    if not isinstance(result, dict):
        return 0
    usage = result.get("usage")
    if not isinstance(usage, dict):
        return 0
    value = usage.get("completion_tokens")
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _pending_approvals(console: Console, session_id: str) -> list[dict[str, Any]]:
    return [approval_json(request) for request in console.broker.pending(session_id)]


def _agent_state(console: Console, session_id: str) -> str:
    entry = console.sessions.get(session_id)
    if entry is None:
        return "not_loaded"
    return "open" if entry.agent is not None else "paused"


# ── Routes ──


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the sessions, turns, events, blobs and submission routes."""

    @app.get("/api/sessions")
    async def list_sessions(
        query: str = Query("", max_length=200),
        limit: int = Query(50, ge=1, le=_SESSION_LIMIT),
        kind: str = Query("chat"),
    ) -> JSONResponse:
        console.require_store()
        if kind not in SESSION_KINDS:
            raise ApiError(
                400,
                "invalid_request",
                f"kind must be one of {', '.join(sorted(SESSION_KINDS))}",
            )
        sessions = await _store_call(
            console,
            console.store.list_sessions(limit=limit + 1, query=query, kinds=(kind,)),
        )
        has_more = len(sessions) > limit
        return JSONResponse(
            content={
                "sessions": [
                    session_json(
                        info,
                        pending_approvals=len(console.broker.pending(info.id)),
                    )
                    for info in sessions[:limit]
                ],
                "has_more": has_more,
            }
        )

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> JSONResponse:
        console.require_store()
        info = await _require_session(console, session_id)
        entry = console.sessions.get(session_id)
        grants = sorted(entry.policy.allowed_tools) if entry is not None else []
        granted_at = (
            {tool: entry.granted_at.get(tool, 0.0) for tool in grants}
            if entry is not None
            else {}
        )
        turns = await _store_call(
            console, console.store.list_turns(session_id), session_id
        )
        tokens = sum(_completion_tokens(turn) for turn in turns)
        return JSONResponse(
            content={
                "session": session_json(
                    info, pending_approvals=len(console.broker.pending(session_id))
                ),
                "pending_approvals": _pending_approvals(console, session_id),
                "grants": grants,
                "granted_at": granted_at,
                "mcp_error": "" if entry is None else entry.mcp_error,
                "agent": _agent_state(console, session_id),
                "tokens": tokens,
            }
        )

    @app.get("/api/sessions/{session_id}/turns")
    async def list_turns(
        session_id: str,
        before: int | None = Query(None, ge=1, le=_EVENT_ID_MAX),
        limit: int = Query(10, ge=1, le=_TURN_LIMIT),
        events: bool = Query(False),
    ) -> JSONResponse:
        console.require_store()
        await _require_session(console, session_id)
        turns = await _store_call(
            console,
            console.store.list_turns(session_id, limit=limit, before=before),
            session_id,
        )
        has_earlier = False
        if turns:
            older = await _store_call(
                console,
                console.store.list_turns(session_id, limit=1, before=turns[0].turn),
                session_id,
            )
            has_earlier = bool(older)
        body: dict[str, Any] = {
            "turns": [
                turn_json(
                    turn,
                    events=(
                        await _turn_events(console, session_id, turn.turn)
                        if events
                        else None
                    ),
                )
                for turn in turns
            ],
            "has_earlier": has_earlier,
        }
        if before is None:
            body["replay_from"] = await _replay_from(console, session_id, turns)
        return JSONResponse(content=body)

    @app.get("/api/sessions/{session_id}/events")
    async def session_events(
        session_id: str,
        after: str | None = Query(None),
        follow: bool = Query(True),
        last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    ) -> Response:
        console.require_store()
        await _require_session(console, session_id)
        # The header wins over ?after, so ?after is judged only without one.
        if last_event_id is not None:
            start = _event_id(last_event_id, "Last-Event-ID")
        elif after is not None:
            start = _event_id(after, "after")
        else:
            start = 0
        high_water: int | None = None
        if not follow:
            high_water = await _store_call(
                console, console.store.latest_event_id(session_id), session_id
            )
        console.stream_opened()
        stream = EventStream(
            console.store,
            console.hub,
            console.shutdown,
            session_id,
            after_id=start,
            follow=follow,
            high_water=high_water,
            on_close=console.stream_closed,
        )
        return EventStreamResponse(
            stream,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/sessions/{session_id}/blobs/{sha256}")
    async def session_blob(session_id: str, sha256: str) -> Response:
        console.require_store()
        if _SHA256_RE.fullmatch(sha256) is None:
            raise ApiError(404, "not_found", "Blob not found.")
        await _require_session(console, session_id)
        blob = await _store_call(
            console, console.store.get_blob(session_id, sha256), session_id
        )
        if blob is None:
            raise ApiError(404, "not_found", "Blob not found.")
        mime_type, data = blob
        if mime_type in _INLINE_IMAGES:
            disposition = "inline"
        else:
            mime_type = "application/octet-stream"
            disposition = f'attachment; filename="{sha256}.bin"'
        return Response(
            content=data,
            media_type=mime_type,
            headers={
                "Content-Disposition": disposition,
                "Content-Security-Policy": "default-src 'none'; sandbox",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
            },
        )

    async def submission_body(session_id: str | None, client_request_id: str) -> Any:
        console.require_store()
        # An unknown session is 404 on every /api/sessions/{id}/... path,
        # before any parameter of the path is judged.
        if session_id is not None:
            await _require_session(console, session_id)
        try:
            check_client_request_id(client_request_id)
        except ValueError as exc:
            raise ApiError(400, "invalid_request", str(exc)) from None
        submission = await _store_call(
            console,
            console.store.get_submission(session_id, client_request_id),
            session_id,
        )
        return submission_json(submission, client_request_id)

    @app.get("/api/sessions/{session_id}/submissions/{client_request_id}")
    async def get_session_submission(
        session_id: str, client_request_id: str
    ) -> JSONResponse:
        return JSONResponse(
            content={"submission": await submission_body(session_id, client_request_id)}
        )

    @app.get("/api/submissions/{client_request_id}")
    async def get_submission(client_request_id: str) -> JSONResponse:
        return JSONResponse(
            content={"submission": await submission_body(None, client_request_id)}
        )


def _event_id(value: str, name: str) -> int:
    """A decimal event id in SQLite's range, or 400 ``invalid_request``."""
    # Decimal digits only: int() would also take "+1", "1_0" and non-ASCII
    # digits, which are not an event id. Leading zeros do not make a valid
    # id too long; more significant digits than a 64-bit integer can hold
    # are refused before int() or SQLite sees them.
    if re.fullmatch(r"[0-9]+", value) is None:
        raise ApiError(
            400,
            "invalid_request",
            f"{name} must be an integer of at least 0.",
        )
    digits = value.lstrip("0") or "0"
    if len(digits) > 19:
        raise ApiError(
            400,
            "invalid_request",
            f"{name} must be an integer of at least 0.",
        )
    number = int(digits)
    if number > _EVENT_ID_MAX:
        raise ApiError(
            400,
            "invalid_request",
            f"{name} must be an integer of at least 0.",
        )
    return number


async def _replay_from(console: Console, session_id: str, turns: list[TurnInfo]) -> int:
    """The oldest returned turn's ``turn.started`` id minus one, or 0."""
    if not turns:
        return 0
    first = turns[0].turn
    events: list[StoredEvent] = await _store_call(
        console,
        console.store.list_events(session_id, limit=1, turns=(first, first)),
        session_id,
    )
    if events and events[0].id is not None:
        return max(events[0].id - 1, 0)
    return 0
