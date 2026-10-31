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

from localmelo.support.gateway.i18n import t
from localmelo.support.sessions.store import (
    SESSION_KINDS,
    SessionInfo,
    SessionStore,
    SessionStoreError,
    StoredEvent,
    TurnInfo,
)
from localmelo.support.sessions.submissions import check_client_request_id

from .cards import CARD_EVENT, CARD_PAGE, card_json
from .errors import ApiError
from .front import ECHO_EVENT
from .running import waiting_by_chat
from .shapes import approval_json, session_json, submission_json, turn_json
from .sse import EventStream, EventStreamResponse
from .task_submissions import created, submission_of
from .worker_reader import WorkerRecord, _total_tokens, decode_cursor, encode_cursor

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
        raise ApiError(404, "not_found", t("errors.sessionNotFound"))
    return info


async def _read_session(
    console: Console, session_id: str
) -> tuple[SessionInfo, SessionStore]:
    """Resolve a gateway session first, then a read-only worker session."""
    info = await _store_call(console, console.store.get_session(session_id), session_id)
    if info is not None:
        return info, console.store
    info = await _store_call(console, console.worker_history.get_session(session_id))
    if info is None:
        raise ApiError(404, "not_found", t("errors.sessionNotFound"))
    return info, console.worker_history


async def _turn_events(
    console: Console, store: SessionStore, session_id: str, turn: int
) -> list[StoredEvent]:
    """Every stored event of one turn, in id order, in pages."""
    events: list[StoredEvent] = []
    after = 0
    while True:
        page = await _store_call(
            console,
            store.list_events(
                session_id, after_id=after, limit=1000, turns=(turn, turn)
            ),
            session_id if store is console.store else None,
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


def _worker_body(
    info: SessionInfo, record: WorkerRecord | None = None
) -> dict[str, Any]:
    meta = info.meta
    body = {**session_json(info, running_here=False), "meta": meta}
    # A chat of the terminal is listed under ``kind=terminal``; the console's
    # own ``kind=chat`` list never holds one.
    if info.kind == "chat":
        body["kind"] = "terminal"
    for key in ("source", "parent_run", "git_branch", "git_head", "exit_code"):
        body[key] = meta.get(key)
    for key in ("status", "steps"):
        body[key] = getattr(record, key) if record is not None else meta.get(key)
    body["tokens"] = record.tokens if record is not None else _total_tokens(meta, {})
    body["started_at"] = record.started_at if record is not None else None
    body["duration_s"] = record.duration_s if record is not None else None
    return body


def agent_state(console: Console, session_id: str) -> str:
    """``open``, ``paused`` or ``not_loaded``, as the session routes report it."""
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
        status: str | None = Query(None, max_length=200),
        source: str | None = Query(None, max_length=200),
        parent_run: str | None = Query(None, max_length=200),
        cursor: str | None = Query(None, max_length=512),
    ) -> JSONResponse:
        console.require_store()
        if kind not in SESSION_KINDS and kind != "terminal":
            raise ApiError(
                400,
                "invalid_request",
                f"kind must be one of {', '.join(sorted(SESSION_KINDS | {'terminal'}))}",
            )
        if kind in ("worker", "terminal"):
            try:
                after = decode_cursor(cursor)
            except ValueError as exc:
                raise ApiError(400, "invalid_request", str(exc)) from None
            records = await _store_call(
                console,
                console.worker_history.workers(
                    limit=limit + 1,
                    query=query,
                    status=status,
                    source=source,
                    parent_run=parent_run,
                    after=after,
                    kind="chat" if kind == "terminal" else "worker",
                ),
            )
            has_more = len(records) > limit
            return JSONResponse(
                content={
                    "sessions": [
                        _worker_body(row.info, row) for row in records[:limit]
                    ],
                    "machine": console.gate.machine_payload(),
                    "has_more": has_more,
                    "next_cursor": (
                        encode_cursor(records[limit - 1]) if has_more else None
                    ),
                }
            )
        if any(value is not None for value in (status, source, parent_run, cursor)):
            raise ApiError(
                400,
                "invalid_request",
                "Worker filters require kind=worker or kind=terminal.",
            )
        # A row with no turn is a create that never got its message (a
        # shutdown in the middle of one); the list starts at one turn, a
        # running one included, so a session appears as it is created. The
        # main chat is there before its first message.
        sessions = await _store_call(
            console,
            console.store.list_sessions(
                limit=limit + 1,
                query=query,
                kinds=(kind,),
                min_turns=0 if kind == "front" else 1,
                # A chat that took a task by code has an echo and no turn.
                or_events=(ECHO_EVENT, CARD_EVENT),
            ),
        )
        has_more = len(sessions) > limit
        bodies = [console.session_body(info) for info in sessions[:limit]]
        waiting = await waiting_by_chat(console) if kind == "chat" else {}
        for body in bodies:
            if waiting.get(body["id"]):
                # A side chat's task waits for the person's answer.
                body["waiting_tasks"] = waiting[body["id"]]
        content: dict[str, Any] = {
            "sessions": bodies,
            # The one shared gate's machine snapshot: the page's header
            # shows the GPU lock and battery, as its /v1 list did (S2b).
            "machine": console.gate.machine_payload(),
            "has_more": has_more,
        }
        if kind == "chat":
            # The chats are the main chat's side chats; the list names it.
            main = await _store_call(console, console.main_body())
            asking = sum(n for owner, n in waiting.items() if owner != console.main_id)
            if main is not None and asking:
                main["waiting_in_side"] = asking  # tasks waiting in side chats
            content["main"] = main
        return JSONResponse(content=content)

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> JSONResponse:
        console.require_store()
        info, store = await _read_session(console, session_id)
        worker = store is console.worker_history
        entry = None if worker else console.sessions.get(session_id)
        grants = sorted(entry.policy.allowed_tools) if entry is not None else []
        granted_at = (
            {tool: entry.granted_at.get(tool, 0.0) for tool in grants}
            if entry is not None
            else {}
        )
        turns = await _store_call(
            console, store.list_turns(session_id), None if worker else session_id
        )
        tokens: int | None = sum(_completion_tokens(turn) for turn in turns)
        if worker and not any(
            isinstance(turn.result, dict)
            and isinstance(turn.result.get("usage"), dict)
            and isinstance(turn.result["usage"].get("completion_tokens"), int)
            and not isinstance(turn.result["usage"].get("completion_tokens"), bool)
            for turn in turns
        ):
            tokens = None
        records = (
            await _store_call(
                console,
                console.worker_history.workers(
                    limit=1, query="", session_id=session_id, kind=info.kind
                ),
            )
            if worker
            else []
        )
        return JSONResponse(
            content={
                "session": (
                    _worker_body(info, records[0] if records else None)
                    if worker
                    else console.session_body(info)
                ),
                "pending_approvals": (
                    [] if worker else _pending_approvals(console, session_id)
                ),
                "grants": grants,
                "granted_at": granted_at,
                "mcp_error": "" if entry is None else entry.mcp_error,
                "agent": "not_loaded" if worker else agent_state(console, session_id),
                "tokens": tokens,
                # Where a client's stream starts to hear only what comes next.
                "last_event_id": await _store_call(
                    console,
                    store.latest_event_id(session_id),
                    None if worker else session_id,
                ),
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
        info, store = await _read_session(console, session_id)
        worker = store is console.worker_history
        turns = await _store_call(
            console,
            store.list_turns(session_id, limit=limit, before=before),
            None if worker else session_id,
        )
        has_earlier = False
        if turns:
            older = await _store_call(
                console,
                store.list_turns(session_id, limit=1, before=turns[0].turn),
                None if worker else session_id,
            )
            has_earlier = bool(older)
        cards: list[StoredEvent] = []
        echoes: list[StoredEvent] = []
        if info.kind in ("front", "chat"):
            # Every console chat's page is as new as what has come back to it.
            await console.cards.flush()
            cards = await _page_cards(console, session_id, turns, before)
            echoes = await _page_cards(console, session_id, turns, before, ECHO_EVENT)
        body: dict[str, Any] = {
            "turns": [
                turn_json(
                    turn,
                    running_here=False if worker else None,
                    events=(
                        await _turn_events(console, store, session_id, turn.turn)
                        if events
                        else None
                    ),
                )
                for turn in turns
            ],
            "has_earlier": has_earlier,
        }
        if before is None:
            body["replay_from"] = await _replay_from(console, store, session_id, turns)
            if (cards or echoes) and not turns:
                # The stream replays from here: the cards and echoes already given.
                first = min(event.id or 1 for event in cards + echoes)
                body["replay_from"] = max(first - 1, 0)
        if info.kind == "front" or cards or echoes:
            # A side chat has them only once it started a task.
            body["cards"] = [card_json(event) for event in cards]
            body["echoes"] = [card_json(event) for event in echoes]
        return JSONResponse(content=body)

    @app.get("/api/sessions/{session_id}/events")
    async def session_events(
        session_id: str,
        after: str | None = Query(None),
        follow: bool = Query(True),
        last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    ) -> Response:
        console.require_store()
        _, store = await _read_session(console, session_id)
        worker = store is console.worker_history
        scope = None if worker else session_id
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
                console, store.latest_event_id(session_id), scope
            )

        async def run_is_live() -> bool:
            info = await store.get_session(session_id)
            return info is not None and info.busy

        console.stream_opened()
        stream = EventStream(
            store,
            console.hub,
            console.shutdown,
            session_id,
            after_id=start,
            follow=follow,
            high_water=high_water,
            on_close=console.stream_closed,
            # The hub hears only this process's commits; a worker run is
            # written by another process, so its stream polls the file.
            live=run_is_live if worker else None,
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
            raise ApiError(404, "not_found", t("errors.blobNotFound"))
        _, store = await _read_session(console, session_id)
        blob = await _store_call(
            console,
            store.get_blob(session_id, sha256),
            None if store is console.worker_history else session_id,
        )
        if blob is None:
            raise ApiError(404, "not_found", t("errors.blobNotFound"))
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
        if submission is None and session_id is None:
            # A create that made a chat for a task is accepted, with the chat and the task.
            made = await created(console, client_request_id)
            if made is not None:
                return submission_of(made[0], made[1], client_request_id)
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


async def _turn_start(console: Console, session_id: str, turn: int) -> int | None:
    """The id of a turn's first event, or ``None``."""
    events: list[StoredEvent] = await _store_call(
        console,
        console.store.list_events(session_id, limit=1, turns=(turn, turn)),
        session_id,
    )
    return events[0].id if events else None


async def _page_cards(
    console: Console,
    session_id: str,
    turns: list[TurnInfo],
    before: int | None,
    type_: str = CARD_EVENT,
) -> list[StoredEvent]:
    """A chat's cards (or echoes) beside a page of its turns, the newest :data:`CARD_PAGE`.

    From the page's oldest turn (or the start, when it is the oldest page)
    to the next page's first turn (or now).
    """
    low = 0
    if turns:
        older = await _store_call(
            console,
            console.store.list_turns(session_id, limit=1, before=turns[0].turn),
            session_id,
        )
        if older:
            low = (await _turn_start(console, session_id, turns[0].turn) or 1) - 1
    high = None if before is None else await _turn_start(console, session_id, before)
    return list(
        await _store_call(
            console,
            console.store.events_of_type(
                session_id, type_, after_id=low, before_id=high, limit=CARD_PAGE
            ),
            session_id,
        )
    )


async def _replay_from(
    console: Console, store: SessionStore, session_id: str, turns: list[TurnInfo]
) -> int:
    """The oldest returned turn's ``turn.started`` id minus one, or 0."""
    if not turns:
        return 0
    first = turns[0].turn
    events: list[StoredEvent] = await _store_call(
        console,
        store.list_events(session_id, limit=1, turns=(first, first)),
        session_id if store is console.store else None,
    )
    if events and events[0].id is not None:
        return max(events[0].id - 1, 0)
    return 0
