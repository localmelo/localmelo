"""API v2 writes: create a session, start, retry, stop and approve turns.

Every route here writes. A turn starts under the session's lock and returns
as soon as its row exists (202); its events arrive on the stream of
``routes_sessions``. A ``client_request_id`` makes a send safe to repeat:
the same key again answers 200 with the turn it started. Console turns run
without the scheduler gate until the next change (3b) adds its provider
wrapper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from localmelo.melo.checker import Checker, GatewayIngressPayload
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.support.sessions.store import (
    SessionBusyError,
    SessionInfo,
    SessionStoreError,
    TurnInfo,
    check_title,
)
from localmelo.support.sessions.submissions import (
    SubmissionError,
    SubmissionRepeated,
    SubmissionWithdrawn,
    check_client_request_id,
)

from .errors import ApiError
from .registry import STOP_REASON
from .shapes import session_json, submission_json, turn_json

if TYPE_CHECKING:
    from localmelo.support.sessions.runner import ResumeInfo

    from . import Console

_MAX_REASON_CHARS = 500
_checker = Checker()


# ── Request bodies ──


class _Attachment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    text: str
    chars: int


class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    attachments: list[_Attachment] = []
    client_request_id: str | None = None


class CreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace: str | None = None
    turn: TurnRequest | None = None


class RetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str


class PatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    workspace: str | None = None


class CancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    attachments: list[_Attachment] = []


class ApprovalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice: Literal["once", "session", "deny"]
    reason: str = ""


# ── Helpers ──


def _key(value: str | None) -> str | None:
    """A checked ``client_request_id``, or ``None`` when absent."""
    if value is None:
        return None
    try:
        return check_client_request_id(value)
    except ValueError as exc:
        raise ApiError(400, "invalid_request", str(exc)) from exc


def _compose(request: TurnRequest | CancelRequest) -> str:
    """The user text of a turn: the typed text, with no file blocks yet."""
    if not request.text.strip():
        raise ApiError(400, "invalid_request", "text: the message must not be blank")
    if request.attachments:
        raise ApiError(
            400,
            "invalid_request",
            "attachments: files are not supported yet; send the text alone",
        )
    return request.text


def _check_text(console: Console, text: str, session_id: str | None) -> None:
    """``check_gateway_ingress``, as every gateway entry point runs it."""
    check = _checker.check_gateway_ingress(
        GatewayIngressPayload(query=text, session_id=session_id)
    )
    if not check.allowed:
        raise ApiError(400, "text_rejected", check.reason)


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
    """Await a registry call, mapping the failures a turn can raise."""
    try:
        return await coro
    except SessionBusyError as exc:
        raise ApiError(
            409, "session_busy", "A turn is already running in this session."
        ) from exc
    except SessionStoreError as exc:
        raise await console.store_api_error(exc, session_id) from exc


async def _require_session(console: Console, session_id: str) -> SessionInfo:
    info: SessionInfo | None = await _store_call(
        console, console.store.get_session(session_id), session_id
    )
    if info is None:
        raise ApiError(404, "not_found", "Session not found.")
    return info


def _submission_error(exc: SubmissionError) -> ApiError:
    """The 409 an :class:`ApiError`-free submission failure answers with."""
    if isinstance(exc, SubmissionWithdrawn):
        return ApiError(
            409,
            "submission_withdrawn",
            "This message was cancelled before it started.",
        )
    return ApiError(
        409,
        "submission_conflict",
        "This message id was already used for a different message.",
    )


def _resume_json(resumed: ResumeInfo | None) -> dict[str, Any]:
    """The three resume keys of a 202, from the open's ``ResumeInfo``."""
    if resumed is None or resumed.turns <= 0:
        return {
            "resumed_turns": None,
            "resume_mode": None,
            "resumed_checkpoint_turns": None,
        }
    return {
        "resumed_turns": resumed.turns,
        "resume_mode": resumed.mode,
        "resumed_checkpoint_turns": resumed.checkpoint_turns,
    }


def _turn_body(turn: TurnInfo, resumed: ResumeInfo | None) -> dict[str, Any]:
    return {"turn_id": turn.id, "turn": turn_json(turn), **_resume_json(resumed)}


def _pending(console: Console, session_id: str) -> int:
    return len(console.broker.pending(session_id))


# ── Routes ──


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the create, turn, retry, stop, session and approval routes."""

    @app.post("/api/sessions")
    async def create_session(body: CreateRequest) -> JSONResponse:
        console.require_store()
        first: tuple[str, str] | None = None
        if body.turn is not None:
            composed = _compose(body.turn)
            _check_text(console, composed, None)
            first = (body.turn.text, composed)
        # The key is judged after the text, as 2.4 orders the checks.
        key = None if body.turn is None else _key(body.turn.client_request_id)
        try:
            entry, turn = await _sessions_call(
                console,
                console.sessions.create(body.workspace, first, client_request_id=key),
            )
        except SubmissionRepeated as exc:
            info = await _require_session(console, exc.turn.session_id)
            return JSONResponse(
                status_code=200,
                content={
                    "session": session_json(
                        info, pending_approvals=_pending(console, info.id)
                    ),
                    **_turn_body(exc.turn, None),
                    "repeated": True,
                },
            )
        except SubmissionError as exc:
            raise _submission_error(exc) from exc
        info = await _require_session(console, entry.session_id)
        content: dict[str, Any] = {
            "session": session_json(
                info, pending_approvals=_pending(console, entry.session_id)
            )
        }
        if turn is not None:
            content.update(_turn_body(turn, None))
        return JSONResponse(status_code=201, content=content)

    @app.post("/api/sessions/{session_id}/turns")
    async def start_turn(session_id: str, body: TurnRequest) -> JSONResponse:
        console.require_store()
        composed = _compose(body)
        _check_text(console, composed, session_id)
        # The key is judged after the text, as 2.4 orders the checks.
        key = _key(body.client_request_id)
        await _require_session(console, session_id)
        try:
            turn, resumed = await _sessions_call(
                console,
                console.sessions.start_turn(
                    session_id,
                    body.text,
                    composed,
                    client_request_id=key,
                ),
                session_id,
            )
        except SubmissionRepeated as exc:
            return JSONResponse(
                status_code=200,
                content={**_turn_body(exc.turn, None), "repeated": True},
            )
        except SubmissionError as exc:
            raise _submission_error(exc) from exc
        return JSONResponse(status_code=202, content=_turn_body(turn, resumed))

    @app.post("/api/sessions/{session_id}/retry")
    async def retry_turn(session_id: str, body: RetryRequest) -> JSONResponse:
        console.require_store()
        await _require_session(console, session_id)
        turn, resumed = await _sessions_call(
            console,
            console.sessions.retry(session_id, body.turn_id),
            session_id,
        )
        return JSONResponse(status_code=202, content=_turn_body(turn, resumed))

    @app.post("/api/turns/{turn_id}/cancel")
    async def cancel_turn(turn_id: str) -> JSONResponse:
        console.require_store()
        entry = console.sessions.find_turn(turn_id)
        if entry is None or not entry.runner.cancel(
            STOP_REASON, cause="user_stop", turn_id=turn_id
        ):
            raise ApiError(
                409,
                "turn_not_running",
                "This turn is not running. It may have finished.",
            )
        return JSONResponse(
            status_code=202, content={"turn_id": turn_id, "cancelling": True}
        )

    async def _cancel_by_key(
        scope: str | None, client_request_id: str, body: CancelRequest
    ) -> JSONResponse:
        console.require_store()
        key = _key(client_request_id)
        assert key is not None
        composed = _compose(body)
        if scope is not None:
            await _require_session(console, scope)
        try:
            submission = await _store_call(
                console,
                console.store.withdraw_submission(scope, key, user_text=composed),
                scope,
            )
        except SubmissionError as exc:
            raise _submission_error(exc) from exc
        cancelling = False
        if (
            submission.state == "accepted"
            and submission.turn_id is not None
            and submission.status == "running"
        ):
            entry = console.sessions.find_turn(submission.turn_id)
            if entry is not None:
                cancelling = entry.runner.cancel(
                    STOP_REASON, cause="user_stop", turn_id=submission.turn_id
                )
            else:
                # The key made a session whose create is still registering
                # its first turn: the stop lands when it does.
                console.sessions.defer_cancel(submission.turn_id)
                cancelling = True
        return JSONResponse(
            status_code=202 if cancelling else 200,
            content={
                "submission": submission_json(submission, key),
                "cancelling": cancelling,
            },
        )

    @app.post("/api/sessions/{session_id}/submissions/{client_request_id}/cancel")
    async def cancel_session_submission(
        session_id: str, client_request_id: str, body: CancelRequest
    ) -> JSONResponse:
        return await _cancel_by_key(session_id, client_request_id, body)

    @app.post("/api/submissions/{client_request_id}/cancel")
    async def cancel_create_submission(
        client_request_id: str, body: CancelRequest
    ) -> JSONResponse:
        return await _cancel_by_key(None, client_request_id, body)

    @app.post("/api/approvals/{request_id}")
    async def answer_approval(request_id: str, body: ApprovalAnswer) -> JSONResponse:
        if body.reason and body.choice != "deny":
            raise ApiError(
                400, "invalid_request", "reason: only a denial takes a reason"
            )
        if len(body.reason) > _MAX_REASON_CHARS:
            raise ApiError(
                400,
                "invalid_request",
                f"reason: at most {_MAX_REASON_CHARS} characters",
            )
        if body.choice == "once":
            decision = ApprovalDecision(True)
        elif body.choice == "session":
            decision = ApprovalDecision(True, remember=True)
        else:
            decision = ApprovalDecision(False, body.reason)
        outcome = console.broker.answer(request_id, decision)
        if outcome == "unknown":
            raise ApiError(404, "not_found", "Approval not found.")
        if outcome == "already_resolved":
            raise ApiError(
                409, "already_answered", "This approval was answered already."
            )
        return JSONResponse(
            content={
                "request_id": request_id,
                "outcome": "allowed" if decision.allow else "denied",
                "remember": decision.remember,
            }
        )

    @app.patch("/api/sessions/{session_id}")
    async def patch_session(session_id: str, body: PatchRequest) -> JSONResponse:
        console.require_store()
        await _require_session(console, session_id)
        # Both changes are judged before either is written, so a refused
        # title or workspace leaves the session as it was.
        title: str | None = None
        if body.title is not None:
            try:
                title = check_title(body.title)
            except ValueError as exc:
                raise ApiError(400, "invalid_request", str(exc)) from exc
        if body.workspace is not None:
            await _sessions_call(
                console,
                console.sessions.change_workspace(session_id, body.workspace),
                session_id,
            )
        if title is not None:
            renamed = await _store_call(
                console,
                console.store.rename_session(session_id, title),
                session_id,
            )
            if not renamed:
                raise ApiError(404, "not_found", "Session not found.")
        info = await _require_session(console, session_id)
        return JSONResponse(
            content={
                "session": session_json(
                    info, pending_approvals=_pending(console, session_id)
                )
            }
        )

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> Response:
        console.require_store()
        await _sessions_call(console, console.sessions.delete(session_id), session_id)
        return Response(status_code=204)


__all__ = ["add_routes"]
