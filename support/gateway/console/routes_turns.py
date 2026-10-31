"""API v2 writes: create a session, start, retry, stop and approve turns.

Every route here writes. A turn starts under the session's lock and returns
as soon as its row exists (202); its events arrive on the stream of
``routes_sessions``. A ``client_request_id`` makes a send safe to repeat:
the same key again answers 200 with the turn it started. A console turn
passes the scheduler gate: ``default_builder`` wraps the Agent's providers
with ``GatedLLM``/``GatedEmbedding``, so its model calls wait for a slot on
the one gate this gateway shares with the workspace page.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from localmelo.melo.checker import Checker, GatewayIngressPayload
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.contracts.review import offers_grant
from localmelo.support.gateway.auto_blocks import (
    BlockError,
    block_to_allow,
    go_ahead_text,
)
from localmelo.support.gateway.i18n import t
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

from .bodies import Body
from .errors import ApiError
from .registry import STOP_REASON
from .shapes import submission_json, turn_json

if TYPE_CHECKING:
    from localmelo.support.sessions.runner import ResumeInfo

    from . import Console

_MAX_REASON_CHARS = 500
_checker = Checker()


# ── Request bodies ──


class _Attachment(Body):
    name: str
    text: str
    chars: int


class TurnRequest(Body):
    text: str
    attachments: list[_Attachment] = []
    client_request_id: str | None = None
    # The composer's Run as a task: the chat starts it as a task.
    as_task: bool = False


class CreateRequest(Body):
    workspace: str | None = None
    turn: TurnRequest | None = None
    # Ask or Auto (#339); absent: the settings' mode for new chats.
    approvals: Literal["ask", "auto"] | None = None


class RetryRequest(Body):
    turn_id: str


class PatchRequest(Body):
    title: str | None = None
    workspace: str | None = None
    # The chat's mode (#339): from its next step.
    approvals: Literal["ask", "auto"] | None = None


class CancelRequest(Body):
    text: str
    attachments: list[_Attachment] = []


class ApprovalAnswer(Body):
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
    """The user text of a turn: the typed text, trimmed as the checker trims it.

    The trim happens here, so the text the store judges by length, keeps in
    a submission and gives the Agent is the one the checker approved.
    """
    text = request.text.strip()
    if not text:
        raise ApiError(400, "invalid_request", t("errors.blankMessage"))
    if request.attachments:
        raise ApiError(
            400,
            "invalid_request",
            t("errors.noAttachments"),
        )
    return text


def _check_text(console: Console, text: str, session_id: str | None) -> str:
    """``check_gateway_ingress``, as every gateway entry point runs it.

    Returns the sanitized text, which is what the turn runs.
    """
    check = _checker.check_gateway_ingress(
        GatewayIngressPayload(query=text, session_id=session_id)
    )
    if not check.allowed:
        raise ApiError(400, "text_rejected", check.reason)
    sanitized = check.sanitized_payload
    assert sanitized is not None
    return str(sanitized.query)


def _turn_text(console: Console, request: TurnRequest, session_id: str | None) -> str:
    """The composed text of a turn: checked, and trimmed by the checker."""
    return _check_text(console, _compose(request), session_id)


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
        raise ApiError(409, "session_busy", t("errors.turnRunning")) from exc
    except SessionStoreError as exc:
        raise await console.store_api_error(exc, session_id) from exc


async def _require_session(console: Console, session_id: str) -> SessionInfo:
    info: SessionInfo | None = await _store_call(
        console, console.store.get_session(session_id), session_id
    )
    if info is None:
        raise ApiError(404, "not_found", t("errors.sessionNotFound"))
    return info


def _submission_error(exc: SubmissionError) -> ApiError:
    """The 409 an :class:`ApiError`-free submission failure answers with."""
    if isinstance(exc, SubmissionWithdrawn):
        return ApiError(
            409,
            "submission_withdrawn",
            t("errors.messageCancelled"),
        )
    return ApiError(
        409,
        "submission_conflict",
        t("errors.messageIdUsed"),
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


# ── Routes ──


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the create, turn, retry, stop, session and approval routes."""

    @app.post("/api/sessions")
    async def create_session(body: CreateRequest) -> JSONResponse:
        console.require_store()
        first: tuple[str, str] | None = None
        if body.turn is not None:
            composed = _turn_text(console, body.turn, None)
            first = (body.turn.text, composed)
        # The key is judged after the text, as 2.4 orders the checks.
        key = None if body.turn is None else _key(body.turn.client_request_id)
        try:
            entry, turn = await _sessions_call(
                console,
                console.sessions.create(
                    body.workspace, first, client_request_id=key, mode=body.approvals
                ),
            )
        except SubmissionRepeated as exc:
            info = await _require_session(console, exc.turn.session_id)
            return JSONResponse(
                status_code=200,
                content={
                    "session": console.session_body(info),
                    **_turn_body(exc.turn, None),
                    "repeated": True,
                },
            )
        except SubmissionError as exc:
            raise _submission_error(exc) from exc
        info = await _require_session(console, entry.session_id)
        content: dict[str, Any] = {"session": console.session_body(info)}
        if turn is not None:
            content.update(_turn_body(turn, None))
        return JSONResponse(status_code=201, content=content)

    @app.post("/api/sessions/{session_id}/turns")
    async def start_turn(session_id: str, body: TurnRequest) -> JSONResponse:
        console.require_store()
        # The session is judged first: a malformed id is 404, as it is on
        # every read of the same path, not the checker's 400.
        info = await _require_session(console, session_id)
        composed = _turn_text(console, body, session_id)
        # The key is judged after the text, as 2.4 orders the checks.
        key = _key(body.client_request_id)
        if info.kind in ("front", "chat"):
            # Code's rules first: a task to start or to change takes no model
            # call. A side chat has the same rules as the main chat, and its
            # tasks are its own.
            echo = await _store_call(
                console,
                console.front_for(session_id).route(
                    body.text, composed, as_task=body.as_task, key=key
                ),
                session_id,
            )
            if echo is not None:
                return JSONResponse(status_code=200, content={"echo": echo})
            if console.asks.holds(session_id):
                # The chat is answering a task's question: the message waits,
                # shown as queued, and runs as the chat's next turn (#369).
                echo = await _store_call(
                    console,
                    console.asks.queue(session_id, body.text, composed, key),
                    session_id,
                )
                return JSONResponse(status_code=200, content={"echo": echo})
        elif body.as_task:
            raise ApiError(400, "invalid_request", "as_task: tasks start from a chat")
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
                t("errors.turnNotRunning"),
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
            # A high-impact call is never granted for the chat: this answer
            # allows the one call, and the reply says so.
            asked = console.broker.request(request_id)
            grant = asked is None or offers_grant(asked.review)
            decision = ApprovalDecision(True, remember=grant)
        else:
            decision = ApprovalDecision(False, body.reason)
        outcome = console.broker.answer(request_id, decision)
        if outcome == "unknown":
            raise ApiError(404, "not_found", t("errors.approvalNotFound"))
        if outcome == "already_resolved":
            raise ApiError(409, "already_answered", t("errors.approvalAnswered"))
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
        info = await _require_session(console, session_id)
        if info.kind == "task":
            raise ApiError(
                409,
                "task_session_read_only",
                t("errors.taskChange"),
            )
        # The changes are judged before any is written, so a refused title,
        # workspace or mode leaves the session as it was.
        if body.approvals is not None and info.kind not in ("chat", "front"):
            raise ApiError(409, "not_a_chat", t("errors.modeNotChat"))
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
        if body.approvals is not None:
            await _sessions_call(
                console,
                console.sessions.set_mode(session_id, body.approvals),
                session_id,
            )
        if title is not None:
            # 2.10 lists PATCH among the lock holders; the workspace change
            # takes the lock itself, so a title alone takes it here.
            locked = (
                console.sessions.lock(session_id)
                if body.workspace is None
                else contextlib.nullcontext()
            )
            async with locked:
                renamed = await _store_call(
                    console,
                    console.store.rename_session(session_id, title),
                    session_id,
                )
            if not renamed:
                raise ApiError(404, "not_found", t("errors.sessionNotFound"))
        info = await _require_session(console, session_id)
        return JSONResponse(content={"session": console.session_body(info)})

    @app.post("/api/sessions/{session_id}/blocks/{request_id}/allow")
    async def allow_block(session_id: str, request_id: str) -> JSONResponse:
        """Allow once a step auto mode blocked, and ask melo to go ahead (#339)."""
        console.require_store()
        info = await _require_session(console, session_id)
        if info.kind == "task":
            raise ApiError(409, "task_session_read_only", t("errors.taskChange"))
        if info.kind not in ("chat", "front"):
            raise ApiError(409, "not_a_chat", t("errors.modeNotChat"))
        try:
            key, categories, tool = await _store_call(
                console,
                block_to_allow(
                    console.store,
                    session_id,
                    request_id,
                    lambda key: console.sessions.allowance_live(session_id, key),
                ),
                session_id,
            )
            text = _check_text(console, go_ahead_text(categories), session_id)
            turn, until = await _sessions_call(
                console,
                console.sessions.allow_once(session_id, key, categories, text, tool),
                session_id,
            )
        except BlockError as exc:
            raise ApiError(exc.status, exc.code, exc.message) from exc
        return JSONResponse(
            status_code=202,
            content={"step_key": key, "until": until, **_turn_body(turn, None)},
        )

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(
        session_id: str, notes: Literal["keep", "forget"] = "keep"
    ) -> Response:
        """Delete a chat; its notes stay unless ``?notes=forget``."""
        console.require_store()
        await _sessions_call(
            console,
            console.sessions.delete(session_id, forget_notes=notes == "forget"),
            session_id,
        )
        await console.forget_front(session_id)
        return Response(status_code=204)


__all__ = ["add_routes"]
