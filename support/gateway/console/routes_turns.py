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
from localmelo.support.gateway.front_rules import short_title
from localmelo.support.gateway.i18n import t
from localmelo.support.gateway.task_policy import TaskPolicyError
from localmelo.support.gateway.task_store import TaskStoreError
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
    check_submission,
)

from .bodies import Body
from .errors import ApiError
from .registry import STOP_REASON
from .shapes import submission_json, turn_json
from .task_submissions import (
    conflict,
    created,
    lock_name,
    started_echo,
    submission_of,
)

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
    # "always" is for a folder outside the workspace (#405): it keeps the
    # folder in the settings' extra folders.
    choice: Literal["once", "session", "always", "deny"]
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


async def _create_for_task(
    console: Console, body: CreateRequest, composed: str, key: str | None
) -> JSONResponse:
    """``POST /api/sessions`` with ``as_task``: the chat is made first, empty, and the message is a task it owns.

    The answer is 201 with the chat and the task's ``echo``, and no turn. A
    task that cannot start leaves no chat behind; one that is admitted keeps
    its chat. With a *key* (the turn's ``client_request_id``) the create is a
    submission like any other: a cancelled key is 409 ``submission_withdrawn``,
    one accepted for another message 409 ``submission_conflict``, and a repeat
    answers 200 with the chat and echo the first made (``repeated``), the echo
    written again if the gateway stopped between the task and it. Creates of a
    key go one at a time, with its cancel.
    """
    assert body.turn is not None
    text = body.turn.text
    # After the creates of a message under the key, and with the cancel of the key;
    # a cancel may still overtake a create of a message while it builds.
    async with console.sessions.keyed(key), console.sessions.keyed(lock_name(key)):
        if key is not None:
            found = await _store_call(console, console.store.get_submission(None, key))
            try:
                check_submission(found, composed)
            except SubmissionRepeated:
                raise conflict() from None
            except SubmissionError as exc:
                raise _submission_error(exc) from exc
            made = await created(console, key, composed)
            if made is not None:
                record, info = made
                echo = await started_echo(console, record, info, text)
                return JSONResponse(
                    content={
                        "session": console.session_body(info),
                        "echo": echo,
                        "repeated": True,
                    }
                )
        entry, _ = await _sessions_call(
            console,
            console.sessions.create(body.workspace, None, mode=body.approvals),
        )
        chat = entry.session_id
        try:
            echo = await _store_call(
                console,
                console.front_for(chat).route(
                    text, composed, as_task=True, key=None, create_key=key
                ),
                chat,
            )
            if echo is None or echo.get("echo") != "started":
                # The task did not start: its echo says why, and the chat has nothing.
                raise ApiError(
                    400, "invalid_request", str((echo or {}).get("line") or "")
                )
            await _store_call(
                console,
                console.store.rename_session(chat, short_title(composed)),
                chat,
            )
        except BaseException:
            if not await _admitted(console, chat, key):
                with contextlib.suppress(Exception):
                    await console.sessions.delete(chat)
            raise
        info = await _require_session(console, chat)
        return JSONResponse(
            status_code=201,
            content={"session": console.session_body(info), "echo": echo},
        )


async def _refuse_task_key(console: Console, key: str | None) -> None:
    """A create of a message under a key a chat was made for a task under: 409 ``submission_conflict``."""
    if key is not None and await console.workers.tasks.by_create_key(key) is not None:
        raise conflict()


async def _admitted(console: Console, chat: str, key: str | None) -> bool:
    """Whether the create already made its task: then its chat stays, and a retry finds both."""
    tasks = console.workers.tasks
    try:
        if key is not None:
            return await tasks.by_create_key(key) is not None
        return any(record.front_id == chat for record in await tasks.list_tasks())
    except Exception:
        return True  # unknown: a chat kept is better than a task with no owner


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
        if body.turn is not None and body.turn.as_task:
            return await _create_for_task(console, body, composed, key)
        try:
            entry, turn = await _sessions_call(
                console,
                console.sessions.create(
                    body.workspace,
                    first,
                    client_request_id=key,
                    mode=body.approvals,
                    admit=lambda: _refuse_task_key(console, key),
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

    @app.post("/api/sessions/{session_id}/undo")
    async def undo_turn(session_id: str, body: RetryRequest) -> JSONResponse:
        console.require_store()
        await _require_session(console, session_id)
        undone = await _sessions_call(
            console, console.sessions.undo(session_id, body.turn_id), session_id
        )
        return JSONResponse(content={"turn_id": undone.id, "undone": True})

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
        # A key that made a chat for a task is stopped with the task, and a cancel
        # waits for a create of the same key that is under way.
        async with console.sessions.keyed(lock_name(key) if scope is None else None):
            made = await created(console, key, composed) if scope is None else None
            if made is not None:
                return await _stop_created(console, made, key)
            return await _withdraw(console, scope, key, composed)

    async def _stop_created(
        console: Console, made: tuple[Any, Any], key: str
    ) -> JSONResponse:
        record, info = made
        try:
            await console.workers.update(record.task_id, state="stop")
            stopped = True
        except (TaskStoreError, TaskPolicyError, ValueError):
            stopped = False  # it ended already
        return JSONResponse(
            status_code=202 if stopped else 200,
            content={
                "submission": submission_of(record, info, key),
                "cancelling": stopped,
            },
        )

    async def _withdraw(
        console: Console, scope: str | None, key: str, composed: str
    ) -> JSONResponse:
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
        asked = console.broker.request(request_id)
        outside = None if asked is None else asked.outside
        if body.choice == "once":
            decision = ApprovalDecision(True)
        elif body.choice in ("session", "always") and outside is not None:
            # A folder outside the workspace (#405): a write, or a folder that
            # may not be granted, is allowed once and granted nothing.
            grant = outside.grantable
            always = grant and body.choice == "always"
            decision = ApprovalDecision(True, remember=grant, always=always)
        elif body.choice == "always":
            raise ApiError(400, "invalid_request", t("errors.alwaysNotFolder"))
        elif body.choice == "session":
            # A high-impact call is never granted for the chat: this answer
            # allows the one call, and the reply says so.
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
                "always": decision.always,
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
        # One transition: its unfinished tasks, chores and waiting cards are the
        # main chat's when it is gone.
        await _sessions_call(
            console,
            console.delete_chat(session_id, forget_notes=notes == "forget"),
            session_id,
        )
        return Response(status_code=204)


__all__ = ["add_routes"]
