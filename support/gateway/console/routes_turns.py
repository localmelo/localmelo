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
from localmelo.melo.checker.validators import MAX_QUERY_LEN
from localmelo.melo.contracts.approvals import ApprovalDecision, grant_offered
from localmelo.support.gateway.auto_blocks import (
    BlockError,
    block_to_allow,
    go_ahead_text,
)
from localmelo.support.gateway.i18n import t
from localmelo.support.gateway.task_policy import TaskPolicyError
from localmelo.support.gateway.task_store import TaskStoreError
from localmelo.support.sessions.attachments import (
    MAX_ATTACHMENTS,
    MAX_TOTAL_BYTES,
    TextAttachment,
    compose_user_text,
    snapshot,
)
from localmelo.support.sessions.display import short_title
from localmelo.support.sessions.picture_input import MAX_PICTURES
from localmelo.support.sessions.store import (
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

from .asks import MAX_NOTE_CHARS
from .bodies import Body
from .errors import ApiError
from .registry import STOP_REASON, TurnExtras
from .shapes import submission_json, turn_json
from .task_submissions import (
    conflict,
    created,
    lock_name,
    started_echo,
    submission_of,
)
from .uploads import Staged, images_of

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
    # Text files the person added, read by the page (at most 8).
    attachments: list[_Attachment] = []
    # Pictures the person added: ids from ``POST /api/uploads`` (at most 4).
    images: list[str] = []
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


class NoteRequest(Body):
    """Words the person adds to a turn that runs, and the pictures that go with them."""

    text: str
    # Pictures the person added: ids from ``POST /api/uploads`` (at most 4).
    images: list[str] = []
    client_request_id: str | None = None


class PatchRequest(Body):
    title: str | None = None
    workspace: str | None = None
    # The chat's mode (#339): from its next step.
    approvals: Literal["ask", "auto"] | None = None


class CancelRequest(Body):
    """A message to withdraw: what its send carried, so the same text is judged."""

    text: str
    attachments: list[_Attachment] = []
    # The staged pictures the send named; only how many is read (the stand-in words of a message with no words).
    images: list[str] = []


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


def _files(items: list[_Attachment]) -> list[TextAttachment]:
    """The text files of a send, cleaned and checked; 400 for one the checker would refuse."""
    attached: list[TextAttachment] = []
    total = 0
    if len(items) > MAX_ATTACHMENTS:
        raise ApiError(
            400, "invalid_request", t("errors.tooManyFiles", n=MAX_ATTACHMENTS)
        )
    for item in items:
        try:
            file = snapshot(item.name, item.text)
        except ValueError as exc:
            raise ApiError(
                400,
                "invalid_request",
                t("errors.fileRefused", name=item.name[:80], why=str(exc)),
            ) from exc
        total += len(file.text.encode("utf-8"))
        attached.append(file)
    if total > MAX_TOTAL_BYTES:
        raise ApiError(
            400,
            "invalid_request",
            t("errors.filesTooLarge", kb=MAX_TOTAL_BYTES // 1024),
        )
    return attached


def _compose(request: CancelRequest) -> str:
    """The user text a withdrawn send had: its words (or the stand-in for none) and its files.

    The store compares this with what the send kept, so it is built as the
    send builds it.
    """
    attached = _files(request.attachments)
    if len(request.images) > MAX_PICTURES:
        raise ApiError(
            400, "invalid_request", t("errors.tooManyPictures", n=MAX_PICTURES)
        )
    words = request.text.strip()
    if not words:
        if not request.images and not attached:
            raise ApiError(400, "invalid_request", t("errors.blankMessage"))
        words = _marker(len(request.images), [file.name for file in attached])
    return compose_user_text(words, attached)


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


async def _staged(console: Console, ids: list[str]) -> list[Staged]:
    """The staged pictures a message or a note names, checked.

    400 for more than the limit, one named twice, or one gone from the stage;
    409 ``model_text_only`` when the chat model is known not to read pictures.
    """
    if len(ids) > MAX_PICTURES:
        raise ApiError(
            400, "invalid_request", t("errors.tooManyPictures", n=MAX_PICTURES)
        )
    if len(set(ids)) != len(ids):
        raise ApiError(400, "invalid_request", t("errors.pictureTwice"))
    staged = console.pictures.resolve(ids)
    if staged and await console.vision.reads() is False:
        raise ApiError(409, "model_text_only", t("errors.modelTextOnly"))
    return staged


class _Sent:
    """A checked send: its words, the whole text a code rule reads, and what it carries."""

    def __init__(self, words: str, whole: str, extras: TurnExtras) -> None:
        self.words = words
        self.whole = whole
        self.extras = extras


def _marker(pictures: int, files: list[str]) -> str:
    """The words of a message sent with a picture or a file and no words."""
    parts = []
    if pictures:
        parts.append(f"[{pictures} picture{'' if pictures == 1 else 's'}]")
    if files:
        parts.append(f"[attached: {', '.join(files)}]")
    return " ".join(parts)


async def _sent(
    console: Console, request: TurnRequest, session_id: str | None
) -> _Sent:
    """The checked send for a turn request: its text, files and pictures.

    400 for a file that is binary, too large or too many, a picture that is
    gone from the stage or too many, and a text over the checker's limit; 409
    ``model_text_only`` when the chat model is known not to read pictures.
    """
    attached = _files(request.attachments)
    staged = await _staged(console, request.images)
    words = request.text.strip()
    if not words:
        if not staged and not attached:
            raise ApiError(400, "invalid_request", t("errors.blankMessage"))
        words = _marker(len(staged), [file.name for file in attached])
    words = _check_text(console, words, session_id)
    whole = compose_user_text(words, attached)
    if len(whole) > MAX_QUERY_LEN:
        raise ApiError(
            400, "invalid_request", t("errors.filesTooLong", n=MAX_QUERY_LEN)
        )
    return _Sent(words, whole, TurnExtras(tuple(attached), tuple(images_of(staged))))


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
            found = await console.store.get_submission(None, key)
            try:
                check_submission(found, composed)
            except SubmissionRepeated:
                raise conflict() from None
            except SubmissionError as exc:
                raise _submission_error(exc) from exc
            made = await created(console, key, composed)
            if made is not None:
                record, info = made
                return JSONResponse(
                    content={
                        "session": console.session_body(info),
                        "echo": await started_echo(console, record, info, text),
                        "repeated": True,
                    }
                )
        entry, _ = await console.sessions.create(
            body.workspace, None, mode=body.approvals
        )
        chat = entry.session_id
        try:
            echo = await console.front_for(chat).route(
                text, composed, as_task=True, key=None, create_key=key
            )
            if echo is None or echo.get("echo") != "started":
                # The task did not start: its echo says why, and the chat has nothing.
                raise ApiError(
                    400, "invalid_request", str((echo or {}).get("line") or "")
                )
            await console.store.rename_session(chat, short_title(composed))
        except BaseException:
            if not await _admitted(console, chat, key):
                with contextlib.suppress(Exception):
                    await console.sessions.delete(chat)
            raise
        info = await console.require_session(chat)
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
        sent: _Sent | None = None
        if body.turn is not None:
            sent = await _sent(console, body.turn, None)
            first = (body.turn.text, sent.words)
        # The key is judged after the text, as 2.4 orders the checks.
        key = None if body.turn is None else _key(body.turn.client_request_id)
        if body.turn is not None and body.turn.as_task:
            assert sent is not None
            if sent.extras.images:
                raise ApiError(400, "invalid_request", t("errors.picturesNoTask"))
            return await _create_for_task(console, body, sent.whole, key)
        try:
            entry, turn = await console.sessions.create(
                body.workspace,
                first,
                client_request_id=key,
                mode=body.approvals,
                admit=lambda: _refuse_task_key(console, key),
                extras=None if sent is None else sent.extras,
            )
        except SubmissionRepeated as exc:
            info = await console.require_session(exc.turn.session_id)
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
        info = await console.require_session(entry.session_id)
        content: dict[str, Any] = {"session": console.session_body(info)}
        if turn is not None:
            content.update(_turn_body(turn, None))
        return JSONResponse(status_code=201, content=content)

    @app.post("/api/sessions/{session_id}/turns")
    async def start_turn(session_id: str, body: TurnRequest) -> JSONResponse:
        console.require_store()
        # The session is judged first: a malformed id is 404, as it is on
        # every read of the same path, not the checker's 400.
        info = await console.require_session(session_id)
        sent = await _sent(console, body, session_id)
        # The key is judged after the text, as 2.4 orders the checks.
        key = _key(body.client_request_id)
        if info.kind in ("front", "chat"):
            pictured = bool(sent.extras.images)
            if pictured and body.as_task:
                raise ApiError(400, "invalid_request", t("errors.picturesNoTask"))
            # Code's rules first: a task to start or to change takes no model
            # call. A side chat has the same rules as the main chat, and its
            # tasks are its own. A message with a picture is a message to the
            # model: no rule reads it, so none can swallow the picture.
            echo = (
                None
                if pictured
                else await console.front_for(session_id).route(
                    body.text, sent.whole, as_task=body.as_task, key=key
                )
            )
            if echo is not None:
                return JSONResponse(status_code=200, content={"echo": echo})
            if console.asks.holds(session_id):
                if pictured:
                    raise ApiError(
                        409, "chat_answering", t("errors.picturesWhileAnswering")
                    )
                # The chat is answering a task's question: the message waits,
                # shown as queued, and runs as the chat's next turn (#369).
                echo = await console.asks.queue(session_id, body.text, sent.whole, key)
                return JSONResponse(status_code=200, content={"echo": echo})
        elif body.as_task:
            raise ApiError(400, "invalid_request", "as_task: tasks start from a chat")
        try:
            turn, resumed = await console.sessions.start_turn(
                session_id,
                body.text,
                sent.words,
                client_request_id=key,
                extras=sent.extras,
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
        await console.require_session(session_id)
        # A turn that carried pictures is not sent again to a model known not to
        # read them (the turn that was refused for them, for one): nothing is undone.
        newest = await console.store.list_turns(session_id, limit=1)
        if (
            newest
            and newest[-1].id == body.turn_id
            and await console.store.turn_images(session_id, newest[-1].turn)
            and await console.vision.reads() is False
        ):
            raise ApiError(409, "model_text_only", t("errors.modelTextOnly"))
        turn, resumed = await console.sessions.retry(session_id, body.turn_id)
        return JSONResponse(status_code=202, content=_turn_body(turn, resumed))

    @app.post("/api/sessions/{session_id}/undo")
    async def undo_turn(session_id: str, body: RetryRequest) -> JSONResponse:
        console.require_store()
        await console.require_session(session_id)
        undone = await console.sessions.undo(session_id, body.turn_id)
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
        # Words added to the turn that its run never read do not start another
        # turn after a Stop: they go back to the composer.
        unread = await console.asks.drop_notes(entry.session_id, turn_id)
        content: dict[str, Any] = {"turn_id": turn_id, "cancelling": True}
        if unread:
            content["unread_notes"] = unread
        return JSONResponse(status_code=202, content=content)

    @app.post("/api/turns/{turn_id}/notes")
    async def add_note(turn_id: str, body: NoteRequest) -> JSONResponse:
        """Add words, and pictures, to a turn that runs: its run reads them at its next step (202).

        If the turn ends before another step, they run as the chat's next turn.
        A note may carry pictures on the terms of a message: the same kinds and
        limits, and no words are needed with a picture. 409 ``turn_not_running``
        when the turn is not running, or is being stopped; a client then sends
        the words as a message.
        """
        console.require_store()
        key = _key(body.client_request_id)
        repeated = console.asks.noted(turn_id, key)
        if repeated is not None:
            return JSONResponse(status_code=202, content={"note": repeated})
        entry = console.sessions.find_turn(turn_id)
        if entry is None:
            raise ApiError(409, "turn_not_running", t("errors.turnNotRunning"))
        info = await console.require_session(entry.session_id)
        current = entry.runner.current_turn
        if info.kind not in ("front", "chat") or (
            current is not None and current.origin
        ):
            raise ApiError(409, "not_your_turn", t("errors.noteInternalTurn"))
        staged = await _staged(console, body.images)
        text = body.text.strip()
        if not text and not staged:
            raise ApiError(400, "invalid_request", t("errors.blankMessage"))
        if len(text) > MAX_NOTE_CHARS:
            raise ApiError(
                400, "invalid_request", t("errors.noteTooLong", n=MAX_NOTE_CHARS)
            )
        # The words a picture stands in for when there are none, as a message has.
        words = text or _marker(len(staged), [])
        composed = _check_text(console, words, entry.session_id)
        note = await console.asks.add_note(
            entry.session_id, turn_id, text, composed, key, staged
        )
        return JSONResponse(status_code=202, content={"note": note})

    async def _cancel_by_key(
        scope: str | None, client_request_id: str, body: CancelRequest
    ) -> JSONResponse:
        console.require_store()
        key = _key(client_request_id)
        assert key is not None
        composed = _compose(body)
        if scope is not None:
            await console.require_session(scope)
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
            submission = await console.store.withdraw_submission(
                scope, key, user_text=composed
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
            # allows the one call, and the reply says so. A site Auto stopped
            # only as not asked for may be (#465).
            grant = asked is None or grant_offered(asked)
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
        info = await console.require_session(session_id)
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
            await console.sessions.change_workspace(session_id, body.workspace)
        if body.approvals is not None:
            await console.sessions.set_mode(session_id, body.approvals)
        if title is not None:
            # 2.10 lists PATCH among the lock holders; the workspace change
            # takes the lock itself, so a title alone takes it here.
            locked = (
                console.sessions.lock(session_id)
                if body.workspace is None
                else contextlib.nullcontext()
            )
            async with locked:
                renamed = await console.store.rename_session(session_id, title)
            if not renamed:
                raise ApiError(404, "not_found", t("errors.sessionNotFound"))
        info = await console.require_session(session_id)
        return JSONResponse(content={"session": console.session_body(info)})

    @app.post("/api/sessions/{session_id}/blocks/{request_id}/allow")
    async def allow_block(session_id: str, request_id: str) -> JSONResponse:
        """Allow once a step auto mode blocked, and ask melo to go ahead (#339)."""
        console.require_store()
        info = await console.require_session(session_id)
        if info.kind == "task":
            raise ApiError(409, "task_session_read_only", t("errors.taskChange"))
        if info.kind not in ("chat", "front"):
            raise ApiError(409, "not_a_chat", t("errors.modeNotChat"))
        try:
            key, categories, tool = await block_to_allow(
                console.store,
                session_id,
                request_id,
                lambda key: console.sessions.allowance_live(session_id, key),
            )
            text = _check_text(console, go_ahead_text(categories), session_id)
            turn, until = await console.sessions.allow_once(
                session_id, key, categories, text, tool
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
        await console.delete_chat(session_id, forget_notes=notes == "forget")
        return Response(status_code=204)


__all__ = ["add_routes"]
