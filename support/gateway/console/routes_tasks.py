"""API v2 task creation, inspection and control."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, TypeVar

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import StrictFloat, StrictInt, StrictStr

from localmelo.support.gateway.auto_blocks import BlockError
from localmelo.support.gateway.resume_evidence import (
    ConfirmationRequiredError,
    RecoveryUnknownError,
    ResumeRefusedError,
    StaleConfirmationError,
)
from localmelo.support.gateway.resume_tokens import TOKEN_TTL_S
from localmelo.support.gateway.task_policy import TaskPolicyError
from localmelo.support.gateway.task_store import (
    TaskConflictError,
    TaskNotFoundError,
    TaskNotPausedError,
    TaskStoreError,
)
from localmelo.support.sessions.store import SessionStoreError

from .activity import left_line
from .bodies import Body
from .errors import ApiError
from .shapes import approval_json

if TYPE_CHECKING:
    from . import Console

T = TypeVar("T")


class CreateTask(Body):
    objective: StrictStr | None = None
    prompt: StrictStr | None = None
    title: StrictStr | None = None
    workspace: StrictStr | None = None
    level: StrictInt | StrictStr | None = None
    deadline: StrictStr | StrictFloat | StrictInt | None = None
    request_key: StrictStr | None = None
    # The chat that starts the task; the main chat when absent.
    front_id: StrictStr | None = None


class UpdateTask(Body):
    level: StrictInt | StrictStr | None = None
    state: StrictStr | None = None


class ResumeTask(Body):
    operation_id: StrictStr
    confirmation_token: StrictStr | None = None


async def _call(console: Console, operation: Awaitable[T]) -> T:
    try:
        return await operation
    except TaskPolicyError as exc:
        raise ApiError(400, exc.code, str(exc)) from exc
    except TaskConflictError as exc:
        raise ApiError(409, "submission_conflict", str(exc)) from exc
    except TaskNotFoundError as exc:
        raise ApiError(404, "not_found", str(exc)) from exc
    except TaskNotPausedError as exc:
        raise ApiError(409, "not_paused", str(exc)) from exc
    except TaskStoreError as exc:
        raise ApiError(409, "invalid_transition", str(exc)) from exc
    except SessionStoreError as exc:
        raise await console.store_api_error(exc) from exc
    except ValueError as exc:
        raise ApiError(400, "invalid_request", str(exc)) from exc


async def _resume_call(console: Console, operation: Awaitable[T]) -> T:
    try:
        return await operation
    except ConfirmationRequiredError as exc:
        raise ApiError(
            409,
            "confirmation_required",
            exc.warning,
            details={
                "reason": "unverified_processes",
                "confirmation_token": exc.token,
                "expires_in_s": int(TOKEN_TTL_S),
                "warning": exc.warning,
            },
        ) from exc
    except StaleConfirmationError as exc:
        raise ApiError(409, "stale_confirmation", str(exc)) from exc
    except RecoveryUnknownError as exc:
        raise ApiError(
            409, "recovery_unknown", str(exc), details={"reason": exc.reason}
        ) from exc
    except ResumeRefusedError as exc:
        raise ApiError(
            409, "invalid_transition", str(exc), details={"reason": exc.reason}
        ) from exc
    except TaskConflictError as exc:
        raise ApiError(409, "invalid_transition", str(exc)) from exc
    except TaskNotFoundError as exc:
        raise ApiError(404, "not_found", str(exc)) from exc
    except TaskStoreError as exc:
        raise ApiError(409, "invalid_transition", str(exc)) from exc
    except SessionStoreError as exc:
        raise await console.store_api_error(exc) from exc
    except ValueError as exc:
        raise ApiError(400, "invalid_request", str(exc)) from exc


def _approvals(console: Console, task: dict[str, Any]) -> dict[str, Any]:
    """*task* with the approvals it waits for; the broker keys them by task id."""
    pending = console.broker.pending(str(task.get("task_id")))
    return {**task, "pending_approvals": [approval_json(r) for r in pending]}


def add_routes(app: FastAPI, console: Console) -> None:
    """Install task routes before the catch-all /api handler."""

    @app.post("/api/tasks")
    async def create(body: CreateTask) -> JSONResponse:
        console.require_store()
        task, repeated = await _call(
            console, console.workers.create(body.model_dump(exclude_unset=True))
        )
        content: dict[str, Any] = {"task": _approvals(console, task)}
        if repeated:
            content["repeated"] = True
        return JSONResponse(content=content, status_code=200 if repeated else 202)

    @app.get("/api/tasks")
    async def list_tasks(
        state: str | None = None, level: int | None = None
    ) -> JSONResponse:
        console.require_store()
        tasks = await _call(console, console.workers.list(state=state, level=level))
        return JSONResponse(content={"tasks": [_approvals(console, t) for t in tasks]})

    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: str) -> JSONResponse:
        console.require_store()
        task = await _call(console, console.workers.get(task_id))
        if task.get("out_of_steps"):
            # What the run says is left, for the page's "What is left" row.
            turns = await console.store.list_turns(task_id, limit=1)
            task["left"] = left_line(turns[-1].answer) if turns else ""
        return JSONResponse(content={"task": _approvals(console, task)})

    @app.patch("/api/tasks/{task_id}")
    async def update(task_id: str, body: UpdateTask) -> JSONResponse:
        console.require_store()
        if body.level is None and body.state is None:
            raise ApiError(400, "invalid_request", "Choose a level or state.")
        task = await _call(
            console, console.workers.update(task_id, level=body.level, state=body.state)
        )
        return JSONResponse(content={"task": _approvals(console, task)})

    @app.post("/api/tasks/{task_id}/continue")
    async def continue_task(task_id: str) -> JSONResponse:
        """Continue a task that ran out of steps, from where it stopped (#341)."""
        console.require_store()
        task, outcome = await _call(console, console.workers.continue_stopped(task_id))
        return JSONResponse(
            status_code=202,
            content={"task": _approvals(console, task), "message": outcome},
        )

    @app.post("/api/tasks/{task_id}/blocks/{request_id}/allow")
    async def allow_block(task_id: str, request_id: str) -> JSONResponse:
        """Allow once a step auto mode blocked in a task, and ask it to go ahead (#339)."""
        console.require_store()
        try:
            task, outcome = await _call(
                console, console.workers.allow_once(task_id, request_id)
            )
        except BlockError as exc:
            raise ApiError(exc.status, exc.code, exc.message) from exc
        return JSONResponse(
            status_code=202,
            content={"task": _approvals(console, task), "message": outcome},
        )

    @app.post("/api/tasks/{task_id}/resume")
    @app.post("/api/v2/tasks/{task_id}/resume", include_in_schema=False)
    async def resume(task_id: str, body: ResumeTask) -> JSONResponse:
        console.require_store()
        task = await _resume_call(
            console,
            console.workers.resume(
                task_id,
                operation_id=body.operation_id,
                confirmation_token=body.confirmation_token,
            ),
        )
        return JSONResponse(
            content={
                "task": _approvals(console, task),
                "operation_id": body.operation_id,
            },
            status_code=202,
        )
