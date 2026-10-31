"""API v2 chores: ``GET/POST /api/chores``, ``PATCH/DELETE /api/chores/{id}``, run now and the check-in switch.

A chore is a saved request with a schedule (:mod:`.chores`). Every chore
carries its ``revision``; ``PATCH`` must name the revision it read and answers
409 ``stale_revision`` when the chore has changed since. ``DELETE`` checks a
``?revision=`` when one is given. ``GET`` lists every chore, or, with the id
of a side chat, that chat's own.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import StrictBool, StrictInt, StrictStr

from localmelo.support.gateway.chore_schedule import Schedule, ScheduleError
from localmelo.support.gateway.chore_store import (
    ChoreLimitError,
    ChoreNotFoundError,
    ChoreStaleError,
)
from localmelo.support.gateway.front_rules import level_number, triage
from localmelo.support.gateway.task_policy import TaskPolicyError, TaskSpec
from localmelo.support.gateway.task_store import TaskStoreError
from localmelo.support.sessions.store import SessionStoreError

from .bodies import Body
from .errors import ApiError

if TYPE_CHECKING:
    from . import Console

T = TypeVar("T")


class CreateChore(Body):
    request: StrictStr
    schedule: dict[str, Any]
    title: StrictStr | None = None
    level: StrictInt | StrictStr | None = None
    # The chat the chore belongs to; the main chat when absent.
    front_id: StrictStr | None = None


class UpdateChore(Body):
    revision: StrictInt
    title: StrictStr | None = None
    request: StrictStr | None = None
    schedule: dict[str, Any] | None = None
    level: StrictInt | StrictStr | None = None
    enabled: StrictBool | None = None


class CheckinSwitch(Body):
    on: StrictBool


async def _call(console: Console, operation: Awaitable[T]) -> T:
    try:
        return await operation
    except ChoreNotFoundError as exc:
        raise ApiError(404, "not_found", str(exc)) from exc
    except ChoreStaleError as exc:
        raise ApiError(409, "stale_revision", str(exc)) from exc
    except ChoreLimitError as exc:
        raise ApiError(409, "chore_limit", str(exc)) from exc
    except (ScheduleError, TaskPolicyError, ValueError) as exc:
        raise ApiError(400, "invalid_request", str(exc)) from exc
    except TaskStoreError as exc:
        raise ApiError(409, "invalid_transition", str(exc)) from exc
    except SessionStoreError as exc:
        raise await console.store_api_error(exc) from exc


def _check(function: Callable[[Any], T], value: Any) -> T:
    """*function*(*value*), a refusal as 400."""
    try:
        return function(value)
    except (ScheduleError, TaskPolicyError, ValueError) as exc:
        raise ApiError(400, "invalid_request", str(exc)) from exc


def _level(value: object) -> int | None:
    """A level code, word or number the chore may have (P1 to P4), or ``None`` when absent."""
    if value is None:
        return None
    number = level_number(value)
    if number is None or not 1 <= number <= 4:
        raise ApiError(400, "invalid_request", "level: P1 to P4")
    return number


async def _owner(console: Console, front_id: str | None) -> str | None:
    """*front_id* when it names a console chat, the main chat when it is absent."""
    if front_id is None or front_id == console.main_id:
        return console.main_id
    info = await console.store.get_session(front_id)
    if info is None or info.kind not in ("chat", "front"):
        raise ApiError(400, "invalid_request", "front_id must name a console chat")
    return front_id


def add_routes(app: FastAPI, console: Console) -> None:
    """Install the chore routes before the catch-all /api handler."""
    runner = console.chore_runner

    @app.get("/api/chores")
    async def list_chores(session_id: str | None = None) -> JSONResponse:
        console.require_store()
        own = None if session_id in (None, "", console.main_id) else session_id
        if own is not None:
            info = await console.store.get_session(own)
            if info is None or info.kind not in ("chat", "front"):
                raise ApiError(404, "not_found", "Not found.")
        chores = await _call(console, runner.store.list_chores(own))
        return JSONResponse(
            content={
                "chores": [await runner.chore_body(c) for c in chores],
                "checkin": await runner.checkin_body(),
                "scope": "all" if own is None else "chat",
            }
        )

    @app.post("/api/chores")
    async def create(body: CreateChore) -> JSONResponse:
        console.require_store()
        request = body.request.strip()
        spec = _check(
            TaskSpec.from_payload, {"objective": request, "title": body.title}
        )
        schedule = _check(Schedule.from_json, body.schedule)
        level = _level(body.level)
        chore = await _call(
            console,
            runner.store.create(
                title=spec.title,
                request=spec.objective,
                schedule=schedule,
                level=triage(request)[0] if level is None else level,
                front_id=await _owner(console, body.front_id),
            ),
        )
        return JSONResponse(
            status_code=201, content={"chore": await runner.chore_body(chore)}
        )

    @app.patch("/api/chores/{chore_id}")
    async def update(chore_id: str, body: UpdateChore) -> JSONResponse:
        console.require_store()
        changes = body.model_dump(exclude_unset=True, exclude={"revision"})
        if not changes:
            raise ApiError(400, "invalid_request", "Choose what to change.")
        schedule = (
            None if body.schedule is None else _check(Schedule.from_json, body.schedule)
        )
        chore = await _call(
            console,
            runner.store.update(
                chore_id,
                revision=body.revision,
                title=body.title,
                request=body.request,
                schedule=schedule,
                level=_level(body.level),
                enabled=body.enabled,
            ),
        )
        return JSONResponse(content={"chore": await runner.chore_body(chore)})

    @app.delete("/api/chores/{chore_id}")
    async def delete(chore_id: str, revision: int | None = None) -> JSONResponse:
        console.require_store()
        await _call(console, runner.store.delete(chore_id, revision=revision))
        return JSONResponse(content={"deleted": chore_id})

    @app.post("/api/chores/{chore_id}/run")
    async def run_now(chore_id: str) -> JSONResponse:
        console.require_store()
        chore = await _call(console, runner.store.get(chore_id))
        task = await _call(console, runner.run_now(chore))
        fresh = await _call(console, runner.store.get(chore_id))
        return JSONResponse(
            status_code=202,
            content={"chore": await runner.chore_body(fresh), "task": task},
        )

    @app.put("/api/checkin")
    async def switch(body: CheckinSwitch) -> JSONResponse:
        console.require_store()
        await _call(console, runner.store.set_checkin(body.on))
        return JSONResponse(content={"checkin": await runner.checkin_body()})


__all__: list[str] = []
