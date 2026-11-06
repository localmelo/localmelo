"""The first-run setup from the welcome card: the plan, one job, and its progress.

The terminal's first run asks melo_serve what this Mac can run, shows a list,
and does it after one yes (:mod:`localmelo.support.server_setup`). The card does
the same through this module, with the same job and the same words
(:mod:`~localmelo.support.server_setup.listing`).

``GET /api/setup/plan`` is the list. Asking melo_serve takes seconds, so one
asker serves every caller at once and an answer is kept for :data:`PLAN_TTL_S`
(a plan that could not run, for :data:`FAILED_TTL_S`); it is not part of
``GET /api/setup/detect``, which the card asks every few seconds. ``?skip=a,b``
words the list for what the person left out, from the last plan served. The
body says ``available: false`` and a ``reason`` when there is nothing to offer.

``POST /api/setup/start`` (202; 409 while a job runs) starts the job in a
thread of its own, and ``POST /api/setup/cancel`` stops it, keeping what was
downloaded. Both check the request against the last plan served, the list the
person said yes to: ``pick`` is that plan's pick or one of its alternatives and
``skip`` holds ids of its items or of the rows the job adds. Commands run as an
argv, never through a shell, and the job registers no login item for the
gateway (installing it would restart the gateway that runs the job). A gateway
that shuts down ends the job ``cancelled``, with the partial files kept.

The job rides on ``GET /api/setup/detect`` as ``job`` (null before the first),
so the card keeps one loop of asking. The routes are under ``/api``: the
gateway's authentication and origin guard hold for them as for the settings.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Annotated, Any, Final, TypeVar

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr, StringConstraints

from localmelo.support.gateway.i18n import t
from localmelo.support.server_setup import listing, run

from .errors import ApiError

if TYPE_CHECKING:
    from . import Console

logger = logging.getLogger(__name__)

PLAN_TTL_S: Final = 60.0
FAILED_TTL_S: Final = 5.0
# How long a gateway that shuts down waits for a job it has stopped.
STOP_WAIT_S: Final = 10.0
# What a person can still leave out besides the plan's items.
JOB_ROWS: Final = (run.MELO_SERVE, run.SERVICE_MELO_SERVE, run.SERVICE_GATEWAY)

T = TypeVar("T")


@dataclass(frozen=True)
class Planned:
    """What :func:`run.plan` found, the version of melo_serve already here, and when."""

    plan: run.Plan
    there: str
    at: float


class Job:
    """One setup: the plan it was started on, the job, and how far it got."""

    def __init__(
        self,
        id: int,
        planned: Planned,
        pick: run.Pick,
        skip: list[str],
        on_end: Callable[[], None],
    ) -> None:
        self.id = id
        self.planned = planned
        self.pick = pick
        self.skip = frozenset(skip)
        self.finished: run.Finished | None = None
        self.stopping = False
        self._on_end = on_end
        # The gateway's own login item is not registered: that restarts the gateway.
        self.job = run.SetupJob(
            planned.plan,
            pick.id,
            skip,
            gateway_service=False,
            on_event=self._event,
        )
        self.thread = threading.Thread(
            target=self._work, name="melo-setup", daemon=True
        )

    def _event(self, event: run.Event) -> None:
        if isinstance(event, run.Finished):
            self.finished = event

    def _work(self) -> None:
        try:
            self.job.run()
        except Exception as exc:  # run() ends in a Finished; this is for a bug
            logger.exception("The setup job stopped on an error")
            self.finished = self.finished or run.Finished("failed", f"failed: {exc}")
        finally:
            self._on_end()

    def stop(self) -> None:
        """Ask the job to end ``cancelled``, keeping what it downloaded."""
        self.stopping = True
        self.job.cancel()


class StartBody(BaseModel):
    """Request body of ``POST /api/setup/start``: the pick and the ids left out."""

    model_config = ConfigDict(extra="forbid")

    pick: StrictStr = Field(min_length=1, max_length=200)
    skip: list[Annotated[StrictStr, StringConstraints(max_length=100)]] = Field(
        default_factory=list, max_length=64
    )


def _off_loop(work: Callable[[], T]) -> asyncio.Future[T]:
    """Run *work* in a daemon thread: a gateway that shuts down does not wait for it.

    ``asyncio.to_thread`` would hold the loop's exit for as long as the plan's
    command runs (its timeout is two minutes).
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()

    def settle(setter: Callable[[Any], None], value: Any) -> None:
        if not future.done():
            setter(value)

    def target() -> None:
        try:
            value = work()
        except BaseException as exc:
            with contextlib.suppress(RuntimeError):  # the loop is gone
                loop.call_soon_threadsafe(settle, future.set_exception, exc)
        else:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(settle, future.set_result, value)

    threading.Thread(target=target, name="melo-setup-plan", daemon=True).start()
    return future


def _ask() -> tuple[run.Plan, str]:
    plan = run.plan()
    return plan, run.installed_version() if plan.available else ""


class SetupFlow:
    """The plan this gateway served last, and the one job it runs at a time."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self.clock: Callable[[], float] = time.monotonic
        self._cached: Planned | None = None
        self._served: Planned | None = None
        self._asking: asyncio.Future[Planned] | None = None
        self.job: Job | None = None
        self._count = 0

    # ── The plan ──

    async def _planned(self, *, any_age: bool = False) -> Planned:
        """The plan: one asker for every caller, kept for :data:`PLAN_TTL_S`.

        With *any_age*, the last plan served, however old (what a start is
        checked against), unless there is none.
        """
        if any_age and self._served is not None:
            return self._served
        found = self._cached
        if found is not None:
            ttl = PLAN_TTL_S if found.plan.available else FAILED_TTL_S
            if self.clock() - found.at < ttl:
                return found
        if self._asking is None:
            self._asking = asyncio.ensure_future(self._ask())
            self._asking.add_done_callback(self._asked)
        return await asyncio.shield(self._asking)

    async def _ask(self) -> Planned:
        plan, there = await _off_loop(_ask)
        found = Planned(plan, there, self.clock())
        self._cached = found
        if plan.available and plan.pick is not None:
            self._served = found
        return found

    def _asked(self, asking: asyncio.Future[Planned]) -> None:
        self._asking = None
        if not asking.cancelled():
            asking.exception()  # read, so that no one has to

    async def plan_json(self, skip: list[str]) -> dict[str, Any]:
        """``GET /api/setup/plan``: the list in the request's language, or why there is none."""
        if sys.platform != "darwin":
            return _unavailable(t("setup.errors.notMac"), "")
        planned = await self._planned(any_age=bool(skip))
        plan = planned.plan
        pick = plan.pick
        if not plan.available or pick is None:
            return _unavailable(plan.reason, plan.hint)
        self._check_skip(plan, skip)
        found = listing.build(plan, skip, planned.there, t, terminal=False)
        resume = any(row.state == "partial" for row in found.rows)
        return {
            "available": True,
            "reason": "",
            "hint": "",
            "pick": {
                "id": pick.id,
                "name": pick.name,
                "backend": pick.backend,
                "backend_name": listing.backend_name(plan, pick.backend),
            },
            "mac": found.this_mac,
            "intro": found.recommended,
            "rows": [asdict(row) for row in found.rows],
            "download": found.download,
            "free": found.free,
            "notes": found.notes,
            "action": listing.setup_row(plan, skip, t, resume=resume),
            "can_set_up": listing.can_set_up(plan, skip),
        }

    @staticmethod
    def _check_skip(plan: run.Plan, skip: list[str]) -> None:
        known = {item.id for item in plan.items} | set(JOB_ROWS)
        for item in skip:
            if item not in known:
                raise ApiError(
                    400,
                    "invalid_request",
                    t("setup.errors.unknownItem", item=item[:60]),
                )

    # ── The job ──

    @property
    def running(self) -> bool:
        return self.job is not None and self.job.finished is None

    async def start(self, body: StartBody) -> Job:
        """Start the setup of the last plan served; 409 while one runs."""
        if self.running:
            raise ApiError(409, "setup_running", t("setup.errors.running"))
        planned = self._served
        if planned is None:
            raise ApiError(409, "no_plan", t("setup.errors.noPlan"))
        plan = planned.plan
        picks = [pick for pick in (plan.pick, *plan.alternatives) if pick]
        pick = next((p for p in picks if p.id == body.pick), None)
        if pick is None:
            raise ApiError(
                400, "invalid_request", t("serve.errors.unknownPick", pick=body.pick)
            )
        self._check_skip(plan, body.skip)
        self._count += 1
        job = Job(self._count, planned, pick, body.skip, self._ended)
        self.job = job
        job.thread.start()
        return job

    def _ended(self) -> None:
        """What is here has changed: the next look at the plan asks again."""
        self._cached = None

    def cancel(self) -> Job:
        """Stop the job that runs, keeping what it downloaded; 409 when none runs."""
        job = self.job
        if job is None or job.finished is not None:
            raise ApiError(409, "no_setup", t("setup.errors.noJob"))
        job.stop()
        return job

    async def aclose(self) -> None:
        """End the job that runs as ``cancelled``, and wait a little for it."""
        job = self.job
        if job is None or job.finished is not None:
            return
        job.stop()
        await asyncio.to_thread(job.thread.join, STOP_WAIT_S)

    # ── What the card reads ──

    def json(self) -> dict[str, Any] | None:
        """``job`` of ``GET /api/setup/detect``; null before the first.

        ``state`` is ``running``, ``done``, ``failed`` or ``cancelled``; for a
        failure ``error`` is its line. ``saved`` is whether the chat model was
        loaded and saved, so the card can connect to it. ``rows`` are the plan's
        rows, then the steps the job adds, then any other row it reported, each
        with its ``state`` (``waiting``, ``running``, ``done``, ``skipped``
        or ``failed``), the line to show and, while it downloads, its progress.
        """
        job = self.job
        if job is None:
            return None
        plan = job.planned.plan
        names = listing.row_names(plan, t)
        found = listing.build(plan, job.skip, job.planned.there, t, terminal=False)
        ids = [row.id for row in found.rows]
        ids += [row for row in _steps(plan, job) if row not in ids]
        items = job.job.items
        ids += [row for row in items if row not in ids]
        finished = job.finished
        rows = [
            _row(row, names, items.get(row), row in job.skip, finished is not None)
            for row in ids
        ]
        saved = items.get(run.SAVE)
        return {
            "id": job.id,
            "state": finished.state if finished else "running",
            "stopping": job.stopping and finished is None,
            "error": finished.error if finished else "",
            "pick": job.pick.id,
            "name": job.pick.name,
            "saved": saved is not None and saved.state == "done",
            "rows": rows,
        }


def _row(
    row: str,
    names: dict[str, str],
    event: run.ItemEvent | None,
    left_out: bool,
    over: bool,
) -> dict[str, Any]:
    """One row of the job: its name, state and line, in the request's language.

    What the person left out reads ``skipped``, a stop reads ``stopped``; the
    other lines are the job's own. A row that was running when the job ended
    was stopped, and one that never started stays ``waiting``.
    """
    if event is None:
        event = run.ItemEvent(row, "skipped" if left_out else "waiting")
    state, line = event.state, event.line
    if state == "skipped" and (left_out or line == run.STOPPED):
        line = t("setup.list.skipped" if left_out else "setup.stopped")
    progress = ""
    if state == "running" and over:
        state, line = "skipped", t("setup.stopped")
    elif state == "running" and event.total_bytes:
        part, whole = listing.amounts(event.done_bytes, event.total_bytes)
        progress = t("setup.progress", part=part, whole=whole)
    return {
        "id": row,
        "name": names.get(row, row.split(":", 1)[-1]),
        "state": state,
        "line": line,
        "done_bytes": event.done_bytes,
        "total_bytes": event.total_bytes,
        "progress": progress,
    }


def _steps(plan: run.Plan, job: Job) -> list[str]:
    """The rows the job adds after the plan's: each model loading, then the config."""
    chat, embedding = run.role_items(plan, job.pick)
    chat_on = not job.skip & chat
    return [
        *([run.LOAD_CHAT] if chat_on else []),
        *(
            [run.LOAD_EMBEDDING]
            if run.embedding_item(plan) and not job.skip & embedding
            else []
        ),
        *([run.SAVE] if chat_on else []),
    ]


def _unavailable(reason: str, hint: str) -> dict[str, Any]:
    return {"available": False, "reason": reason, "hint": hint}


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the setup card's plan, start and cancel routes on *app*."""

    @app.get("/api/setup/plan")
    async def api_setup_plan(
        skip: Annotated[str, Query(max_length=4000)] = "",
    ) -> JSONResponse:
        ids = [part for part in (p.strip() for p in skip.split(",")) if part]
        return JSONResponse(content=await console.setup.plan_json(ids))

    @app.post("/api/setup/start")
    async def api_setup_start(body: StartBody) -> JSONResponse:
        await console.setup.start(body)
        return JSONResponse(status_code=202, content={"job": console.setup.json()})

    @app.post("/api/setup/cancel")
    async def api_setup_cancel() -> JSONResponse:
        console.setup.cancel()
        return JSONResponse(content={"job": console.setup.json()})
