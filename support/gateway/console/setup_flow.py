"""The first-run setup from the welcome card: the plan, one job, and its progress.

The terminal's first run asks melo_serve what this Mac can run, shows a list,
and does it after one yes (:mod:`localmelo.support.server_setup`). The card does
the same through this module, with the same job and the same words
(:mod:`~localmelo.support.server_setup.listing`).

``GET /api/setup/plan`` is the list. Asking melo_serve takes seconds, so one
asker serves every caller at once and an answer is kept for :data:`PLAN_TTL_S`
(a plan that could not run, for :data:`FAILED_TTL_S`); it is not part of
``GET /api/setup/detect``, which the card asks every few seconds. ``?skip=a,b``
words the list for what the person left out. There is one plan at a time, the
last one asked for: it is dropped when a job starts or ends (what is here has
changed), and an answer that was asked before that is not kept. When there is
nothing to offer the body says ``available: false``, a ``code`` for the cause, a
``reason`` in the request's language and, in ``detail`` and ``hint``,
melo_serve's or the tool's own words, which are not translated.

``POST /api/setup/start`` (202; 409 while a job runs) starts the job in a
thread of its own, and ``POST /api/setup/cancel`` stops it, keeping what was
downloaded. A start is checked against that plan, the list the person said yes
to, and refused (409 ``no_plan``) once it is gone, older than :data:`START_TTL_S`
(a person may read the list that long; :data:`PLAN_TTL_S` is only how often
melo_serve is asked again) or unavailable, or (409 ``no_room``) when the list
does not fit the disk and nothing is left out. ``pick`` must be the plan's
pick, and ``skip`` may hold only what the list lets a person turn off.
Commands run as an argv, never through a shell, and the job registers no login
item for the gateway (installing it would restart the gateway that runs the
job). A gateway that shuts down ends the job ``cancelled`` and starts at once,
before any graceful wait: the child gets SIGTERM, then SIGKILL after
:data:`KILL_AFTER_S`, which fits inside the 8 s the macOS app gives a gateway
it quits. A child that handles SIGTERM goes on with its own cleanup (its grace
is 30 s) even if the gateway is gone by then; the partial files are kept.

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
# A plan can be started this long after it was asked for, within its epoch: a person
# reads the list, and PLAN_TTL_S is only how often melo_serve is asked again.
START_TTL_S: Final = 900.0
# From the start of a gateway's shutdown to the SIGKILL of a setup child that ignores
# SIGTERM. The macOS app kills a gateway 8 s after its SIGTERM (main.swift): this leaves
# the gateway time to record the job before that.
KILL_AFTER_S: Final = 5.0
# How long a gateway that shuts down waits for a child that was killed.
KILL_WAIT_S: Final = 1.0

T = TypeVar("T")


@dataclass(frozen=True)
class Planned:
    """What :func:`run.plan` found, the version of melo_serve already here, when, and for which state.

    *epoch* is the flow's :meth:`SetupFlow.epoch` when the plan was asked for:
    a plan is current only while it is still that.
    """

    plan: run.Plan
    there: str
    at: float
    epoch: tuple[int, bool]


class Job:
    """One setup: the plan it was started on, the job, and how far it got."""

    def __init__(self, id: int, planned: Planned, pick: str, skip: list[str]) -> None:
        self.id = id
        self.planned = planned
        self.skip = frozenset(skip)
        self.finished: run.Finished | None = None
        self.stopping = False
        self._killer: threading.Timer | None = None
        # The gateway's own login item is not registered: that restarts the gateway.
        self.job = run.SetupJob(
            planned.plan,
            pick,
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

    def stop(self) -> None:
        """Ask the job to end ``cancelled``, keeping what it downloaded."""
        self.stopping = True
        self.job.cancel()

    def stop_for_shutdown(self) -> None:
        """:meth:`stop`, and SIGKILL after :data:`KILL_AFTER_S` for a child that does not listen."""
        if self.finished is not None or self._killer is not None:
            return
        self.stop()
        self._killer = threading.Timer(KILL_AFTER_S, self.kill)
        self._killer.daemon = True
        self._killer.start()

    def kill(self) -> None:
        """End the job now: SIGKILL to the child that did not stop."""
        self.stopping = True
        self.job.kill()


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
    """The plan this gateway asked for last, and the one job it runs at a time."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self.clock: Callable[[], float] = time.monotonic
        self._plan: Planned | None = None
        self._asking: tuple[tuple[int, bool], asyncio.Future[Planned]] | None = None
        self.job: Job | None = None
        self._count = 0

    @property
    def epoch(self) -> tuple[int, bool]:
        """Which state of the machine a plan describes: how many jobs have started, and whether one runs.

        A job that starts or ends changes what is here, so a plan asked for
        before is not the list to say yes to. It is read from the job's own
        state, so no thread has to tell the flow that a job ended.
        """
        return self._count, self.running

    # ── The plan ──

    def _current(self) -> Planned | None:
        """The plan, while it is the answer to what is here now and young enough to be served."""
        found = self._plan
        if found is None or found.epoch != self.epoch:
            return None
        ttl = PLAN_TTL_S if found.plan.available else FAILED_TTL_S
        return found if self.clock() - found.at < ttl else None

    def _startable(self) -> Planned | None:
        """The plan a start may use: current in its epoch, available, and asked for within :data:`START_TTL_S`."""
        found = self._plan
        if found is None or found.epoch != self.epoch or found.plan.pick is None:
            return None
        young = self.clock() - found.at < START_TTL_S
        return found if found.plan.available and young else None

    async def _planned(self) -> Planned:
        """The plan: one asker for every caller, kept for :data:`PLAN_TTL_S`."""
        found = self._current()
        if found is not None:
            return found
        epoch = self.epoch
        if self._asking is None or self._asking[0] != epoch:
            asking = asyncio.ensure_future(self._ask(epoch))
            asking.add_done_callback(self._asked)
            self._asking = (epoch, asking)
        return await asyncio.shield(self._asking[1])

    async def _ask(self, epoch: tuple[int, bool]) -> Planned:
        plan, there = await _off_loop(_ask)
        found = Planned(plan, there, self.clock(), epoch)
        if epoch == self.epoch:  # a job that started or ended meanwhile made it old
            self._plan = found
        return found

    def _asked(self, asking: asyncio.Future[Planned]) -> None:
        if self._asking is not None and self._asking[1] is asking:
            self._asking = None
        if not asking.cancelled():
            asking.exception()  # read, so that no one has to

    async def plan_json(self, skip: list[str]) -> dict[str, Any]:
        """``GET /api/setup/plan``: the list in the request's language, or why there is none."""
        if sys.platform != "darwin":
            return _unavailable("not_mac", "", "")
        planned = await self._planned()
        plan = planned.plan
        pick = plan.pick
        if not plan.available or pick is None:
            code = plan.code or "failed"
            return _unavailable(code, "" if code == "uv" else plan.reason, plan.hint)
        _check_skip(plan, skip)
        found = listing.build(plan, skip, planned.there, t, terminal=False)
        resume = any(row.state == "partial" for row in found.rows)
        return {
            "available": True,
            "code": "",
            "reason": "",
            "detail": "",
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

    # ── The job ──

    @property
    def running(self) -> bool:
        return self.job is not None and self.job.finished is None

    async def start(self, body: StartBody) -> Job:
        """Start the setup of the plan the person said yes to.

        409 while a job runs, and 409 ``no_plan`` unless the plan is current in
        its epoch, asked for within :data:`START_TTL_S` and available (the card
        asks again), or ``no_room`` when it does not fit the disk and nothing is
        left out.
        """
        if self.running:
            raise ApiError(409, "setup_running", t("setup.errors.running"))
        planned = self._startable()
        pick = planned.plan.pick if planned else None
        if planned is None or pick is None:
            raise ApiError(409, "no_plan", t("setup.errors.noPlan"))
        plan = planned.plan
        if body.pick != pick.id:
            raise ApiError(400, "invalid_request", t("setup.errors.pick", pick=pick.id))
        _check_skip(plan, body.skip)
        if not listing.can_set_up(plan, body.skip):
            raise ApiError(409, "no_room", t("setup.errors.noRoom"))
        job = Job(self._count + 1, planned, body.pick, body.skip)
        job.thread.start()  # a thread that cannot start leaves no job that runs forever
        self._count += 1
        self.job = job
        return job

    def cancel(self) -> Job:
        """Stop the job that runs, keeping what it downloaded; 409 when none runs."""
        job = self.job
        if job is None or job.finished is not None:
            raise ApiError(409, "no_setup", t("setup.errors.noJob"))
        job.stop()
        return job

    def begin_shutdown(self) -> None:
        """Stop the job that runs, now: SIGTERM, and SIGKILL after :data:`KILL_AFTER_S`.

        The gateway calls this when it starts to shut down, before it waits for
        its connections: that wait, and the other parts that close first, are
        not spent with a child that could already be stopping.
        """
        if self.job is not None:
            self.job.stop_for_shutdown()

    async def aclose(self) -> None:
        """Wait for the job that was stopped (:meth:`begin_shutdown`, if it was not yet) to end."""
        job = self.job
        if job is None or job.finished is not None:
            return
        job.stop_for_shutdown()
        await asyncio.to_thread(job.thread.join, KILL_AFTER_S + KILL_WAIT_S)
        if (
            job.finished is None
        ):  # the timer is a daemon thread: do not rely on it alone
            job.kill()
            await asyncio.to_thread(job.thread.join, KILL_WAIT_S)

    # ── What the card reads ──

    def json(self) -> dict[str, Any] | None:
        """``job`` of ``GET /api/setup/detect``; null before the first.

        ``state`` is ``running``, ``done``, ``failed`` or ``cancelled``; for a
        failure ``error`` is its line. ``saved`` is whether the chat model was
        loaded and saved, so the card can connect to it. ``rows`` are the plan's
        rows, then the steps the job adds, then any other row it reported, each
        with its ``state`` (``waiting``, ``running``, ``done``, ``skipped``,
        ``stopped`` or ``failed``), the job's own ``line`` (not translated) and,
        while it downloads, its progress.
        """
        job = self.job
        if job is None:
            return None
        plan = job.planned.plan
        names = listing.row_names(plan, t)
        found = listing.build(plan, job.skip, job.planned.there, t, terminal=False)
        ids = [row.id for row in found.rows]
        ids += [row for row in job.job.steps if row not in ids]
        items = job.job.items
        ids += [row for row in items if row not in ids]
        finished = job.finished
        rows = [
            _row(row, items.get(row), names, row in job.skip, finished) for row in ids
        ]
        saved = items.get(run.SAVE)
        return {
            "id": job.id,
            "state": finished.state if finished else "running",
            "stopping": job.stopping and finished is None,
            "error": finished.error if finished else "",
            "pick": job.job.pick.id,
            "name": job.job.pick.name,
            "saved": saved is not None and saved.state == "done",
            "rows": rows,
        }


def _row(
    row: str,
    event: run.ItemEvent | None,
    names: dict[str, str],
    left_out: bool,
    finished: run.Finished | None,
) -> dict[str, Any]:
    """One row of the job: its name, state and line.

    The ``state`` is the stable word of the row and the screen words it in its
    language; ``line`` is the job's or melo_serve's own text, shown as it is
    (it is empty when the state says it all). What the person left out reads
    ``skipped``; a row that was running when the job ended, or that a stop
    ended, reads ``stopped`` (melo_serve's line for it names a command, which
    the card's user does not run, so it is left off); one that never started
    stays ``waiting``.
    """
    if event is None:
        event = run.ItemEvent(row, "skipped" if left_out else "waiting")
    state, line = event.state, event.line
    cancelled = finished is not None and finished.state == "cancelled"
    if state == "skipped":
        if left_out or line == run.SKIPPED:
            line = ""
        elif cancelled:
            state, line = "stopped", ""
    progress = ""
    if state == "running" and finished is not None:
        state, line = "stopped", ""
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


def _unavailable(code: str, detail: str, hint: str) -> dict[str, Any]:
    """The body of a plan there is none of: *code* worded in the request's language, then the words of whoever said no.

    *detail* and *hint* are melo_serve's (or the failed tool's) own, shown as
    they are, in smaller type under the sentence.
    """
    return {
        "available": False,
        "code": code,
        "reason": t(f"setup.unavailable.{code}", url=run.UV_HINT),
        "detail": detail,
        "hint": hint,
    }


def _check_skip(plan: run.Plan, skip: list[str]) -> None:
    """400 for an id the list does not let a person turn off (what is here, a login item it does not add, or nothing)."""
    allowed = listing.skippable_ids(plan, gateway=False)
    for item in skip:
        if item not in allowed:
            raise ApiError(
                400, "invalid_request", t("setup.errors.cannotSkip", item=item[:60])
            )


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
