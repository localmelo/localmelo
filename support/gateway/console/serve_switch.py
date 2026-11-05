"""Switch the chat model melo_serve serves, for every chat of this gateway.

A switch holds the chats first. It waits until no chat of this gateway is in
the middle of a turn, and a turn that starts meanwhile waits for the switch
before it calls the model (its ``held`` says so) and then runs on the new
model. The wait is bounded by :data:`WAIT_S` and can be cancelled. Then the
switch asks melo_serve to load the pick, naming the generation it was chosen
at: a switch made elsewhere meanwhile is reported, not undone. A load that
has started goes on in melo_serve whatever happens here.

The state rides on ``GET /api/health`` as ``switch``. ``GET /api/serve``
lists the chat picks, each worded by
:func:`~localmelo.support.melo_serve.pick_words` in the request's language;
``POST /api/serve/switch`` starts a switch and ``POST /api/serve/switch/cancel``
ends its wait. The routes are under ``/api``: the gateway's authentication
and origin guard hold for them as for the settings.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from localmelo.support import melo_serve
from localmelo.support.gateway.i18n import current, said, said_text, t

from .errors import ApiError

if TYPE_CHECKING:
    from . import Console

logger = logging.getLogger(__name__)

# How long a switch waits for the chats that are mid-turn, in seconds.
WAIT_S: Final = 600.0
POLL_S: Final = 0.25
# The hold reason of a chat whose turn waits for a switch.
HELD: Final = "Switching"
ACTIVE: Final = ("waiting", "loading")


@dataclass
class Switch:
    """One switch: the pick, the generation it was chosen at, and how far it got.

    *state* is ``waiting`` (for the chats mid-turn), ``loading``, ``done``,
    ``failed`` or ``cancelled``; *error* is melo_serve's words or the
    catalog's (:func:`said`), read in each request's language.
    """

    id: int
    pick: str
    name: str
    load_s: float
    generation: int
    state: str = "waiting"
    waiting_for: int = 0
    error: str | dict[str, Any] = ""
    cancel: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    ended: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


class SwitchBody(BaseModel):
    """Request body of ``POST /api/serve/switch``: a pick's id from ``GET /api/serve``."""

    model_config = ConfigDict(extra="forbid")

    pick: StrictStr = Field(min_length=1, max_length=200)


class ServeSwitch:
    """The one switch a gateway runs at a time, and the turns it holds."""

    def __init__(
        self, console: Console, *, wait_s: float = WAIT_S, poll_s: float = POLL_S
    ) -> None:
        self._console = console
        self.wait_s, self.poll_s = wait_s, poll_s
        self.last: Switch | None = None
        self._count = 0
        self._held: set[str] = set()
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def active(self) -> bool:
        return self.last is not None and self.last.state in ACTIVE

    def held(self, session_id: str) -> str:
        """:data:`HELD` while the chat's turn waits for a switch, else ``""``."""
        return HELD if session_id in self._held else ""

    def working(self) -> int:
        """How many chats of this gateway are mid-turn; a held turn is not."""
        return sum(
            1
            for entry in self._console.sessions.entries()
            if entry.runner.busy and entry.session_id not in self._held
        )

    async def hold(self, session_id: str) -> bool:
        """Before a turn calls the model: wait for the switch that runs. True when one did."""
        found = self.last
        if found is None or found.state not in ACTIVE:
            return False
        self._held.add(session_id)
        try:
            await found.ended.wait()
        finally:
            self._held.discard(session_id)
        return True

    # ── What the surfaces read ──

    def json(self) -> dict[str, Any] | None:
        """``switch`` of ``GET /api/health``; null before the first switch.

        ``title`` and ``detail`` are its two lines in the request's language
        (``Switching to Qwen3 8B…``, ``Waiting for 1 chat to finish``).
        """
        found = self.last
        if found is None:
            return None
        detail = ""
        if found.state == "waiting" and found.waiting_for:
            detail = t("serve.waiting", n=found.waiting_for)
        elif found.state == "loading":
            detail = (
                t("serve.loadingAbout", s=round(found.load_s))
                if found.load_s
                else t("serve.loadingNow")
            )
        error = found.error
        return {
            "id": found.id,
            "pick": found.pick,
            "name": found.name,
            "state": found.state,
            "waiting_for": found.waiting_for,
            "load_s": found.load_s,
            "error": said_text(error) if isinstance(error, dict) else error,
            "title": t("serve.switching", name=found.name),
            "detail": detail,
        }

    async def picks_json(self) -> dict[str, Any]:
        """``GET /api/serve``: the chat picks as every surface words them, the foot and the switch."""
        self._require_melo_serve()
        language = current()
        developer = self._console.cfg.ui.mode == "developer"
        rows = []
        for pick in await self._picks():
            words, fit = melo_serve.pick_words(pick, language, developer=developer)
            rows.append(
                {
                    "id": pick.id,
                    "short": pick.short,
                    "name": pick.name,
                    "backend_name": pick.backend_name,
                    "model": pick.model,
                    "current": pick.current,
                    "loadable": pick.loadable,
                    "words": words,
                    "fit": fit,
                    "tight": pick.verdict == "alone",
                }
            )
        return {"picks": rows, "foot": t("serve.foot"), "switch": self.json()}

    # ── Starting and cancelling ──

    def _require_melo_serve(self) -> None:
        if self._console.cfg.chat_backend != "melo_serve":
            raise ApiError(409, "not_melo_serve", t("serve.errors.notServe"))

    def _require_idle(self) -> None:
        if self.last is not None and self.active:
            raise ApiError(
                409, "switch_running", t("serve.errors.running", name=self.last.name)
            )

    async def _picks(self) -> list[melo_serve.Pick]:
        try:
            return await melo_serve.picks()
        except melo_serve.MeloServeError as exc:
            raise ApiError(502, "melo_serve_unavailable", str(exc)) from None

    async def start(self, pick_id: str) -> Switch:
        """Start the switch to *pick_id*, one of the chat picks; 409 while one runs."""
        self._require_melo_serve()
        self._require_idle()
        found = next((p for p in await self._picks() if p.id == pick_id), None)
        if found is None:
            raise ApiError(
                400, "invalid_request", t("serve.errors.unknownPick", pick=pick_id)
            )
        if not found.loadable:
            reason = melo_serve.pick_words(found, current())[0]
            raise ApiError(
                409,
                "pick_cannot_load",
                t("serve.errors.cannotLoad", name=found.name, reason=reason),
            )
        snap = await asyncio.to_thread(melo_serve.snapshot)
        self._require_idle()  # another start may have won the awaits
        self._count += 1
        switch = Switch(
            self._count, found.id, found.name, found.load_s, snap.generation
        )
        self.last = switch
        task = asyncio.create_task(self._run(switch))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return switch

    def cancel(self) -> None:
        """End the wait of the switch that runs; a load that has started goes on."""
        found = self.last
        if found is None or found.state not in ACTIVE:
            raise ApiError(409, "no_switch", t("serve.errors.noSwitch"))
        if found.state == "loading":
            raise ApiError(
                409, "switch_loading", t("serve.errors.loading", name=found.name)
            )
        found.state = "cancelled"
        found.cancel.set()
        found.ended.set()  # the held turns go on, on the model they have

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # ── The switch ──

    async def _run(self, switch: Switch) -> None:
        try:
            if await self._quiet(switch):
                switch.state, switch.waiting_for = "loading", 0
                await self._load(switch)
        except Exception as exc:
            logger.exception("The switch to %s failed", switch.pick)
            switch.state, switch.error = "failed", str(exc) or type(exc).__name__
        finally:
            if switch.state in ACTIVE:  # shut down meanwhile
                switch.state = "cancelled"
            switch.ended.set()

    async def _quiet(self, switch: Switch) -> bool:
        """Wait until no chat is mid-turn: True then, False once cancelled or out of time."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.wait_s
        while not switch.cancel.is_set():
            busy = self.working()
            switch.waiting_for = busy
            if not busy:
                return True
            if loop.time() >= deadline:
                switch.state = "failed"
                switch.error = said("serve.stillWorking", n=busy)
                return False
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(switch.cancel.wait(), self.poll_s)
        return False

    async def _load(self, switch: Switch) -> None:
        try:
            await melo_serve.switch(switch.pick, expected_generation=switch.generation)
        except melo_serve.MeloServeError as exc:
            switch.state, switch.error = "failed", str(exc)
            if exc.kind == "stale_generation":
                # Switched elsewhere meanwhile: read what is served now and say so.
                snap = await asyncio.to_thread(melo_serve.snapshot)
                served = snap.roles.get("chat")
                switch.generation = snap.generation
                if served is not None:
                    switch.error = said("serve.stale", name=served.name or served.model)
            return
        switch.state = "done"


def add_routes(app: FastAPI, console: Console) -> None:
    """Register ``GET /api/serve`` and the switch's two routes on *app*."""

    @app.get("/api/serve")
    async def api_serve() -> JSONResponse:
        return JSONResponse(content=await console.serve.picks_json())

    @app.post("/api/serve/switch")
    async def api_serve_switch(body: SwitchBody) -> JSONResponse:
        await console.serve.start(body.pick)
        return JSONResponse(status_code=202, content={"switch": console.serve.json()})

    @app.post("/api/serve/switch/cancel")
    async def api_serve_cancel() -> JSONResponse:
        console.serve.cancel()
        return JSONResponse(content={"switch": console.serve.json()})
