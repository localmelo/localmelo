"""The web console's server half: API v2 under ``/api``.

:class:`Console` owns the one gateway :class:`~localmelo.support.sessions.store.SessionStore`,
its :class:`~localmelo.support.sessions.hub.SessionHub`, an
:class:`~localmelo.support.sessions.approvals.ApprovalBroker` and the open
sessions. It opens the store when the gateway starts, answers 503
``store_unavailable`` on the routes that need it while it is not open, and
retries the open every :data:`STORE_RETRY_S` seconds. Health is served
either way. The console page itself arrives in PR 5; this PR mounts only
``/api``.

The gateway mounts the console only when it listens on a loopback address
(see :class:`~localmelo.support.gateway.guard.LoopbackGuard`); API v2 has
no login.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.routing import Match

from localmelo.support.gateway.chat_gate import ChatGate, call_log_dir
from localmelo.support.gateway.machine_monitor import MachineMonitor
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.sessions import display
from localmelo.support.sessions.approvals import ApprovalBroker
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import (
    FAILED_ERROR,
    GATEWAY_SESSIONS_DB,
    RETENTION_SWEEP_S,
    SessionInfo,
    SessionStore,
    SessionStoreError,
    StoreOwnedError,
)

from . import (
    errors,
    routes_details,
    routes_page,
    routes_sessions,
    routes_settings,
    routes_tasks,
    routes_turns,
)
from .errors import ApiError
from .probe import ModelServerProbe, Probe
from .registry import SWEEP_INTERVAL_S, AgentBuilder, ConsoleSessions
from .shapes import session_json
from .worker_reader import WorkerReader

if TYPE_CHECKING:
    from localmelo.support.config import Config

logger = logging.getLogger(__name__)

# Seconds between retries of a store open that failed.
STORE_RETRY_S: Final = 5.0

# What an API v2 message says when another process holds the store's owner
# lock. It never names the other process: spec 2.2 keeps lease holder
# strings out of API messages. The terminal keeps the store's own text.
STORE_OWNED_MESSAGE: Final = (
    "Another melo gateway is using the console's session file. "
    "Stop it to use the console here."
)


def store_message(exc: SessionStoreError) -> str:
    """A store failure as an API v2 message, with the home folder written ``~``."""
    if isinstance(exc, StoreOwnedError):
        return STORE_OWNED_MESSAGE
    return display.home_label(str(exc))


@dataclass(frozen=True)
class ConsoleOptions:
    """What a test may replace; production reads every default.

    *store_path* is ``~/.cache/localmelo/gateway/sessions.db`` when None,
    *worker_store_path* defaults to the CLI sessions file, *builder* is
    :func:`~.registry.default_builder`, and *probe* is the
    real cached model-server probe. *dev_reload* is read by the static
    bundle of PR 5.
    """

    store_path: str | None = None
    worker_store_path: str | None = None
    builder: AgentBuilder | None = None
    probe: Probe | None = None
    # None builds the console's gate from the ``[scheduler]`` table; a test
    # passes a gate of its own (no call log, a fake clock).
    gate: ChatGate | None = None
    dev_reload: bool = False


class Console:
    """The console's server state and the ``/api`` routes over it."""

    def __init__(
        self,
        cfg: Config,
        *,
        port: int,
        options: ConsoleOptions | None = None,
    ) -> None:
        self.cfg = cfg
        self.port = port
        self.options = options or ConsoleOptions()
        self.store = SessionStore(self.options.store_path, writer="gateway")
        self.worker_history = WorkerReader(self.options.worker_store_path)
        self.hub = SessionHub(self.store)
        self.broker = ApprovalBroker(timeout_s=None)
        self.gate = self.options.gate or self._build_gate()
        # A gate the console built is its own to close; a shared one belongs
        # to the gateway, which closes it after both surfaces' sessions.
        self._owns_gate = self.options.gate is None
        self._monitor = None if self.options.gate is not None else self._build_monitor()
        self.sessions = ConsoleSessions(
            self.store,
            self.broker,
            lambda: self.cfg,
            builder=self.options.builder,
            gate=self.gate,
        )
        self.workers = TaskWorkers(
            self.store, self.gate, lambda: self.cfg, self.hub, broker=self.broker
        )
        self.probe: Probe = self.options.probe or ModelServerProbe(cfg)
        self._started_at = time.time()
        self._store_ok = False
        self._store_error = ""
        self._store_open_reason = "open-error"
        self._closed = False
        self._shutdown = asyncio.Event()
        self._retry_task: asyncio.Task[None] | None = None
        self._retention_task: asyncio.Task[None] | None = None
        self._sweep_task: asyncio.Task[None] | None = None
        self._streams = 0
        self.app: FastAPI | None = None

    # ── Shapes ──

    def session_body(self, info: SessionInfo) -> dict[str, Any]:
        """One session as API v2 gives it, with this console's live fields."""
        return session_json(
            info,
            pending_approvals=len(self.broker.pending(info.id)),
            level=self.gate.level(info.id),
            held=self.gate.held(info.id),
        )

    # ── The gate ──

    def _build_gate(self) -> ChatGate:
        """The console's scheduler gate, from the ``[scheduler]`` table."""
        scheduler = self.cfg.scheduler
        return ChatGate(
            policy=scheduler.policy,
            n_srv=scheduler.n_srv,
            r_f=scheduler.r_f,
            n_fg=scheduler.n_fg,
            call_log_dir=call_log_dir(scheduler),
            lock_gate=scheduler.lock_gate,
        )

    def _build_monitor(self) -> MachineMonitor | None:
        """The GPU-lock watcher a ``lock_gate = "wait"`` needs.

        Only ``wait`` holds a call, so only ``wait`` needs to know the
        machine; the battery is not read here: no console route shows it.
        """
        scheduler = self.cfg.scheduler
        if scheduler.lock_gate != "wait" or not scheduler.gpu_lock_file:
            return None
        return MachineMonitor(
            lock_path=scheduler.gpu_lock_file,
            lock_self=scheduler.lock_self,
            lock_gate=scheduler.lock_gate,
            write=self.gate.write_machine,
            on_change=self._on_machine,
        )

    def _on_machine(self, snapshot: Any) -> None:
        """Feed one observed snapshot to the gate; no await between the two."""
        self.gate.set_machine(snapshot.machine, payload=snapshot.payload())

    # ── Mounting ──

    def mount(self, app: FastAPI) -> None:
        """Register the error handlers and the ``/api`` routes on *app*."""
        self.app = app
        app.state.console = self
        errors.install(app)
        routes_settings.add_routes(app, self)
        routes_sessions.add_routes(app, self)
        routes_turns.add_routes(app, self)
        routes_details.add_routes(app, self)
        routes_page.add_routes(app, self)
        routes_tasks.add_routes(app, self)
        _add_not_found(app)

    # ── Lifecycle ──

    async def start(self) -> None:
        """Open the store, or start retrying it; start the sweeps.

        A failure does not raise: the gateway still serves ``/v1`` and
        ``/api/health``, and ``/api`` answers 503 ``store_unavailable``
        until an open succeeds.
        """
        self._started_at = time.time()
        if self._monitor is not None:
            await self._monitor.start()
        if await self._try_open():
            self._start_retention()
        else:
            self._retry_task = asyncio.create_task(self._retry_loop())
        self._sweep_task = asyncio.create_task(self._sweep_loop())

    async def _try_open(self) -> bool:
        try:
            await self.store.open()
            await self.workers.open()
        except SessionStoreError as exc:
            self._store_ok = False
            self._store_open_reason = (
                "owned" if isinstance(exc, StoreOwnedError) else "open-error"
            )
            self._store_error = store_message(exc)
            logger.warning("The session store is unavailable: %s", self._store_error)
            return False
        self._store_ok = True
        self._store_error = ""
        return True

    def _start_retention(self) -> None:
        if self._retention_task is not None and not self._retention_task.done():
            return
        self._retention_task = asyncio.create_task(self._retention_loop())

    async def _retry_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(STORE_RETRY_S)
            if self._closed or self.store.failed is not None:
                return
            if await self._try_open():
                self._start_retention()
                return

    async def _retention_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(RETENTION_SWEEP_S)
            try:
                await self.store.apply_retention()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Session retention failed: %s", exc)

    async def _sweep_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(SWEEP_INTERVAL_S)
            try:
                await self.sessions.close_idle()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("The console's idle sweep failed")

    def begin_shutdown(self) -> None:
        """End every event stream; the uvicorn server calls this first."""
        self._shutdown.set()

    async def aclose(self) -> None:
        """Close the sessions, the hub and the store, releasing the owner lock last."""
        if self._closed:
            return
        self._closed = True
        self._shutdown.set()
        tasks = [
            task
            for task in (self._retry_task, self._retention_task, self._sweep_task)
            if task is not None
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._monitor is not None:
            with contextlib.suppress(Exception):
                await self._monitor.stop()
        try:
            try:
                await self.workers.aclose()
            finally:
                await self.sessions.close_all()
        finally:
            # Every live call leaves its one line; the log's files close.
            # A shared gate stays open: the gateway closes it last.
            if self._owns_gate:
                with contextlib.suppress(Exception):
                    await self.gate.close()
            try:
                await self.hub.aclose()
            finally:
                try:
                    await self.store.aclose()
                except Exception:  # never opened, or already closed
                    logger.debug("Closing the session store failed", exc_info=True)
                with contextlib.suppress(Exception):
                    await self.worker_history.aclose()
                with contextlib.suppress(Exception):
                    await self.probe.aclose()

    # ── Store state ──

    @property
    def shutdown(self) -> asyncio.Event:
        """Set when the gateway starts to shut down; every stream waits on it."""
        return self._shutdown

    def uptime_s(self) -> float:
        return max(0.0, time.time() - self._started_at)

    def store_state(self) -> str:
        """``ok``, ``unavailable`` or ``failed``, as health reports it."""
        if self.store.failed is not None:
            return "failed"
        return "ok" if self._store_ok else "unavailable"

    def store_reason(self) -> str | None:
        """A stable machine-readable reason for the current store state."""
        if self.store.failed is not None:
            return "failed"
        return None if self._store_ok else self._store_open_reason

    def store_error(self) -> str:
        """Why the store is not ``ok``, with the home folder as ``~``."""
        if self.store.failed is not None:
            return display.home_label(str(self.store.failed))
        if self._store_ok:
            return ""
        return self._store_error

    def require_store(self) -> None:
        """Raise ``503 store_unavailable`` unless the store is open and usable."""
        state = self.store_state()
        if state == "ok":
            return
        if state == "failed":
            raise ApiError(503, "store_unavailable", FAILED_ERROR)
        raise ApiError(
            503,
            "store_unavailable",
            self._store_error or "The session store is not available.",
        )

    async def store_api_error(
        self, exc: SessionStoreError, session_id: str | None = None
    ) -> ApiError:
        """Map a store failure as API v2 says, by state and never by message text.

        A failure while ``store.failed`` is set is a 503 without a second
        read. Any other failure makes the route read the row again: a
        missing row is 404 ``This session was deleted.``, anything else 503.
        """
        if self.store.failed is not None:
            return ApiError(503, "store_unavailable", FAILED_ERROR)
        if session_id is not None:
            try:
                info = await self.store.get_session(session_id)
            except SessionStoreError:
                return ApiError(
                    503, "store_unavailable", self._store_error or store_message(exc)
                )
            if info is None:
                return ApiError(404, "not_found", "This session was deleted.")
        return ApiError(
            503,
            "store_unavailable",
            self._store_error or store_message(exc),
        )

    # ── Event streams ──

    def stream_opened(self) -> None:
        self._streams += 1

    def stream_closed(self) -> None:
        self._streams = max(0, self._streams - 1)

    @property
    def open_streams(self) -> int:
        return self._streams


def _add_not_found(app: FastAPI) -> None:
    """Answer an unknown ``/api`` path with the API v2 404 shape."""
    methods = [
        "GET",
        "HEAD",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
        "TRACE",
        "CONNECT",
    ]

    @app.api_route("/api", methods=methods)
    async def api_root_not_found() -> JSONResponse:
        raise ApiError(404, "not_found", "Not found.")

    @app.api_route("/api/{rest:path}", methods=methods)
    async def api_not_found(request: Request, rest: str) -> JSONResponse:
        allowed = _allowed_methods(request.app, str(request.url.path))
        if allowed:
            raise HTTPException(
                status_code=405,
                detail="Method Not Allowed",
                headers={"Allow": ", ".join(allowed)},
            )
        raise ApiError(404, "not_found", "Not found.")


def _allowed_methods(app: FastAPI, path: str) -> list[str]:
    """The methods the real ``/api`` routes serve for *path*.

    The catch-all runs after every real route; Starlette's router would have
    answered 405 with an ``Allow`` header for a known path with the wrong
    method, but the catch-all matches the path first. Any real route whose
    path matches is that answer.
    """
    methods: set[str] = set()
    scope = {
        "type": "http",
        "path": path,
        "root_path": "",
        "method": "GET",
        "headers": [],
    }
    for route in app.router.routes:
        endpoint = getattr(route, "endpoint", None)
        if getattr(endpoint, "__name__", "") in {
            "api_not_found",
            "api_root_not_found",
        }:
            continue
        match, _ = route.matches(scope)
        if match in (Match.FULL, Match.PARTIAL):
            methods.update(getattr(route, "methods", None) or ())
    return sorted(methods)


# Re-exported for the gateway and for tests: the path the console uses when
# no ConsoleOptions.store_path is given.
DEFAULT_STORE_PATH: Final = GATEWAY_SESSIONS_DB
__all__ = [
    "Console",
    "ConsoleOptions",
    "DEFAULT_STORE_PATH",
    "STORE_RETRY_S",
    "ApiError",
]
