from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
except ImportError as e:
    raise ImportError(
        "Gateway requires fastapi and uvicorn: pip install fastapi uvicorn"
    ) from e

from localmelo.melo.checker import Checker, GatewayIngressPayload, SessionTransition
from localmelo.support import config
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.guard import LoopbackGuard, is_loopback
from localmelo.support.gateway.session import SessionBusyError, SessionManager

logger = logging.getLogger("localmelo.gateway")

# Seconds uvicorn waits for open connections after it starts shutting down.
GRACEFUL_SHUTDOWN_S = 3.0


# ── Gateway Server ──


class GatewayServer:
    def __init__(
        self, cfg: config.Config, *, console_options: ConsoleOptions | None = None
    ) -> None:
        self.cfg = cfg
        host = cfg.gateway.host
        # A URL writes an IPv6 address in brackets, as in [::1]; the socket
        # takes it without them.
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        self.host = host
        self.port = cfg.gateway.port
        self._started_at: float | None = None
        self._checker = Checker()

        # SessionManager now builds providers from Config directly
        self.sessions = SessionManager(
            max_sessions=50,
            idle_ttl=3600.0,
            config=cfg,
        )

        # No Swagger or ReDoc page: /docs loads Swagger UI from a CDN, which
        # the console's no-CDN rule forbids. API v2 is documented in
        # docs/api.md instead.
        self.app = FastAPI(
            title="localmelo gateway",
            lifespan=self._lifespan,
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )
        self._register_routes()

        from localmelo.support.gateway.webapp import mount as mount_webapp

        mount_webapp(self.app, self.cfg)

        # The console is mounted only on loopback: API v2 has no login. Its
        # error middleware must be added before the guard, so the guard stays
        # outermost and refuses a foreign request before any route sees it.
        self.console: Console | None = None
        if is_loopback(self.host):
            self.console = Console(self.cfg, port=self.port, options=console_options)
            self.console.mount(self.app)
            self.app.add_middleware(LoopbackGuard, port=self.port)
        else:
            logger.warning(
                "Gateway host %s is not localhost or a loopback IP address, so "
                "requests are not checked. The gateway has no authentication: "
                "anything that can reach it, web pages through DNS rebinding "
                'included, can run tasks with tools. Use host = "127.0.0.1" '
                "(the default).",
                self.host,
            )

    # ── Lifespan ──

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncIterator[None]:  # type: ignore[type-arg]
        self._started_at = time.time()
        if self.console is not None:
            await self.console.start()
        logger.info("gateway ready on %s:%d", self.host, self.port)
        try:
            yield
        finally:
            logger.info("gateway shutting down")
            # Each close must run even when an earlier one raises: the
            # console's aclose releases the gateway store's owner lock last.
            try:
                await self.sessions.close_all()
            finally:
                try:
                    # FastAPI does not run on_event handlers when an explicit
                    # lifespan is installed. Close the web workspace's
                    # providers here as well.
                    playground = getattr(app.state, "smoke_playground", None)
                    if playground is not None:
                        await playground.close_all()
                finally:
                    if self.console is not None:
                        await self.console.aclose()

    # ── Routes ──

    def _register_routes(self) -> None:
        app = self.app

        @app.post("/v1/agent/run")
        async def agent_run(body: dict[str, Any]) -> JSONResponse:
            query = body.get("query", "")
            session_id = body.get("session_id")

            # ── Ingress validation ──
            ingress_check = self._checker.check_gateway_ingress(
                GatewayIngressPayload(query=query or "", session_id=session_id)
            )
            if not ingress_check.allowed:
                return JSONResponse(
                    status_code=400,
                    content={"error": ingress_check.reason},
                )

            sanitized = ingress_check.sanitized_payload
            query = sanitized.query
            session_id = sanitized.session_id

            session = await self.sessions.get_or_create(session_id)

            # Per-session serialization: only one run at a time per session
            async with session._lock:
                # ── Session transition check: idle → running ──
                trans_check = self._checker.check_session_transition(
                    SessionTransition(
                        from_status=session.status,
                        to_status="running",
                        session_id=session.session_id,
                    )
                )
                if not trans_check.allowed:
                    return JSONResponse(
                        status_code=409,
                        content={
                            "error": trans_check.reason,
                            "session_id": session.session_id,
                        },
                    )

                session.status = "running"
                session.touch()

                try:
                    result = await session.agent.run(query)

                    # ── Session transition: running → idle ──
                    self._checker.check_session_transition(
                        SessionTransition(
                            from_status="running",
                            to_status="idle",
                            session_id=session.session_id,
                        )
                    )
                    session.status = "idle"
                    session.touch()
                    return JSONResponse(
                        content={
                            "result": result,
                            "session_id": session.session_id,
                        }
                    )
                except Exception as e:
                    session.status = "idle"
                    logger.exception("agent run failed")
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": str(e),
                            "session_id": session.session_id,
                        },
                    )
                finally:
                    # Cancellation is a BaseException, so an except Exception
                    # alone leaves a disconnected request permanently running.
                    session.status = "idle"
                    session.touch()

        @app.get("/v1/sessions")
        async def list_sessions() -> JSONResponse:
            return JSONResponse(
                content={
                    "sessions": self.sessions.list_sessions(),
                }
            )

        @app.delete("/v1/sessions/{session_id}")
        async def close_session(session_id: str) -> JSONResponse:
            try:
                closed = await self.sessions.close(session_id)
            except SessionBusyError as e:
                return JSONResponse(
                    status_code=409,
                    content={"error": str(e), "session_id": session_id},
                )
            if not closed:
                return JSONResponse(
                    status_code=404,
                    content={"error": "session not found"},
                )
            return JSONResponse(content={"closed": session_id})

        @app.get("/v1/health")
        async def health() -> JSONResponse:
            sessions = self.sessions.list_sessions()
            busy_count = sum(1 for s in sessions if s.get("is_busy"))
            return JSONResponse(
                content={
                    "status": "ok",
                    "uptime": round(time.time() - (self._started_at or time.time()), 1),
                    "sessions": {
                        "active": len(sessions),
                        "busy": busy_count,
                        "max": self.sessions._max,
                    },
                    "chat_backend": self.cfg.chat_backend,
                    "embedding_backend": self.cfg.embedding_backend,
                }
            )

    # ── Run ──

    def uvicorn_server(self) -> uvicorn.Server:
        """The server ``run()`` uses, whose shutdown ends the event streams first."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            timeout_graceful_shutdown=int(GRACEFUL_SHUTDOWN_S),
        )
        return _ConsoleServer(config, console=self.console)

    def run(self) -> None:
        self.uvicorn_server().run()


def start_gateway(cfg: config.Config) -> None:
    cfg.validate_or_raise()
    server = GatewayServer(cfg)
    server.run()


class _ConsoleServer(uvicorn.Server):
    """A uvicorn server that ends the console's streams before it waits.

    uvicorn waits for open connections to close before it runs the lifespan
    shutdown, and an ``EventSource`` never closes by itself. Setting the
    console's shutdown event first makes every stream return, so Ctrl+C and
    SIGTERM do not hang on an open console tab.
    """

    def __init__(self, config: uvicorn.Config, *, console: Console | None) -> None:
        super().__init__(config)
        self._console = console

    async def shutdown(self, sockets: Any = None) -> None:
        if self._console is not None:
            self._console.begin_shutdown()
        await super().shutdown(sockets=sockets)
