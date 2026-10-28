from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
except ImportError as e:
    raise ImportError(
        "Gateway requires fastapi and uvicorn: pip install fastapi uvicorn"
    ) from e

from localmelo.melo.checker import Checker, GatewayIngressPayload, SessionTransition
from localmelo.support import config
from localmelo.support.gateway.auth import COOKIE_TTL, GatewayAuth
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.guard import LoopbackGuard, app_instance, is_loopback
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
        if not is_loopback(self.host):
            raise ValueError("The gateway must listen on a loopback address.")
        self.auth = GatewayAuth()
        # The per-launch identifier the macOS app passes to its child; empty
        # for a gateway the user started directly.
        self.instance = app_instance()

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

        mount_webapp(
            self.app,
            self.cfg,
            enable_playground=getattr(cfg, "_enable_playground", False),
        )

        # The guard stays outermost, before console error middleware, so it
        # rejects foreign and unauthenticated requests before any route runs.
        self.console: Console | None = Console(
            self.cfg,
            port=self.port,
            options=self._console_options(console_options),
        )
        self.console.mount(self.app)
        self.app.add_middleware(
            LoopbackGuard,
            port=self.port,
            auth=self.auth,
            instance=self.instance,
        )

    def _console_options(self, options: ConsoleOptions | None) -> ConsoleOptions:
        """*options*, with the workspace page's gate when it built one.

        One scheduler, one gate: the workspace page and the console share a
        process-wide slot budget, so a call waits behind the other surface's
        calls. A test may pass a gate of its own and keep it.
        """
        if options is not None and options.gate is not None:
            return options
        playground = getattr(self.app.state, "smoke_playground", None)
        gate = getattr(playground, "gate", None)
        if gate is None:
            return options or ConsoleOptions()
        return replace(options or ConsoleOptions(), gate=gate)

    # ── Lifespan ──

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncIterator[None]:  # type: ignore[type-arg]
        self._started_at = time.time()
        # The web workspace's machine monitor starts before the gateway
        # reports ready, so a lock present at startup is observed at once.
        playground = getattr(app.state, "smoke_playground", None)
        starter = getattr(playground, "start", None)
        if starter is not None:
            await starter()
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
                        await playground.close_sessions()
                finally:
                    try:
                        if self.console is not None:
                            await self.console.aclose()
                    finally:
                        # The one shared gate goes last: a console turn
                        # waiting for its slot must end `cancelled`, cause
                        # `shutdown`, not fail on a closed gate.
                        playground = getattr(app.state, "smoke_playground", None)
                        if playground is not None:
                            await playground.gate.close()

    # ── Routes ──

    def _register_routes(self) -> None:
        app = self.app

        @app.post("/api/auth/login")
        async def login(body: dict[str, Any]) -> Response:
            code = body.get("code")
            cookie = (
                self.auth.consume_login(code, self.port, instance=self.instance)
                if isinstance(code, str)
                else None
            )
            if cookie is None:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "code": "unauthorized",
                            "message": "Invalid or expired login code.",
                            "details": {},
                        }
                    },
                    headers={"Cache-Control": "no-store"},
                )
            response = Response(status_code=204, headers={"Cache-Control": "no-store"})
            response.set_cookie(
                f"melo_session_{self.port}",
                cookie,
                max_age=COOKIE_TTL,
                httponly=True,
                samesite="strict",
                path="/",
            )
            return response

        @app.post("/api/auth/logout")
        async def logout(request: Request) -> Response:
            name = f"melo_session_{self.port}"
            cookie = request.cookies.get(name, "")
            if cookie:
                self.auth.revoke_cookie(cookie)
                self.auth.invalidate_streams()
            response = Response(status_code=204, headers={"Cache-Control": "no-store"})
            response.delete_cookie(name, path="/", httponly=True, samesite="strict")
            return response

        @app.post("/api/auth/rotate")
        async def rotate(request: Request) -> Response:
            header = request.headers.get("authorization", "")
            if not header.startswith("Bearer ") or not self.auth.valid_bearer(
                header[7:]
            ):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "code": "unauthorized",
                            "message": "Bearer token required.",
                            "details": {},
                        }
                    },
                    headers={"Cache-Control": "no-store", "WWW-Authenticate": "Bearer"},
                )
            self.auth.rotate()
            return Response(status_code=204, headers={"Cache-Control": "no-store"})

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

            try:
                session = await self.sessions.get_or_create(session_id)
            except SessionBusyError as exc:
                return JSONResponse(
                    status_code=409, content={"error": str(exc), "code": "session_busy"}
                )

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
                    content={
                        "error": str(e),
                        "code": "session_busy",
                        "session_id": session_id,
                    },
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


def start_gateway(cfg: config.Config, *, allow_unconfigured: bool = False) -> None:
    if not allow_unconfigured:
        cfg.validate_or_raise()
    elif cfg.gateway.port <= 0 or cfg.gateway.port > 65535:
        raise config.ConfigError(
            "Configuration errors:\n"
            f"  - [gateway] port={cfg.gateway.port} is out of range (1-65535)"
        )
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
