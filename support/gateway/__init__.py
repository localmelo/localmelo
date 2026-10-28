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
from localmelo.support.gateway.guard import LoopbackGuard, is_loopback
from localmelo.support.gateway.session import SessionBusyError, SessionManager

logger = logging.getLogger("localmelo.gateway")


# ── Gateway Server ──


class GatewayServer:
    def __init__(self, cfg: config.Config) -> None:
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

        self.app = FastAPI(title="localmelo gateway", lifespan=self._lifespan)
        self._register_routes()

        from localmelo.support.gateway.webapp import mount as mount_webapp

        mount_webapp(self.app, self.cfg)

        # The guard sees each request before routing, so it covers every
        # route, error responses and routes added later. Middleware added
        # after it would run before it: add middleware above this point.
        if is_loopback(self.host):
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
        # The web workspace's machine monitor starts before the gateway
        # reports ready, so a lock present at startup is observed at once.
        playground = getattr(app.state, "smoke_playground", None)
        if playground is not None:
            await playground.start()
        logger.info("gateway ready on %s:%d", self.host, self.port)
        try:
            yield
        finally:
            logger.info("gateway shutting down")
            await self.sessions.close_all()
            # FastAPI does not run on_event handlers when an explicit lifespan
            # is installed. Close the web workspace's providers here as well.
            if playground is not None:
                await playground.close_all()

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

    def run(self) -> None:
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")


def start_gateway(cfg: config.Config) -> None:
    cfg.validate_or_raise()
    server = GatewayServer(cfg)
    server.run()
