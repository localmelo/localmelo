"""Smoke playground web UI and API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from localmelo.melo.contracts.providers import ProviderError
from localmelo.support.config import Config, SchedulerConfig
from localmelo.support.gateway.chat_gate import DEFAULT_CALL_LOG_DIR
from localmelo.support.gateway.playground import SmokePlayground

_STATIC = Path(__file__).with_name("static")


def _call_log_dir(scheduler: SchedulerConfig) -> Path | None:
    """The configured call-log directory, or None when the file is off."""
    if not scheduler.call_log:
        return None
    if scheduler.call_log_dir:
        return Path(scheduler.call_log_dir).expanduser()
    return DEFAULT_CALL_LOG_DIR


def _manager(app: FastAPI) -> SmokePlayground:
    manager = getattr(app.state, "smoke_playground", None)
    if manager is None:
        cfg = getattr(app.state, "config", None)
        scheduler = cfg.scheduler if isinstance(cfg, Config) else SchedulerConfig()
        manager = SmokePlayground(
            policy=scheduler.policy,
            n_srv=scheduler.n_srv,
            r_f=scheduler.r_f,
            n_fg=scheduler.n_fg,
            call_log_dir=_call_log_dir(scheduler),
        )
        app.state.smoke_playground = manager
    return manager


def _body_text(body: dict[str, Any], key: str, default: str = "") -> str:
    value = body.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def mount(app: FastAPI, cfg: Config | None = None) -> None:
    """Register the smoke playground routes on the app."""
    if getattr(app.state, "_localmelo_webapp_mounted", False):
        return
    app.state._localmelo_webapp_mounted = True
    app.state.config = cfg

    from localmelo.support.gateway.playground import SessionBusyError

    _manager(app)
    app.mount("/assets", StaticFiles(directory=_STATIC), name="workspace-assets")

    @app.on_event("shutdown")
    async def _close_smoke_playground() -> None:
        await _manager(app).close_all()

    @app.get("/", response_class=HTMLResponse)
    async def chat_ui() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.post("/v1/playground/presence")
    async def playground_presence(body: dict[str, Any]) -> JSONResponse:
        session_id = body.get("session_id")
        if not isinstance(session_id, str):
            return JSONResponse(
                status_code=400, content={"error": "session_id must be a string"}
            )
        try:
            _manager(app).presence(session_id)
            return JSONResponse(content={"ok": True})
        except KeyError:
            return JSONResponse(status_code=404, content={"error": "session not found"})

    @app.post("/v1/playground/session/{session_id}/level")
    async def playground_level(session_id: str, body: dict[str, Any]) -> JSONResponse:
        level = body.get("level")
        if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 4:
            return JSONResponse(
                status_code=400,
                content={"error": "level must be an integer from 1 to 4"},
            )
        try:
            return JSONResponse(content=_manager(app).set_level(session_id, level))
        except KeyError:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    @app.get("/v1/playground/scenarios")
    async def playground_scenarios() -> JSONResponse:
        return JSONResponse(content={"scenarios": _manager(app).scenarios_payload()})

    @app.post("/v1/playground/discover")
    async def playground_discover(body: dict[str, Any]) -> JSONResponse:
        try:
            payload = await _manager(app).discover(
                chat_url=_body_text(body, "chat_url", ""),
                embedding_url=_body_text(body, "embedding_url", ""),
                adapter=_body_text(body, "adapter", "auto"),
            )
            return JSONResponse(content=payload)
        except SessionBusyError as e:
            return JSONResponse(status_code=409, content={"error": str(e)})
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except httpx.HTTPError as e:
            return JSONResponse(status_code=502, content={"error": str(e)})

    @app.post("/v1/playground/session")
    async def playground_session(body: dict[str, Any]) -> JSONResponse:
        try:
            payload = await _manager(app).create_session(
                chat_url=_body_text(body, "chat_url", ""),
                chat_model=_body_text(body, "chat_model", ""),
                embedding_url=_body_text(body, "embedding_url", ""),
                embedding_model=_body_text(body, "embedding_model", ""),
                adapter=_body_text(body, "adapter", "auto"),
                scenario_id=_body_text(body, "scenario_id", ""),
            )
            return JSONResponse(content=payload)
        except SessionBusyError as e:
            return JSONResponse(status_code=409, content={"error": str(e)})
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        # Seeding a scenario embeds its memories, and the embedding
        # provider reports a failed request as ProviderError.
        except (httpx.HTTPError, ProviderError) as e:
            return JSONResponse(status_code=502, content={"error": str(e)})

    @app.post("/v1/playground/session/{session_id}/run")
    async def playground_run(session_id: str, body: dict[str, Any]) -> JSONResponse:
        try:
            payload = await _manager(app).run(session_id, _body_text(body, "query", ""))
            return JSONResponse(content=payload)
        except KeyError:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        except SessionBusyError as e:
            return JSONResponse(status_code=409, content={"error": str(e)})
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except TimeoutError:
            return JSONResponse(
                status_code=504, content={"error": "Agent timed out (600s limit)"}
            )
        except httpx.HTTPError as e:
            return JSONResponse(status_code=502, content={"error": str(e)})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/v1/playground/sessions")
    async def playground_sessions() -> JSONResponse:
        return JSONResponse(content={"sessions": _manager(app).list_sessions()})

    @app.get("/v1/playground/session/{session_id}")
    async def playground_get(session_id: str) -> JSONResponse:
        try:
            return JSONResponse(content=_manager(app).get_session(session_id))
        except KeyError:
            return JSONResponse(status_code=404, content={"error": "session not found"})

    @app.post("/v1/playground/session/{session_id}/cancel")
    async def playground_cancel(session_id: str) -> JSONResponse:
        try:
            cancelled = await _manager(app).cancel(session_id)
            return JSONResponse(
                content={"cancelled": cancelled, "session_id": session_id}
            )
        except KeyError:
            return JSONResponse(status_code=404, content={"error": "session not found"})

    @app.delete("/v1/playground/session/{session_id}")
    async def playground_close(session_id: str) -> JSONResponse:
        try:
            closed = await _manager(app).close(session_id)
        except SessionBusyError as e:
            return JSONResponse(status_code=409, content={"error": str(e)})
        if not closed:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return JSONResponse(content={"closed": session_id})
