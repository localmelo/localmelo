"""Smoke playground web UI and API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from localmelo.melo.contracts.providers import ProviderError
from localmelo.support import config
from localmelo.support.config import Config, SchedulerConfig
from localmelo.support.gateway.chat_gate import call_log_dir
from localmelo.support.gateway.guard import app_instance
from localmelo.support.gateway.i18n import SETTINGS, page_language
from localmelo.support.gateway.machine_monitor import read_pmset
from localmelo.support.gateway.playground import LockHeldError, SmokePlayground
from localmelo.support.gateway_endpoint import GATEWAY_HEADER

_STATIC = Path(__file__).with_name("static")


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
            focus_s=scheduler.focus_s,
            call_log_dir=call_log_dir(scheduler),
            lock_path=scheduler.gpu_lock_file or None,
            lock_self=scheduler.lock_self,
            lock_gate=scheduler.lock_gate,
            # The gateway always reads the battery; the lock is optional.
            battery_runner=read_pmset,
        )
        app.state.smoke_playground = manager
    return manager


def _language_setting(app: FastAPI) -> str:
    """``[console] language`` as the console holds it now (Settings may change it); a bad value is ``auto``."""
    console = getattr(app.state, "console", None)
    cfg = getattr(console, "cfg", None) or getattr(app.state, "config", None)
    setting = cfg.console.language if isinstance(cfg, Config) else "auto"
    return setting if setting in SETTINGS else "auto"


def _page(request: Request, setting: str) -> str:
    """``index.html`` with its first guess at the reader's language on ``<html>``.

    ``data-language`` carries the setting for the page to apply. While the
    guess is not English, ``data-i18n-pending`` keeps the words the page will
    translate hidden (for a second at most), so they never flash in English.
    """
    language = page_language(setting, request.scope.get("headers", ()))
    pending = "" if language == "en" else " data-i18n-pending"
    return (
        (_STATIC / "index.html")
        .read_text(encoding="utf-8")
        .replace(
            '<html lang="en" data-language="auto">',
            f'<html lang="{language}" data-language="{setting}"{pending}>',
            1,
        )
    )


def _body_text(body: dict[str, Any], key: str, default: str = "") -> str:
    value = body.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _saved_endpoints(app: FastAPI, body: dict[str, Any]) -> dict[str, str]:
    if any(
        key in body
        for key in (
            "chat_url",
            "chat_model",
            "embedding_url",
            "embedding_model",
            "adapter",
            "backend",
            "base_url",
        )
    ):
        raise ValueError(
            "Playground endpoints and models must come from saved settings"
        )
    saved = config.load() if Path(config.config_path()).is_file() else app.state.config
    if not isinstance(saved, Config) or saved.chat_backend not in {
        "mlc",
        "ollama",
        "vllm",
        "sglang",
    }:
        raise ValueError("Playground requires a saved local chat backend")
    chat = getattr(saved, saved.chat_backend)
    embedding = getattr(saved, saved.embedding_backend, None)
    return {
        "chat_url": chat.chat_url,
        "chat_model": chat.chat_model,
        "embedding_url": getattr(embedding, "embedding_url", ""),
        "embedding_model": getattr(embedding, "embedding_model", ""),
        "adapter": saved.chat_backend,
    }


def mount(
    app: FastAPI, cfg: Config | None = None, *, enable_playground: bool = False
) -> None:
    """Mount the public shell and, when explicitly enabled, playground routes."""
    if getattr(app.state, "_localmelo_webapp_mounted", False):
        return
    app.state._localmelo_webapp_mounted = True
    app.state.config = cfg

    app.mount("/assets", StaticFiles(directory=_STATIC), name="workspace-assets")

    # The header identifies this app-started instance; it is not authentication.
    instance = app_instance()

    @app.get("/", response_class=HTMLResponse)
    async def chat_ui(request: Request) -> HTMLResponse:
        headers = (
            {"X-LocalMelo-Instance": instance, "Cache-Control": "no-store"}
            if instance
            else {}
        )
        # The id in the gateway's endpoint file: a reader confirms the file is live.
        gateway_id = getattr(app.state, "gateway_id", "")
        if gateway_id:
            headers = {**headers, GATEWAY_HEADER: gateway_id}
        return HTMLResponse(_page(request, _language_setting(app)), headers=headers)

    if not enable_playground:
        return

    from localmelo.support.gateway.playground import SessionBusyError

    _manager(app)

    @app.on_event("startup")
    async def _start_smoke_playground() -> None:
        await _manager(app).start()

    @app.on_event("shutdown")
    async def _close_smoke_playground() -> None:
        await _manager(app).close_all()

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
            endpoints = _saved_endpoints(app, body)
            payload = await _manager(app).discover(
                chat_url=endpoints["chat_url"],
                embedding_url=endpoints["embedding_url"],
                adapter=endpoints["adapter"],
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
                **_saved_endpoints(app, body),
                scenario_id=_body_text(body, "scenario_id", ""),
            )
            return JSONResponse(content=payload)
        except LockHeldError as e:
            return JSONResponse(status_code=409, content={"error": str(e)})
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
        manager = _manager(app)
        return JSONResponse(
            content={
                "sessions": manager.list_sessions(),
                "machine": manager.machine_payload(),
            }
        )

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
            return JSONResponse(
                status_code=409, content={"error": str(e), "code": "session_busy"}
            )
        if not closed:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return JSONResponse(content={"closed": session_id})
