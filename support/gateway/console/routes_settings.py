"""``GET /api/health``: readiness, versions, limits and the store's state.

Health is served while the store is unavailable: the console's ``Chats are
unavailable`` banner needs it. The model server's reachability comes from
the cached probe and never makes the request wait for a network answer.
Settings, key status, model listing and the connection test are PR 4 and
belong in this module with health.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from localmelo.melo.checker.validators import MAX_QUERY_LEN
from localmelo.melo.contracts.events import EVENT_SCHEMA_VERSION
from localmelo.melo.schema import SESSION_TEXT_CHARS
from localmelo.support.backends.registry import get_backend
from localmelo.support.sessions import display
from localmelo.support.sessions.runner import DEFAULT_RESUME_MODE

if TYPE_CHECKING:
    from . import Console


def model_json(cfg: Any) -> dict[str, Any]:
    """The settings' chat model, with the registry's label for its backend."""
    backend = cfg.chat_backend
    section = getattr(cfg, backend, None) if backend else None
    name = getattr(section, "chat_model", "") if section is not None else ""
    try:
        label = get_backend(backend).display_name
    except KeyError:
        label = backend
    return {
        "backend": backend,
        "backend_label": label,
        "name": name,
        "embedding": cfg.embedding_backend or "none",
    }


def health_json(console: Console) -> dict[str, Any]:
    """The whole ``GET /api/health`` body."""
    entries = console.sessions.entries()
    return {
        "ok": True,
        "api": 2,
        "events_schema": EVENT_SCHEMA_VERSION,
        "version": display.localmelo_version(),
        "configured": console.cfg.is_configured,
        "config_errors": console.cfg.validate(),
        "model": model_json(console.cfg),
        "model_server": console.probe.snapshot(),
        "limits": {
            "text_chars": MAX_QUERY_LEN,
            "attachment_budget_chars": SESSION_TEXT_CHARS,
        },
        "sessions": {
            "open": len(entries),
            "running": sum(1 for entry in entries if entry.runner.busy),
            "pending_approvals": sum(
                len(console.broker.pending(entry.session_id)) for entry in entries
            ),
            "store": console.store_state(),
            "store_error": console.store_error(),
            "resume_mode": DEFAULT_RESUME_MODE,
        },
        "uptime_s": round(console.uptime_s(), 1),
    }


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the health route on *app*."""

    @app.get("/api/health")
    async def api_health() -> JSONResponse:
        return JSONResponse(content=health_json(console))
