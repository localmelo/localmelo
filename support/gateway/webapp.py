"""The web console's shell: its page and static assets."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from localmelo.support.config import Config
from localmelo.support.gateway.guard import app_instance
from localmelo.support.gateway.i18n import SETTINGS, page_language
from localmelo.support.gateway_endpoint import GATEWAY_HEADER

_STATIC = Path(__file__).with_name("static")


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


def mount(app: FastAPI, cfg: Config | None = None) -> None:
    """Mount the page and its static assets."""
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
