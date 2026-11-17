"""The web console's shell: its page and static assets."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

from localmelo.support.config import Config
from localmelo.support.gateway.guard import app_instance
from localmelo.support.gateway.i18n import SETTINGS, page_language
from localmelo.support.gateway_endpoint import (
    BUILD_HEADER,
    BUSY_HEADER,
    GATEWAY_HEADER,
)

_STATIC = Path(__file__).with_name("static")


class _Assets(StaticFiles):
    """The page's own files, which a browser asks about on every use.

    Without a ``Cache-Control`` a browser may reuse a file it saved before an
    update, by its own guess at how long it stays fresh. ``no-cache`` keeps the
    saved copy but asks first; the ETag makes an unchanged file a 304.
    """

    def file_response(
        self,
        full_path: str | os.PathLike[str],
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = "no-cache"
        return response


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

    app.mount("/assets", _Assets(directory=_STATIC), name="workspace-assets")

    # The header identifies this app-started instance; it is not authentication.
    instance = app_instance()

    @app.get("/", response_class=HTMLResponse)
    async def chat_ui(request: Request) -> HTMLResponse:
        headers = (
            {"X-LocalMelo-Instance": instance, "Cache-Control": "no-store"}
            if instance
            else {"Cache-Control": "no-cache"}
        )
        # The id in the gateway's endpoint file: a reader confirms the file is live.
        gateway_id = getattr(app.state, "gateway_id", "")
        if gateway_id:
            headers = {**headers, GATEWAY_HEADER: gateway_id}
            # Which build this is and whether anything runs in it (see gateway_endpoint).
            if build := getattr(app.state, "gateway_build", ""):
                headers[BUILD_HEADER] = build
            if busy := getattr(app.state, "gateway_busy", None):
                headers[BUSY_HEADER] = "1" if await busy() else "0"
        return HTMLResponse(_page(request, _language_setting(app)), headers=headers)
