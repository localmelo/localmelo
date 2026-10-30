"""API v2 pictures: ``GET /api/sessions/{id}/pictures?path=…``.

The one route that serves a file of the folder a chat works in. It answers a
picture only: the real path has to be inside the session's folder, the name and
the content have to be a PNG, JPEG, GIF, WebP or SVG, and the file at most 20 MB
(:func:`~localmelo.support.sessions.pictures.read` holds every check). It writes
nothing and lists nothing. Every answer carries ``nosniff`` and a policy that
keeps an SVG from running as a page.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final

from fastapi import FastAPI, Query
from fastapi.responses import Response

from localmelo.support.sessions import pictures
from localmelo.support.sessions.store import SessionStoreError

from .errors import ApiError

if TYPE_CHECKING:
    from . import Console

_RASTER_POLICY: Final = "default-src 'none'; sandbox"
_SVG_POLICY: Final = "sandbox; default-src 'none'; style-src 'unsafe-inline'"


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the picture route on *app*."""

    @app.get("/api/sessions/{session_id}/pictures")
    async def session_picture(
        session_id: str, path: str = Query(..., min_length=1, max_length=4096)
    ) -> Response:
        console.require_store()
        try:
            info = await console.store.get_session(session_id)
        except SessionStoreError as exc:
            raise await console.store_api_error(exc, session_id) from exc
        if info is None:
            raise ApiError(404, "not_found", "Session not found.")
        if not info.workspace:
            raise ApiError(404, "not_found", "This chat has no folder yet.")
        try:
            picture = await asyncio.to_thread(pictures.read, info.workspace, path)
        except pictures.PictureError as exc:
            raise ApiError(exc.status, exc.code, exc.message) from None
        svg = picture.mime == "image/svg+xml"
        return Response(
            content=picture.data,
            media_type=picture.mime,
            headers={
                "Content-Disposition": "inline",
                "Content-Security-Policy": _SVG_POLICY if svg else _RASTER_POLICY,
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private",
            },
        )


__all__: list[str] = []
