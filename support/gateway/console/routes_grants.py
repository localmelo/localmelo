"""API v2 grants: every "allow ... for this chat" in one list, each with Revoke.

``GET /api/grants`` lists the live grants of the gateway's table in the
order the Settings page and ``/grants`` number them
(:func:`~localmelo.support.sessions.grants.listed`): what the person
allowed in the main chat counts everywhere and comes first, then each
chat's own. ``DELETE /api/grants/{id}`` takes one back; every chat's check
stops using it at once. Both sit behind the gateway's guard and token like
every other route. Money and account steps, typing, sending a form and an
Allow once are never grants, so they are never listed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Final

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from localmelo.support.gateway.i18n import t
from localmelo.support.sessions.grants import listed
from localmelo.support.sessions.store import MAIN_CHAT_TITLE

from .errors import ApiError

if TYPE_CHECKING:
    from . import Console

# A grant's id, as the table mints it.
_GRANT_ID: Final = re.compile(r"[0-9a-f]{12}")


async def grants_json(console: Console) -> list[dict[str, Any]]:
    """The live grants, listed, each with the title of the chat it was given in."""
    titles: dict[str, str | None] = {}
    rows: list[dict[str, Any]] = []
    for grant in listed(console.sessions.book.live()):
        if grant.session_id not in titles:
            info = await console.store.get_session(grant.session_id)
            titles[grant.session_id] = (
                None
                if info is None
                else MAIN_CHAT_TITLE if info.kind == "front" else info.title
            )
        title = titles[grant.session_id]
        if title is None:
            continue  # its chat went: the row went with it
        rows.append({**grant.to_json(), "chat_title": title})
    return rows


def add_routes(app: FastAPI, console: Console) -> None:
    """Register the grants routes."""

    @app.get("/api/grants")
    async def list_grants() -> JSONResponse:
        console.require_store()
        return JSONResponse(content={"grants": await grants_json(console)})

    @app.delete("/api/grants/{grant_id}")
    async def revoke_grant(grant_id: str) -> Response:
        """Take back one grant: the next step that needs it asks again."""
        console.require_store()
        if not _GRANT_ID.fullmatch(grant_id):
            raise ApiError(404, "not_found", t("errors.noGrant"))
        gone = await console.sessions.book.revoke(grant_id)
        if gone is None:
            raise ApiError(404, "not_found", t("errors.noGrant"))
        return Response(status_code=204)


__all__ = ["add_routes", "grants_json"]
