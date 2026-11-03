"""The live view of a chat's browser: ``GET /api/sessions/{id}/browser`` and the stream ``/browser/live``.

``GET`` says whether the chat has a page open in melo's browser (the current tab, its
address and title) and whether the live view is available, so the console can
open its Browser panel when melo starts browsing. It starts nothing.

``/browser/live`` is a WebSocket for one chat's browser (see
:mod:`localmelo.support.browser.live`). The server sends:

* text frames, ``{"type": "meta", ...}`` JSON: ``state`` (``idle``, ``loading``,
  ``live``, ``stale`` or ``closed``), ``tab``, ``url``, ``title``, ``browsing``
  (melo is in the middle of a browser step), ``owner`` (``melo``) and, once a
  picture came, ``w`` and ``h`` (the page view in CSS pixels), ``offsetTop`` and
  ``scale``; ``reason`` says why a state is ``closed``;
* binary frames, one JPEG each: only the newest picture, a viewer that is slow
  loses the pictures in between.

The client sends nothing yet that the server acts on. The optional ``tab=tN``
query pins the view to one tab; without it the view follows the tab melo works
in. Only a tab of this chat can be named: another chat's tab, an unknown
session and a chat with the live view turned off are refused before the
handshake completes. The stream runs only while somebody watches, and nothing
it carries is written to disk or to the session store.

Like every ``/api`` route it is behind the loopback guard (Host, a same-origin
Origin, and the bearer or the console's cookie), and it ends when the
gateway's token rotates.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

from localmelo.support.browser.live import LiveRefusedError, LiveView
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.sessions.store import SessionStoreError

from .routes_sessions import _require_session

if TYPE_CHECKING:
    from . import Console

logger = logging.getLogger(__name__)

# Close codes: normal, policy violation (a refusal), message too big, store down.
_NORMAL = 1000
_POLICY = 1008
_TOO_BIG = 1009
_RETRY = 1013
# What the client may send for now is small; more ends the stream.
MAX_CLIENT_MESSAGE = 4096
# Seconds a send to a viewer may take before the viewer is let go.
SEND_TIMEOUT_S = 15.0
_TAB = re.compile(r"t[0-9]{1,6}")


def live_of(console: Console, session_id: str) -> LiveView | None:
    """The live view of chat *session_id*'s browser, or None when the chat is not open or has no streaming browser."""
    entry = console.sessions.get(session_id)
    agent = entry.agent if entry is not None else None
    if agent is None:
        return None
    for provider in agent.tool_providers:
        if isinstance(provider, BrowserToolProvider):
            return provider.live
    return None


def available(console: Console) -> bool:
    """Whether a chat's browser can be watched: Chrome is the browser and the live view is on."""
    browser = console.cfg.browser
    return browser.driver == "chrome" and browser.live is True


def add_routes(app: FastAPI, console: Console) -> None:
    @app.get("/api/sessions/{session_id}/browser")
    async def browser_state(session_id: str) -> JSONResponse:
        console.require_store()
        await _require_session(console, session_id)
        live = live_of(console, session_id)
        state: dict[str, Any] = (
            await live.snapshot()
            if live is not None
            else {
                "open": False,
                "tab": None,
                "url": "",
                "title": "",
                "browsing": False,
                "viewers": 0,
            }
        )
        state["available"] = available(console)
        state["driver"] = console.cfg.browser.driver
        return JSONResponse(content=state, headers={"Cache-Control": "no-store"})

    @app.websocket("/api/sessions/{session_id}/browser/live")
    async def browser_live(
        websocket: WebSocket, session_id: str, tab: str | None = None
    ) -> None:
        if console.store_state() != "ok":
            await websocket.close(code=_RETRY)
            return
        if tab is not None and not _TAB.fullmatch(tab):
            await websocket.close(code=_POLICY, reason="not_a_tab")
            return
        try:
            known = await console.store.get_session(session_id)
        except SessionStoreError:
            await websocket.close(code=_RETRY)
            return
        if known is None:
            await websocket.close(code=_POLICY, reason="not_found")
            return
        if not available(console):
            await websocket.close(code=_POLICY, reason="not_available")
            return
        live = live_of(console, session_id)
        if live is None:
            # The chat has no streaming browser (yet): say so and end calmly.
            await websocket.accept()
            await websocket.send_text(
                json.dumps({"type": "meta", "state": "closed", "reason": "unavailable"})
            )
            await websocket.close(code=_NORMAL)
            return
        try:
            viewer = await live.watch(tab)
        except LiveRefusedError:
            await websocket.close(code=_POLICY, reason="refused")
            return
        try:
            await websocket.accept()
            await _serve(websocket, viewer)
        finally:
            await live.release(viewer)


async def _serve(websocket: WebSocket, viewer: Any) -> None:
    """Send the viewer's meta and pictures until it ends or the client goes."""
    sender = asyncio.ensure_future(_send(websocket, viewer))
    receiver = asyncio.ensure_future(_receive(websocket))
    try:
        await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (sender, receiver):
            task.cancel()
        await asyncio.gather(sender, receiver, return_exceptions=True)
    if websocket.application_state == WebSocketState.CONNECTED:
        with contextlib.suppress(Exception):
            await websocket.close(code=_NORMAL)


async def _send(websocket: WebSocket, viewer: Any) -> None:
    while True:
        item = await viewer.next()
        if item is None:
            return
        kind, payload = item
        try:
            async with asyncio.timeout(SEND_TIMEOUT_S):
                if kind == "meta":
                    await websocket.send_text(
                        json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
                    )
                else:
                    await websocket.send_bytes(payload.data)
        except (TimeoutError, RuntimeError, WebSocketDisconnect):
            return


async def _receive(websocket: WebSocket) -> None:
    """Read what the client sends until it goes; nothing it sends is acted on yet."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        size = len(message.get("text") or "") + len(message.get("bytes") or b"")
        if size > MAX_CLIENT_MESSAGE:
            with contextlib.suppress(Exception):
                await websocket.close(code=_TOO_BIG)
            return
