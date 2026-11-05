"""The live view of a chat's browser: ``GET /api/sessions/{id}/browser`` and the stream ``/browser/live``.

``GET`` says whether the chat has a page open in melo's browser (the current tab, its
address and title), whether melo is in the middle of a browser step
(``browsing``), whether the person has one of its pages (``yours``), and whether
the live view is available, so the console can open its Browser panel when melo
starts browsing. It starts nothing. ``available`` is about the browser the chat
browses in now: an open chat's own (a chat whose turn ran when Settings changed
the browser keeps the one it has until its next turn), or, for a chat with no
browser open, the one the settings name; ``driver`` is the setting.

``/browser/live`` is a WebSocket for one chat's browser (see
:mod:`localmelo.support.browser.live`). The server sends:

* text frames, ``{"type": "meta", ...}`` JSON: ``state`` (``idle``, ``loading``,
  ``live``, ``stale`` or ``closed``), ``tab``, ``url``, ``title``, ``browsing``
  (melo is in the middle of a browser step), ``owner`` (``melo``) and, once a
  picture came, ``w`` and ``h`` (the page view in CSS pixels), ``offsetTop`` and
  ``scale``; ``reason`` says why a state is ``closed``;
* binary frames, one JPEG each: only the newest picture, a viewer that is slow
  loses the pictures in between.

The meta also says ``owner``: ``melo``, or ``you`` once the person took the tab
over (with ``grace``, the seconds the tab stays theirs once no panel shows it, and
``dialog`` for a box the page shows them); ``takeover`` says a viewer may take it.

The client sends JSON text messages of at most :data:`MAX_CLIENT_MESSAGE` bytes
(more ends the stream, 1009): ``take`` and ``give`` hand the tab to the person
and back; while they have it, ``mouse``, ``wheel``, ``key``, ``text``,
``compose`` and ``paste`` are their input and ``dialog`` their answer to a box
(see :mod:`localmelo.support.browser.takeover`). Anything else, and input while
melo has the tab, is dropped without a word. Nothing a message carries is
logged, stored or shown to a model. The optional ``tab=tN``
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

from localmelo.support.browser.live import LiveRefusedError, LiveView, Viewer
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.browser.takeover import INPUT_TYPES, quiet_frames
from localmelo.support.sessions.store import SessionStoreError

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


def available(console: Console, session_id: str) -> bool:
    """Whether chat *session_id*'s browser can be watched now.

    The live view must be on (turning it off holds at once). Then a chat with
    an Agent open is watched in the browser that Agent has, which streams or
    not; a chat with none open will get the browser the settings name:
    Chrome streams, Safari's windows do not.
    """
    browser = console.cfg.browser
    if browser.live is not True:
        return False
    entry = console.sessions.get(session_id)
    if entry is not None and entry.agent is not None:
        return live_of(console, session_id) is not None
    return browser.driver == "chrome"


def add_routes(app: FastAPI, console: Console) -> None:
    # The websocket server's debug line for each frame would show what the person types.
    quiet_frames()

    @app.get("/api/sessions/{session_id}/browser")
    async def browser_state(session_id: str) -> JSONResponse:
        console.require_store()
        await console.require_session(session_id)
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
                "yours": False,
                "viewers": 0,
            }
        )
        state["available"] = available(console, session_id)
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
        if not available(console, session_id):
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
            await _serve(websocket, viewer, live)
        finally:
            await live.release(viewer)


async def _serve(websocket: WebSocket, viewer: Viewer, live: LiveView) -> None:
    """Send the viewer's meta and pictures until it ends or the client goes; act on what the client sends."""
    sender = asyncio.ensure_future(_send(websocket, viewer))
    receiver = asyncio.ensure_future(_receive(websocket, viewer, live))
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


async def _receive(websocket: WebSocket, viewer: Viewer, live: LiveView) -> None:
    """Read what the client sends until it goes, and act on it (see the module docstring).

    Nothing a message carries is logged: a message that is not one the panel
    sends is dropped without a word, and so is input the live view does not take.
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        text = message.get("text")
        size = len((text or "").encode("utf-8")) + len(message.get("bytes") or b"")
        if size > MAX_CLIENT_MESSAGE:
            with contextlib.suppress(Exception):
                await websocket.close(code=_TOO_BIG)
            return
        if not text:
            continue
        try:
            said = json.loads(text)
        except ValueError:
            continue
        if not isinstance(said, dict):
            continue
        kind = said.get("type")
        try:
            if kind == "take":
                await live.take(viewer)
            elif kind == "give":
                await live.give(viewer)
            elif kind == "dialog":
                answer = said.get("text")
                await live.answer_dialog(
                    viewer,
                    said.get("accept") is True,
                    answer if isinstance(answer, str) else "",
                )
            elif kind in INPUT_TYPES:
                live.send_input(viewer, said)
        except (
            Exception
        ):  # noqa: BLE001 - one message never ends the view, and is never logged
            logger.warning("A message from the Browser panel could not be handled")
