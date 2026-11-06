"""The live view of a browser: ``GET /api/sessions/{id}/browser``, the stream ``/browser/live``, and the person's Done.

``{id}`` is the source of the browser: a chat, or a running task whose card a chat shows. ``GET`` says whether the
chat has a page open in melo's browser (the current tab, its
address and title), whether melo is in the middle of a browser step
(``browsing``), whether the person has one of its pages (``yours``, and
``place``: ``window`` or ``app``), and whether
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
over (with ``place``: ``window``, a real Chrome window this stream is a live copy
of, or ``app``, this panel; ``form`` and ``failed`` say why it is in the app; and
``dialog`` for a box the page shows them); ``takeover`` says a viewer may take it.

``POST /browser/done`` is the person's Done from anywhere (the app's pill, the
terminal, the console): every page they have comes back to melo, and the run
goes on. ``POST /browser/window`` moves their page to a real Chrome window, or
brings that window to the front; sent while a take-over card waits, its "Take
over" opens the window. Both answer ``{"ok": bool}``: false when the chat has no
page the person has (or no card waits).

**One source.** Every one of these routes asks :func:`resolve` which browser it is
for. A client that shows a chat's browser card names its source with the query
``gen`` (the browser tools' generation, ``browser_session_id`` of the card) and
``run`` (the card's run): a tab id such as ``t1`` means something only inside one
generation, so a card of an old Agent can never reach a new one, nor an ended run
a later one. A source that is not the chat's now is refused: ``409
stale_source`` on the HTTP routes, and a close with 1008 and the reason
``stale_source`` before the stream starts. Nothing falls back to the chat's
current browser. A running task's browser takes all three parts or nothing: ``gen``,
``run`` and ``owner``, the chat the card shows in, as the task's record says it now
(a card from a chat the task moved away from is refused like any stale source), so
Done, Show window and the stream reach that task's run from its card alone. A stream that names a ``run`` is for that run alone: when the run
ends, the chat begins another or its Agent is rebuilt, the gateway sends the meta
``closed`` with the reason ``browser`` and ends the stream (1000), and the stream
never carries a picture or takes a message for the next run. A client that names no
run (an older one, or the generation alone) keeps following the chat's browser, as
before. None of the routes starts a browser: a ``GET`` only reads what is open.

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
from localmelo.support.gateway.i18n import t
from localmelo.support.sessions.store import SessionStoreError

from .errors import ApiError

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
# What a source's parts look like (the generation is the browser tools' own, the run id the agent's).
_GENERATION = re.compile(r"[0-9a-f]{8}")
_RUN = re.compile(r"[A-Za-z0-9._:-]{1,80}")


class StaleSourceError(Exception):
    """A request names a source (a browser generation, a run) that the chat does not have now."""


def resolve(
    console: Console,
    session_id: str,
    *,
    gen: str | None = None,
    run: str | None = None,
    owner: str | None = None,
) -> BrowserToolProvider | None:
    """The browser tools a request is for: the chat's or the task's own, or None when it has none open.

    The one resolver of state, the stream, Done and Show window. A *gen* (the
    browser tools' generation), a *run* or an *owner* that does not fit raises
    :class:`StaleSourceError`: nothing falls back to whatever the chat has now.
    A running task's browser is reached only with all three. It starts nothing.
    """
    browser = browser_of(console, session_id)
    works_for = console.browser_cards.owner_of(session_id)
    if works_for is not None and (gen is None or run is None or owner is None):
        raise StaleSourceError
    if owner is not None and owner != (works_for or session_id):
        raise StaleSourceError
    if gen is not None and (
        browser is None
        or _GENERATION.fullmatch(gen) is None
        or browser.generation != gen
    ):
        raise StaleSourceError
    if run is not None and (
        _RUN.fullmatch(run) is None
        or not console.browser_cards.is_current(session_id, run)
    ):
        raise StaleSourceError
    return browser


def stale_source() -> ApiError:
    return ApiError(409, "stale_source", t("errors.staleBrowser"))


def browser_of(console: Console, session_id: str) -> BrowserToolProvider | None:
    """The browser tools of open chat or running task *session_id*, or None when it has none."""
    entry = console.sessions.get(session_id)
    agent = entry.agent if entry is not None else None
    if agent is None:
        return console.browser_cards.task_browser(session_id)
    for provider in agent.tool_providers:
        if isinstance(provider, BrowserToolProvider):
            return provider
    return None


def live_of(console: Console, session_id: str) -> LiveView | None:
    """The live view of chat *session_id*'s browser, or None when the chat is not open or has no streaming browser."""
    browser = browser_of(console, session_id)
    return browser.live if browser is not None else None


def available(console: Console, session_id: str) -> bool:
    """Whether the browser of *session_id*, or of the work it will start, can be watched now.

    The live view must be on (turning it off holds at once). Then a source with
    a browser open is watched in that browser, which streams or not; one with
    none open gets the browser the settings name: Chrome streams, Safari's
    windows do not.
    """
    browser = console.cfg.browser
    if browser.live is not True:
        return False
    found = browser_of(console, session_id)
    return found.live is not None if found is not None else browser.driver == "chrome"


def add_routes(app: FastAPI, console: Console) -> None:
    # The websocket server's debug line for each frame would show what the person types.
    quiet_frames()

    @app.get("/api/sessions/{session_id}/browser")
    async def browser_state(
        session_id: str,
        gen: str | None = None,
        run: str | None = None,
        owner: str | None = None,
    ) -> JSONResponse:
        console.require_store()
        await console.require_session(session_id)
        try:
            resolved = resolve(console, session_id, gen=gen, run=run, owner=owner)
        except StaleSourceError:
            raise stale_source() from None
        live = resolved.live if resolved is not None else None
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
                "place": None,
                "viewers": 0,
            }
        )
        state["available"] = available(console, session_id)
        state["driver"] = console.cfg.browser.driver
        return JSONResponse(content=state, headers={"Cache-Control": "no-store"})

    @app.post("/api/sessions/{session_id}/browser/done")
    async def browser_done(
        session_id: str,
        gen: str | None = None,
        run: str | None = None,
        owner: str | None = None,
    ) -> JSONResponse:
        console.require_store()
        await console.require_session(session_id)
        try:
            browser = resolve(console, session_id, gen=gen, run=run, owner=owner)
        except StaleSourceError:
            raise stale_source() from None
        return JSONResponse({"ok": browser is not None and browser.hand_back()})

    @app.post("/api/sessions/{session_id}/browser/window")
    async def browser_window(
        session_id: str,
        gen: str | None = None,
        run: str | None = None,
        owner: str | None = None,
    ) -> JSONResponse:
        console.require_store()
        await console.require_session(session_id)
        try:
            browser = resolve(console, session_id, gen=gen, run=run, owner=owner)
        except StaleSourceError:
            raise stale_source() from None
        return JSONResponse({"ok": browser is not None and browser.show_to_person()})

    @app.websocket("/api/sessions/{session_id}/browser/live")
    async def browser_live(
        websocket: WebSocket,
        session_id: str,
        tab: str | None = None,
        gen: str | None = None,
        run: str | None = None,
        owner: str | None = None,
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
        try:
            resolved = resolve(console, session_id, gen=gen, run=run, owner=owner)
        except StaleSourceError:
            await websocket.close(code=_POLICY, reason="stale_source")
            return
        # A stream for a run ends with it: pinned before anything is awaited, so no end goes unheard.
        ends = console.browser_cards.pin(session_id, run) if run is not None else None
        live = resolved.live if resolved is not None else None
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
            await _serve(websocket, viewer, live, ends)
        finally:
            await live.release(viewer)


async def _serve(
    websocket: WebSocket,
    viewer: Viewer,
    live: LiveView,
    ends: asyncio.Event | None = None,
) -> None:
    """Send the viewer's meta and pictures until it ends or the client goes; act on what the client sends.

    *ends* is set when the run the stream is for is over: the stream then says so and ends, and serves nothing more.
    """
    sender = asyncio.ensure_future(_send(websocket, viewer, ends))
    receiver = asyncio.ensure_future(_receive(websocket, viewer, live, ends))
    tasks: set[asyncio.Future[Any]] = {sender, receiver}
    if ends is not None:
        tasks.add(asyncio.ensure_future(ends.wait()))
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    if websocket.application_state == WebSocketState.CONNECTED:
        with contextlib.suppress(Exception):
            if ends is not None and ends.is_set():
                async with asyncio.timeout(SEND_TIMEOUT_S):
                    await websocket.send_text(
                        json.dumps(
                            {"type": "meta", "state": "closed", "reason": "browser"}
                        )
                    )
            await websocket.close(code=_NORMAL)


async def _send(
    websocket: WebSocket, viewer: Any, ends: asyncio.Event | None = None
) -> None:
    while True:
        item = await viewer.next()
        if item is None or (ends is not None and ends.is_set()):
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


async def _receive(
    websocket: WebSocket,
    viewer: Viewer,
    live: LiveView,
    ends: asyncio.Event | None = None,
) -> None:
    """Read what the client sends until it goes, and act on it (see the module docstring).

    Nothing a message carries is logged: a message that is not one the panel
    sends is dropped without a word, and so is input the live view does not take.
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect" or (
            ends is not None and ends.is_set()
        ):
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
