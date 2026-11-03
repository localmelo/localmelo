"""The live view: what melo's background browser shows, for the person watching.

A driver that can stream (:class:`Screencaster`, as :mod:`~localmelo.support.browser.chrome`
does) turns a tab's page into JPEG pictures, :class:`Frame`. :class:`LiveView`
is the relay between that stream and the people watching one chat's browser:

* **Only while watched.** The stream of a tab runs while at least one
  :class:`Viewer` looks at it, and stops when the last one goes. A page that
  does not change sends nothing, so a watched still page costs nothing either.
* **Newest picture only.** Each viewer holds one slot. A picture that arrives
  before the last one was sent replaces it, so a slow viewer sees the present
  and never a queue of the past, and never slows another viewer or the browser.
* **In memory only.** The last picture of a watched tab is kept for a viewer who
  joins late, and dropped when the stream stops. No picture is written to disk
  or to the session store, and none reaches the model (``browser_screenshot``
  pictures are a separate path).
* **Follows the tab melo works in.** A viewer that names no tab follows the chat's
  current tab; one that names a tab watches only that tab, and only a tab this chat
  opened can be named (:class:`LiveRefusedError` otherwise).

What a viewer is told besides pictures is a *meta* message: the tab, its address
and title, the size of the page the picture shows, whether melo is in the middle
of a browser step, who has the browser (``melo``), and a calm state:

``idle``
    The chat has no page open.
``loading``
    A page is open and its first picture has not come yet.
``live``
    Pictures come as the page changes.
``stale``
    The stream of the tab could not be started: the last picture is kept but is not
    updating.
``closed``
    The tab, the browser or the chat is gone (``reason`` says which).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from localmelo.support.browser.driver import BrowserError, TabInfo

logger = logging.getLogger(__name__)

# The rate the pictures are held to while melo has the browser, a setting.
DEFAULT_FPS = 15.0
MIN_FPS = 1.0
MAX_FPS = 60.0
# JPEG quality of a picture, and the most viewers of one chat's browser.
DEFAULT_QUALITY = 70
MAX_VIEWERS = 4
# Seconds between looks at the address and title of the watched tabs.
POLL_S = 1.0

State = Literal["idle", "loading", "live", "stale", "closed"]


class LiveRefusedError(Exception):
    """A viewer that may not watch: a tab this chat does not have, or too many viewers."""


@dataclass(frozen=True)
class Frame:
    """One picture of a tab's page.

    ``width`` and ``height`` are the size of the page view in CSS pixels, not of
    the JPEG: a point of the picture maps to a point of the page by the ratio of
    the two, and ``offset_top`` and ``scale`` are the browser's own words for a
    bar above the page and the page's zoom.
    """

    data: bytes
    width: float = 0.0
    height: float = 0.0
    offset_top: float = 0.0
    scale: float = 1.0
    scroll_x: float = 0.0
    scroll_y: float = 0.0

    @property
    def geometry(self) -> tuple[float, float, float, float]:
        return (self.width, self.height, self.offset_top, self.scale)


class Screencaster(Protocol):
    """A driver that can stream the pictures of a tab."""

    async def start_screencast(
        self,
        tab: str,
        on_frame: Callable[[Frame], None],
        on_end: Callable[[], None],
        *,
        fps: float = DEFAULT_FPS,
        quality: int = DEFAULT_QUALITY,
    ) -> None: ...

    async def stop_screencast(self, tab: str) -> None: ...

    async def tabs(self) -> list[TabInfo]: ...


class LiveHost(Protocol):
    """What the relay asks of the browser tools it serves: the chat's tabs and what melo is doing."""

    def live_current(self) -> str | None:
        """The short id of the tab melo works in, or None."""
        ...

    def live_raw(self, short: str) -> str | None:
        """The driver's id of the chat's tab *short*, or None when the chat has no such tab."""
        ...

    def live_busy(self) -> bool:
        """Whether melo is in the middle of a browser step."""
        ...


class Viewer:
    """One person's view of a chat's browser: one slot for the newest picture and one for the newest meta.

    :meth:`next` gives the meta first when there is one, then the picture. A
    meta is dropped for a newer one the same way a picture is.
    """

    def __init__(self, pinned: str | None = None) -> None:
        self.pinned = pinned
        # The tab it watches now (short id, driver id), set by the relay.
        self.target: tuple[str | None, str | None] = (None, None)
        self.meta: dict[str, Any] = {}
        self.replaced = 0
        self.sent = 0
        self.closed = False
        self.offered: Frame | None = None
        self._frame: Frame | None = None
        self._queued: dict[str, Any] | None = None
        self._wake = asyncio.Event()

    def offer(self, frame: Frame) -> None:
        """Put *frame* in the slot, in place of a picture that was not sent."""
        if self._frame is not None:
            self.replaced += 1
        self.offered = self._frame = frame
        self._wake.set()

    def tell(self, meta: dict[str, Any]) -> bool:
        """Queue *meta* when it differs from the last one told; whether it did."""
        if meta == self.meta:
            return False
        self.meta = meta
        self._queued = meta
        self._wake.set()
        return True

    def close(self) -> None:
        self.closed = True
        self._wake.set()

    async def next(self) -> tuple[Literal["meta", "frame"], Any] | None:
        """The next thing to send, or None once the viewer is closed and everything told was sent."""
        while True:
            if self._queued is not None:
                meta, self._queued = self._queued, None
                return "meta", meta
            if self.closed:
                return None
            if self._frame is not None:
                frame, self._frame = self._frame, None
                self.sent += 1
                return "frame", frame
            self._wake.clear()
            await self._wake.wait()


class _TabCast:
    """The stream of one tab: the newest picture, and what the driver said of it."""

    def __init__(self) -> None:
        self.frame: Frame | None = None
        self.geometry: tuple[float, float, float, float] | None = None
        self.failed = False
        self.ended = False


class LiveView:
    """The relay between one chat's browser and the people watching it. See the module docstring.

    *driver* streams (:class:`Screencaster`); *host* answers for the chat's tabs;
    *fps* holds the stream to that many pictures a second (1 to 60) at JPEG *quality*.
    """

    def __init__(
        self,
        driver: Screencaster,
        host: LiveHost,
        *,
        fps: float = DEFAULT_FPS,
        quality: int = DEFAULT_QUALITY,
        poll_s: float = POLL_S,
        max_viewers: int = MAX_VIEWERS,
    ) -> None:
        self._driver = driver
        self._host = host
        self.fps = min(max(float(fps), MIN_FPS), MAX_FPS)
        self.quality = min(max(int(quality), 10), 95)
        self._poll_s = poll_s
        self._max = max_viewers
        self._viewers: list[Viewer] = []
        self._casts: dict[str, _TabCast] = {}
        self._info: dict[str, TabInfo] = {}
        self._gone: set[str] = set()
        self._stopped = ""  # why the browser is gone, when it is
        self._lock = asyncio.Lock()
        self._ticker: asyncio.Task[None] | None = None
        self._syncing: asyncio.Task[None] | None = None
        self._again = False
        self._closed = False

    # ── Reading ──

    @property
    def viewers(self) -> int:
        return len(self._viewers)

    @property
    def streaming(self) -> int:
        """How many tabs stream now."""
        return len(self._casts)

    async def snapshot(self) -> dict[str, Any]:
        """Whether the chat has a page open and which: the current tab, its address and title.

        Nothing streams for this; the address and title are asked of the browser.
        """
        short = self._host.live_current()
        raw = self._host.live_raw(short) if short else None
        url = title = ""
        if raw:
            with contextlib.suppress(BrowserError):
                for tab in await self._driver.tabs():
                    if tab.id == raw:
                        url, title = tab.url, tab.title
                        break
                else:
                    raw = None
        return {
            "open": raw is not None,
            "tab": short if raw else None,
            "url": url,
            "title": title,
            "browsing": bool(self._host.live_busy()),
            "viewers": len(self._viewers),
        }

    # ── Viewers ──

    async def watch(self, tab: str | None = None) -> Viewer:
        """A viewer of tab *tab*, or of the tab melo works in when None.

        :class:`LiveRefusedError` for a tab this chat does not have or when too many
        watch. The viewer is told its first meta before this returns.
        """
        if self._closed:
            raise LiveRefusedError("The chat's browser is closed.")
        if tab is not None and self._host.live_raw(tab) is None:
            raise LiveRefusedError("That is not a tab of this chat.")
        if len(self._viewers) >= self._max:
            raise LiveRefusedError("Too many are watching this chat's browser.")
        viewer = Viewer(tab)
        self._viewers.append(viewer)
        self._set_target(viewer)
        self._refresh(viewer)
        if self._ticker is None:
            self._ticker = asyncio.ensure_future(self._tick())
        await self._sync()
        return viewer

    async def release(self, viewer: Viewer) -> None:
        """*viewer* looks no more; the last one to go stops the stream."""
        viewer.close()
        if viewer in self._viewers:
            self._viewers.remove(viewer)
        if not self._viewers:
            await self._stop_ticker()
        await self._sync()

    async def aclose(self) -> None:
        """The chat's browser is over: every viewer is told, and every stream stops."""
        if self._closed:
            return
        self._closed = True
        self._stopped = "chat"
        await self._stop_ticker()
        for viewer in list(self._viewers):
            self._refresh(viewer)
            viewer.close()
        self._viewers.clear()
        await self._sync()

    def poke(self) -> None:
        """The chat's tabs or what melo does changed: look again soon (never blocks)."""
        if not self._viewers and not self._casts:
            return
        if self._syncing is not None and not self._syncing.done():
            self._again = True
            return
        with contextlib.suppress(RuntimeError):
            self._syncing = asyncio.ensure_future(self._run_sync())

    # ── The relay ──

    async def _run_sync(self) -> None:
        try:
            while True:
                self._again = False
                await self._sync()
                if not self._again:
                    return
        except Exception:  # a failed look never ends the relay
            logger.warning("The live view could not follow a change", exc_info=True)

    def _set_target(self, viewer: Viewer) -> None:
        short = (
            viewer.pinned if viewer.pinned is not None else self._host.live_current()
        )
        raw = self._host.live_raw(short) if short else None
        viewer.target = (short if raw else None, raw)

    async def _sync(self) -> None:
        """Make the streams the ones the viewers need and tell the viewers what changed."""
        async with self._lock:
            for viewer in self._viewers:
                self._set_target(viewer)
            await self._look()
            need = {
                raw
                for v in self._viewers
                if (raw := v.target[1]) and raw not in self._gone
            }
            for raw in [r for r in self._casts if r not in need]:
                await self._stop(raw)
            for raw in sorted(need):
                cast = self._casts.get(raw)
                if cast is None or cast.failed:
                    await self._start(raw)
            for viewer in self._viewers:
                self._refresh(viewer)
                cast = self._casts.get(viewer.target[1] or "")
                if (
                    cast is not None
                    and cast.frame is not None
                    and cast.frame is not viewer.offered
                ):
                    viewer.offer(cast.frame)

    async def _start(self, raw: str) -> None:
        cast = self._casts.get(raw) or _TabCast()
        cast.failed = False
        self._casts[raw] = cast

        def on_frame(frame: Frame) -> None:
            self._on_frame(raw, frame)

        def on_end() -> None:
            self._on_end(raw)

        try:
            await self._driver.start_screencast(
                raw, on_frame, on_end, fps=self.fps, quality=self.quality
            )
        except BrowserError as exc:
            logger.info("The live view could not start a tab's stream: %s", exc)
            cast.failed = True

    async def _stop(self, raw: str) -> None:
        # The picture goes with the stream: a viewer who comes later gets a new one.
        self._casts.pop(raw, None)
        self._info.pop(raw, None)
        with contextlib.suppress(BrowserError):
            await self._driver.stop_screencast(raw)

    def _on_frame(self, raw: str, frame: Frame) -> None:
        cast = self._casts.get(raw)
        if cast is None:
            return
        first = cast.frame is None
        moved = cast.geometry != frame.geometry
        cast.frame, cast.geometry = frame, frame.geometry
        for viewer in self._viewers:
            if viewer.target[1] != raw:
                continue
            if first or moved:
                self._refresh(viewer)
            viewer.offer(frame)

    def _on_end(self, raw: str) -> None:
        cast = self._casts.get(raw)
        if cast is not None:
            cast.ended = True
            cast.failed = True
        self.poke()

    async def _tick(self) -> None:
        """Look at the watched tabs' addresses and titles while somebody watches."""
        try:
            while True:
                await asyncio.sleep(self._poll_s)
                await self._sync()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("The live view stopped looking at the tabs", exc_info=True)

    async def _stop_ticker(self) -> None:
        ticker, self._ticker = self._ticker, None
        if ticker is not None and ticker is not asyncio.current_task():
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)

    async def _look(self) -> None:
        """Ask the browser for the watched tabs' addresses and titles, and which are gone."""
        raws = {v.target[1] for v in self._viewers if v.target[1]}
        if not raws:
            self._stopped = "chat" if self._closed else ""
            self._gone.clear()
            return
        try:
            listed = {tab.id: tab for tab in await self._driver.tabs()}
        except BrowserError:
            self._stopped = "browser"
            return
        self._stopped = ""
        self._gone = {raw for raw in raws if raw not in listed}
        for raw in raws & listed.keys():
            self._info[raw] = listed[raw]

    # ── What a viewer is told ──

    def _state(self, raw: str | None) -> tuple[State, str]:
        if self._stopped:
            return "closed", self._stopped
        if raw is None:
            return "idle", ""
        if raw in self._gone:
            return "closed", "tab"
        cast = self._casts.get(raw)
        if cast is None:
            return "loading", ""
        if cast.failed:
            return "stale", ""
        if cast.frame is None:
            return "loading", ""
        return "live", ""

    def _meta(self, viewer: Viewer | None, short: str | None) -> dict[str, Any]:
        raw = (
            viewer.target[1] if viewer is not None else self._host.live_raw(short or "")
        )
        state, reason = self._state(raw)
        if viewer is not None and viewer.pinned is not None and raw is None:
            state, reason = "closed", self._stopped or "tab"
        tab = self._info.get(raw or "")
        cast = self._casts.get(raw or "")
        meta: dict[str, Any] = {
            "type": "meta",
            "state": state,
            "tab": short if raw else None,
            "url": tab.url if tab else "",
            "title": tab.title if tab else "",
            "browsing": bool(self._host.live_busy()),
            "owner": "melo",
        }
        if reason:
            meta["reason"] = reason
        if cast is not None and cast.geometry is not None:
            width, height, offset_top, scale = cast.geometry
            meta.update(w=width, h=height, offsetTop=offset_top, scale=scale)
        return meta

    def _refresh(self, viewer: Viewer) -> None:
        viewer.tell(self._meta(viewer, viewer.target[0]))
