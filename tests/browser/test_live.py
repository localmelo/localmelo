"""The live view: the stream of a tab's pictures, and the relay that serves the people watching.

Three layers, none with a real browser. The relay (``LiveView``) over a driver
and a host written here; the relay under the browser tools (``BrowserToolProvider``)
over a scripted driver that streams; and the Chrome driver's screencast over the
fake Chrome peer (acknowledgements held to a rate, stop, end, bad pictures).
"""

from __future__ import annotations

import asyncio
import base64
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.browser import chrome
from localmelo.support.browser.chrome import ChromeBackgroundDriver
from localmelo.support.browser.driver import BrowserError, TabInfo
from localmelo.support.browser.live import Frame, LiveRefusedError, LiveView
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.browser.takeover import Owners
from localmelo.tests.browser.fakes import FakeDriver, link, page
from localmelo.tests.browser.test_chrome import FAKE, Peer
from localmelo.tests.browser.test_provider import run

HOME = "https://example.com/"


def picture(n: int, w: float = 1280.0, h: float = 713.0) -> Frame:
    return Frame(b"\xff\xd8" + str(n).encode(), width=w, height=h)


async def settle(times: int = 6) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


async def next_of(viewer: Any, kind: str | None = None) -> Any:
    """The next thing a viewer is to send (of *kind*, skipping others), at most a second."""
    async with asyncio.timeout(1):
        while True:
            item = await viewer.next()
            assert item is not None
            if kind is None or item[0] == kind:
                return item


# ── The relay, with a driver and a host of its own ──


class Stream:
    """A driver that streams: records starts and stops, and lets a test push pictures."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.live: dict[str, tuple[Callable[[Frame], None], Callable[[], None]]] = {}
        self.listed: dict[str, TabInfo] = {}
        self.refuse = False
        self.tabs_fail = False

    async def start_screencast(
        self,
        tab: str,
        on_frame: Callable[[Frame], None],
        on_end: Callable[[], None],
        *,
        fps: float = 15.0,
        quality: int = 70,
    ) -> None:
        if self.refuse:
            raise BrowserError("no stream")
        self.started.append(tab)
        self.live[tab] = (on_frame, on_end)

    async def stop_screencast(self, tab: str) -> None:
        self.stopped.append(tab)
        self.live.pop(tab, None)

    async def tabs(self) -> list[TabInfo]:
        if self.tabs_fail:
            raise BrowserError("Chrome stopped")
        return list(self.listed.values())

    def push(self, tab: str, frame: Frame) -> None:
        self.live[tab][0](frame)


class Host:
    """A chat's tabs: short ids to the driver's."""

    def __init__(self, driver: Stream) -> None:
        self.driver = driver
        self.tabs: dict[str, str] = {}
        self.current: str | None = None
        self.busy = False

    def add(self, short: str, raw: str, url: str = HOME, title: str = "Home") -> None:
        self.tabs[short] = raw
        self.driver.listed[raw] = TabInfo(raw, url, title)
        self.current = short

    def live_current(self) -> str | None:
        return self.current if self.current in self.tabs else None

    def live_raw(self, short: str) -> str | None:
        return self.tabs.get(short)

    def live_busy(self) -> bool:
        return self.busy


@pytest.fixture
def stream() -> Stream:
    return Stream()


@pytest.fixture
def host(stream: Stream) -> Host:
    return Host(stream)


@pytest.fixture
async def view(stream: Stream, host: Host) -> AsyncIterator[LiveView]:
    found = LiveView(stream, host, poll_s=3600)
    yield found
    await found.aclose()


async def test_a_viewer_with_no_page_open_sees_idle_and_nothing_streams(
    view: LiveView, stream: Stream
) -> None:
    viewer = await view.watch()
    _, meta = await next_of(viewer, "meta")
    assert meta["state"] == "idle" and meta["tab"] is None and meta["owner"] == "melo"
    assert stream.started == [] and view.streaming == 0


async def test_a_page_with_no_picture_yet_is_loading_and_then_live(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1", HOME, "Home")
    viewer = await view.watch()
    _, meta = await next_of(viewer, "meta")
    assert (meta["state"], meta["tab"], meta["url"], meta["title"]) == (
        "loading",
        "t1",
        HOME,
        "Home",
    )
    assert "w" not in meta
    stream.push("R1", picture(1))
    _, told = await next_of(viewer, "meta")
    assert (told["state"], told["w"], told["h"], told["scale"]) == (
        "live",
        1280.0,
        713.0,
        1.0,
    )
    kind, frame = await next_of(viewer)
    assert kind == "frame" and frame.data == b"\xff\xd81"


async def test_only_the_newest_picture_is_kept_for_a_viewer_that_is_slow(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1")
    viewer = await view.watch()
    await next_of(viewer, "meta")
    for n in range(1, 21):
        stream.push("R1", picture(n))
    await next_of(viewer, "meta")  # the first picture changed what the meta says
    kind, frame = await next_of(viewer)
    assert kind == "frame" and frame.data == b"\xff\xd820"
    assert viewer.replaced == 19 and viewer.sent == 1


async def test_a_slow_viewer_does_not_hold_back_another(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1")
    slow, quick = await view.watch(), await view.watch()
    for n in range(1, 6):
        stream.push("R1", picture(n))
        _, frame = await next_of(quick, "frame")
        assert frame.data == b"\xff\xd8" + str(n).encode()
    assert slow.replaced == 4


async def test_the_stream_runs_while_somebody_watches_and_stops_with_the_last_viewer(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1")
    first = await view.watch()
    second = await view.watch()
    assert stream.started == ["R1"] and view.viewers == 2
    await view.release(first)
    assert stream.stopped == [] and view.streaming == 1
    await view.release(second)
    assert stream.stopped == ["R1"] and view.streaming == 0 and view.viewers == 0


async def test_a_viewer_who_joins_late_gets_the_last_picture_and_a_new_stream_starts_fresh(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1")
    first = await view.watch()
    stream.push("R1", picture(7))
    late = await view.watch()
    _, frame = await next_of(late, "frame")
    assert frame.data == b"\xff\xd87"
    await view.release(first)
    await view.release(late)
    # Nothing of the old stream is kept: the next viewer sees no picture until a new one comes.
    again = await view.watch()
    assert stream.started == ["R1", "R1"]
    await settle()
    assert again.offered is None


async def test_a_viewer_follows_the_tab_melo_works_in(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1", HOME, "Home")
    viewer = await view.watch()
    stream.push("R1", picture(1))
    host.add("t2", "R2", "https://example.org/", "Other")
    view.poke()
    async with asyncio.timeout(1):
        while "R2" not in stream.started:
            await asyncio.sleep(0.01)
    assert stream.stopped == ["R1"]
    _, meta = await next_of(viewer, "meta")
    while meta["tab"] != "t2":
        _, meta = await next_of(viewer, "meta")
    assert (meta["url"], meta["title"], meta["state"]) == (
        "https://example.org/",
        "Other",
        "loading",
    )


async def test_a_viewer_pinned_to_a_tab_stays_on_it(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1")
    host.add("t2", "R2")
    viewer = await view.watch("t1")
    assert viewer.target == ("t1", "R1") and stream.started == ["R1"]
    host.current = "t2"
    view.poke()
    await asyncio.sleep(0.05)
    assert viewer.target == ("t1", "R1") and stream.started == ["R1"]


async def test_a_tab_the_chat_does_not_have_is_refused(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1")
    with pytest.raises(LiveRefusedError):
        await view.watch("t2")
    assert view.viewers == 0 and stream.started == []


async def test_too_many_viewers_are_refused(stream: Stream, host: Host) -> None:
    host.add("t1", "R1")
    view = LiveView(stream, host, poll_s=3600, max_viewers=2)
    try:
        await view.watch()
        await view.watch()
        with pytest.raises(LiveRefusedError):
            await view.watch()
    finally:
        await view.aclose()


async def test_a_stream_that_cannot_start_is_stale_and_starts_again_when_it_can(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1")
    stream.refuse = True
    viewer = await view.watch()
    assert viewer.meta["state"] == "stale"
    stream.refuse = False
    view.poke()
    async with asyncio.timeout(1):
        while "R1" not in stream.started:
            await asyncio.sleep(0.01)
    await settle()
    assert viewer.meta["state"] == "loading"


async def test_a_tab_that_goes_and_a_browser_that_stops_are_closed(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1")
    viewer = await view.watch()
    stream.listed.pop("R1")
    view.poke()
    async with asyncio.timeout(1):
        while viewer.meta["state"] != "closed":
            await asyncio.sleep(0.01)
    assert viewer.meta["reason"] == "tab"
    stream.listed["R1"] = TabInfo("R1", HOME, "Home")
    stream.tabs_fail = True
    view.poke()
    async with asyncio.timeout(1):
        while viewer.meta.get("reason") != "browser":
            await asyncio.sleep(0.01)
    stream.tabs_fail = False
    view.poke()
    async with asyncio.timeout(1):
        while viewer.meta["state"] == "closed":
            await asyncio.sleep(0.01)


async def test_a_pinned_viewer_is_closed_when_its_tab_is_forgotten(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1")
    viewer = await view.watch("t1")
    host.tabs.pop("t1")
    view.poke()
    async with asyncio.timeout(1):
        while viewer.meta["state"] != "closed":
            await asyncio.sleep(0.01)
    assert viewer.meta["reason"] == "tab"


async def test_the_stream_ending_is_told_to_the_viewer_as_stale(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1")
    viewer = await view.watch()
    stream.push("R1", picture(1))
    stream.live["R1"][1]()  # the browser says the stream ended
    stream.refuse = True
    async with asyncio.timeout(1):
        while viewer.meta["state"] != "stale":
            await asyncio.sleep(0.01)


async def test_a_meta_is_sent_only_when_it_changes(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1")
    viewer = await view.watch()
    stream.push("R1", picture(1))
    metas = 0
    await next_of(viewer, "frame")
    for n in range(2, 12):
        stream.push("R1", picture(n))  # the same size: nothing new to tell
    view.poke()
    await asyncio.sleep(0.05)
    while viewer._queued is not None:  # noqa: SLF001
        await viewer.next()
        metas += 1
    assert metas == 0
    stream.push("R1", picture(12, 1000.0, 600.0))  # the page view changed size
    _, meta = await next_of(viewer, "meta")
    assert (meta["w"], meta["h"]) == (1000.0, 600.0)


async def test_busy_says_melo_is_in_the_middle_of_a_browser_step(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1")
    viewer = await view.watch()
    assert viewer.meta["browsing"] is False
    host.busy = True
    view.poke()
    async with asyncio.timeout(1):
        while viewer.meta["browsing"] is not True:
            await asyncio.sleep(0.01)


async def test_closing_the_view_tells_every_viewer_and_stops_the_stream(
    view: LiveView, stream: Stream, host: Host
) -> None:
    host.add("t1", "R1")
    viewer = await view.watch()
    await view.aclose()
    kind, meta = await next_of(viewer)
    assert kind == "meta" and meta["state"] == "closed" and meta["reason"] == "chat"
    assert await viewer.next() is None
    assert stream.stopped == ["R1"]
    with pytest.raises(LiveRefusedError):
        await view.watch()


async def test_the_snapshot_says_which_page_is_open_and_streams_nothing(
    view: LiveView, stream: Stream, host: Host
) -> None:
    assert (await view.snapshot())["open"] is False
    host.add("t1", "R1", HOME, "Home")
    snap = await view.snapshot()
    assert (snap["open"], snap["tab"], snap["url"], snap["title"]) == (
        True,
        "t1",
        HOME,
        "Home",
    )
    assert snap["viewers"] == 0 and stream.started == []


async def test_the_snapshot_says_whether_the_person_has_a_page(
    stream: Stream, host: Host
) -> None:
    # A console that opens the chat again shows the page the person took over.
    owners = Owners(
        later=lambda seconds, fn: asyncio.get_running_loop().call_later(seconds, fn)
    )
    view = LiveView(stream, host, poll_s=3600, owners=owners)
    try:
        host.add("t1", "R1", HOME, "Home")
        assert (await view.snapshot())["yours"] is False
        owners.take("t1")
        assert (await view.snapshot())["yours"] is True
        owners.give("t1")
        assert (await view.snapshot())["yours"] is False
        no_takeover = LiveView(stream, host, poll_s=3600)
        assert (await no_takeover.snapshot())["yours"] is False
        await no_takeover.aclose()
    finally:
        await view.aclose()


# ── The relay under the browser tools ──


class Streaming(FakeDriver):
    """A scripted browser that can stream, as the Chrome driver does."""

    label = "Chrome"

    def __init__(self, *pages: Any) -> None:
        super().__init__(*pages)
        self.cast = Stream()

    async def start_screencast(self, tab: str, *args: Any, **kwargs: Any) -> None:
        await self.cast.start_screencast(tab, *args, **kwargs)

    async def stop_screencast(self, tab: str) -> None:
        await self.cast.stop_screencast(tab)


def site() -> Streaming:
    return Streaming(
        page(HOME, "Welcome.", link("News", HOME + "news"), title="Home"),
        page(HOME + "news", "News.", title="News"),
    )


async def test_the_browser_tools_serve_a_live_view_that_follows_their_tabs() -> None:
    driver = site()
    provider = BrowserToolProvider(driver)  # type: ignore[arg-type]
    tools = {t.definition.name: t for t in await provider.start()}
    live = provider.live
    assert live is not None and provider.live is live
    viewer = await live.watch()
    assert viewer.meta["state"] == "idle"
    await run(tools, "browser_open", url=HOME)
    async with asyncio.timeout(1):
        while viewer.meta["state"] == "idle":
            await asyncio.sleep(0.01)
    assert viewer.target[0] == "t1"
    await run(tools, "browser_open", url=HOME + "news", tab="new")
    async with asyncio.timeout(1):
        while viewer.target[0] != "t2":
            await asyncio.sleep(0.01)
    assert len(driver.cast.started) == 2
    await provider.close()
    assert await viewer.next() is not None
    assert viewer.closed


async def test_a_chat_with_no_live_view_has_none() -> None:
    driver = site()
    assert BrowserToolProvider(driver, live=False).live is None  # type: ignore[arg-type]
    plain = FakeDriver(page(HOME, "Welcome."))
    assert BrowserToolProvider(plain).live is None  # type: ignore[arg-type]


async def test_melo_is_browsing_while_a_browser_step_runs() -> None:
    driver = site()
    provider = BrowserToolProvider(driver)  # type: ignore[arg-type]
    tools = {t.definition.name: t for t in await provider.start()}
    seen: list[bool] = []
    real = driver.open

    async def open_(*args: Any, **kwargs: Any) -> Any:
        seen.append(provider.live_busy())
        return await real(*args, **kwargs)

    driver.open = open_  # type: ignore[method-assign,assignment]
    assert provider.live_busy() is False
    await run(tools, "browser_open", url=HOME)
    assert seen == [True] and provider.live_busy() is False


# ── The Chrome driver's screencast, over the fake Chrome ──


@pytest.fixture
def peer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Peer:
    found = Peer(tmp_path)
    monkeypatch.setenv("FAKE_CHROME_LOG", str(found.path))
    monkeypatch.delenv("FAKE_CHROME_MODE", raising=False)
    monkeypatch.delenv("FAKE_CHROME_FRAMES", raising=False)
    return found


@pytest.fixture
async def driver(peer: Peer) -> AsyncIterator[ChromeBackgroundDriver]:
    found = ChromeBackgroundDriver(
        command=[sys.executable, FAKE], profile=peer.profile, version="156.0.8078.80"
    )
    await found.start()
    try:
        yield found
    finally:
        await found.close()


async def collect(
    driver: ChromeBackgroundDriver, tab: str, count: int, *, fps: float
) -> list[Frame]:
    got: list[Frame] = []
    await driver.start_screencast(tab, got.append, lambda: None, fps=fps)
    async with asyncio.timeout(10):
        while len(got) < count:
            await asyncio.sleep(0.005)
    return got


async def test_chrome_streams_a_tab_and_the_acknowledgements_are_held_to_the_rate(
    driver: ChromeBackgroundDriver, peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = await driver.open("https://example.com/", new_tab=True)
    loop = asyncio.get_running_loop()
    begun = loop.time()
    got = await collect(driver, tab.id, 12, fps=20)
    took = loop.time() - begun
    assert got[0].data.startswith(b"\xff\xd8")
    assert (got[0].width, got[0].height, got[0].offset_top, got[0].scale) == (
        1280.0,
        713.0,
        0.0,
        1.0,
    )
    started = peer.calls("Page.startScreencast")[0]["params"]
    assert started["format"] == "jpeg" and started["quality"] == 70
    assert (started["maxWidth"], started["maxHeight"]) == (1280, 800)
    # Three pictures come at once, and each next one only after an acknowledgement,
    # which waits its turn: 12 pictures at 20 a second cannot come in less than 0.4 s.
    assert took >= 0.4
    acks = [c["t"] for c in peer.calls("Page.screencastFrameAck")]
    assert len(acks) >= 9
    gaps = [b - a for a, b in zip(acks, acks[1:], strict=False)]
    assert sorted(gaps)[len(gaps) // 2] >= 0.04
    assert peer.calls("Page.screencastFrameAck")[0]["params"] == {"sessionId": 1}


async def test_a_faster_rate_lets_more_pictures_through(
    driver: ChromeBackgroundDriver,
) -> None:
    tab = await driver.open("https://example.com/", new_tab=True)
    loop = asyncio.get_running_loop()
    begun = loop.time()
    await collect(driver, tab.id, 24, fps=60)
    assert loop.time() - begun < 1.5


async def test_stopping_a_stream_stops_the_pictures_and_the_waiting_acknowledgements(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    tab = await driver.open("https://example.com/", new_tab=True)
    got = await collect(driver, tab.id, 4, fps=5)
    await driver.stop_screencast(tab.id)
    assert len(peer.calls("Page.stopScreencast")) == 1
    seen, acked = len(got), len(peer.calls("Page.screencastFrameAck"))
    await asyncio.sleep(0.5)
    assert len(got) == seen
    assert len(peer.calls("Page.screencastFrameAck")) == acked
    await driver.stop_screencast(tab.id)  # nothing streams: nothing happens
    assert len(peer.calls("Page.stopScreencast")) == 1


async def test_starting_again_replaces_the_receiver_and_does_not_start_twice(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    tab = await driver.open("https://example.com/", new_tab=True)
    first: list[Frame] = []
    second: list[Frame] = []
    await driver.start_screencast(tab.id, first.append, lambda: None, fps=60)
    await driver.start_screencast(tab.id, second.append, lambda: None, fps=60)
    async with asyncio.timeout(5):
        while len(second) < 3:
            await asyncio.sleep(0.005)
    assert len(peer.calls("Page.startScreencast")) == 1


async def test_a_page_that_does_not_change_sends_one_picture_and_costs_nothing(
    driver: ChromeBackgroundDriver, peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CHROME_FRAMES", "1")
    other = ChromeBackgroundDriver(
        command=[sys.executable, FAKE],
        profile=str(Path(peer.profile).with_name("still")),
        version="156.0.8078.80",
    )
    await other.start()
    try:
        tab = await other.open("https://example.com/", new_tab=True)
        got = await collect(other, tab.id, 1, fps=15)
        await asyncio.sleep(0.3)
        assert len(got) == 1
    finally:
        await other.close()


async def test_streaming_does_not_change_the_tab_melo_works_in(
    driver: ChromeBackgroundDriver,
) -> None:
    first = await driver.open("https://example.com/", new_tab=True)
    await driver.open("https://example.com/news", new_tab=True)
    before = driver._current  # noqa: SLF001
    await collect(driver, first.id, 1, fps=15)
    assert driver._current == before != first.id  # noqa: SLF001


async def test_a_stream_that_chrome_cannot_start_is_an_error_and_leaves_nothing_running(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CHROME_MODE", "cast_error")
    found = ChromeBackgroundDriver(
        command=[sys.executable, FAKE], profile=peer.profile, version="156.0.8078.80"
    )
    await found.start()
    try:
        tab = await found.open("https://example.com/", new_tab=True)
        with pytest.raises(BrowserError):
            await found.start_screencast(tab.id, lambda f: None, lambda: None)
        assert found._browser is not None and not found._browser.casts  # noqa: SLF001
    finally:
        await found.close()


async def test_the_end_of_chrome_ends_the_stream(
    driver: ChromeBackgroundDriver,
) -> None:
    tab = await driver.open("https://example.com/", new_tab=True)
    ended: list[int] = []
    await driver.start_screencast(
        tab.id, lambda f: None, lambda: ended.append(1), fps=1
    )
    assert driver._browser is not None  # noqa: SLF001
    driver._browser.kill()  # noqa: SLF001
    async with asyncio.timeout(5):
        while not ended:
            await asyncio.sleep(0.01)


async def test_a_closed_tab_leaves_no_stream_behind(
    driver: ChromeBackgroundDriver,
) -> None:
    tab = await driver.open("https://example.com/", new_tab=True)
    await collect(driver, tab.id, 1, fps=15)
    await driver.close_tab(tab.id)
    assert driver._browser is not None and not driver._browser.casts  # noqa: SLF001


def test_a_picture_must_be_a_jpeg_of_a_sane_size() -> None:
    good = base64.b64encode(b"\xff\xd8\xff\xe0abc\xff\xd9").decode()
    meta = {"deviceWidth": 1280, "deviceHeight": 713, "pageScaleFactor": 2}
    frame = chrome._picture({"data": good, "metadata": meta})  # noqa: SLF001
    assert frame is not None and (frame.width, frame.height, frame.scale) == (
        1280.0,
        713.0,
        2.0,
    )
    png = base64.b64encode(b"\x89PNG\r\n").decode()
    for data in (png, "not base64!", "", 7, None):
        assert chrome._picture({"data": data}) is None  # noqa: SLF001
    huge = base64.b64encode(b"\xff\xd8" + bytes(chrome.MAX_FRAME_BYTES)).decode()
    assert chrome._picture({"data": huge}) is None  # noqa: SLF001
    odd = {"deviceWidth": True, "deviceHeight": float("nan"), "pageScaleFactor": 0}
    plain = chrome._picture({"data": good, "metadata": odd})  # noqa: SLF001
    assert plain is not None
    assert (plain.width, plain.height, plain.scale) == (0.0, 0.0, 1.0)
