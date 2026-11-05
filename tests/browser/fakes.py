"""A scripted browser for the browsing tests: pages by address, no real browser."""

from __future__ import annotations

import dataclasses
import socket
from typing import Any

from localmelo.support.browser import desktop
from localmelo.support.browser.driver import (
    Act,
    ActResult,
    BrowserError,
    BrowserSetupError,
    Changes,
    CoveredError,
    Element,
    PointHit,
    RawPage,
    Shot,
    StaleElementError,
    TabInfo,
    View,
)
from localmelo.support.browser.sites import origin_of

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kg"
PNG += "AAAABJRU5ErkJggg=="
# What a name answers when a test gives it no answer of its own: a public address.
PUBLIC_IP = "93.184.216.34"


def forget_helper_status() -> None:
    """Forget the desktop helper's newest status, so the next read starts fresh."""
    desktop._last = None


class FakeDns:
    """Stands in for the system's name lookup (``socket.getaddrinfo``).

    ``answers`` maps a name to its addresses, or to None when it is not
    found; any other name answers :data:`PUBLIC_IP`. ``asked`` lists the
    names looked up, in order.
    """

    def __init__(self) -> None:
        self.answers: dict[str, list[str] | None] = {}
        self.asked: list[str] = []

    def getaddrinfo(
        self, host: Any, port: Any, *args: Any, **kwargs: Any
    ) -> list[tuple[Any, ...]]:
        name = host.decode() if isinstance(host, bytes) else str(host)
        self.asked.append(name)
        found = self.answers.get(name, [PUBLIC_IP])
        if found is None:
            raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided")
        number, infos = int(port or 0), []
        for address in found:
            v6 = ":" in address
            family = socket.AF_INET6 if v6 else socket.AF_INET
            where = (address, number, 0, 0) if v6 else (address, number)
            infos.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", where))
        return infos


def link(label: str, href: str) -> Element:
    return Element("link", label, href=href)


def button(label: str, **more: Any) -> Element:
    return Element("button", label, **more)


def safari_id(n: int) -> str:
    """A tab id in the shape Safari's automation server gives: 41 characters, too long for a short field."""
    return f"page-635DB8F8-27ED-4CA5-81A2-7EAEE1DF{n:04X}"


def page(
    url: str,
    text: str = "",
    *elements: Element,
    title: str = "T",
    frames: tuple[str, ...] = (),
) -> RawPage:
    return RawPage(url, title, text, tuple(elements), doc="d1", frames=frames)


class FakeDriver:
    """Pages by address; ``redirects`` maps an address to where it lands.

    ``ids`` makes the tab ids: ``t1``, ``t2`` by default, or ``safari_id``
    for the long ones the real helper gives. An action (``act``) records
    what it was asked and lands where ``lands`` says (a label, or
    ``enter:<label>``, to an address); a link goes to its address. ``then``
    maps a label to the address the page moves its own tab to a moment after
    the action (``location`` answers the old address once, then the new one);
    ``fails`` maps a label to an address the tab is moved to before the action
    raises; ``stopped`` is what ``act`` answers (the page's blocked sends);
    ``docs`` maps a tab to the document id its page shows. ``popups``
    maps a label to an address that opens in a new tab. ``lose`` forgets a tab
    as the real helper does when it loses one. ``via`` is the way an act says
    it used (``refuse_native`` makes it the page script, with the browser's own
    input failed for the site), ``changes`` maps a label to what an act on it
    changed, and ``redraw`` maps a label to the address whose page an act on it
    finds redrawn (stale, once). ``hits`` maps a rounded point to what is there.
    ``covers`` maps a label to what the browser says when something stands over
    it (the click is stopped and nothing is done). ``held_already`` is what
    ``holds`` answers (a condition that held before any waiting).
    """

    name = "fake"

    def __init__(
        self,
        *pages: RawPage,
        redirects: dict[str, str] | None = None,
        ids: Any = None,
    ) -> None:
        self.pages = {p.url: p for p in pages}
        self.redirects = redirects or {}
        self.calls: list[tuple[str, str]] = []
        self.tabs_: dict[str, list[str]] = {}  # tab id -> history of addresses
        self.current: str | None = None
        self.setup_problem: str | None = None
        self.started = 0
        self.ids = ids or (lambda n: f"t{n}")
        self.made = 0
        self.open_in: list[str | None] = []  # the tab each non-new open asked for
        self.acts: list[Act] = []
        self.lands: dict[str, str] = {}
        self.popups: dict[str, str | list[str]] = {}
        self.then: dict[str, str] = {}
        self.fails: dict[str, str] = {}
        self.stopped = ""
        self.docs: dict[str, str] = {}
        self.queue: dict[str, list[str]] = {}  # scripted answers of location()
        self.blank_fails = False
        self.tabs_fail = False
        self.location_empty: set[str] = set()
        self.via = "browser"
        self.refuse_native = False
        self.changes: dict[str, Changes] = {}
        self.redraw: dict[str, RawPage] = {}
        self.covers: dict[str, str] = {}
        self.held_already = False
        self.hits: dict[tuple[int, int], PointHit] = {}
        self.marks: list[tuple[View, float, float] | None] = []
        self.shot_view: View | None = View("", "d1", 1000.0, 500.0)

    def _new_tab(self) -> str:
        self.made += 1
        tab = str(self.ids(self.made))
        self.tabs_[tab] = []
        return tab

    def lose(self, tab: str) -> None:
        """The helper loses *tab*: it is no longer listed and answers nothing."""
        self.tabs_.pop(tab, None)
        if self.current == tab:
            self.current = None

    def _url(self, tab: str | None) -> str:
        return self.tabs_[tab or self.current or ""][-1]

    def _info(self, tab: str) -> TabInfo:
        url = self.tabs_[tab][-1]
        return TabInfo(tab, url, self.pages[url].title if url in self.pages else "")

    async def start(self) -> None:
        if self.setup_problem:
            raise BrowserSetupError(self.setup_problem)
        self.started += 1

    async def close(self) -> None:
        self.calls.append(("close", ""))

    def kill(self) -> None:
        self.calls.append(("kill", ""))

    async def open(self, url: str, *, new_tab: bool, tab: str | None = None) -> TabInfo:
        self.calls.append(("open", url))
        landed = self.redirects.get(url, url)
        if not new_tab:
            self.open_in.append(tab)
            if tab is not None and tab not in self.tabs_:
                raise BrowserError(
                    "Safari's automation helper reported a problem: "
                    "Could not find browsing context."
                )
            self.current = tab or self.current
        if new_tab or self.current is None:
            self.current = self._new_tab()
        self.tabs_[self.current].append(landed)
        return self._info(self.current)

    async def page(self, tab: str | None) -> RawPage:
        self.calls.append(("page", tab or ""))
        here = tab or self.current or ""
        raw = self.pages[self._url(tab)]
        return dataclasses.replace(raw, doc=self.docs.get(here, raw.doc))

    async def back(self, tab: str | None) -> TabInfo:
        self.calls.append(("back", tab or ""))
        history = self.tabs_[tab or self.current or ""]
        if len(history) > 1:
            history.pop()
        return self._info(tab or self.current or "")

    async def tabs(self) -> list[TabInfo]:
        if self.tabs_fail:
            raise BrowserError(
                "Safari's automation helper reported a problem: no list."
            )
        return [self._info(tab) for tab in self.tabs_]

    async def switch(self, tab: str) -> TabInfo:
        if tab not in self.tabs_:
            raise BrowserError(f"no tab {tab}")
        self.current = tab
        return self._info(tab)

    async def close_tab(self, tab: str) -> None:
        self.calls.append(("close_tab", tab))
        if tab not in self.tabs_:
            raise BrowserError(
                "Safari's automation helper reported a problem: closeBrowsingContext: "
                "Could not find browsing context."
            )
        self.tabs_.pop(tab)
        if self.current == tab:
            self.current = None

    async def location(self, tab: str | None) -> str:
        here = tab or self.current or ""
        if here in self.location_empty:
            return ""
        if self.queue.get(here):
            return self.queue[here].pop(0)
        return self._url(tab)

    async def wait(
        self,
        tab: str | None,
        until: str,
        text: str,
        seconds: float,
        *,
        origin: str = "",
    ) -> bool:
        self.calls.append(("wait", f"{until}:{text}:{seconds:g}"))
        if origin and origin_of(self._url(tab)) != origin:
            return False  # the condition holds only on the page's own site
        return text != "never"

    async def holds(
        self, tab: str | None, until: str, text: str, *, origin: str = ""
    ) -> bool:
        self.calls.append(("holds", f"{until}:{text}"))
        return self.held_already

    async def screenshot(
        self, tab: str | None, mark: tuple[View, float, float] | None = None
    ) -> Shot:
        self.marks.append(mark)
        view = self.shot_view
        if view is not None:
            view = dataclasses.replace(view, url=self._url(tab))
        return Shot("image/png", PNG, view)

    async def hit(self, tab: str | None, view: View, x: float, y: float) -> PointHit:
        self.calls.append(("hit", f"{x:g},{y:g}"))
        if self._url(tab) != view.url:
            raise StaleElementError("The page is not the one that was read.", "page")
        return self.hits.get((round(x), round(y)), PointHit("none"))

    async def blank(self, tab: str | None) -> None:
        self.calls.append(("blank", tab or ""))
        if self.blank_fails:
            raise BrowserError("Safari's automation helper reported a problem: blank.")
        here = tab or self.current or ""
        if here in self.tabs_:
            # As the real driver does it, with location.replace: the page's own
            # entry is swapped, so Back never returns to the page that was refused.
            self.tabs_[here][-1:] = ["about:blank"]

    async def act(self, tab: str | None, act: Act) -> ActResult:
        self.calls.append(("act", f"{act.kind}:{act.element.label}"))
        here = tab or self.current or ""
        if self._url(tab) != act.url:
            raise StaleElementError("The page is not the one that was read.", "page")
        if act.doc and act.doc != self.docs.get(here, "d1"):
            raise StaleElementError(
                "The page was reloaded or replaced since it was read.", "doc"
            )
        label = act.element.label
        if label in self.redraw and act.point is None:
            # The page redrew itself since it was read: the element is there again,
            # under another place, and the act on the old reading is stale.
            self.pages[self._url(tab)] = self.redraw.pop(label)
            raise StaleElementError("That element is no longer on the page.", "gone")
        if label in self.covers:
            raise CoveredError(self.covers[label])
        self.acts.append(act)
        key = ("enter:" if act.enter else "") + label
        via = "point" if act.point is not None else self.via
        failed = False
        if self.refuse_native and act.native and act.point is None:
            via, failed = "page-script", True
        elif not act.native and act.point is None:
            via = "page-script"
        result = ActResult(
            self.stopped,
            via,  # type: ignore[arg-type]
            "the browser's own input did not reach it" if failed else "",
            self.changes.get(label, Changes()),
            native_failed=failed,
        )
        if label in self.fails:
            self.tabs_[here].append(self.fails[label])
            raise BrowserError("The page did not take that action.")
        if act.kind == "click" and label in self.popups:
            wanted = self.popups[label]
            for address in [wanted] if isinstance(wanted, str) else wanted:
                self.current = self._new_tab()
                self.tabs_[self.current].append(address)
            return result
        landed = self.lands.get(key)
        if landed is None and act.kind == "click" and act.element.kind == "link":
            landed = act.element.href
        if landed is not None:
            self.tabs_[here].append(landed)
        if key in self.then:
            self.queue[here] = [self._url(here), self.then[key]]
            self.tabs_[here].append(self.then[key])
        return result
