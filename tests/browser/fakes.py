"""A scripted browser for the browsing tests: pages by address, no real browser."""

from __future__ import annotations

from localmelo.support.browser.driver import (
    BrowserError,
    BrowserSetupError,
    Element,
    RawPage,
    Shot,
    TabInfo,
)

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kg"
PNG += "AAAABJRU5ErkJggg=="


def link(label: str, href: str) -> Element:
    return Element("link", label, href=href)


def page(url: str, text: str = "", *elements: Element, title: str = "T") -> RawPage:
    return RawPage(url, title, text, tuple(elements))


class FakeDriver:
    """Pages by address; ``redirects`` maps an address to where it lands."""

    name = "fake"

    def __init__(
        self, *pages: RawPage, redirects: dict[str, str] | None = None
    ) -> None:
        self.pages = {p.url: p for p in pages}
        self.redirects = redirects or {}
        self.calls: list[tuple[str, str]] = []
        self.tabs_: dict[str, list[str]] = {}  # tab id -> history of addresses
        self.current: str | None = None
        self.setup_problem: str | None = None
        self.started = 0

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

    async def open(self, url: str, *, new_tab: bool) -> TabInfo:
        self.calls.append(("open", url))
        landed = self.redirects.get(url, url)
        if new_tab or self.current is None:
            self.current = f"t{len(self.tabs_) + 1}"
            self.tabs_[self.current] = []
        self.tabs_[self.current].append(landed)
        return self._info(self.current)

    async def page(self, tab: str | None) -> RawPage:
        self.calls.append(("page", tab or ""))
        return self.pages[self._url(tab)]

    async def back(self, tab: str | None) -> TabInfo:
        self.calls.append(("back", tab or ""))
        history = self.tabs_[tab or self.current or ""]
        if len(history) > 1:
            history.pop()
        return self._info(tab or self.current or "")

    async def tabs(self) -> list[TabInfo]:
        return [self._info(tab) for tab in self.tabs_]

    async def switch(self, tab: str) -> TabInfo:
        if tab not in self.tabs_:
            raise BrowserError(f"no tab {tab}")
        self.current = tab
        return self._info(tab)

    async def close_tab(self, tab: str) -> None:
        self.calls.append(("close_tab", tab))
        self.tabs_.pop(tab, None)
        if self.current == tab:
            self.current = None

    async def wait(
        self, tab: str | None, until: str, text: str, seconds: float
    ) -> bool:
        self.calls.append(("wait", f"{until}:{text}:{seconds:g}"))
        return text != "never"

    async def screenshot(self, tab: str | None) -> Shot:
        return Shot("image/png", PNG)

    async def blank(self, tab: str | None) -> None:
        self.calls.append(("blank", tab or ""))
