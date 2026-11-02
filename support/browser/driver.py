"""What melo's browser tools need from a browser, and what they get back.

A driver turns a page into a :class:`RawPage` and moves tabs; everything the
model reads (budgets, the nonce markers, sensitive fields) is decided in
:mod:`localmelo.support.browser.render`, not in a driver. A driver reports a
problem as :class:`BrowserError`, in words a person and the model can read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

ElementKind = Literal["link", "button", "textbox", "checkbox", "select"]
# Why a field is never read or typed into: "" for an ordinary one.
Sensitive = Literal["", "password", "code", "card"]


class BrowserError(Exception):
    """A browser call that failed; the message says what to do about it."""


class BrowserSetupError(BrowserError):
    """The browser is not set up for melo; the message says what to turn on."""


@dataclass(frozen=True)
class Element:
    """One thing on a page: a link, a button or a form field.

    ``href`` is absolute and only for links. ``value`` is only for text boxes
    and selects, and a driver leaves it empty for a ``sensitive`` field (the
    renderer drops it again).
    """

    kind: ElementKind
    label: str
    href: str = ""
    value: str = ""
    checked: bool | None = None
    sensitive: Sensitive = ""


@dataclass(frozen=True)
class RawPage:
    """The readable text of a page and its elements, in document order."""

    url: str
    title: str
    text: str
    elements: tuple[Element, ...] = ()


@dataclass(frozen=True)
class TabInfo:
    id: str
    url: str = ""
    title: str = ""


@dataclass(frozen=True)
class Shot:
    """An image of the page: ``data`` is base64 without a ``data:`` prefix."""

    mime_type: str
    data: str


class BrowserDriver(Protocol):
    """A browser melo reads pages in. ``tab`` None means the current tab."""

    name: str

    async def start(self) -> None:
        """Connect. Idempotent; raises :class:`BrowserSetupError` when not set up."""

    async def close(self) -> None: ...

    def kill(self) -> None: ...

    async def open(self, url: str, *, new_tab: bool) -> TabInfo:
        """Load *url* (in a new tab or the current one) and return where it landed."""

    async def page(self, tab: str | None) -> RawPage: ...

    async def back(self, tab: str | None) -> TabInfo: ...

    async def tabs(self) -> list[TabInfo]: ...

    async def switch(self, tab: str) -> TabInfo: ...

    async def close_tab(self, tab: str) -> None: ...

    async def location(self, tab: str | None) -> str:
        """The address the tab shows now, read in the tab itself (``location.href``)."""

    async def wait(
        self,
        tab: str | None,
        until: str,
        text: str,
        seconds: float,
        *,
        origin: str = "",
    ) -> bool:
        """Wait for ``load``, ``text`` or ``gone``; whether it happened in time.

        With *origin*, the condition counts only while the page is on that
        origin (``location.origin``), read in the same step as the condition,
        so a page that goes elsewhere for a moment answers nothing about it.
        """

    async def screenshot(self, tab: str | None) -> Shot: ...

    async def blank(self, tab: str | None) -> None:
        """Leave the page: load an empty one (a redirect the person refused)."""
