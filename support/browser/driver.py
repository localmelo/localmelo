"""What melo's browser tools need from a browser, and what they get back.

A driver turns a page into a :class:`RawPage` and moves tabs; everything the
model reads (budgets, the nonce markers, sensitive fields) is decided in
:mod:`localmelo.support.browser.render`, not in a driver. A driver reports a
problem as :class:`BrowserError`, in words a person and the model can read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

ElementKind = Literal["link", "button", "textbox", "checkbox", "select"]
# Why a field is never read or typed into: "" for an ordinary one.
Sensitive = Literal["", "password", "code", "card"]


class BrowserError(Exception):
    """A browser call that failed; the message says what to do about it."""


class BrowserSetupError(BrowserError):
    """The browser is not set up for melo; the message says what to turn on."""


@dataclass(frozen=True)
class FieldAttrs:
    """How a page describes a text box, as the page script read it.

    ``words`` joins what names the field: its name, id, ``aria-label``, the text
    ``aria-labelledby`` and ``aria-describedby`` point at, title, placeholder,
    the text of its labels, its fieldset's legend and the text just before it.
    :mod:`localmelo.support.browser.fields`
    decides from these whether melo may type into it, here when the page is
    read and again where it types, so a field the page re-described since
    is not typed into on the old reading.
    """

    type: str = ""
    autocomplete: str = ""
    words: str = ""
    inputmode: str = ""
    maxlength: int = 0
    masked: bool = False


@dataclass(frozen=True)
class Element:
    """One thing on a page: a link, a button or a form field.

    ``href`` is absolute and only for links. ``value`` is only for text boxes
    and selects, and a driver leaves it empty for a ``sensitive`` field (the
    renderer drops it again). ``also`` is another name the page gives it
    (an ``aria-label`` next to its visible text, a title), searched by
    ``browser_find``. ``at`` is where the driver finds it again on the page it
    read and ``fp`` what identifies it there (its tag, kind, key attributes and
    place in the document); neither is part of what two elements are.
    ``submit`` is the address a click on a button, or Enter in a text box,
    would send a form to, and is empty for anything that sends nothing (a text
    box that is in no form, where Enter is the page's own business, has none);
    ``form`` is the name and value of every field in that form, with nothing
    secret in it, as the question about sending it shows; ``options`` are the
    choices of a select; ``attrs`` describes a text box or a select (see
    :class:`FieldAttrs`).
    """

    kind: ElementKind
    label: str
    href: str = ""
    value: str = ""
    checked: bool | None = None
    sensitive: Sensitive = ""
    also: str = ""
    submit: str = ""
    options: tuple[str, ...] = ()
    attrs: FieldAttrs | None = None
    at: int = field(default=-1, compare=False)
    fp: str = field(default="", compare=False)
    form: tuple[tuple[str, str], ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class RawPage:
    """The readable text of a page and its elements, in document order.

    ``url`` is the address the browser itself reports for the tab (never the
    page's own word for it). ``doc`` identifies the loaded document: it changes
    when the page is reloaded or replaced. ``frames`` are the addresses of the
    page's frames as far as they can be read.
    """

    url: str
    title: str
    text: str
    elements: tuple[Element, ...] = ()
    skipped: int = (
        0  # candidates the page script did not list (hidden, disabled, over the cap)
    )
    doc: str = ""
    frames: tuple[str, ...] = ()


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


ActionKind = Literal["click", "type", "select"]


@dataclass(frozen=True)
class Act:
    """One thing to do on a page: click, type or choose, on an element melo read.

    ``url`` is the address the page had when it was read, ``doc`` the document
    it was read in and ``element`` what was read there; a driver does nothing
    unless the page still shows the same address in the same document and, at
    ``element.at``, the same element (its fingerprint, label, address, the
    address its form is sent to and, for a text box, the same
    :class:`FieldAttrs`), and says so with :class:`StaleElementError`. ``text``
    is what to type, or the option to choose; ``enter`` presses Enter after
    typing. ``send`` says the action is an approved send of ``element.submit``
    (a click on its submit button, or Enter): any other attempt by the page to
    send a form while the action runs is stopped.
    """

    kind: ActionKind
    url: str
    element: Element
    text: str = ""
    enter: bool = False
    doc: str = ""
    send: bool = False


class StaleElementError(BrowserError):
    """The page is not the one that was read, or the element is no longer there."""


class BrowserDriver(Protocol):
    """A browser melo reads pages in. ``tab`` None means the current tab."""

    name: str

    async def start(self) -> None:
        """Connect. Idempotent; raises :class:`BrowserSetupError` when not set up."""

    async def close(self) -> None: ...

    def kill(self) -> None: ...

    async def open(self, url: str, *, new_tab: bool, tab: str | None = None) -> TabInfo:
        """Load *url* in a new tab, or in *tab* (the current one when None); return where it landed."""

    async def page(self, tab: str | None) -> RawPage: ...

    async def back(self, tab: str | None) -> TabInfo: ...

    async def tabs(self) -> list[TabInfo]: ...

    async def switch(self, tab: str) -> TabInfo: ...

    async def close_tab(self, tab: str) -> None: ...

    async def location(self, tab: str | None) -> str:
        """The address the browser reports for the tab now, or ``""`` when it cannot say.

        It comes from the browser's own tab list, never from a script running
        in the page, which could say anything.
        """

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

    async def act(self, tab: str | None, act: Act) -> str:
        """Do *act* on the page of *tab*, then let the page settle.

        Returns a sentence about what the driver stopped the page from doing
        meanwhile (sending a form nobody asked about), or ``""``. Raises
        :class:`StaleElementError` when the page is not the one that was read,
        and :class:`BrowserError`, in words for the model, when the element
        cannot be used (hidden, disabled, a field melo never types in, an
        option that is not there).
        """
