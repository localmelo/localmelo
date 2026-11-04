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
# How an action was done, best first: the browser's own input on the element melo
# read (trusted events), a page script on that element, or the browser's own click
# at a point of the last picture.
Way = Literal["browser", "page-script", "point"]
# The most characters one text is typed with. The question that asks to type shows
# all of it, so it is short enough to read before it is allowed.
MAX_TYPED_CHARS = 1000
# Why a field is never read or typed into: "" for an ordinary one.
Sensitive = Literal["", "password", "code", "card"]


class BrowserError(Exception):
    """A browser call that failed; the message says what to do about it."""


class BrowserSetupError(BrowserError):
    """The browser is not set up for melo; the message says what to turn on."""


@dataclass(frozen=True)
class FieldAttrs:
    """How a page describes a text box, as the page script read it.

    ``maxlength`` is the longest text the field takes: its ``maxlength``, or the
    length a digits-only ``pattern`` or an all-9s ``max`` gives it (0 when
    unknown); a digits-only ``pattern`` makes ``inputmode`` ``numeric``.
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
    :class:`FieldAttrs`). ``frame`` is where it is when not in the page itself
    (a frame of the same site, or ``x<id>`` in a box from another site, whose
    ``site`` is that box's origin); ``plain`` marks an element that only looks
    clickable. ``context`` says where it belongs (what it is, its row or the
    heading above it, the text next to it), ``""`` when nothing does: a ref the
    page redrew is found again only with the same.
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
    frame: str = field(default="", compare=False)
    site: str = field(default="", compare=False)
    plain: bool = field(default=False, compare=False)
    context: str = field(default="", compare=False)


@dataclass(frozen=True)
class RawPage:
    """The readable text of a page and its elements, in document order.

    ``url`` is the address the browser itself reports for the tab (never the
    page's own word for it). ``doc`` identifies the loaded document: it changes
    when the page is reloaded or replaced. ``frames`` are the addresses of the
    page's frames as far as they can be read. ``unreached`` are the addresses
    of the visible frames from another site whose insides melo cannot read or
    act in (a sign-in box from another site, say): the model is told they are
    there and that a picture and a click at a point can still reach them.
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
    unreached: tuple[str, ...] = ()


@dataclass(frozen=True)
class TabInfo:
    id: str
    url: str = ""
    title: str = ""


@dataclass(frozen=True)
class View:
    """What a picture of a page showed: the document, and the part of it in view.

    ``width`` and ``height`` are the size of the view in the page's own pixels
    (CSS pixels), ``x`` and ``y`` how far it was scrolled, ``ratio`` the
    screen's pixels to a page pixel. A point on the picture means the same place
    on the page only while all of these hold.
    """

    url: str
    doc: str
    width: float
    height: float
    x: float = 0.0
    y: float = 0.0
    ratio: float = 2.0


@dataclass(frozen=True)
class Shot:
    """An image of the page: ``data`` is base64 without a ``data:`` prefix.

    ``view`` is the view the picture showed, when the driver could read it.
    """

    mime_type: str
    data: str
    view: View | None = None


@dataclass(frozen=True)
class PointHit:
    """What is at a point of the page, as the page script found it there.

    ``element`` is the control the point resolves to (a label moves to its
    field), in the same terms as a read element, or None when the point is on
    nothing melo can name. ``surface`` says what it is: ``control``; ``canvas``
    (a drawing, picture or video that takes clicks itself); ``frame`` (a box
    from another site, ``site`` its origin, whose inside melo cannot see); or
    ``none`` (plain text or background). ``consent`` is set when the control's
    words agree to terms, consent or a sign-in with another account, which
    melo asks the person about. ``at`` is where to click instead (page pixels),
    when the point was on a label and its field can take the click itself.
    """

    surface: Literal["control", "canvas", "frame", "none"]
    element: Element | None = None
    site: str = ""
    consent: bool = False
    at: tuple[float, float] | None = None


@dataclass(frozen=True)
class Changes:
    """What an action changed on the page, as an observer set before it saw.

    ``texts`` is new visible text (a notice, a toast, an error), even text that
    was gone again by the end; ``target`` the acted-on element's new state in
    words (``"now checked, disabled"``, ``"gone from the page"``), ``""`` when
    it did not change. ``changed`` is whether the page changed at all (its
    document tree, or the element's state). ``quiet`` is False when the page
    was still changing when melo stopped waiting. These are words from the page.
    """

    texts: tuple[str, ...] = ()
    target: str = ""
    quiet: bool = True
    changed: bool = False


@dataclass(frozen=True)
class Dialog:
    """A box the page put up while melo acted, and how melo answered it.

    ``kind`` is ``alert``, ``confirm``, ``prompt`` or ``beforeunload``;
    ``message`` is the page's own words (data, cut short); ``accepted`` is
    whether melo answered OK (or Leave), as against Cancel.
    """

    kind: str
    message: str
    accepted: bool


@dataclass(frozen=True)
class ActResult:
    """What :meth:`BrowserDriver.act` did.

    ``note`` is a sentence about what the driver stopped the page from doing
    meanwhile (a send nobody asked about), or ``""``; ``via`` the way it acted
    and ``why`` the reason a better way was not used (``""`` when none was
    passed over); ``changes`` what it saw change, or None when the page became
    another document. ``native_failed`` says the browser's own input did not
    work there, so the site starts with the page script next time.
    ``handoff`` is an address of another app's kind (``chat://...``) the page
    asked the browser to open meanwhile, which the browser did not open
    (``""``: none); melo asks the person before it opens it itself.
    ``dialogs`` are the boxes the page put up meanwhile and how they were
    answered.
    """

    note: str = ""
    via: Way = "page-script"
    why: str = ""
    changes: Changes | None = None
    native_failed: bool = False
    handoff: str = ""
    dialogs: tuple[Dialog, ...] = ()


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
    (a click on its submit button, or Enter): a driver tries to stop any other
    attempt by the page to send a form while the action runs. That is best
    effort and holds only that long; once a text is in a page, the page may send
    it itself. ``native`` lets the driver use the browser's own input first;
    False starts with the page script (a site where the browser's own input
    did not work earlier). ``point`` is a click at that place of ``view`` (page
    pixels), with the browser's own click, on the control ``element`` that was
    found there; the driver does nothing unless the point still resolves to it
    and the view has not scrolled.
    """

    kind: ActionKind
    url: str
    element: Element
    text: str = ""
    enter: bool = False
    doc: str = ""
    send: bool = False
    native: bool = True
    point: tuple[float, float] | None = None
    view: View | None = None


class StaleElementError(BrowserError):
    """The page is not the one that was read, or the element is no longer there.

    ``why`` is the page script's word for it: ``page`` (another address),
    ``doc`` (reloaded), ``gone`` (no longer there as it was read), ``dest`` (it
    leads or sends elsewhere now), ``field``, ``grew`` or ``frame``. Only a
    page that redrew itself in the same document (``gone``) may have the same
    element under another ref.
    """

    def __init__(self, message: str, why: str = "") -> None:
        super().__init__(message)
        self.why = why


class CoveredError(StaleElementError):
    """A real click or key press would land on another element, so melo stopped it.

    Nothing was done and the page is as it was read, so the readings stay (the
    refs work as before). The message says what stands over the element when
    the page said so. ``why`` is ``covered``.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, "covered")


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

    async def holds(
        self, tab: str | None, until: str, text: str, *, origin: str = ""
    ) -> bool:
        """Whether the condition of :meth:`wait` holds right now: one look, no waiting."""

    async def screenshot(
        self, tab: str | None, mark: tuple[View, float, float] | None = None
    ) -> Shot:
        """A picture of the view, with the view it showed.

        *mark* is a point melo clicked (the view it was clicked in, and the
        point in page pixels): it is drawn on the picture when the page still
        shows that view, so the person and the model see where the click went.
        """

    async def hit(self, tab: str | None, view: View, x: float, y: float) -> PointHit:
        """What is at (*x*, *y*) of *view* (page pixels), the view of the last picture.

        Raises :class:`StaleElementError` when the page is not that one, or it scrolled.
        """

    async def blank(self, tab: str | None) -> None:
        """Leave the page: load an empty one (a redirect the person refused)."""

    async def act(self, tab: str | None, act: Act) -> ActResult:
        """Do *act* on the page of *tab*, then let the page settle.

        Returns what it did (see :class:`ActResult`). Raises
        :class:`StaleElementError` when the page is not the one that was read,
        and :class:`BrowserError`, in words for the model, when the element
        cannot be used (hidden, disabled, a field melo never types in, an
        option that is not there).
        """
