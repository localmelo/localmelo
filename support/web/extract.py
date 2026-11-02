"""An HTML page as readable text: its title, headings, main text and links (standard library only).

Scripts, styles, templates, embedded documents and form choices are left
out, and so is what a page hides from people, since hidden text is a common place
for instructions aimed at a model. Two things are dropped, and nothing a person
sees:

* an element the page does not draw, with everything in it: the ``hidden``
  attribute, ``aria-hidden="true"``, ``display: none``, ``visibility: hidden`` or
  ``collapse`` (also an element a browser closes by itself, such as ``<p hidden>``),
  by its own style or by a simple rule of a ``<style>`` block (``.class``, ``#id`` or
  ``tag``);
* a piece of text whose font size is tiny (2 px or less), whose colour is clear
  (``transparent``, an alpha of 0), or whose colour is white with no coloured
  background behind it, following the inline styles down the page (the nearest font
  size, colour and background set on an open element win, so a child that resets a
  size or sits on a dark background shows its own text). A background image or
  gradient counts as a dark background, since white text over a photo is a common
  design; the note gate's order filter still applies to that text.

``opacity: 0`` hides nothing here (a script usually fades such text in), and a
colour or background that comes from a style sheet is not seen. The stricter rule
for a raw file the note gate reads is :func:`hidden_style` and :func:`hides`.
When the page marks its main part (``<main>``, ``role="main"`` or
``<article>``) with enough text in it, only that part is kept, so menus and
footers do not crowd it out. Headings read ``# Title``, list items ``- item``
and table cells are joined with `` | ``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import NamedTuple
from urllib.parse import urljoin, urlsplit

# Text a main part needs to be kept alone, in characters other than spaces.
MAIN_MIN_CHARS = 200
MAX_LINKS = 200

SKIP = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "math",
        "canvas",
        "iframe",
        "object",
        "embed",
        "select",
        "datalist",
    }
)
VOID = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
# Elements a browser closes by itself. A hidden one is skipped as far as its end
# tag, the tag that closes it or the end of its parent (``_Reader._in_auto``),
# not as far as an end tag that may never come: the rest of the page would go with it.
AUTO_CLOSED = frozenset(
    {
        "p",
        "li",
        "dt",
        "dd",
        "tr",
        "td",
        "th",
        "option",
        "optgroup",
        "thead",
        "tbody",
        "tfoot",
        "colgroup",
        "rb",
        "rt",
        "rp",
    }
)
# Elements that stand apart with a blank line, and those that only start a line.
PARAGRAPH = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "details",
        "dl",
        "fieldset",
        "figure",
        "footer",
        "form",
        "header",
        "hr",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "ul",
    }
)
LINE = frozenset(
    {
        "body",
        "center",
        "dd",
        "dialog",
        "div",
        "dt",
        "figcaption",
        "html",
        "li",
        "summary",
        "tr",
    }
)
# The elements whose ``/>`` closes them (foreign content); on any other a ``/>`` is
# ignored, as in a browser.
FOREIGN = frozenset({"svg", "math"})
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
# The start tags that end an open element of an auto-closed kind besides one of its
# own name: a block opens, so the paragraph before it is over.
_ENDED_BY = {
    "p": PARAGRAPH | {"div", "dl", "h1", "h2", "h3", "h4", "h5", "h6", "figcaption"},
    "dt": frozenset({"dd"}),
    "dd": frozenset({"dt"}),
    "td": frozenset({"th", "tr"}),
    "th": frozenset({"td", "tr"}),
}

# ── What a page hides from people ──
# Two rules share the style parser. The strict one, :func:`hidden_style` and
# :func:`hides`, is for raw files, where only the markup is known: the note gate
# uses it, and text it takes for hidden costs at most a note. The page reader
# (:class:`_Reader`) is what the model reads a page by, so it drops only what no
# person sees: it leaves out an element that is not drawn at all (``hidden``,
# ``aria-hidden="true"``, ``display: none``, ``visibility: hidden`` or ``collapse``,
# also by a simple rule of a ``<style>`` block), and it follows how font size, text
# colour and background cascade through the inline styles, dropping a piece of text
# whose size is tiny, whose colour is clear, or whose colour is white with no coloured
# background behind it. ``opacity: 0`` hides nothing there: a script fades such text in.
#
# Every parse here is bounded: a style value is read to its first
# :data:`MAX_VALUE_CHARS` characters, and no pattern backtracks over a run.

_CSS_COMMENT = re.compile(r"/\*.*?(?:\*/|\Z)", re.DOTALL)
# The longest style value that is read, in characters; the rest is ignored.
MAX_VALUE_CHARS = 2_000
# Font sizes in pixels per unit; a number with no unit is a size only when it is 0. A
# relative unit (``em``, ``%``, ``ex``, ``ch``) is a share of the size around it, 16 px
# at the top. Zero is nothing in any unit, a viewport unit included.
_PIXELS = {
    "px": 1.0,
    "pt": 4 / 3,
    "pc": 16.0,
    "q": 0.945,
    "em": 16.0,
    "rem": 16.0,
    "%": 0.16,
    "ex": 8.0,
    "ch": 8.0,
    "mm": 3.78,
    "cm": 37.8,
    "in": 96.0,
}
_RELATIVE = frozenset({"em", "%", "ex", "ch"})
# A number, with an optional sign and exponent, and its unit. ``\d+(?:\.\d*)?`` cannot
# split a run of digits two ways, so a long run is read in one pass.
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
_SIZE = re.compile(rf"({_NUMBER})\s*([a-z%]*)")
_OPACITY = re.compile(rf"{_NUMBER}%?")
# Colours are read after their white space is folded to single spaces.
_FULL = r"(?:255|100%)"
_WHITE = re.compile(
    r"white|#f{3}|#f{4}|#f{6}|#f{8}"
    rf"|rgba?\( ?{_FULL} ?[, ] ?{_FULL} ?[, ] ?{_FULL} ?(?:[,/][^)]*)?\)"
    r"|hsla?\( ?[^,) ]+ ?[, ] ?[^,) ]+ ?[, ] ?100% ?(?:[,/][^)]*)?\)"
)
_COLOUR_FUNCTION = re.compile(r"(?:rgb|hsl)a?\( ?([^)]*)\)")
_ALPHA_ZERO = re.compile(r"\+?(?:0+(?:\.0*)?|\.0+)%?")
_SHORT_HEX = re.compile(r"#[0-9a-f]{3}0|#[0-9a-f]{6}00")
# The largest font size, in pixels, that counts as hidden, and the usual one.
TINY_PIXELS = 2.0
_NORMAL_PIXELS = 16.0
_INHERITED = frozenset({"inherit", "unset", "currentcolor"})
# What a background property says when it puts nothing behind the text.
_NO_BACKGROUND = frozenset({"", "none", "inherit", "initial", "unset", "currentcolor"})


def _size_px(value: str, base: float) -> float | None:
    """A font size in pixels, a relative unit taken against *base*; ``None`` when
    *value* is no size this reads (a keyword, ``calc()``, a viewport unit, a negative
    number a browser ignores)."""
    found = _SIZE.fullmatch(value)
    if not found:
        return None
    number, unit = float(found.group(1)), found.group(2)
    if number == 0:
        return 0.0
    if not unit or unit not in _PIXELS or number < 0:
        return None
    return number * _PIXELS[unit] * (base / _NORMAL_PIXELS if unit in _RELATIVE else 1)


def _tiny(value: str) -> bool:
    size = _size_px(value, _NORMAL_PIXELS)
    return size is not None and size <= TINY_PIXELS


def _folded(value: str) -> str:
    return " ".join(value.lower().split())[:MAX_VALUE_CHARS]


def _is_white(value: str) -> bool:
    return _WHITE.fullmatch(_folded(value)) is not None


def _is_clear(value: str) -> bool:
    """Whether a colour *value* is ``transparent`` or has an alpha of 0 (``rgba(0, 0,
    0, 0)``, ``rgb(0 0 0 / 0%)``, ``#fff0``, ``#ffffff00``): it draws nothing."""
    folded = _folded(value)
    if folded == "transparent" or _SHORT_HEX.fullmatch(folded):
        return True
    found = _COLOUR_FUNCTION.fullmatch(folded)
    if found is None:
        return False
    body = found.group(1)
    if "/" in body:
        alpha = body.rsplit("/", 1)[1].strip()
    else:
        parts = [part for part in re.split(r"[, ]+", body.strip()) if part]
        alpha = parts[3] if len(parts) == 4 else ""
    return _ALPHA_ZERO.fullmatch(alpha) is not None


def _font_size_of(value: str) -> str | None:
    """The size a ``font`` shorthand names (``"0/0 serif"`` gives ``"0"``), if any."""
    for token in value.split():
        size = token.partition("/")[0]
        found = _SIZE.fullmatch(size)
        if found and (found.group(2) or float(found.group(1)) == 0):
            return size
    return None


def _unimportant(value: str) -> str:
    """*value* without a trailing ``!important`` (in any spacing)."""
    bang = value.rfind("!")
    if bang >= 0 and value[bang + 1 :].strip() == "important":
        return value[:bang].rstrip()
    return value


def _declarations(style: str) -> list[tuple[str, str]]:
    """The ``(property, value)`` pairs of an inline *style* in order, both in lower
    case, the value without ``!important`` and read to its first
    :data:`MAX_VALUE_CHARS` characters; CSS comments are dropped."""
    if not style:
        return []
    pairs = []
    for declaration in _CSS_COMMENT.sub("", style).split(";"):
        name, _, value = declaration.partition(":")
        pairs.append(
            (
                name.strip().lower(),
                _unimportant(value.strip().lower())[:MAX_VALUE_CHARS],
            )
        )
    return pairs


def _hides_text(name: str, value: str) -> bool:
    """Whether one style declaration keeps the text of its element from being drawn,
    by the strict rule."""
    if name == "display":
        return value == "none"
    if name == "visibility":
        return value in ("hidden", "collapse")
    if name == "opacity":
        found = _OPACITY.fullmatch(value)
        return found is not None and float(value.rstrip("%")) <= 0
    if name == "font-size":
        return _tiny(value)
    if name == "font":
        size = _font_size_of(value)
        return size is not None and _tiny(size)
    if name == "color":
        return _is_white(value) or _is_clear(value)
    return False


def hidden_style(style: str) -> bool:
    """Whether the inline *style* (an attribute value, character references already
    decoded) keeps the text of its element from being drawn, as a raw file's markup
    says it: ``display: none``, ``visibility: hidden`` or ``collapse``, a font of 0
    to :data:`TINY_PIXELS` pixels (also in a ``font`` shorthand, and a zero of any
    unit, sign or exponent), ``opacity: 0``, or a text colour of white (``white``,
    ``#fff``, ``#ffffff``, ``rgb(255, 255, 255)``, ``rgb(100%, 100%, 100%)``,
    ``hsl(0, 0%, 100%)``) or clear (``transparent``, an alpha of 0). Names and values
    are read in any case and spacing, with ``!important`` and CSS comments, and a
    value to its first :data:`MAX_VALUE_CHARS` characters. This is the strict rule: it
    knows nothing of a dark background behind white text or of a child that sets its
    own size, which the page reader follows."""
    return any(_hides_text(name, value) for name, value in _declarations(style))


def _drawn_never(name: str, value: str) -> bool:
    """Whether one style declaration leaves its element out of the page altogether."""
    return (name == "display" and value == "none") or (
        name == "visibility" and value in ("hidden", "collapse")
    )


# ── Rules of a style sheet that hide an element ──
#
# A ``<style>`` block can hide an element by its class, id or tag. Only simple
# selectors are read: ``.class``, ``#id``, ``tag`` and comma lists of them; anything
# else (``.a .b``, ``a:hover``, ``p.x``) changes nothing.

_SIMPLE_SELECTOR = re.compile(r"([.#]?)(-?[_A-Za-z][\w-]*)")
_BRACES = re.compile(r"[{}]")
# The at-rules whose rules apply under a condition: read by the strict rule only.
_CONDITIONAL = frozenset({"media", "supports", "layer", "container", "document"})
_STYLE_OPEN = re.compile(r"<style(?=[\s/>])", re.IGNORECASE)
_STYLE_CLOSE = re.compile(r"</style(?=[\s/>])", re.IGNORECASE)
_MEDIA = re.compile(
    r"""\bmedia\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE
)


class Sheet(NamedTuple):
    """The elements a style sheet hides: by class, by id and by tag name. *shown*
    holds the classes, ids and tags a rule gives some other display, so an element
    that also carries one of them (``d-none d-md-block``) is not taken as hidden by
    the page reader."""

    classes: frozenset[str] = frozenset()
    ids: frozenset[str] = frozenset()
    tags: frozenset[str] = frozenset()
    shown: frozenset[str] = frozenset()

    def hides(self, tag: str, attrs: Mapping[str, str | None]) -> bool:
        """Whether a rule of the sheet hides an element of this *tag* and these
        attributes (names in lower case, values decoded)."""
        if not (self.classes or self.ids or self.tags):
            return False
        names = (attrs.get("class") or "").split()
        ident = (attrs.get("id") or "").strip()
        if not (
            tag in self.tags
            or ident in self.ids
            or any(name in self.classes for name in names)
        ):
            return False
        return not (
            tag in self.shown
            or ident in self.shown
            or any(name in self.shown for name in names)
        )


def _rules(css: str) -> list[tuple[str, str, bool]]:
    """The rules of a style sheet as ``(selector, declarations, conditional)``, the
    last true for a rule inside ``@media`` and the like. Rules inside any other
    at-rule (``@font-face``, ``@keyframes``) and nested rules are not read."""
    css = _CSS_COMMENT.sub("", css)
    rules: list[tuple[str, str, bool]] = []
    stack: list[str] = []  # "group", "skip" or "rule": what each open brace opens
    starts: list[int] = []
    prelude = 0
    for found in _BRACES.finditer(css):
        if found.group() == "{":
            head = css[prelude : found.start()].rsplit(";", 1)[-1].strip()
            if "rule" in stack or "skip" in stack:
                stack.append("skip")
            elif head.startswith("@"):
                word = re.match(r"@([A-Za-z-]+)", head)
                kind = (
                    "group"
                    if word and word.group(1).lower() in _CONDITIONAL
                    else "skip"
                )
                stack.append(kind)
            else:
                stack.append("rule")
                starts.append(found.end())
                rules.append((head, "", "group" in stack))
        elif stack:
            if stack.pop() == "rule":
                selector, _, conditional = rules[-1]
                rules[-1] = (selector, css[starts.pop() : found.start()], conditional)
        prelude = found.end()
    # A rule never closed has no declarations.
    return [(s, d, c) for s, d, c in rules if d and "{" not in d]


def sheet_of(css: Iterable[str], *, strict: bool) -> Sheet:
    """The :class:`Sheet` of some style blocks' text. The strict reading (a raw file,
    by the note gate) takes every rule :func:`hidden_style` flags, the ones under
    ``@media`` too; the page reader takes only ``display: none`` and ``visibility:
    hidden`` or ``collapse`` in a rule outside any at-rule."""
    classes: set[str] = set()
    ids: set[str] = set()
    tags: set[str] = set()
    shown: set[str] = set()
    for block in css:
        for selector, body, conditional in _rules(block):
            pairs = _declarations(body)
            if strict:
                hiding = any(_hides_text(name, value) for name, value in pairs)
            else:
                hiding = not conditional and any(
                    _drawn_never(name, value) for name, value in pairs
                )
            display = any(
                (name == "display" and value != "none")
                or (name == "visibility" and value == "visible")
                for name, value in pairs
            )
            for item in selector.split(","):
                found = _SIMPLE_SELECTOR.fullmatch(item.strip())
                if found is None:
                    continue
                kind, word = found.groups()
                word = word if kind else word.lower()
                target = classes if kind == "." else ids if kind == "#" else tags
                if hiding:
                    target.add(word)
                if display and not strict:
                    shown.add(word)
    return Sheet(frozenset(classes), frozenset(ids), frozenset(tags), frozenset(shown))


def style_blocks(html: str, *, screen_only: bool = True) -> list[str]:
    """The text of each ``<style>`` block of *html* in order (one never closed runs to
    the end), the ones for the screen only unless *screen_only* is off: a block with
    a ``media`` that names neither ``screen`` nor ``all`` is for another medium. A
    scan for the block's tags, not a parser."""
    blocks: list[str] = []
    at = 0
    while True:
        opening = _STYLE_OPEN.search(html, at)
        if opening is None:
            return blocks
        end = html.find(">", opening.end())
        if end < 0:
            return blocks
        closing = _STYLE_CLOSE.search(html, end + 1)
        stop = closing.start() if closing else len(html)
        media = _MEDIA.search(html, opening.end(), end)
        medium = next((g for g in (media.groups() if media else ()) if g), "")
        if (
            not screen_only
            or not medium
            or any(word in medium.lower() for word in ("all", "screen"))
        ):
            blocks.append(html[end + 1 : stop])
        if closing is None:
            return blocks
        at = closing.end()


def hides(
    tag: str, attrs: Mapping[str, str | None], sheet: Sheet | None = None
) -> bool:
    """Whether an element of this *tag* and these attributes (names in lower case,
    values decoded) is not shown, by the strict rule for a raw file: the ``hidden``
    attribute, ``aria-hidden="true"``, a style that hides it (:func:`hidden_style`),
    the old ``<font color=white>`` (or a clear one), or a rule of the page's
    :class:`Sheet`."""
    return (
        "hidden" in attrs
        or (attrs.get("aria-hidden") or "").strip().lower() == "true"
        or hidden_style(attrs.get("style") or "")
        or (
            tag == "font"
            and (
                _is_white(attrs.get("color") or "")
                or _is_clear(attrs.get("color") or "")
            )
        )
        or (sheet is not None and sheet.hides(tag, attrs))
    )


def _never_drawn(
    tag: str,
    attrs: Mapping[str, str | None],
    style: list[tuple[str, str]],
    sheet: Sheet,
) -> bool:
    """Whether the page reader leaves out the element and all it holds: the
    ``hidden`` attribute, ``aria-hidden="true"``, ``display: none``, ``visibility:
    hidden`` or ``collapse``, by its own style or by a simple rule of the page's
    style sheet (which an inline ``display`` or ``visibility`` overrides)."""
    if (
        "hidden" in attrs
        or (attrs.get("aria-hidden") or "").strip().lower() == "true"
        or any(_drawn_never(name, value) for name, value in style)
    ):
        return True
    if not sheet.hides(tag, attrs):
        return False
    return not any(name in ("display", "visibility") for name, _ in style)


class _Look(NamedTuple):
    """What cascades through the open elements of a page: the font size in pixels,
    whether the text colour is white or clear, and whether the nearest background set
    is one other than white."""

    size: float = _NORMAL_PIXELS
    white: bool = False
    dark: bool = False
    clear: bool = False

    @property
    def shown(self) -> bool:
        """Whether text here is seen: not tiny, not clear, and not white with no
        background."""
        return (
            self.size > TINY_PIXELS
            and not self.clear
            and not (self.white and not self.dark)
        )


def _background(value: str) -> bool | None:
    """What a ``background``, ``background-color`` or ``background-image`` value
    does: ``True`` for a colour that is not white or an image, ``False`` for white,
    ``None`` for nothing (it leaves what is behind as it was, as ``transparent`` and
    an alpha of 0 do)."""
    if "url(" in value or "gradient(" in value:
        return True
    first = re.match(r"(?:rgb|hsl)a?\([^)]*\)|\S+", value)
    if first is None or first.group(0) in _NO_BACKGROUND:
        return None
    token = first.group(0)
    if _is_clear(token):
        return None
    return not _is_white(token)


def _size_after(value: str, around: float) -> float:
    """The font size *value* sets inside an element of size *around*: a size it
    reads, else the one around it (inherit, larger, a negative size a
    browser ignores), else the usual 16 px (a keyword, calc(), a viewport unit)."""
    if value in _INHERITED or value in ("larger", "smaller"):
        return around
    size = _size_px(value, around)
    if size is not None:
        return size
    return around if value.startswith("-") else _NORMAL_PIXELS


def _look_of(
    around: _Look,
    tag: str,
    attrs: Mapping[str, str | None],
    style: list[tuple[str, str]],
) -> _Look:
    """The :class:`_Look` of an element inside one that looks as *around*: what its
    inline style (and ``<font color>``, ``bgcolor``) sets, else what it inherits.
    A value this does not read is taken as the visible one."""
    size, white, dark, clear = around
    if tag == "font" and attrs.get("color"):
        colour = attrs["color"] or ""
        clear = _is_clear(colour)
        white = _is_white(colour) and not clear
    background = (attrs.get("bgcolor") or "").strip().lower()
    if background and (state := _background(background)) is not None:
        dark = state
    clipped = any(
        name in ("background-clip", "-webkit-background-clip") and value == "text"
        for name, value in style
    )
    for name, value in style:
        if name == "font-size":
            size = _size_after(value, size)
        elif name == "font":
            named = _font_size_of(value)
            size = _size_after(named, size) if named else _size_after(value, size)
        elif name == "color" and value not in _INHERITED:
            # Text with a clear colour over a clipped background is the background.
            clear = _is_clear(value) and not clipped
            white = _is_white(value) and not clear
        elif (
            name in ("background", "background-color", "background-image")
            and (state := _background(value)) is not None
        ):
            dark = state
    return _Look(size, white, dark, clear)


@dataclass
class Link:
    text: str
    href: str


@dataclass
class Extracted:
    """What :func:`extract` read: the title, the text and the links of the part kept."""

    title: str = ""
    text: str = ""
    links: list[Link] = field(default_factory=list)
    description: str = ""
    scripts: int = 0


@dataclass
class _Piece:
    text: str
    main: bool
    article: bool
    # A break before what follows: 1 starts a line, 2 leaves a blank line.
    brk: int = 0


class _Reader(HTMLParser):
    def __init__(self, base: str, sheet: Sheet | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.sheet = sheet or Sheet()
        self.pieces: list[_Piece] = []
        self.links: list[tuple[Link, bool, bool]] = []
        self.title = ""
        self.description = ""
        self.scripts = 0
        self._skip: list[str] = []  # the open elements whose content is left out
        # The open elements inside a hidden element a browser closes by itself, the
        # hidden one first; ``None`` when there is none.
        self._auto: list[str] | None = None
        self._auto_count: Counter[str] = Counter()
        # The open elements that are drawn, with how each looks (:class:`_Look`),
        # and how many of each name are open.
        self._looks: list[tuple[str, _Look]] = []
        self._open_count: Counter[str] = Counter()
        self._in_title = False
        self._pre = 0
        self._main = 0
        self._article = 0
        self._region: list[tuple[str, str]] = []  # (tag, "main" | "article")
        self._link: tuple[str, list[str]] | None = None
        self._cells = 0

    # ── helpers ──

    def _put(self, text: str) -> None:
        self.pieces.append(_Piece(text, self._main > 0, self._article > 0))

    def _newline(self, level: int = 1) -> None:
        self.pieces.append(_Piece("", self._main > 0, self._article > 0, level))

    def _look(self) -> _Look:
        return self._looks[-1][1] if self._looks else _Look()

    def _pop(self) -> None:
        name, _ = self._looks.pop()
        self._open_count[name] -= 1

    def _open(
        self,
        tag: str,
        values: Mapping[str, str | None],
        style: list[tuple[str, str]],
    ) -> None:
        """Open a drawn element: first end the auto-closed ones this tag ends, then
        give it its look from the one around it."""
        while self._looks:
            top = self._looks[-1][0]
            if top in AUTO_CLOSED and (tag == top or tag in _ENDED_BY.get(top, ())):
                self._pop()
            else:
                break
        if tag not in VOID:
            self._looks.append((tag, _look_of(self._look(), tag, values, style)))
            self._open_count[tag] += 1

    def _close(self, tag: str) -> None:
        """Close the nearest open element of this name, and what is open inside it;
        an end tag with no start tag open ends nothing."""
        if self._open_count[tag] > 0:
            while True:
                name = self._looks[-1][0]
                self._pop()
                if name == tag:
                    break

    def _auto_pop(self, stack: list[str]) -> str:
        name = stack.pop()
        self._auto_count[name] -= 1
        return name

    def _auto_end(self) -> None:
        self._auto = None
        self._auto_count.clear()

    def _in_auto(self, tag: str, *, closing: bool) -> bool:
        """Whether *tag*, a start or an end tag, is still inside the hidden element a
        browser closes by itself that is open (``_auto``). It ends at its own end
        tag, at the start tag that closes it (the next ``<li>``, the next block after
        a ``<p>``) or at the end tag of an element that was open before it, its
        parent: the page past that point is shown, so one never closed does not
        take the rest of the page with it. When it ends at a tag that is not its
        own, that tag is not inside it."""
        stack = self._auto
        if stack is None:
            return False
        if closing:
            if self._auto_count[tag] > 0:
                while self._auto_pop(stack) != tag:
                    pass
                if not stack:
                    self._auto_end()
                return True
            self._auto_end()  # the parent ends, and the hidden element with it
            return False
        if tag in VOID:
            return True
        while (
            stack
            and stack[-1] in AUTO_CLOSED
            and (tag == stack[-1] or tag in _ENDED_BY.get(stack[-1], ()))
        ):
            self._auto_pop(stack)
        if not stack:
            self._auto_end()
            return False
        stack.append(tag)
        self._auto_count[tag] += 1
        return True

    # ── the parser's calls ──

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): value for name, value in attrs}
        if self._in_auto(tag, closing=False):
            return
        if self._skip:
            if tag == self._skip[-1] and tag not in VOID:
                self._skip.append(tag)
            return
        if tag == "script":
            self.scripts += 1
        if tag == "meta":
            name = (values.get("name") or values.get("property") or "").lower()
            if name in ("description", "og:description") and not self.description:
                self.description = (values.get("content") or "").strip()
            return
        if tag == "title":
            self._in_title = not self.title  # the first title is the page's
            return
        style = _declarations(values.get("style") or "")
        never = _never_drawn(tag, values, style, self.sheet)
        if tag in SKIP or (tag not in VOID and tag not in AUTO_CLOSED and never):
            if tag not in VOID:
                self._skip.append(tag)
            return
        if tag in AUTO_CLOSED and never:
            self._auto = [tag]
            self._auto_count = Counter({tag: 1})
            return
        self._open(tag, values, style)
        shown = self._look().shown
        role = (values.get("role") or "").lower()
        if tag == "main" or role == "main":
            self._main += 1
            self._region.append((tag, "main"))
        elif tag == "article":
            self._article += 1
            self._region.append((tag, "article"))
        if tag in HEADINGS:
            self._newline(2)
            if shown:
                self._put("#" * HEADINGS[tag] + " ")
        elif tag == "li":
            self._newline()
            if shown:
                self._put("- ")
        elif tag == "br":
            self._newline()
        elif tag == "tr":
            self._newline()
            self._cells = 0
        elif tag in ("td", "th"):
            if shown:
                if self._cells:
                    self._put(" | ")
                self._cells += 1
        elif tag in PARAGRAPH:
            self._newline(2)
        elif tag in LINE:
            self._newline()
        if tag == "pre":
            self._pre += 1
        if tag == "a":
            href = (values.get("href") or "").strip()
            self._link = (href, []) if href and shown else None

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # A ``/>`` on an element that is not void opens it, as a browser reads it
        # (``<div hidden/>`` hides what follows); only ``svg`` and ``math``, whose
        # elements are foreign content, close themselves.
        self.handle_starttag(tag, attrs)
        if tag in FOREIGN:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._in_auto(tag, closing=True):
            return
        if self._skip:
            if tag == self._skip[-1]:
                self._skip.pop()
            return
        if tag == "title":
            self._in_title = False
            return
        self._close(tag)
        if tag == "a" and self._link is not None:
            href, words = self._link
            self._link = None
            self._keep_link(href, " ".join(" ".join(words).split()))
        if self._region and self._region[-1][0] == tag:
            _, kind = self._region.pop()
            if kind == "main":
                self._main -= 1
            else:
                self._article -= 1
        if tag == "pre" and self._pre:
            self._pre -= 1
        if tag in HEADINGS or tag in PARAGRAPH:
            self._newline(2)
        elif tag in LINE:
            self._newline()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip or self._auto is not None or not self._look().shown:
            return
        if self._link is not None and data.strip():
            self._link[1].append(data)
        if self._pre:
            self._put(data)
            return
        text = " ".join(data.split())
        if not text:
            if (
                data
                and self.pieces
                and self.pieces[-1].text[-1:] not in ("", " ", "\n")
            ):
                self._put(" ")
            return
        lead = " " if data[:1].isspace() else ""
        trail = " " if data[-1:].isspace() else ""
        self._put(f"{lead}{text}{trail}")

    def _keep_link(self, href: str, text: str) -> None:
        if len(self.links) >= MAX_LINKS or href.startswith("#"):
            return
        try:
            url = urljoin(self.base, href)
            scheme = urlsplit(url).scheme.lower()
        except ValueError:
            return
        if scheme not in ("http", "https"):
            return
        self.links.append(
            (Link(text, url.split("#", 1)[0]), self._main > 0, self._article > 0)
        )


def _joined(pieces: list[_Piece]) -> str:
    """The text of *pieces*: each run of breaks is its strongest one, and lines are tidied."""
    out: list[str] = []
    pending = 0
    for piece in pieces:
        if piece.brk:
            pending = max(pending, piece.brk)
        elif piece.text.strip() or not pending:
            if pending and out:
                out.append("\n" * pending)
            pending = 0
            out.append(piece.text)
    lines: list[str] = []
    for line in "".join(out).split("\n"):
        line = " ".join(line.split())
        if line or (lines and lines[-1]):
            lines.append(line)
    return "\n".join(lines).strip()


def _amount(pieces: list[_Piece]) -> int:
    return sum(len("".join(piece.text.split())) for piece in pieces)


def extract(html: str, base: str) -> Extracted:
    """The title, text and links of *html*, its links made absolute against *base*."""
    reader = _Reader(base, sheet_of(style_blocks(html), strict=False))
    try:
        reader.feed(html)
        reader.close()
    except (AssertionError, ValueError):  # a page the parser cannot finish
        pass
    main = [piece for piece in reader.pieces if piece.main]
    article = [piece for piece in reader.pieces if piece.article]
    if _amount(main) >= MAIN_MIN_CHARS:
        pieces, links = main, [link for link, in_main, _ in reader.links if in_main]
    elif _amount(article) >= MAIN_MIN_CHARS:
        pieces, links = article, [link for link, _, in_art in reader.links if in_art]
    else:
        pieces, links = reader.pieces, [link for link, _, _ in reader.links]
    seen: set[str] = set()
    unique: list[Link] = []
    for link in links:
        if link.href not in seen:
            seen.add(link.href)
            unique.append(link)
    return Extracted(
        title=" ".join(reader.title.split()),
        text=_joined(pieces),
        links=unique,
        description=" ".join(reader.description.split()),
        scripts=reader.scripts,
    )


__all__ = ["Extracted", "Link", "extract", "hidden_style", "hides"]
