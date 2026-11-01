"""An HTML page as readable text: its title, headings, main text and links (standard library only).

Scripts, styles, templates, embedded documents and form choices are left
out, and so is what a page hides from people (the ``hidden`` attribute,
``aria-hidden="true"``, ``display: none`` or ``visibility: hidden`` in its
style), since hidden text is a common place for instructions aimed at a
model. When the page marks its main part (``<main>``, ``role="main"`` or
``<article>``) with enough text in it, only that part is kept, so menus and
footers do not crowd it out. Headings read ``# Title``, list items ``- item``
and table cells are joined with `` | ``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
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
# Elements a browser closes by itself; a hidden one is not skipped, since its
# end may never come and the rest of the page would go with it.
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
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden")


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
    def __init__(self, base: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.pieces: list[_Piece] = []
        self.links: list[tuple[Link, bool, bool]] = []
        self.title = ""
        self.description = ""
        self.scripts = 0
        self._skip: list[str] = []  # the open elements whose content is left out
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

    @staticmethod
    def _hidden(attrs: dict[str, str | None]) -> bool:
        if "hidden" in attrs:
            return True
        if (attrs.get("aria-hidden") or "").strip().lower() == "true":
            return True
        return bool(_HIDDEN_STYLE.search((attrs.get("style") or "").lower()))

    # ── the parser's calls ──

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): value for name, value in attrs}
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
        if tag in SKIP or (
            tag not in VOID and tag not in AUTO_CLOSED and self._hidden(values)
        ):
            if tag not in VOID:
                self._skip.append(tag)
            return
        role = (values.get("role") or "").lower()
        if tag == "main" or role == "main":
            self._main += 1
            self._region.append((tag, "main"))
        elif tag == "article":
            self._article += 1
            self._region.append((tag, "article"))
        if tag in HEADINGS:
            self._newline(2)
            self._put("#" * HEADINGS[tag] + " ")
        elif tag == "li":
            self._newline()
            self._put("- ")
        elif tag == "br":
            self._newline()
        elif tag == "tr":
            self._newline()
            self._cells = 0
        elif tag in ("td", "th"):
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
            self._link = (href, []) if href else None

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._skip:
            if tag == self._skip[-1]:
                self._skip.pop()
            return
        if tag == "title":
            self._in_title = False
            return
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
        if self._skip:
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
    reader = _Reader(base)
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


__all__ = ["Extracted", "Link", "extract"]
