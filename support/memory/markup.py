"""Raw markup, read for the note gate by an allow-list that fails closed (#442).

A file, or a body, that the chat read may carry markup. What a note may rest on is
what a reader of the page plainly sees. This module does not list the ways a page can
hide text (there are too many, and a style sheet can add more); it lists what is
plainly shown, and everything else supports nothing. A note refused by mistake costs
one note; a note supported by hidden text is the attack. (The page reader of
``web_fetch`` is a separate, lenient parser that is not touched; it flattens a page to
text before the gate sees it.)

The text is cut into tokens left to right (:func:`_tokens`), and nothing is read inside
a token: a comment, a bogus comment, a raw text element (``script``, ``style``,
``textarea``, ``title``, ``xmp``, ``noscript``, ``iframe``, ``noembed``, ``noframes``
with all they hold, to their own end tag or to the end of the text; ``plaintext``
always to the end), or a tag with its attributes. A Markdown autolink
(``<https://example.com>``, ``<user@example.com>``), ``a < b`` and ``<3`` are no tags
and stay text. A name that runs to the end of the text is an unfinished tag, dropped.
A comment ends only at ``-->`` (``<!-->`` and ``--!>`` do not end it early).

Then the rule, with no style, no sheet and no colour read at all:

- **Sheets.** If the text has a ``<style>`` element, a ``<link>`` whose ``rel`` has
  ``stylesheet`` in it, or an XML style sheet instruction, the rules of a sheet could
  hide anything, so nothing in the text supports a note (:func:`without_markup`
  returns an empty text): a styled HTML file supports no note.
- **No sheets.** An element's content counts only when its tag is in
  :data:`ALLOWED_TAGS` and every attribute it carries is in :data:`ALLOWED_ATTRS`
  (and none of them is a ``width`` or ``height`` of 0). Any other tag (unknown names,
  ``font``, ``svg``, ``select``, ``dialog``, ``details``, ``template`` ...) or any
  other attribute (``style``, ``hidden``, ``aria-hidden``, ``bgcolor``, ``color``,
  ``data-*``, ``on*`` ...) hides the element's whole content. Text counts only when
  every open element is allowed.
- **Extents fail closed.** A disallowed element ends only at an end tag of its own
  name, and only when every element opened inside it has been closed in order. On
  anything else (a mismatched end tag, an unclosed child, ``</body>``, the end of the
  text) it runs to the end of the text.

Every tag that is left is replaced by a space. Elements are tracked with a stack of the
names opened inside a disallowed element, so the pass is linear in the text.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from typing import Final, NamedTuple

# The tags whose content is plainly shown. Nothing else is: not ``head``, ``font``,
# ``center``, ``svg``, ``details``, ``template``, ``select``, ``label``, ``button``...
ALLOWED_TAGS: Final = frozenset(
    {
        "html",
        "body",
        "main",
        "article",
        "section",
        "header",
        "footer",
        "nav",
        "aside",
        "div",
        "span",
        "p",
        "br",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "td",
        "th",
        "caption",
        "blockquote",
        "pre",
        "code",
        "kbd",
        "samp",
        "b",
        "i",
        "u",
        "s",
        "em",
        "strong",
        "small",
        "mark",
        "sub",
        "sup",
        "cite",
        "q",
        "abbr",
        "time",
        "address",
        "figure",
        "figcaption",
        "a",
        "img",
        "wbr",
    }
)
# The attributes an allowed element may carry. Any other (``style``, ``hidden``,
# ``aria-hidden``, ``bgcolor``, ``color``, ``text``, ``fill``, ``data-*``, ``on*``...)
# hides the element's content.
ALLOWED_ATTRS: Final = frozenset(
    {
        "href",
        "title",
        "lang",
        "dir",
        "class",
        "id",
        "colspan",
        "rowspan",
        "headers",
        "scope",
        "alt",
        "src",
        "width",
        "height",
        "start",
        "reversed",
        "value",
        "datetime",
        "cite",
        "name",
        "target",
        "rel",
    }
)

# A ``width`` or ``height`` of zero (any sign, unit or exponent) or less.
_ZERO_SIZE = re.compile(r"\s*[+-]?(?:0+(?:\.0*)?|\.0+)(?![0-9.])|\s*-")
# The names of the attributes that hide an element drawn as an autolink: the text of
# ``<https://x/style=display:none>`` is a tag as a browser reads it.
_HIDING_NAMES: Final = frozenset({"style", "hidden", "popover", "aria-hidden"})

# ── Tokens ──

_RAW: Final = frozenset(
    {
        "script",
        "style",
        "textarea",
        "title",
        "xmp",
        "noscript",
        "iframe",
        "noembed",
        "noframes",
        "plaintext",
    }
)
# What follows a tag name in a tag: attributes up to the first ``>`` that is not in a
# quoted value. A quote opens a value only after ``=``; elsewhere it is part of a name,
# and an unquoted value runs to white space or ``>``. A quote never closed runs to the
# end of the text. Possessive, so a run is never scanned twice.
_ATTRS: Final = (
    r"(?:(?:[^>=\"'/]|/(?!>))++"
    r"|=\s*+(?:\"[^\"]*+(?:\"|\Z)|'[^']*+(?:'|\Z)|[^\s>]*+)"
    r"|[\"'])*+"
)
# The end tag of a raw text element: ``</name`` and then white space (as HTML spells
# it), ``/`` or ``>``, then attributes (a quoted ``>`` does not end it) up to ``>``.
_RAW_END: Final = {
    name: re.compile(
        rf"</{name}(?=[\t\n\f\r />]){_ATTRS}/?+(?:>|\Z)", re.IGNORECASE | re.DOTALL
    )
    for name in _RAW
}
# One token. A comment ends at ``-->`` only, never earlier: ``<!-->``, ``<!--->`` and
# ``--!>`` end it in a browser, but a reader that waits for ``-->`` hides more, never
# less. A bogus comment is ``<!...>``, ``<?...>`` or ``</`` and a non-letter, each to
# the first ``>``. An autolink is the only thing that starts with ``<`` and is
# no tag: a scheme of :data:`_SCHEMES`, ``:`` and up to a quote, white space or ``<``
# (``<https://example.com>``), or an address with no ``/`` (``<user@example.com>``).
# A tag is ``<``, an optional ``/``, a name that runs to white space (as HTML spells
# it), ``/`` or ``>`` (a NUL included), and attributes. A tag or quote never ended
# runs to the end of the text, as a browser drops it.
_SCHEMES: Final = "https?|mailto|ftps?|file|tel|sms"
_TAG_PATTERN: Final = (
    r"<(?P<close>/?+)(?P<name>[A-Za-z][^\t\n\f\r />]*+)"
    rf"(?P<attrs>{_ATTRS})"
    r"(?P<self>/?+)(?:>|\Z)"
)
_TOKEN = re.compile(
    r"(?P<comment><!--.*?(?:-->|\Z))"
    r"|(?P<bogus><[!?][^>]*+(?:>|\Z)|</(?![A-Za-z])[^>]*+(?:>|\Z))"
    rf"|(?P<auto><(?:(?:{_SCHEMES}):(?P<rest>[^\s<>\"']*+)"
    r"|[^\s<>@/\"']++@[^\s<>/\"'.]++(?:\.[^\s<>/\"'.]++)++)>)"
    rf"|(?P<tag>{_TAG_PATTERN})",
    re.DOTALL | re.IGNORECASE,
)
_TAG_AT = re.compile(_TAG_PATTERN, re.DOTALL)
_ATTRIBUTE = re.compile(
    r"([^\s/>=][^\s/>=]*+)"
    r"(?:\s*+=\s*+(?:\"([^\"]*+)\"?+|'([^']*+)'?+|([^\s>]*+)))?+"
)
# Elements that have no content and no end tag.
_VOID: Final = frozenset(
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
# Elements whose ``/>`` closes them (foreign content); only :func:`has_stray_closer`
# reads this, to count what a part of a page opens and closes.
_FOREIGN: Final = frozenset({"svg", "math"})
# What can load or hold a style sheet outside a ``<style>`` element: an XML style
# sheet instruction, a ``<style>`` anywhere (a comment, a script or a ``<noscript>``
# included, since a reader may run it), a ``<link>`` with ``stylesheet`` in it.
_SHEET_HINT = re.compile(
    r"<style\b|xml-stylesheet|<link\b[^>]{0,300}stylesheet", re.IGNORECASE
)
_LINK_START = re.compile(r"<link\b", re.IGNORECASE)

_TAG: Final = 1
_DROP: Final = 2


class _Token(NamedTuple):
    kind: int  # _TAG, or _DROP (a comment, a bogus comment, a raw text element)
    start: int
    end: int
    name: str  # lower case; of a raw text element too
    attrs: str
    close: bool
    self_closing: bool


def _tokens(text: str) -> list[_Token]:
    """The tokens of *text*, left to right. Nothing is read inside one: a ``<!--`` in a
    quoted attribute value or in a script starts no comment, and a raw text element
    ends only at its own end tag."""
    out: list[_Token] = []
    at = 0
    size = len(text)
    while True:
        found = _TOKEN.search(text, at)
        if found is None:
            return out
        kind = found.lastgroup
        if kind == "auto":
            rest = found["rest"]
            if rest is None or not _HIDING_NAMES.intersection(_attributes(rest)):
                at = found.end()
                continue
            # ``<https://x/style=display:none>`` is an element to a browser.
            found = _TAG_AT.match(text, found.start())
            if found is None:  # not reachable: an autolink is a tag too
                at = text.find(">", at) + 1 or size
                continue
            kind = "tag"
        at = found.end()
        if kind in ("comment", "bogus"):
            out.append(_Token(_DROP, found.start(), at, "", "", False, False))
            continue
        name = found["name"].lower()
        close = bool(found["close"])
        if name in _RAW and not close:
            end = size
            if name != "plaintext":
                closing = _RAW_END[name].search(text, at)
                stop = closing.start() if closing else size
                # A ``<!--`` before the end tag may change where the element ends in
                # a browser (a script that escapes itself): in doubt it runs to the end.
                if closing and text.find("<!--", at, stop) < 0:
                    end = closing.end()
            out.append(_Token(_DROP, found.start(), end, name, "", False, False))
            at = end
            continue
        out.append(
            _Token(
                _TAG,
                found.start(),
                at,
                name,
                found["attrs"],
                close,
                bool(found["self"]),
            )
        )


def _attributes(text: str) -> dict[str, str]:
    """The attributes of a tag's attribute text, names in lower case and values with
    their character references decoded; the first of a name wins, as in a browser."""
    values: dict[str, str] = {}
    for found in _ATTRIBUTE.finditer(text):
        value = next((g for g in found.groups()[1:] if g is not None), "")
        values.setdefault(found.group(1).lower(), html.unescape(value))
    return values


def shown(name: str, attributes: Mapping[str, str]) -> bool:
    """Whether the content of an element of this tag *name* (lower case) with these
    attributes (names in lower case, values decoded) is plainly shown: the tag is in
    :data:`ALLOWED_TAGS`, every attribute is in :data:`ALLOWED_ATTRS`, and no
    ``width`` or ``height`` is 0. Anything else hides it."""
    if name not in ALLOWED_TAGS:
        return False
    for key, value in attributes.items():
        if key not in ALLOWED_ATTRS:
            return False
        if key in ("width", "height") and _ZERO_SIZE.match(value):
            return False
    return True


def has_sheets(text: str, tokens: Iterable[_Token] | None = None) -> bool:
    """Whether *text* has, or may have, a style sheet: a ``<style>`` anywhere, an XML
    style sheet instruction, a ``<link>`` with ``stylesheet`` in its ``rel`` (also one
    in a ``<noscript>``)."""
    if _SHEET_HINT.search(text):
        return True
    for token in _tokens(text) if tokens is None else tokens:
        if token.kind == _TAG:
            if token.name == "link" and not token.close:
                rel = _attributes(token.attrs).get("rel") or ""
                if "stylesheet" in rel.lower():
                    return True
        elif token.name == "noscript" and _LINK_START.search(
            text, token.start, token.end
        ):
            return True
    return False


# ── Cutting ──


def cut(text: str, spans: Iterable[tuple[int, int, str]]) -> str:
    """*text* with each span ``(start, end, replacement)`` (in order, not overlapping)
    replaced. A line that held only what is removed (an empty replacement) goes with
    its line break, so the paragraph around it does not split in two. One pass: each
    span costs only the white space beside it."""
    kept: list[str] = []
    done = 0
    size = len(text)
    for start, end, replacement in spans:
        if start < done:
            continue
        kept.append(text[done:start])
        kept.append(replacement)
        done = end
        if replacement:
            continue
        before = start
        while before > 0 and text[before - 1] != "\n" and text[before - 1].isspace():
            before -= 1
        if before > 0 and text[before - 1] != "\n":
            continue
        after = end
        while after < size and text[after] != "\n" and text[after].isspace():
            after += 1
        if after < size and text[after] == "\n":
            done = after + 1
    kept.append(text[done:])
    return "".join(kept)


def _spans(text: str, tokens: list[_Token]) -> Iterator[tuple[int, int, str]]:
    """What of *text* goes, in order: comments and raw text elements for nothing, a
    disallowed element (with all it holds) for nothing, every other tag for a space.

    A disallowed element ends at the end tag of its own name when every element opened
    inside it (:data:`_VOID` ones aside) has been closed in order. On any other end
    tag, or at the end of the text, it runs to the end of the text."""
    region = ""  # the name of the disallowed element we are in
    begins = 0
    inside: list[str] = []  # the elements opened inside it, innermost last
    for token in tokens:
        if not region:
            if token.kind != _TAG:
                yield token.start, token.end, ""
            elif token.close:
                yield token.start, token.end, " "
            else:
                attributes = _attributes(token.attrs) if token.attrs.strip() else {}
                if shown(token.name, attributes):
                    yield token.start, token.end, " "
                elif token.name in _VOID:
                    yield token.start, token.end, ""
                else:
                    region, begins = token.name, token.start
                    inside.clear()
            continue
        if token.kind != _TAG:
            if text.startswith("<![CDATA[", token.start):
                break  # CDATA is text in foreign content, markup elsewhere: unclear
            continue
        if not token.close:
            if token.name not in _VOID:
                inside.append(token.name)
        elif inside:
            if inside[-1] != token.name:
                break
            inside.pop()
        elif token.name == region:
            yield begins, token.end, ""
            region = ""
        else:
            break
    if region:
        yield begins, len(text), ""


def without_markup(text: str) -> str:
    """What of *text* a reader plainly sees, without its markup: comments, bogus
    comments and raw text elements (``script``, ``style``, ``textarea``, ``title``,
    ``noscript`` ...) are left out; so is every element that is not plainly shown
    (:func:`shown`), with what it holds; every other tag is replaced by a space. A
    text with a style sheet (:func:`has_sheets`) supports nothing, so it comes back
    empty. A comment, tag or element never closed runs to the end of the text. Text
    with no ``<`` comes back as it is."""
    if "<" not in text:
        return text
    tokens = _tokens(text)
    if not tokens:
        return text
    if has_sheets(text, tokens):
        return ""
    return cut(text, _spans(text, tokens))


def has_stray_closer(text: str) -> bool:
    """Whether *text*, a part of a page or file read without the part before it, holds
    a closer with no opener in it: a ``-->`` outside a comment, or an end tag of an
    element that is not open in the part. Such a part may begin inside a comment or a
    hidden element."""
    last = 0
    open_names: list[str] = []
    count: Counter[str] = Counter()
    for token in _tokens(text):
        if text.find("-->", last, token.start) >= 0:
            return True
        last = token.end
        if token.kind != _TAG:
            continue
        name = token.name
        if token.close:
            if count[name] <= 0:
                return True
            while True:
                popped = open_names.pop()
                count[popped] -= 1
                if popped == name:
                    break
        elif name not in _VOID and not (
            token.self_closing
            and (name in _FOREIGN or count["svg"] + count["math"] > 0)
        ):
            open_names.append(name)
            count[name] += 1
    return text.find("-->", last) >= 0
