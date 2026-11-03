"""Search result pages as lists of hits: a title, an address and a snippet (standard library only).

One reader per kind of page. Each takes the page's text and returns the hits
it can read, in the page's order, never more than :data:`MAX_HITS`; a page it
cannot read (a block page, a changed layout) gives an empty list, never an
error. An address is the page's own address for the result, and a result
without one is left out. Only ``http`` and ``https`` addresses are kept.

The readers do not clean text for the model; :mod:`.search` does that when
it frames the hits, the way ``web_fetch`` frames a page.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlsplit

from localmelo.support.web.extract import SKIP, VOID

# The most hits a reader returns, and the most elements it builds from a page.
MAX_HITS = 30
MAX_NODES = 150_000
MAX_DEPTH = 200


@dataclass(frozen=True)
class Hit:
    """One search result as the engine showed it."""

    title: str
    url: str
    snippet: str


def _words(text: str) -> str:
    return " ".join(text.split())


def _public(url: str) -> str:
    """*url* when it is an absolute http or https address, else ``""``."""
    url = url.strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    if parts.scheme.lower() in ("http", "https") and parts.netloc:
        return url
    return ""


# ── a small element tree ──


class Node:
    """An element of a page: its tag, attributes and children (elements or text)."""

    __slots__ = ("tag", "attrs", "kids", "parent")

    def __init__(
        self, tag: str, attrs: dict[str, str], parent: Node | None = None
    ) -> None:
        self.tag = tag
        self.attrs = attrs
        self.kids: list[Node | str] = []
        self.parent = parent

    def classes(self) -> frozenset[str]:
        return frozenset(self.attrs.get("class", "").split())

    def walk(self) -> Iterator[Node]:
        """This element and every element below it, in document order."""
        stack: list[Node] = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(kid for kid in reversed(node.kids) if isinstance(kid, Node))

    def find(self, match: Callable[[Node], bool]) -> Node | None:
        return next((n for n in self.walk() if n is not self and match(n)), None)

    def text(self, skip: Callable[[Node], bool] | None = None) -> str:
        """The text below this element, scripts and styles left out, white space joined."""
        out: list[str] = []
        stack: list[Node | str] = [self]
        while stack:
            item = stack.pop()
            if isinstance(item, str):
                out.append(item)
                continue
            if item.tag in SKIP or (
                skip is not None and item is not self and skip(item)
            ):
                continue
            stack.extend(reversed(item.kids))
        return _words(" ".join(out))


class _Tree(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", {})
        self._open = self.root
        self._depth = 0
        self._count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._count += 1
        if self._count > MAX_NODES:
            return
        node = Node(tag, {k: v or "" for k, v in attrs}, self._open)
        self._open.kids.append(node)
        if tag not in VOID and self._depth < MAX_DEPTH:
            self._open = node
            self._depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._count += 1
        if self._count <= MAX_NODES:
            self._open.kids.append(
                Node(tag, {k: v or "" for k, v in attrs}, self._open)
            )

    def handle_endtag(self, tag: str) -> None:
        node: Node | None = self._open
        while node is not None and node.tag != tag:
            node = node.parent
        if node is not None and node.parent is not None:
            self._open = node.parent
            depth, walk = 0, self._open
            while walk.parent is not None:
                depth, walk = depth + 1, walk.parent
            self._depth = depth

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._open.kids.append(data)


def tree(html: str) -> Node:
    """The elements of *html*; a damaged page gives what could be read."""
    parser = _Tree()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - a page that cannot be read gives what was read
        pass
    return parser.root


# ── Bing ──


def parse_bing_rss(body: str) -> list[Hit]:
    """The results of Bing's RSS answer (``search?format=rss``)."""
    # An XML parser expands entities a page declares; a feed has none.
    if "<!DOCTYPE" in body or "<!ENTITY" in body:
        return []
    try:
        root = ET.fromstring(body.lstrip("﻿").encode("utf-8"))
    except (ET.ParseError, ValueError):
        return []
    hits: list[Hit] = []
    for item in root.iter("item"):
        title = _words(item.findtext("title") or "")
        url = _public(item.findtext("link") or "")
        if title and url:
            hits.append(Hit(title, url, _words(item.findtext("description") or "")))
        if len(hits) >= MAX_HITS:
            break
    return hits


# ── Baidu ──


def parse_baidu(body: str) -> list[Hit]:
    """The results of Baidu's page: ``div.c-container`` with a title and the page's own address (``mu``)."""
    hits: list[Hit] = []
    seen: set[str] = set()
    for node in tree(body).walk():
        if node.tag != "div" or "c-container" not in node.classes():
            continue
        if "result" not in node.classes() and "result-op" not in node.classes():
            continue
        url = _public(node.attrs.get("mu", ""))
        head = node.find(lambda n: n.tag == "h3")
        if not url or head is None or url in seen:
            continue
        host = (urlsplit(url).hostname or "").lower()
        if host.endswith(".ubs.baidu.com") or host == "nourl.ubs.baidu.com":
            continue
        title = head.text()
        rest = node.text(skip=lambda n: n.tag == "h3")
        if title and rest:
            seen.add(url)
            hits.append(Hit(title, url, rest))
        if len(hits) >= MAX_HITS:
            break
    return hits


# ── any result page, from what a browser read ──

# A result link's label is a title: at least this many characters.
MIN_TITLE = 12
# The text after a title that is taken as its snippet, at most.
SNIPPET_SPAN = 600


_ADDRESS_LIKE = re.compile(r"(?:https?://)?[\w-]+(?:\.[\w-]+)+(?:[/?#]\S*)?")


def _address_like(label: str) -> bool:
    """Whether *label* is an address shown as a link (``example.com/a``), not a title."""
    return _ADDRESS_LIKE.fullmatch(label) is not None or " › " in label


def _own(host: str, own: frozenset[str]) -> bool:
    return any(host == name or host.endswith("." + name) for name in own)


def read_links(
    text: str, links: list[tuple[str, str]], own: frozenset[str]
) -> list[Hit]:
    """Hits from a page read as text and links, for a browser that gives no markup.

    *links* are ``(label, address)`` pairs in page order and *own* the engine's
    own domains. A result is a link to another site whose label reads as a
    title; its snippet is the text between its title and the next result's.
    The first link to an address wins, so a site's breadcrumb and image links
    do not repeat it.
    """
    text = _words(text)  # a label that the page wraps over lines still matches
    chosen: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, href in links:
        label, url = _words(label), _public(href)
        if len(label) < MIN_TITLE or not url or url in seen or _address_like(label):
            continue
        host = (urlsplit(url).hostname or "").lower()
        if not host or _own(host, own):
            continue
        seen.add(url)
        chosen.append((label, url))
    hits: list[Hit] = []
    at = 0
    for index, (label, url) in enumerate(chosen):
        found = text.find(label, at)
        if found < 0:
            hits.append(Hit(label, url, ""))
            continue
        start = found + len(label)
        end = len(text)
        if index + 1 < len(chosen):
            following = text.find(chosen[index + 1][0], start)
            end = following if following >= 0 else end
        hits.append(
            Hit(label, url, _words(text[start : min(end, start + SNIPPET_SPAN)]))
        )
        at = start
        if len(hits) >= MAX_HITS:
            break
    return hits


__all__ = [
    "MAX_HITS",
    "Hit",
    "Node",
    "parse_baidu",
    "parse_bing_rss",
    "read_links",
    "tree",
]
