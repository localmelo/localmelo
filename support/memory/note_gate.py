"""The note gate: where a note the chat's model keeps with ``remember`` may come from.

Developer setting ``[memory] note_gate`` (#442), four cumulative levels. A note is
kept when something the chat shows supports it, and the gate tags the note with that
source (source monitoring): what the user typed (``typed``), material the user
pointed to (``pointed``), a finished task's result (``result``) or the model's own
inference from what the user typed (``inferred``), in that order of trust.

- ``typed``, the default, is the rule that always was: the note's words are in what
  the user typed in this chat or in a finished task's result shown in it
  (:func:`records.unsupported`). This module is not asked at all.
- ``pointed`` adds the page or file the user pointed to: one the chat fetched or read
  (:func:`pointed_material`, from the chat's stored tool events) whose address or
  file name the user typed, or that the user's words in that turn call theirs
  (:data:`THEIRS`). A page the user did not point to is no source at any level, and a
  page the user called someone else's supports no note about the user.
- ``meaning`` adds support by meaning: the note's key tokens (:func:`key_tokens`:
  numbers, dates, addresses, e-mail addresses, names) must be found in a source, and
  the rest of it may say the same in other words or another language, judged by the
  cosine between its embedding and the best window of the sources
  (:func:`windows`). Without an embedding model that is the ``typed`` word check.
- ``model`` adds the model's own judgement: a note about the user that no source
  supports, whose key tokens are all found and whose meaning is close to what the user
  typed (the bar less :data:`INFERRED_SLACK`), is kept as ``inferred``.

Every level but ``typed`` also refuses a note that is an instruction to the assistant
(:func:`is_instruction`), not a fact about the user. Secrets are refused at every
level before this module is asked.

What a page, a file or a finished task's result may support is what it *states*, not
what it hides or asks the assistant to do. Material (a page or a file) goes through
:func:`material_text`: one left-to-right scan (:mod:`markup`) leaves out the comments,
the raw text elements and every element that is not plainly drawn, then the tags; then
every paragraph that orders the assistant is dropped. The scan is an allow-list and
fails closed (no style or colour is parsed): text counts only outside every element
that is not a plain one (:data:`markup.ALLOWED_TAGS`, :data:`markup.ALLOWED_ATTRS`),
and a text with a style sheet supports no note at all. A note wrongly refused costs
little and a forged one costs more. A fetched page has been flattened to text by the
page reader, which is not changed: white, tiny, transparent or class-hidden text on a
fetched page reaches this module as plain text. A part of a page or file read without
the part before it is dropped when it holds a closer with no opener (``-->``, a stray
end tag), since it may begin inside a comment or a hidden element; every later part of
a read in pieces needs the one before it, in order. A finished task's result is melo's
own answer, so :func:`result_text` reads no markup in it and drops only the paragraphs
that carry an unmistakable injection form (an order to ignore the rules, a role
header, an order to save to the assistant's memory, a lead that names an AI). A page
read in the browser supports no note, since the browser's text is not the page's
markup. What the user typed is never cleaned.
"""

from __future__ import annotations

import bisect
import contextlib
import html
import itertools
import os
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, NamedTuple
from urllib.parse import unquote, urlsplit

import numpy as np

from localmelo.melo.agent._step_review import named_sites
from localmelo.support.config import MEMORY_NOTE_GATE
from localmelo.support.memory import markup, records
from localmelo.support.memory.index import CHUNK_OVERLAP

# The sources a note is tagged with, most trusted first.
TYPED: Final = "typed"
POINTED: Final = "pointed"
RESULT: Final = "result"
INFERRED: Final = "inferred"

# The default bar for the meaning of a note against its source, and how much lower the
# bar for a note the model infers is.
MIN_COSINE: Final = 0.55
INFERRED_SLACK: Final = 0.15
# A window of a source, in characters, and the most that one call compares: the user's
# newest typed windows, the pointed material's windows that hold most key tokens, and
# the results'.
WINDOW_CHARS: Final = 300
MIN_WINDOW_CHARS: Final = 8
TYPED_WINDOWS: Final = 16
POINTED_WINDOWS: Final = 40
RESULT_WINDOWS: Final = 8
# The pages and files of one chat that count, newest first.
MAX_MATERIAL: Final = 20

NOT_A_FACT: Final = "Not remembered: an instruction, not a fact about the user."
NOT_BROWSER: Final = (
    "Not remembered: that came from a page read in the browser, which shows melo "
    "text a reader does not see (labels, hidden controls), so it supports no note. "
    "Keep only what the user told you, or a page fetched with web_fetch."
)
NOT_THEIRS: Final = (
    "Not remembered: that came from a page or file that is not the user's own, "
    "so it says nothing about the user. Keep only what it says about someone else."
)
_WHERE: Final = (
    "what the user typed in this chat, a page or file they pointed to, or a "
    "finished task's result shown here"
)
NOT_SUPPORTED: Final = f"Not remembered: no source supports it. Words not in {_WHERE}: "
NOT_FOUND: Final = f"not found in {_WHERE}."
TOO_FAR: Final = (
    f"Not remembered: its meaning is not close enough to {_WHERE}. Keep only what "
    "those show."
)

# ── The pointed material ──

# What the user says of a page or file that makes it theirs, as a fold of the typed
# words holds it (lower case, accents dropped: "résumé" is "resume").
THEIRS: Final = (
    "my site",
    "my website",
    "my web site",
    "my web page",
    "my webpage",
    "my page",
    "my homepage",
    "my home page",
    "my resume",
    "my cv",
    "my portfolio",
    "my blog",
    "my profile",
    "我的网站",
    "我的简历",
    "我的主页",
)
_THEIRS = re.compile(
    "|".join(rf"\b{re.escape(p)}\b" if p.isascii() else re.escape(p) for p in THEIRS)
)
# "My profile at github.com/jane", "my resume.pdf": the user names what the phrase
# means, in a few words after it, so the phrase does not stand for any other page.
_BOUND = re.compile(
    "(?:"
    + "|".join(rf"\b{re.escape(p)}\b" if p.isascii() else re.escape(p) for p in THEIRS)
    + r")(?:\.[a-z0-9]{1,5}\b|(?:[\s,:;()\[\]\-\u2013\u2014]|\b(?:at|on|in|is|are|its?|here|see"
    r"|which|that|located|hosted|url|link|address|page|the|this)\b|['\u2019]s)*"
    r"(?:https?://)?(?:[\w-]+\.)+[a-z0-9]{2,}\b)"
)
_THING = (
    "page|site|website|web page|webpage|profile|resume|cv|homepage|portfolio|blog"
    "|file|document"
)
# What the user says of a page or file that makes it someone else's ("a page about
# Bob", "Bob's site", "his resume"), as a fold of the typed words holds it.
_OTHERS = re.compile(
    rf"\b(?:a|an|the|this|that)\s+(?:{_THING})\s+(?:about|of|for|from|by)\s+"
    r"(?!me\b|my\b|myself\b)\w"
    r"|\b(?!(?:it|that|there|here|what|who|he|she|let|where|when|how|my|our|your)"
    rf"['’]s\b)[a-z][\w-]*['’]s\s+(?:{_THING})\b"
    rf"|\b(?:his|her|their|someone else's|another person's)\s+(?:{_THING})\b"
    r"|(?:他|她|他们|她们|别人|朋友|同事|老师|客户|对方)的(?:网站|简历|主页|博客|页面)"
    r"|关于\S{1,12}的(?:页面|网页|网站|简历)"
)
# The tools that read a page. The address is in the call, or, for a page the browser
# already holds, in the page block's ``origin=``.
_PAGE_CALLS: Final = {"web_fetch": "url", "browser_open": "url"}
_PAGE_BLOCKS: Final = frozenset({"browser_read", "browser_follow", "browser_back"})
_FILE_TOOLS: Final = frozenset({"file_read", "read", "read_file", "read_text_file"})
_FILE_KEYS: Final = ("path", "file_path", "filename")
_PAGE_ALL: Final = frozenset(_PAGE_CALLS) | _PAGE_BLOCKS
# The tools that read a page in the browser: the page text they return is the
# browser's own view, not the page's markup.
BROWSER_TOOLS: Final = frozenset({"browser_open"}) | _PAGE_BLOCKS
_ORIGIN = re.compile(r"<<page [^>\n]*?\borigin=([^\s>]+)")
_TAB = re.compile(r"<<page [^>\n]*?\btab=([^\s>]+)")
_MARKER = re.compile(r"^<<(?:end )?page .*>>[ \t]*$", re.MULTILINE)
# What ends a typed address: the user's sentence goes on after it.
_ADDRESS_END = "\\s<>\"'`，。）)、；：！？"
_PATH_END = ".,;:!?)]}>'\"/"


class Material(NamedTuple):
    """A page or file the user pointed to, as this chat read it.

    *ref* is the address (a page) or the file name, *text* what the tool returned
    that may support a note (:func:`material_text`: no hidden text, no order to the
    assistant, never empty) and *other* says the user called it someone else's.
    *browser* says the page was read in the browser (:data:`BROWSER_TOOLS`): what the
    browser shows melo is not what a reader sees (its labels and hidden controls), so
    such a page supports no note and its *text* only tells why a note is refused.
    """

    kind: str  # "page" or "file"
    ref: str
    text: str
    other: bool
    browser: bool = False


def fold(text: str) -> str:
    """*text* in lower case without accents, for comparing words."""
    plain = unicodedata.normalize("NFKD", text)
    return "".join(c for c in plain if not unicodedata.combining(c)).casefold()


class Place(NamedTuple):
    """Where a page is: its *host* (as :func:`named_sites` has it) and its *path*.

    The path is ``"/jane"`` or ``"/p?id=3"``: percent-decoded, without a trailing
    slash, so the root is ``""``. A page the browser read without its address has no
    path (``None``). In the words the user typed, an empty path is the whole host.
    """

    host: str
    path: str | None


def _tail(path: str, query: str) -> str:
    """A path and its query as places compare them."""
    return (unquote(path) + (f"?{query}" if query else "")).rstrip(_PATH_END)


def _place_of(address: str) -> Place | None:
    """The :class:`Place` of a page address (the scheme may be left out)."""
    text = address.strip()
    try:
        parts = urlsplit(text if "://" in text else "//" + text)
        netloc = parts.netloc.rpartition("@")[2]
    except ValueError:
        return None
    host = next(iter(named_sites("http://" + netloc)), "") if netloc else ""
    return Place(host, _tail(parts.path, parts.query)) if host else None


def _spellings(host: str) -> set[str]:
    """The ways a host may be typed: as is, and in Unicode if it is punycode."""
    names = {host}
    with contextlib.suppress(UnicodeError):
        names.add(host.encode("ascii").decode("idna"))
    return names


def typed_places(text: str) -> list[Place]:
    """The places *text* (what the user typed) names: a host alone is every page of
    it (the path ``""``), an address with a path is that page and what is below it.

    The scheme, ``www.``, a trailing slash, a fragment and the punctuation after
    the address are not part of it: ``see https://www.github.com/jane/.`` names
    ``github.com`` at ``/jane``.
    """
    places: list[Place] = []
    for host in sorted(named_sites(text)):
        names = "|".join(re.escape(name) for name in sorted(_spellings(host)))
        pattern = re.compile(
            r"(?<![\w.@-])(?:https?://)?(?:www\.)?(?:" + names + r")(?::\d+)?"
            rf"(?P<path>/[^{_ADDRESS_END}]*)?",
            re.IGNORECASE,
        )
        found = list(pattern.finditer(text))
        for match in found:
            rest = urlsplit(match.group("path") or "")
            places.append(Place(host, _tail(rest.path, rest.query)))
        if not found:
            places.append(Place(host, ""))
    return places


def _covers(typed: Place, page: Place) -> bool:
    """Whether the place the user typed holds *page*: the same host and, for a typed
    path, that path or one below it (``/jane`` holds ``/jane/repo``, not ``/janet``)."""
    prefix = typed.path or ""
    if typed.host != page.host:
        return False
    if not prefix:
        return True
    path = page.path
    if path is None or not path.startswith(prefix):
        return False
    return len(path) == len(prefix) or path[len(prefix)] in "/?&"


class _Seen(NamedTuple):
    """A page or file a turn read: for a page its host, path and the browser's view
    (one landing of a tab: a read of the same tab page repeats the view)."""

    kind: str
    key: str
    path: str | None
    view: int | None


def _again(first: _Seen, seen: _Seen) -> bool:
    """Whether *seen* is the page or file that *first* is (or a read of it again)."""
    if first == seen:
        return True
    if first.kind != seen.kind or first.key != seen.key:
        return False
    if seen.kind == "file":
        return True
    return (seen.view is not None and seen.view == first.view) or (
        seen.path is not None and seen.path == first.path
    )


def _file_of(call: Mapping[str, Any]) -> str:
    """The path a file-reading call named, or ``""``."""
    arguments = call.get("arguments")
    if str(call.get("name") or "") not in _FILE_TOOLS or not isinstance(
        arguments, Mapping
    ):
        return ""
    return next(
        (str(arguments[k]) for k in _FILE_KEYS if isinstance(arguments.get(k), str)),
        "",
    )


def reads(name: object) -> bool:
    """Whether a tool of this name reads a page or a file (:func:`pointed_material`
    looks at the calls of these only)."""
    return isinstance(name, str) and (
        name in _PAGE_CALLS or name in _PAGE_BLOCKS or name in _FILE_TOOLS
    )


class _Tabs:
    """The browser's tabs as the chat's page reads show them, in order.

    A page block names its tab and host, not its address. Opening an address puts
    that page in the tab; reading the tab again shows the same page; following a
    link or going back lands on a page whose address is not known. So a page read
    is placed by what the chat did to its tab, and a read after a move has no path.
    """

    def __init__(self) -> None:
        self._views = itertools.count(1)
        self._tabs: dict[str, tuple[int, Place]] = {}

    def land(
        self, call: Mapping[str, Any], output: str
    ) -> tuple[Place, int | None, str] | None:
        """The place a finished page read is of, the browser's view of it (``None``
        for a fetch) and the address to show; ``None`` when it has none."""
        name, arguments = str(call.get("name") or ""), call.get("arguments")
        asked = ""
        if isinstance(arguments, Mapping):
            value = arguments.get(_PAGE_CALLS.get(name, ""))
            asked = value.strip() if isinstance(value, str) else ""
        if name == "web_fetch":
            place = _place_of(asked)
            return (place, None, asked) if place else None
        origin = _ORIGIN.search(output)
        landed = _place_of(origin.group(1)) if origin else None
        tab = _TAB.search(output)
        tab_id = tab.group(1) if tab else ""
        if name == "browser_open":
            place = _place_of(asked)
            if place is None or (landed is not None and landed.host != place.host):
                return None  # not the page that was asked for: a redirect led away
            return place, self._set(tab_id, place), asked
        if landed is None or origin is None:
            return None
        here = self._tabs.get(tab_id)
        if name == "browser_read" and here is not None and here[1].host == landed.host:
            return here[1], here[0], origin.group(1)
        return (
            Place(landed.host, None),
            self._set(tab_id, Place(landed.host, None)),
            origin.group(1),
        )

    def _set(self, tab: str, place: Place) -> int:
        view = next(self._views)
        if tab:
            self._tabs[tab] = (view, place)
        return view


class _Read(NamedTuple):
    """One finished read that the user pointed to: what the tool returned, and where
    in the page or file it began (*start*: the part, or the line, 1 for the first)
    and where the one after it would (*after*: ``None`` when that is not known)."""

    kind: str
    ref: str
    text: str
    other: bool
    browser: bool
    start: int
    after: int | None


@dataclass(eq=False)
class _Group:
    """The reads of one page or file, one after the other, that make one text: where
    the first began (*begin*), where the last one did (*last*) and where the one after
    it would (*after*, ``None`` when that is not known)."""

    kind: str
    ref: str
    texts: list[str]
    other: bool
    browser: bool
    begin: int
    last: int
    after: int | None


def _number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _span_of(call: Mapping[str, Any]) -> tuple[int, int | None]:
    """Where a read began and where the next one in order would: the part of a
    ``web_fetch`` (default 1), the first and last line of a file read (``start_line``,
    default 1, and ``end_line``); ``(1, None)`` for any other call."""
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping):
        return 1, None
    name = str(call.get("name") or "")
    if name == "web_fetch":
        part = _number(arguments.get("part")) or 1
        return max(part, 1), max(part, 1) + 1
    if name in _FILE_TOOLS:
        start = _number(arguments.get("start_line")) or 1
        end = _number(arguments.get("end_line"))
        return max(start, 1), None if end is None else end + 1
    return 1, None


def _assembled(reads: Sequence[_Read], limit: int = MAX_MATERIAL) -> list[Material]:
    """The materials of the *reads*, oldest first, the newest *limit* of them: the
    reads of one page or file in parts, each part right after the one before it (the
    one before it must say where it ended), are joined and read as one text
    (:func:`material_text`). Any other read starts a text of its own.

    A text that does not start at part 1 or line 1 may begin inside a comment or an
    element that the part before it, not read, opened: it is left out whole when it
    holds any ``-->`` or a closer with no opener in it (:func:`markup.has_stray_closer`).
    The newest text is read first and the reading stops when *limit* of them have
    something left, so a chat with many reads costs no more than *limit* of them. A
    read with nothing left is no material."""
    groups: list[_Group] = []
    latest: dict[tuple[str, str, bool], _Group] = {}
    for read in reads:
        key = (read.kind, read.ref, read.browser)
        group = latest.get(key)
        if (
            group is None
            or read.browser
            or read.start <= 1
            or group.after is None
            or not group.last < read.start <= group.after
        ):
            group = _Group(
                read.kind,
                read.ref,
                [read.text],
                read.other,
                read.browser,
                read.start,
                read.start,
                read.after,
            )
            groups.append(group)
            latest[key] = group
            continue
        group.texts.append(read.text)
        group.other = group.other or read.other
        group.last = read.start
        group.after = read.after
        groups.remove(group)
        groups.append(group)
    found: list[Material] = []
    for group in reversed(groups):
        whole = "\n".join(group.texts)
        if group.begin > 1 and ("-->" in whole or markup.has_stray_closer(whole)):
            continue
        stated = material_text(whole)
        if stated.strip():
            found.append(
                Material(group.kind, group.ref, stated, group.other, group.browser)
            )
            if len(found) >= limit:
                break
    found.reverse()
    return found


def pointed_material(
    events: Iterable[tuple[int | None, Mapping[str, Any]]], typed: Mapping[int, str]
) -> list[Material]:
    """The pages and files this chat read that the user pointed to, newest last.

    *events* are the chat's stored events, oldest first, as ``(turn, event)``;
    *typed* maps each turn the user began (not an undone one, not a task's
    question) to what they typed in it, the turn under way included. A tool call
    counts when it finished with a result and the user pointed to its page or file:

    - a page, when the user typed its address (:func:`typed_places`) in some turn up
      to now: a host alone points to every page of it, an address with a path to that
      page and what is below it (``github.com/jane`` is not ``github.com/evil``);
    - a file, when its name is in the typed words of some turn up to now;
    - the first page or file a turn read, when the typed words of that turn call it
      theirs (:data:`THEIRS`) without saying which: "my profile at github.com/jane"
      means that address and no other. A later read of that turn is the user's only
      if it is the same page or file again, or the user typed its address or name.

    It is someone else's (``other``) when the words of that turn, or of a turn that
    names it, say so. A page or file the user did not point to is left out, as is
    every call of a turn the user did not begin. The text of each is what
    :func:`material_text` leaves of what the tool returned, and a read that leaves
    nothing is left out.

    The parts of one page or file read one after the other (``web_fetch`` with
    ``part``, ``file_read`` with ``start_line`` and ``end_line``) are one text, read as
    a whole, so a comment or element that one part opens and a later one closes is one
    thing. A text that does not start at part 1 or line 1 is left out whole when it
    holds a ``-->`` or a closer with no opener in it (:func:`_assembled`). A page read
    in the browser comes
    back with ``browser`` set (:data:`BROWSER_TOOLS`).
    """
    folded = {turn: fold(text) for turn, text in typed.items()}
    places = {turn: typed_places(text) for turn, text in typed.items()}
    started: dict[tuple[int | None, str], Mapping[str, Any]] = {}
    tabs = _Tabs()
    first: dict[int, _Seen] = {}
    reads: list[_Read] = []
    for began, event in events:
        data = event.get("data")
        if not isinstance(data, Mapping):
            continue
        key = (began, str(data.get("call_id") or ""))
        if event.get("type") == "tool.started":
            started[key] = data
            continue
        if event.get("type") != "tool.finished" or data.get("status") != "success":
            continue
        call = started.pop(key, None)
        output = data.get("output")
        if call is None or not isinstance(output, str):
            continue
        text = _MARKER.sub("", output).strip()
        if not text:
            continue
        # The tabs follow every page the chat read, the user's turns or not.
        landing = (
            tabs.land(call, output) if str(call.get("name")) in _PAGE_ALL else None
        )
        if began is None or began not in typed:
            continue
        named: list[int] = []
        if landing is not None:
            place, view, ref = landing
            kind = "page"
            seen = _Seen(kind, place.host, place.path, view)
            named = [
                t
                for t, wanted in places.items()
                if any(_covers(p, place) for p in wanted)
            ]
        elif path := _file_of(call):
            name = os.path.basename(path.rstrip("/\\"))
            if not name:
                continue
            kind, ref = "file", name
            seen = _Seen(kind, name, None, None)
            wanted_name = fold(name)
            pattern = re.escape(wanted_name)
            if wanted_name.isascii():
                pattern = rf"(?<![\w-]){pattern}(?!\w)"
            named = [t for t, words in folded.items() if re.search(pattern, words)]
        else:
            continue
        opener = first.setdefault(began, seen)
        # "My site at example.com": the address typed is what the phrase means, so
        # no other page is "my site".
        theirs = (
            _THEIRS.search(folded[began]) is not None
            and _BOUND.search(folded[began]) is None
            and _again(opener, seen)
        )
        if not named and not theirs:
            continue
        other = any(_OTHERS.search(folded[t]) for t in {began, *named})
        browser = str(call.get("name")) in BROWSER_TOOLS
        start, after = (1, None) if browser else _span_of(call)
        reads.append(_Read(kind, ref, text, other, browser, start, after))
    return _assembled(reads)


def name_of(source: str) -> str:
    """What a note's ``from`` shows: a page's host without ``www.``, or a file name."""
    if "://" in source:
        host = next(iter(named_sites(source)), "")
        return host or source
    return source


def best_material(note: str, material: Sequence[Material]) -> Material:
    """The page or file of *material* (not empty) whose text holds most of the words
    of *note*, the earlier one when two tie."""
    return min(material, key=lambda m: len(records.missing_terms(note, [m.text])))


# ── Key tokens ──

_MONTH_WORDS: Final = (
    "january february march april may june july august september october november "
    "december"
)
_DAY_WORDS: Final = "monday tuesday wednesday thursday friday saturday sunday"
_MONTHS: Final = _MONTH_WORDS.split()
_DAYS: Final = _DAY_WORDS.split()
# A month or a weekday is compared by its first three letters, so a source that
# abbreviates ("Oct 30") holds a note that does not.
_MONTH_ABBREVIATIONS: Final = frozenset({m[:3] for m in _MONTHS} | {"sept"})
_CALENDAR: Final = frozenset(
    {*_MONTHS, *_DAYS, *_MONTH_ABBREVIATIONS, *(d[:3] for d in _DAYS)}
)
_URL = re.compile(r"https?://[^\s<>\"'`，。）)]+", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NUMBER = re.compile(r"(?<!\d)\d+(?:\.\d+)?(?!\d)")
_GROUPED = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_WORD = re.compile(r"[^\W\d_]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n|[。！？]")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
# Words a note may write capitalised that are no name: the assistant, and the user.
_NOT_NAMES: Final = frozenset({"melo", "localmelo", "you", "your", "yours", "user"})


class Key(NamedTuple):
    """One key token of a note: what it is, as the note wrote it, and how it is
    compared (``fold`` of it; a month or weekday by its first three letters)."""

    kind: str  # number, address, email, name, date
    text: str
    norm: str


def _address(url: str) -> str:
    """An address as compared: no scheme, no ``www.``, lower case, no trailing mark."""
    plain = re.sub(r"^https?://(?:www\.)?", "", url.strip(), flags=re.IGNORECASE)
    return plain.rstrip("/.,;:!?)").lower()


def keys_of(note: str) -> list[Key]:
    """The key tokens of *note* in order, each once."""
    text = _GROUPED.sub("", note)
    found: list[tuple[int, Key]] = []
    for pattern, kind in ((_URL, "address"), (_EMAIL, "email")):
        for hit in pattern.finditer(text):
            norm = _address(hit.group(0)) if kind == "address" else hit.group(0).lower()
            found.append(
                (hit.start(), Key(kind, hit.group(0).rstrip("/.,;:!?)"), norm))
            )
    rest = _EMAIL.sub(lambda m: " " * len(m.group(0)), text)
    rest = _URL.sub(lambda m: " " * len(m.group(0)), rest)
    for host in sorted(named_sites(rest)):
        named = re.search(rf"(?i)(?:www\.)?{re.escape(host)}", rest)
        if named:
            found.append((named.start(), Key("address", host, host)))
            rest = (
                rest[: named.start()] + " " * len(named.group(0)) + rest[named.end() :]
            )
    for hit in _NUMBER.finditer(rest):
        found.append((hit.start(), Key("number", hit.group(0), hit.group(0))))
    start = 0
    for sentence in _SENTENCE.split(rest):
        begin = rest.find(sentence, start)
        start = begin + len(sentence)
        words = list(_WORD.finditer(sentence))
        free = _CJK.search(sentence) is not None
        for n, hit in enumerate(words):
            word = hit.group(0)
            low = fold(word)
            at = begin + hit.start()
            if (
                low in _DAYS  # a weekday written out
                or (
                    low in _MONTHS and word[0].isupper() and (low != "may" or n or free)
                )
                or (
                    low in _MONTH_ABBREVIATIONS
                    and word[0].isupper()
                    and re.match(r"\.?\s*\d", sentence[hit.end() :])
                )
            ):
                found.append((at, Key("date", word, low[:3])))
            elif (
                len(word) > 1
                and low not in _NOT_NAMES
                and unicodedata.name(word[0], "").startswith("LATIN")
                and (word.isupper() or (word[0].isupper() and (n or free)))
            ):
                found.append((at, Key("name", word, low)))
    keys: list[Key] = []
    seen: set[tuple[str, str]] = set()
    for _, key in sorted(found, key=lambda pair: pair[0]):
        if (key.kind, key.norm) not in seen:
            seen.add((key.kind, key.norm))
            keys.append(key)
    return keys


def key_tokens(note: str) -> list[str]:
    """The key tokens of *note*, in order, each once: the numbers (digit runs and
    years), the dates (months and weekdays written out), the web and e-mail
    addresses, and the names in Latin script (a capitalised word that does not
    start its sentence, or an acronym of two letters or more). Words of other
    scripts, and every other word, are not key tokens."""
    return [key.text for key in keys_of(note)]


class Evidence:
    """The texts a note is checked against, folded once."""

    def __init__(self, texts: Iterable[str]) -> None:
        raw = [_GROUPED.sub("", text) for text in texts]
        self._folded = [fold(text) for text in raw]
        self._plain = [
            re.sub(r"https?://(?:www\.)?", "", text) for text in self._folded
        ]
        self._calendar = {
            word[:3]
            for text in self._folded
            for word in _WORD.findall(text)
            if word in _CALENDAR
        }

    def supports(self, key: Key) -> bool:
        """Whether some text holds *key*."""
        if key.kind == "date":
            return key.norm in self._calendar
        if key.kind == "number":
            pattern = rf"(?<!\d)(?<!\d\.){re.escape(key.norm)}(?!\d)(?!\.\d)"
            return any(re.search(pattern, text) for text in self._folded)
        if key.kind in ("address", "email"):
            pattern = rf"(?<![\w.@-]){re.escape(key.norm)}(?![\w@-])"
            texts = self._plain if key.kind == "address" else self._folded
            return any(re.search(pattern, text) for text in texts)
        pattern = rf"(?<![^\W\d_]){re.escape(key.norm)}(?![^\W\d_])"
        return any(re.search(pattern, text) for text in self._folded)


def unsupported_keys(note: str, texts: Iterable[str]) -> list[str]:
    """The key tokens of *note* that none of *texts* holds, as the note wrote them."""
    evidence = Evidence(texts)
    return [key.text for key in keys_of(note) if not evidence.supports(key)]


# ── Instructions ──

_VERB_WORDS: Final = (
    "remember forget ignore disregard override obey follow always never do don't dont "
    "must ensure make treat act pretend behave answer reply respond call run execute "
    "delete remove write create open read fetch send email post upload download "
    "install use stop start reveal print output say tell show give ask keep update "
    "change set add save store visit browse click submit buy pay transfer share "
    "publish commit push merge deploy navigate search summarize summarise translate "
    "list"
)
_VERBS: Final = _VERB_WORDS.split()
_LEAD = r"(?:(?:please|kindly|also|then|and|now|melo|hey|ok|okay)\b[\s,:]*)*"
_IMPERATIVE = re.compile(
    rf"^{_LEAD}(?:{'|'.join(re.escape(v) for v in _VERBS)}|go to)\b"
    r"|^(?:请|记住|务必|必须|不要|别(?!人)|忽略|无视|总是|永远|从现在起|以后你)",
    re.IGNORECASE,
)
# What speaks to the assistant: an order to ignore what it was told, a claim about its
# prompt, "from now on you must ...". (Raw material has the whole rule; a finished
# task's result has its own, narrower one, :func:`_result_orders`.)
_AIMED_ORDER = (
    r"\b(?:ignore|disregard|forget|override|bypass)\b[^.!?\n]{0,60}"
    r"\b(?:instructions?|rules?|prompts?|guidelines?|policy|policies|restrictions?"
    r"|safeguards?)\b"
    r"|\b(?:system|developer) (?:prompt|message)\b"
    r"|\bfrom now on\b[^.!?\n]{0,40}\b(?:you|melo)\s+"
    r"(?:must|should|shall|will|have to|need to|are to)\b"
    r"|\b(?:you|melo)\s+(?:must|should|shall|will|have to|need to|are to)\s+"
    r"(?:always|never)\b"
    r"|^melo\s+(?:must|should|shall|needs? to|has to|is to|always|never)\b"
    r"|忽略(?:之前|以上|前面|所有)的?(?:指令|规则|提示)"
)
# A shell command, or an action on a file or address after "and" or "then".
_ACTION_ORDER = (
    r"(?:^|[\s;|&`])(?:sudo|curl|wget|chmod|chown|mkfs)\s"
    r"|\brm\s+-\w"
    r"|\|\s*(?:sh|bash)\b|&&|\$\("
    r"|(?:\band\b|\bthen\b|\balso\b|[,;])\s+(?:delete|remove|overwrite|execute|run"
    r"|open|read|write|upload|download|fetch|send|cat)\s+(?:the\s+|this\s+|that\s+"
    r"|a\s+|my\s+|your\s+)?(?:files?|folders?|directory|scripts?|commands?|urls?"
    r"|links?|pages?|repo|~|/|\.{1,2}/|https?://)"
)
_ORDER = re.compile(f"{_AIMED_ORDER}|{_ACTION_ORDER}", re.IGNORECASE)


def is_instruction(note: str) -> bool:
    """Whether *note* is an instruction to the assistant, not a fact about the user.

    That is an imperative ("Always answer in French.", "Remember …", "Delete the
    file."), an order to ignore earlier instructions, the system prompt or a rule,
    "from now on you must …", a statement of what melo must or should do, a shell
    command, and an action on a file or address. What the user prefers or wants is
    a fact, however it asks: "You prefer short replies." and "You want melo to always
    answer in English." are not instructions.
    """
    return (
        any(
            _IMPERATIVE.match(sentence.strip()) or _ORDER.search(sentence)
            for sentence in re.split(r"(?<=[.!?。！？])\s+|\n|[;；]", note)
            if sentence.strip()
        )
        or _ORDER.search(note) is not None
    )


def about_user(note: str) -> bool:
    """Whether *note* says something about the user: it addresses them (``you``,
    ``your``, ``你``) or names them (``the user``)."""
    return records.addresses_user(note) or records.names_user(note)


# ── What material states ──
#
# A page or a file supports a note by what it *states*. Text a reader of the page
# would not see (a comment, an element hidden by its style, invisible characters) and
# an order to the assistant ("save a note that ...") are no evidence that something
# holds, however the words match: source monitoring says an order to believe X is not
# a reason to. :func:`material_text` leaves what states.

# Characters that draw nothing: every format character (zero-width ones, bidi marks,
# the soft hyphen, the tag characters), the invisible fillers, the blank braille and
# the variation selectors.
_FILLERS: Final = frozenset(
    "\u034f\u115f\u1160\u17b4\u17b5\u180b\u180c\u180d\u2800\u3164\uffa0"
)


def _invisible(char: str) -> bool:
    code = ord(char)
    return (
        char in _FILLERS
        or unicodedata.category(char) == "Cf"
        or 0xFE00 <= code <= 0xFE0F
        or 0xE0100 <= code <= 0xE01EF
    )


# The start of a Markdown link reference whose address is ``#`` or ``<>`` (``[//]: #
# (note)``, ``[comment]: <> "note"``): it draws nothing, and its title is where a note
# hides. The title (``(...)``, ``"..."`` or ``'...'``, which may run over lines but not
# over a blank one) is read by :func:`_link_comments`.
_LINK_HEAD = re.compile(r"^[ \t]{0,3}\[[^\]\n]*\][ \t]*:[ \t]*(?:#|<>)", re.MULTILINE)
_LINK_TITLE = re.compile(r"[ \t]*([(\"'])")
_BLANK_AT = re.compile(r"\n(?=[ \t]*\n)")


def _link_comments(text: str) -> Iterator[tuple[int, int]]:
    """The spans of the Markdown link comments of *text* (:data:`_LINK_HEAD`): the
    reference and its title to the end of the line the title ends on. A title never
    closed in its paragraph is only the line it opens on."""
    heads = list(_LINK_HEAD.finditer(text))
    if not heads:
        return
    blanks = [found.start() for found in _BLANK_AT.finditer(text)]
    found_at: dict[str, tuple[int, int]] = {}
    last = 0
    for head in heads:
        if head.start() < last:
            continue
        end = head.end()
        title = _LINK_TITLE.match(text, end)
        if title is not None:
            closer = ")" if title.group(1) == "(" else title.group(1)
            begin = title.end()
            seen = found_at.get(closer)
            if (
                seen is not None
                and seen[0] <= begin
                and (seen[1] < 0 or seen[1] >= begin)
            ):
                close = seen[1]
            else:
                close = text.find(closer, begin)
                found_at[closer] = (begin, close)
            room = bisect.bisect_left(blanks, begin)
            limit = blanks[room] if room < len(blanks) else len(text)
            if 0 <= close < limit:
                end = close + 1
        line = text.find("\n", end)
        last = len(text) if line < 0 else line
        yield head.start(), last


_KEEP_VERBS: Final = (
    "save|store|record|log|add|write|put|update|create|make|take|keep|commit|append"
    "|insert"
)
# A request to keep something in the assistant's memory.
_ASKS_TO_KEEP = re.compile(
    rf"\b(?:{_KEEP_VERBS})\b[^.!?\n]{{0,40}}?\b(?:notes?|memory|memories)\b(?!-)"
    r"|\b(?:remember|memori[sz]e)\s+(?:that|this|these|the\s+following|the\s+fact|it|to)\b"
    r"|\b(?:please|kindly)\s+(?:remember|note)\b"
    r"|\bnote\s+(?:down\s+)?(?:that\s+)?(?:the\s+user|users?['’]s|you|your|melo)\b"
    r"|\b(?:jot|note|write)\s+(?:this\s+|that\s+|it\s+)?down\b"
    r"|\b(?:keep|bear)\s+(?:this\s+|that\s+|it\s+)?in\s+mind\b"
    "|记住|记下|写入记忆|存入记忆|保存到记忆|更新(?:你的|我的)?记忆",
    re.IGNORECASE,
)
_AI_WORD = r"(?:ai|a\.i\.)"
_AI = r"(?:ai|a\.i\.|llms?|language\s+models?|chatbots?|assistants?|agents?|bots?|melo)"
# Words that speak to the assistant that reads the text ("note to any AI reading
# this", "if you are an AI", "dear assistant").
# A sentence about the assistants that read the text ("AI agents reading this page"):
# in a page it is addressed to them; a summary of a page may only say it.
_ADDRESSED_ABOUT = (
    rf"(?<!\w){_AI}(?!\w)[^.!?\n]{{0,25}}?\b(?:reading|processing|summari[sz]ing"
    r"|parsing|scanning|analy[sz]ing|browsing)\s+(?:this|it|the\s+(?:page|file"
    r"|document|text|content))\b"
)
# A sentence that speaks to the assistant ("if you are an AI", "note to any AI",
# "dear assistant").
_ADDRESSED_TO = (
    rf"\bif\s+you(?:'re|’re|\s+are)\s+(?:an?\s+|the\s+)?{_AI}(?!\w)"
    r"|(?<!\w)(?:note|message|instructions?|attention|important|reminder|p\.?s\.?"
    r"|n\.?b\.?)(?!\w)[^.!?\n]{0,15}?\b(?:to|for)\s+(?:the\s+|an?\s+|any\s+|all\s+"
    rf"|every\s+|this\s+)?{_AI}(?!\w)"
    rf"|\b(?:hey|hi|hello|dear|attention)[\s,]+(?:the\s+|any\s+|all\s+)?{_AI}(?!\w)"
)
_ADDRESSED = re.compile(f"{_ADDRESSED_ABOUT}|{_ADDRESSED_TO}", re.IGNORECASE)
# A sentence that opens by naming the assistant it speaks to ("AI assistant: ...",
# "Any AI agent, ...", "Dear assistant, ...", "To the model: ...", "Note to any AI:").
# Words that are ordinary nouns of a sentence ("The model, trained on ...", "Model:
# ResNet", "AI: hype or help?") name an addressee only with a lead, ``AI`` or both.
_TO = r"(?:note\s+to(?:\s+(?:the|any))?|to(?:\s+(?:the|any))?)"
_ADDRESS_LEAD = rf"(?:dear|any|attention|{_TO})\s+"
_ADDRESSED_AS = r"(?:assistants?|chatbots?|bots?|llms?|melo)"
_ADDRESSED_NOUN = r"(?:agent|model|system|developer)"
_ADDRESSEE = (
    rf"(?:{_ADDRESS_LEAD}(?:{_AI_WORD}\s+)?{_ADDRESSED_AS}\s*[:,]"
    rf"|(?:(?:{_ADDRESS_LEAD}|(?:the|an?)\s+))?{_AI_WORD}\s+(?:{_ADDRESSED_AS}|{_ADDRESSED_NOUN})\s*:"
    rf"|{_ADDRESS_LEAD}{_AI_WORD}\s+(?:{_ADDRESSED_AS}|{_ADDRESSED_NOUN})\s*,"
    rf"|{_TO}\s+{_AI_WORD}\s*[:,]"
    rf"|{_TO}\s+{_ADDRESSED_NOUN}\s*:"
    r"|(?:the|an?)\s+(?:assistant|chatbot|llm|melo)\s*:)"
)
# A role the text claims ("SYSTEM: ...", "[assistant] ...", "<system>"), or an
# assistant it names as the one it speaks to.
_ROLE = re.compile(
    r"[\s>*+\-•#\[(<]*(?:"
    r"(?:system|developer|assistant|melo)"
    r"(?:\s+(?:message|prompt|note|instructions?))?\s*[:：\])>]"
    rf"|{_ADDRESSEE})",
    re.IGNORECASE,
)
_CLAUSES = re.compile(r"(?<=[.!?。！？])\s+|\n|[;；]")
# The ``>`` that quotes a line, and a paragraph break: a blank line, a quoted one
# (only ``>`` marks) included.
_QUOTED = re.compile(r"^[ \t]*(?:>[ \t]*)+", re.MULTILINE)
_BREAK = re.compile(r"(\r?\n(?:[ \t]*(?:>[ \t]*)*\r?\n)+)")


def _forms(text: str) -> list[str]:
    """*text* as written and, when it differs, as a reader or a parser folds it
    (compatibility forms such as full-width letters, character references)."""
    folded = html.unescape(unicodedata.normalize("NFKC", text))
    return [text] if folded == text else [text, folded]


def _orders(paragraph: str) -> bool:
    """Whether *paragraph* holds an order to the assistant: a request to keep
    something in memory, a line that speaks to the assistant or claims a role, a
    sentence that orders it (:data:`_ORDER`: ``ignore previous instructions``, a shell
    command), or one that only starts with a verb (:func:`is_instruction`: "Use ...",
    "Open ..."). The ``>`` that quotes its lines does not count."""
    for form in _forms(_QUOTED.sub("", paragraph)):
        if _ASKS_TO_KEEP.search(form) or _ADDRESSED.search(form):
            return True
        for sentence in _CLAUSES.split(form):
            sentence = sentence.strip()
            if not sentence:
                continue
            if _ROLE.match(sentence) or _ORDER.search(sentence):
                return True
            if is_instruction(sentence):
                return True
    return False


def _without_paragraphs(text: str, orders: Callable[[str], bool]) -> str:
    """*text* without its paragraphs for which *orders* holds: the facts an order
    wants kept follow it in the same paragraph as plain statements."""
    parts = _BREAK.split(text)
    kept = [part for n, part in enumerate(parts) if n % 2 or not orders(part)]
    return "".join(kept) if len(kept) < len(parts) else text


def material_text(text: str) -> str:
    """What *text*, a page's or a file's (a body, or a text the chat read), states:
    what is left for a note to rest on.

    Left out, in this order: the markup, by one strict left-to-right scan that allows
    only what is plainly shown (:func:`markup.without_markup`: comments and other
    ``<!...>``, ``<?...>``; ``script``, ``style``, ``textarea``, ``title`` and the
    other raw text elements; every element that is not a plain one, tags and text, so
    ``style``, ``hidden``, ``font``, ``svg``, ``details`` and the like hide what they
    hold; the whole text when it has a style sheet; every other tag, with its
    attributes, replaced by a space); then the characters that draw nothing (zero-width
    and other format characters, which includes the joiners of some scripts; they are
    stripped after the markup is cut, so none can split a tag); the Markdown link
    comments (``[//]: # (...)``); and every paragraph (text between blank lines; in a
    quoted block a line of only ``>`` is blank) with an order to the assistant in it
    (:func:`_orders`). A comment, tag or element never closed hides the rest of the
    text, as in a browser.

    Text with no markup, no order and no invisible character comes back unchanged. It
    is a small scan, not an HTML engine, it never raises, and it takes time linear
    in the text. What the user typed is not material and is never passed through it.
    """
    if not text:
        return text
    text = markup.without_markup(text)
    if not text.isascii():
        text = "".join(char for char in text if not _invisible(char))
    text = markup.cut(text, ((a, b, "") for a, b in _link_comments(text)))
    return _without_paragraphs(text, _orders)


# ── What a result states ──
#
# A finished task's result is melo's own answer, in Markdown, and "you" in it is the
# user. The part of it that is not trusted is page text the page reader already
# flattened, so it holds no markup to read. Only the plainly injected forms take a
# paragraph out of it: an order to ignore what the assistant was told, a role header,
# an order to write to the assistant's memory, a line that opens by addressing an AI.
# Advice ("Run the tests.", "You should always check in online."), a fact about a
# system prompt, a "Developer:" line and "Note that your lease ends June 30" are the
# answer, and a note may rest on them.

_ANSWER: Final = "\nmelo: "
_IGNORE_ORDER = re.compile(
    r"\b(?:ignore|disregard|forget|override)\s+(?:(?:all|any|the)\s+)?"
    r"(?:(?:previous|prior|above|earlier|preceding)\s+"
    r"|your\s+(?:(?:previous|prior|above|earlier|preceding)\s+)?)"
    r"(?:instructions?|rules?|prompts?)\b"
    "|忽略(?:之前|以上|前面|所有)的?(?:指令|规则|提示)",
    re.IGNORECASE,
)
# What orders the assistant to keep something in its memory: "save this to your
# memory", "update your memory", "remember that the user ...", "save a note that the
# user ...". Not "save the notes to notes.md", "add more memory" or "remember to ...".
_MEMORY_ORDER = re.compile(
    r"\b(?:save|store|record|write|add)\b[^.!?\n]{0,60}?\b(?:to|in|into)\s+"
    r"(?:your|its|the\s+assistant['’]?s|long[- ]term)\s+memory\b(?![-\w])"
    r"|\bupdate\s+your\s+memory\b(?![-\w])"
    r"|\bremember\s+that\s+the\s+users?\b"
    r"|\b(?:save|store|record|write|add)\b[^.!?\n]{0,40}?\b(?:note|memory)\s+"
    r"(?:that|saying|stating)\s+the\s+users?\b"
    "|写入记忆|存入记忆|保存到记忆|更新(?:你的|我的)?记忆",
    re.IGNORECASE,
)
# A role header at the start of a line ("System: ...", "[assistant]", "<system>"); a
# list item ("- System: macOS 27.2") and ``Developer:`` are not one.
_RESULT_ROLE = re.compile(
    r"^[ \t>]*+#*+[ \t]*+(?:(?:system|assistant)[ \t]*[:：]"
    r"|[\[<](?:system|assistant)[\]>](?!\())",
    re.IGNORECASE | re.MULTILINE,
)
# A line that opens by addressing an AI and then has a comma or a colon: "Dear AI
# assistant, ...", "Note to any AI reading this: ...". Not "Dear Assistant Professor Lee, ...",
# and not "Hello, ..." with no AI named.
_RESULT_ADDRESSEE = re.compile(
    r"^[ \t>#]*+(?:dear|hello|hi|hey|attention|note\s+to|message\s+to|to)\s+"
    r"(?:(?:any|all|the|every)\s+)?"
    r"(?:ai|a\.i\.|llms?|language\s+models?|chatbots?)"
    r"(?:\s+(?:assistants?|agents?|models?|systems?|bots?))?"
    r"(?:\s+reading\s+(?:this|it))?[ \t]*[,:]",
    re.IGNORECASE | re.MULTILINE,
)
_FENCE = re.compile(
    r"^[ \t]*(```|~~~).*?(?:^[ \t]*\1[^\n]*$|\Z)", re.DOTALL | re.MULTILINE
)
_CODE_SPAN = re.compile(r"`[^`\n]*+`")


def _code_blanked(text: str) -> str:
    """*text* with the code in it (fenced blocks and code spans) replaced by ``~``, so
    what it holds is no order and the paragraphs around it stay as they were."""

    def blank(found: re.Match[str]) -> str:
        return re.sub(r"[^\n]", "~", found.group())

    return _CODE_SPAN.sub(blank, _FENCE.sub(blank, text))


def _result_orders(paragraph: str) -> bool:
    """Whether *paragraph* of a result holds a plainly injected form: an order to
    ignore instructions, an order to write to the assistant's memory, a role header at
    the start of a line, or a line that opens by addressing an AI. The ``>`` that
    quotes its lines does not count."""
    for form in _forms(_QUOTED.sub("", paragraph)):
        if (
            _IGNORE_ORDER.search(form)
            or _MEMORY_ORDER.search(form)
            or _RESULT_ROLE.search(form)
            or _RESULT_ADDRESSEE.search(form)
        ):
            return True
    return False


def result_text(text: str) -> str:
    """What a finished task's indexed result (:func:`index.document_text`: its task
    line, what the user asked, ``melo: `` and the answer) states: the answer without
    its paragraphs that are aimed at an assistant (:func:`_result_orders`), code
    skipped. No markup is read in it. What the user asked of the task is theirs and
    stays as it is."""
    asked, label, answer = text.partition(_ANSWER)
    if not label:
        return text
    blanked = _code_blanked(answer)
    parts = _BREAK.split(blanked)
    if not any(_result_orders(part) for part in parts[::2]):
        return text
    kept: list[str] = []
    at = 0
    for n, part in enumerate(parts):
        piece = answer[at : at + len(part)]
        at += len(part)
        if n % 2 or not _result_orders(part):
            kept.append(piece)
    return asked + label + "".join(kept)


def _following(answer: str, more: str) -> str:
    """What the chunk *more* adds to *answer*, the text of the chunks so far: the
    chunk starts inside the overlap with the one before (:func:`index.document_chunks`
    makes the next window start at the first word start after the last
    ``CHUNK_OVERLAP`` characters of the one before), so the text it adds is what
    follows that overlap. When the overlap cannot be found (a secret was redacted in
    one of the two) the whole chunk is added after a line break."""
    back = len(answer) - CHUNK_OVERLAP
    if back > 0:
        gap = re.search(r"\s", answer[back:])
        at = back + (gap.end() if gap else 0)
        if more.startswith(answer[at:]):
            return more[len(answer) - at :]
    for at in range(max(0, back), len(answer)):
        if more.startswith(answer[at:]):
            return more[len(answer) - at :]
    return "\n" + more


def joined_result(chunks: Sequence[str]) -> str:
    """The text of a result's chunks (:func:`index.document_chunks`, in order) as one,
    from the first chunk that holds the ``melo: `` label (a long question fills the
    first chunks with itself): the answer of each later chunk from where it goes
    past the end of the text so far, so a paragraph that one chunk starts and a later
    one ends is one paragraph when :func:`result_text` reads it."""
    first = next((n for n, chunk in enumerate(chunks) if _ANSWER in chunk), None)
    if first is None:
        return chunks[0] if chunks else ""
    asked, label, answer = chunks[first].partition(_ANSWER)
    for chunk in chunks[first + 1 :]:
        more = chunk.partition(_ANSWER)[2]
        if more:
            answer += _following(answer, more)
    return asked + label + answer


# ── Meaning ──


def windows(text: str, size: int = WINDOW_CHARS) -> list[str]:
    """*text* in windows of whole sentences of at most *size* characters (a sentence
    longer than that is cut), the shortest dropped."""
    out: list[str] = []
    current = ""
    for piece in re.split(r"(?<=[。！？])|(?<=[.!?])\s+|\n+", text):
        piece = " ".join(piece.split())
        while len(piece) > size:
            if current:
                out.append(current)
                current = ""
            out.append(piece[:size])
            piece = piece[size:]
        if not piece:
            continue
        if current and len(current) + 1 + len(piece) > size:
            out.append(current)
            current = piece
        else:
            current = f"{current} {piece}" if current else piece
    if current:
        out.append(current)
    return [window for window in out if len(window) >= MIN_WINDOW_CHARS]


def recent_windows(texts: Sequence[str], limit: int) -> list[str]:
    """The last *limit* windows of *texts*, taken in order."""
    return [w for text in texts for w in windows(text)][-limit:]


def keyed_windows(
    material: Sequence[Material], keys: Sequence[Key], limit: int
) -> list[tuple[str, str]]:
    """At most *limit* windows of *material* as ``(ref, window)``: those that hold
    most of *keys* first, the others in order; the chosen ones stay in order."""
    every = [(m.ref, w) for m in material for w in windows(m.text)]
    if len(every) <= limit:
        return every
    score = [sum(Evidence([w]).supports(key) for key in keys) for _, w in every]
    best = sorted(range(len(every)), key=lambda n: (-score[n], n))[:limit]
    return [every[n] for n in sorted(best)]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """The cosine of two vectors (0 when either has no length)."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / norm if norm else 0.0


def best_cosine(vector: Sequence[float], group: Sequence[Sequence[float]]) -> float:
    """The highest cosine of *vector* to a vector of *group* (0 for none)."""
    return max((cosine(vector, other) for other in group), default=0.0)


def level_at_least(level: str, wanted: str) -> bool:
    """Whether the gate *level* includes *wanted*; an unknown level is ``typed``."""
    ranks = {name: n for n, name in enumerate(MEMORY_NOTE_GATE)}
    return ranks.get(level, 0) >= ranks[wanted]
