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
"""

from __future__ import annotations

import contextlib
import itertools
import os
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final, NamedTuple
from urllib.parse import unquote, urlsplit

import numpy as np

from localmelo.melo.agent._step_review import named_sites
from localmelo.support.config import MEMORY_NOTE_GATE
from localmelo.support.memory import records

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
_ORIGIN = re.compile(r"<<page [^>\n]*?\borigin=([^\s>]+)")
_TAB = re.compile(r"<<page [^>\n]*?\btab=([^\s>]+)")
_MARKER = re.compile(r"^<<(?:end )?page .*>>[ \t]*$", re.MULTILINE)
# What ends a typed address: the user's sentence goes on after it.
_ADDRESS_END = "\\s<>\"'`，。）)、；：！？"
_PATH_END = ".,;:!?)]}>'\"/"


class Material(NamedTuple):
    """A page or file the user pointed to, as this chat read it.

    *ref* is the address (a page) or the file name, *text* what the tool returned
    and *other* says the user called it someone else's.
    """

    kind: str  # "page" or "file"
    ref: str
    text: str
    other: bool


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
    every call of a turn the user did not begin.
    """
    folded = {turn: fold(text) for turn, text in typed.items()}
    places = {turn: typed_places(text) for turn, text in typed.items()}
    started: dict[tuple[int | None, str], Mapping[str, Any]] = {}
    tabs = _Tabs()
    first: dict[int, _Seen] = {}
    found: list[Material] = []
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
        found.append(Material(kind, ref, text, other))
    return found[-MAX_MATERIAL:]


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
_ORDER = re.compile(
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
    # A shell command, or an action on a file or address after "and" or "then".
    r"|(?:^|[\s;|&`])(?:sudo|curl|wget|chmod|chown|mkfs)\s"
    r"|\brm\s+-\w"
    r"|\|\s*(?:sh|bash)\b|&&|\$\("
    r"|(?:\band\b|\bthen\b|\balso\b|[,;])\s+(?:delete|remove|overwrite|execute|run"
    r"|open|read|write|upload|download|fetch|send|cat)\s+(?:the\s+|this\s+|that\s+"
    r"|a\s+|my\s+|your\s+)?(?:files?|folders?|directory|scripts?|commands?|urls?"
    r"|links?|pages?|repo|~|/|\.{1,2}/|https?://)",
    re.IGNORECASE,
)


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
