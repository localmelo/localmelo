"""The page block: how a web page reaches the model.

A page becomes text between two markers that carry the same random nonce, then
a short list of numbered elements::

    <<page id=s7 nonce=4f1c02aa tab=t2 origin=https://example.com title="Home" part=1/2>>
    main text ...
    elements:
    [e12 link "Next" -> /page/2]  [e13 button "Search"]
    <<end page s7 nonce=4f1c02aa>>

The nonce is drawn when the block is rendered, after the page was read, so
nothing on the page can write the closing marker. The system note (see
:data:`localmelo.support.browser.provider.PAGE_NOTE`) tells the model that text
between the markers is data. Text from a page is also cleaned here: control
and direction-override characters go, and ``<<`` and ``>>`` are broken up, so
a page cannot imitate a marker. Fields that hold secrets (passwords, one-time
codes, card numbers) never show a value, whatever the driver sent.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

from localmelo.support.browser.driver import Element, RawPage

# A part of a page's text holds this many units: a Latin character is one, a
# CJK character three, since a Chinese character costs about as many tokens as
# three Latin ones. A part stays near 2k tokens in either script.
TEXT_BUDGET = 6_000
MAX_ELEMENTS = 60
FIND_LIMIT = 20
LABEL_CHARS = 80
VALUE_CHARS = 60
HREF_CHARS = 120
TITLE_CHARS = 120

_UNSAFE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f‎‏‪-‮⁦-⁩]")
_SENSITIVE_WORDS = {
    "password": "password field",
    "code": "one-time code field",
    "card": "payment field",
}
# Where a part may end, best first: a blank line, a line, a sentence, a clause.
_BREAKS = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    ". ",
    "! ",
    "? ",
    "；",
    "; ",
    "，",
    ", ",
    " ",
)


def weight(char: str) -> int:
    """Units a character costs: 3 for CJK, kana, hangul and fullwidth forms, else 1."""
    code = ord(char)
    wide = (
        0x2E80 <= code <= 0xA4CF
        or 0xAC00 <= code <= 0xD7A3
        or 0xF900 <= code <= 0xFAFF
        or 0xFE30 <= code <= 0xFE4F
        or 0xFF00 <= code <= 0xFFEF
        or 0x20000 <= code <= 0x3FFFF
    )
    return 3 if wide else 1


def clean(text: str) -> str:
    """*text* from a page, safe to put between markers."""
    text = _UNSAFE.sub(" ", text.replace(" ", "\n").replace(" ", "\n"))
    return text.replace("<<", "< <").replace(">>", "> >")


def field_text(text: str, limit: int) -> str:
    """One line for a label, title or value: quotes softened, cut with an ellipsis."""
    line = " ".join(clean(text).split()).replace('"', "'")
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _tidy(text: str) -> str:
    out: list[str] = []
    for line in clean(text).replace("\r", "").split("\n"):
        line = re.sub(r"[ \t ]{2,}", " ", line.strip())
        if line:
            out.append(line)
        elif out and out[-1]:
            out.append("")
    return "\n".join(out).strip()


def _cut(text: str, start: int, limit: int) -> int:
    """Where the part that began at *start* ends, given it may not pass *limit*."""
    floor = start + (limit - start) // 2
    for brk in _BREAKS:
        at = text.rfind(brk, floor, limit)
        if at != -1:
            return at + len(brk)
    return limit


def split_parts(text: str, budget: int = TEXT_BUDGET) -> list[str]:
    """*text* in parts of at most *budget* units, cut at the best break near the end."""
    text = _tidy(text)
    parts: list[str] = []
    start, size = 0, len(text)
    while start < size:
        used, index = 0, start
        while index < size and used + weight(text[index]) <= budget:
            used += weight(text[index])
            index += 1
        end = size if index >= size else _cut(text, start, index)
        parts.append(text[start:end].strip())
        start = end
    return [part for part in parts if part] or [""]


@dataclass
class Snapshot:
    """One reading of a tab: its parts and its numbered elements."""

    id: str
    tab: str
    url: str
    origin: str
    title: str
    parts: list[str]
    elements: dict[str, Element] = field(default_factory=dict)
    payment: bool = False


def make_snapshot(
    raw: RawPage, *, sid: str, tab: str, origin: str, first_ref: int
) -> Snapshot:
    """A :class:`Snapshot` of *raw*. Refs are ``e<first_ref>`` onward, never reused."""
    snap = Snapshot(sid, tab, raw.url, origin, raw.title, [""])
    kept: list[Element] = []
    for item in raw.elements:
        if item.sensitive:
            item = Element(item.kind, "", sensitive=item.sensitive)
        elif not (item.label.strip() or item.href):
            continue
        if kept and kept[-1] == item and item.kind == "link":
            continue
        kept.append(item)
    if any(item.sensitive == "card" for item in kept):
        snap.payment = True  # a payment page is not read at all
        return snap
    snap.parts = split_parts(raw.text)
    snap.elements = {f"e{first_ref + i}": item for i, item in enumerate(kept)}
    return snap


def block(ident: str, attrs: str, body: str) -> str:
    """*body* between markers with a fresh nonce."""
    nonce = secrets.token_hex(4)
    extra = f" {attrs}" if attrs else ""
    return (
        f"<<page id={ident} nonce={nonce}{extra}>>\n{body}\n"
        f"<<end page {ident} nonce={nonce}>>"
    )


def element_line(ref: str, item: Element, origin: str = "") -> str:
    if item.sensitive:
        return f"[{ref} {_SENSITIVE_WORDS[item.sensitive]}: melo never types here]"
    label = field_text(item.label, LABEL_CHARS)
    if item.kind == "link":
        href = item.href
        if origin and href.startswith(origin + "/"):
            href = href[len(origin) :]
        return f'[{ref} link "{label}" -> {field_text(href, HREF_CHARS)}]'
    if item.kind == "checkbox":
        state = "checked" if item.checked else "unchecked"
        return f'[{ref} checkbox "{label}" {state}]'
    if item.kind in ("textbox", "select"):
        return f'[{ref} {item.kind} "{label}" value="{field_text(item.value, VALUE_CHARS)}"]'
    return f'[{ref} {item.kind} "{label}"]'


def render_page(snap: Snapshot, *, part: int = 1, view: str = "all") -> str:
    """The page block for *part* of *snap*; *view* is ``text``, ``elements`` or ``all``."""
    total = len(snap.parts)
    attrs = f'tab={field_text(snap.tab, 40)} origin={snap.origin} title="{field_text(snap.title, TITLE_CHARS)}"'
    if total > 1 and view != "elements":
        attrs += f" part={part}/{total}"
    if snap.payment:
        body = "[melo does not read pages with payment fields.]"
        return block(snap.id, attrs, body)
    lines: list[str] = []
    if view != "elements":
        lines.append(snap.parts[part - 1] or "(no readable text)")
    if view != "text" and (part == 1 or view == "elements") and snap.elements:
        shown = list(snap.elements.items())[:MAX_ELEMENTS]
        lines.append("elements:")
        lines.extend(element_line(ref, item, snap.origin) for ref, item in shown)
        if len(snap.elements) > len(shown):
            more = len(snap.elements) - len(shown)
            lines.append(f"[{more} more elements: browser_find searches them]")
    if view != "elements" and part < total:
        lines.append(
            f"[part {part} of {total}: browser_read part={part + 1} continues]"
        )
    return block(snap.id, attrs, "\n".join(lines))


def find_elements(
    snap: Snapshot, query: str, role: str = "any"
) -> tuple[list[tuple[str, Element]], int]:
    """The first :data:`FIND_LIMIT` elements of *snap* that match, and how many match."""
    words = query.lower().split()
    found = [
        (ref, item)
        for ref, item in snap.elements.items()
        if role in ("any", item.kind)
        and all(w in f"{item.label} {item.href}".lower() for w in words)
    ]
    return found[:FIND_LIMIT], len(found)


def render_find(snap: Snapshot, query: str, role: str = "any") -> str:
    found, total = find_elements(snap, query, role)
    lines = [
        f'{total} of {len(snap.elements)} elements match "{field_text(query, 60)}":'
    ]
    lines.extend(element_line(ref, item, snap.origin) for ref, item in found)
    attrs = f"tab={field_text(snap.tab, 40)} origin={snap.origin}"
    return block(snap.id, attrs, "\n".join(lines))
