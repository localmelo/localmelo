"""A password the person gives in their own message, kept out of everything but the page (#547).

melo signs in with the details the person writes in a message of this turn, so
it does not stop to ask for what it was handed. The value must never reach a
model, a record, a log or a memory, so it is cut out of the message the moment
the message arrives and a handle stands in its place: ``my password is
hunter2`` reaches the model, the history, the events and the stores as ``my
password is [password-1]``. :func:`hide_passwords` is the one function that does
it, called where the person's text comes in (a gateway route, the terminal's
prompt, a session's runner, an agent's run): before the text is routed,
queued, titled, stored or read by a model, and running it again on its own
output changes nothing. :data:`VAULT` keeps the values in this process only and
gives one back only to code that has checked who may use it
(``StepReviewer.sign_in_value``): a text box of a password, on a site the
person's own words name, in a turn the person began. The model types the
handle; the browser tool types the value.

What counts as given is decided here, by code, from the person's text: a
password word (:data:`CUES`) followed by a value, never by the model and never
by a page (a page's text never reaches :func:`hide`). When the text is not
plainly a password given, nothing is cut out and nothing can be typed: the page
is handed to the person, as before. A card number or a one-time code is never
cut out or typed: those stay the person's. A value is read whole or not at all:
one over :data:`MAX_CHARS` is cut out too, with :data:`SPENT` in its place and
nothing kept (melo could not type it). A shell's working folder (``PWD=/tmp
make``, ``$PWD``, ``pwd /tmp``) is no password and is read as it is.

Text that is kept and never typed from is read by two functions that only
read: :func:`password_spans` says where a text gives a password, for memory to
refuse the text or leave it out, and :func:`without_passwords` cuts the values
out, for chats loaded from a file. Neither keeps a value or makes a handle.

This module imports only the standard library.
"""

from __future__ import annotations

import re
import threading
from typing import Final

# The words that say a value is a password.
CUES: Final = ("password", "passwd", "pwd", "pw", "密码", "口令")
# A value is 4 to 128 characters with no space (a quoted one may hold spaces). A
# longer one is cut out whole and never kept: nothing can type it (see Vault.hide).
MIN_CHARS: Final = 4
MAX_CHARS: Final = 128
# Passwords the vault keeps at once: the newest. A handle is never used for two
# values (the numbers only go up); the value of an older one is forgotten, and its
# handle then stands for nothing.
MAX_KEPT: Final = 16

HANDLE: Final = re.compile(r"\[password-[1-9]\d{0,8}\]")
# What stands where a password was, once nothing there can be typed.
SPENT: Final = "[…]"

_CUE = "(?:" + "|".join(CUES) + ")"
# A quoted value runs to the first closing quote on its line, a bare one to the
# next space, quote or end of text; neither is ever cut short (see _values).
_QUOTES = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("「", "」"))
_CLOSING = dict(_QUOTES)
_OPENING = "".join(_CLOSING)
_QUOTE_CHARS = "".join(left + right for left, right in _QUOTES)
# A password word (not inside a longer word: "keypassword"), a joint that says
# a value follows, and the start of the value: an opening quote or a bare run.
# After ":" or "=" any value counts; after "is" or only a space a bare value
# counts only when it is no plain word (it holds a digit or a symbol), so "the
# password manager" gives nothing.
_GIVEN = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<cue>{_CUE})(?![A-Za-z0-9_])"
    r"(?:(?P<strong>\s*[:：=]\s*)|\s+(?:is|are)\s+|\s*[为是]\s*|\s+)"
    rf"(?:(?P<open>[{_OPENING}])|(?P<bare>[^\s{_QUOTE_CHARS}`]{{{MIN_CHARS},}}))",
    re.IGNORECASE,
)
_TRAILING = ".,;，。；)）"
# A shell line says ``pwd`` too, and what it gives is a folder, not a password.
# Read as the shell reads them, in these shapes only (each narrower than a shell
# would allow, so that an ODBC string ``PWD=...`` or a password that starts with
# ``-`` or ``$`` is still hidden): a variable (``$PWD``, ``${PWD:-/tmp}``), an
# assignment (``PWD=/some/folder make``: upper case, no space round the ``=``,
# a folder or a variable after it) and the command (``pwd /some/folder``: lower
# case, only blanks between, a path or an option after it).
_FOLDER = re.compile(r"/|~/|\.{1,2}/|\$(?:[({]|[A-Z_])")
_ARGUMENT = re.compile(r"/|~/|\.{1,2}/|--?[A-Za-z][A-Za-z-]*$")


def _shell(match: re.Match[str], text: str, value: str, *, quoted: bool) -> bool:
    """Whether the word of *match* is the shell's own, with *value* after it."""
    cue, at = match.group("cue"), match.start("cue")
    if cue.lower() == "pwd" and text.endswith(("$", "${"), 0, at):
        return True
    if cue == "PWD":
        return match.group("strong") == "=" and _FOLDER.match(value) is not None
    if cue == "pwd" and not quoted and match.group("strong") is None:
        joint = text[match.end("cue") : match.start("bare")]
        return not joint.strip(" \t") and _ARGUMENT.match(value) is not None
    return False


def _first(text: str, char: str, start: int, memo: dict[str, int]) -> int:
    """Where *char* first stands in *text* at or after *start*, or -1.

    *start* only grows from call to call, so an answer found before holds while it
    is not behind *start*, and "nowhere" holds for good: a line of opening quotes
    with no closing one is read once, not once per quote.
    """
    at = memo.get(char, -2)
    if at != -1 and at < start:
        memo[char] = at = text.find(char, start)
    return at


def _values(text: str) -> list[tuple[int, int, str]]:
    """The passwords *text* gives, as ``(start, end, value)`` of the value, in order.

    A value is a whole run: a bare one to the next space, quote or end of text, a
    quoted one to its closing quote on the same line; never a part of one. A run
    over :data:`MAX_CHARS` is a value too (the caller keeps none of it).
    """
    found: list[tuple[int, int, str]] = []
    memo: dict[str, int] = {}
    pos = 0
    while (match := _GIVEN.search(text, pos)) is not None:
        # Where the next look starts when nothing was read: after the word.
        pos = match.end("cue")
        opening = match.group("open")
        if opening is not None:
            start = match.end()
            end = _first(text, _CLOSING[opening], start, memo)
            line = _first(text, "\n", start, memo)
            if end < 0 or 0 <= line < end or end - start < MIN_CHARS:
                continue
            value = text[start:end]
            after = end + 1
        else:
            start, end = match.span("bare")
            value = text[start:end].rstrip(_TRAILING)
            after = end
            end = start + len(value)
            if len(value) < MIN_CHARS:
                pos = after
                continue
            # A bare word after "is" or a space is no password by its shape.
            if match.group("strong") is None and value.isalpha():
                pos = after
                continue
        if _shell(match, text, value, quoted=opening is not None):
            continue
        pos = after
        if HANDLE.fullmatch(value) or value == SPENT:
            continue
        found.append((start, end, value))
    return found


class Vault:
    """The passwords the person gave, by handle; in this process's memory only.

    :meth:`hide` cuts them out of a message and keeps them; :meth:`value` gives
    one back by its handle, and :meth:`forget` drops them all. Nothing is
    written anywhere: a restart forgets them, and ``repr`` shows none. Safe to
    call from any thread.
    """

    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        self._made = 0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"Vault({len(self._values)} password(s))"

    def hide(self, text: str) -> str:
        """*text* with each password it gives replaced by its handle.

        The same value always has the same handle while it is kept, so hiding
        a text again (or the same message twice) changes nothing the second
        time. A value over :data:`MAX_CHARS` is cut out whole and kept nowhere:
        :data:`SPENT` stands in its place, for no handle could be typed from it.
        """
        spans = _values(text)
        if not spans:
            return text
        with self._lock:
            handles = [
                self._handle(value) if len(value) <= MAX_CHARS else SPENT
                for _, _, value in spans
            ]
        out = text
        for (start, end, _), handle in reversed(list(zip(spans, handles, strict=True))):
            out = out[:start] + handle + out[end:]
        return out

    def _handle(self, value: str) -> str:
        for handle, kept in self._values.items():
            if kept == value:
                return handle
        self._made += 1
        handle = f"[password-{self._made}]"
        self._values[handle] = value
        while len(self._values) > MAX_KEPT:
            del self._values[next(iter(self._values))]
        return handle

    def value(self, handle: str) -> str | None:
        """The password *handle* stands for, or ``None``."""
        with self._lock:
            return self._values.get(handle)

    def forget(self) -> None:
        """Drop every password and number the handles from 1 again.

        Nothing in melo calls it: a restart forgets them all. A test does, to
        start from an empty vault.
        """
        with self._lock:
            self._values.clear()
            self._made = 0


# The passwords of the person who uses this process: one vault, so a message is
# hidden once, at the door, whichever chat or entry it comes in by. Only a
# handle the person's newest message of the chat holds can be typed there.
VAULT: Final = Vault()


def hide_passwords(text: str) -> str:
    """*text*, a message of the person's, with each password it gives replaced by its handle."""
    return VAULT.hide(text)


def password_spans(text: str) -> list[tuple[int, int]]:
    """Where *text* gives a password: the ``(start, end)`` of each value, in order.

    The places :func:`hide_passwords` would hide, a run over :data:`MAX_CHARS`
    whole. It only reads: nothing is kept and no handle is made. A handle is not
    a value.
    """
    return [(start, end) for start, end, _ in _values(text)]


def without_passwords(text: str) -> str:
    """*text* with each password it gives cut out and :data:`SPENT` in its place.

    What :func:`hide_passwords` would hide, left as :func:`spent` leaves a
    handle. It only reads: nothing is kept and no handle is made. For text that
    is kept and never typed from.
    """
    out = text
    for start, end in reversed(password_spans(text)):
        out = out[:start] + SPENT + out[end:]
    return out


def spent(text: str) -> str:
    """*text* with its handles gone: it reads the same, and nothing in it can be typed."""
    return HANDLE.sub(SPENT, text)


def handles_in(text: str) -> frozenset[str]:
    """The handles of passwords that *text* holds and the vault still keeps."""
    return frozenset(h for h in HANDLE.findall(text) if VAULT.value(h) is not None)
