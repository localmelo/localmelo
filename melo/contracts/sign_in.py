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
cut out or typed: those stay the person's.

This module imports only the standard library.
"""

from __future__ import annotations

import re
import threading
from typing import Final

# The words that say a value is a password.
CUES: Final = ("password", "passwd", "pwd", "pw", "密码", "口令")
# A value is 4 to 128 characters with no space (a quoted one may hold spaces).
MIN_CHARS: Final = 4
MAX_CHARS: Final = 128
# Passwords the vault keeps at once: the newest. A handle is never used for two
# values (the numbers only go up); the value of an older one is forgotten, and its
# handle then stands for nothing.
MAX_KEPT: Final = 16

HANDLE: Final = re.compile(r"\[password-[1-9]\d{0,8}\]")

_CUE = "(?:" + "|".join(CUES) + ")"
_SIZE = "{" + f"{MIN_CHARS},{MAX_CHARS}" + "}"
# The quoted forms, one group each, and a bare value.
_QUOTES = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("「", "」"))
_QUOTED = "|".join(
    f"{left}(?P<q{n}>[^{right}\\n]{_SIZE}){right}"
    for n, (left, right) in enumerate(_QUOTES)
)
_BARE = rf"(?P<bare>[^\s\"'“”‘’「」`]{_SIZE})"
# A password word (not inside a longer word: "keypassword"), a joint that says
# a value follows, and the value. After ":" or "=" any value counts; after
# "is" or only a space a bare value counts only when it is no plain word (it
# holds a digit or a symbol), so "the password manager" gives nothing.
_GIVEN = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<cue>{_CUE})(?![A-Za-z0-9_])"
    r"(?:(?P<strong>\s*[:：=]\s*)|\s+(?:is|are)\s+|\s*[为是]\s*|\s+)"
    rf"(?:{_QUOTED}|{_BARE})",
    re.IGNORECASE,
)
_TRAILING = ".,;，。；)）"


def _values(text: str) -> list[tuple[int, int, str]]:
    """The passwords *text* gives, as ``(start, end, value)`` of the value, in order."""
    found: list[tuple[int, int, str]] = []
    for match in _GIVEN.finditer(text):
        name = next(
            (
                n
                for n in (*(f"q{i}" for i in range(len(_QUOTES))), "bare")
                if match.group(n)
            ),
            "",
        )
        if not name:
            continue
        value, start, end = match.group(name), match.start(name), match.end(name)
        if name == "bare":
            stripped = value.rstrip(_TRAILING)
            end -= len(value) - len(stripped)
            value = stripped
            if len(value) < MIN_CHARS:
                continue
            # A bare word after "is" or a space is no password by its shape.
            if match.group("strong") is None and value.isalpha():
                continue
        if HANDLE.fullmatch(value):
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
        time.
        """
        spans = _values(text)
        if not spans:
            return text
        with self._lock:
            handles = [self._handle(value) for _, _, value in spans]
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


def handles_in(text: str) -> frozenset[str]:
    """The handles of passwords that *text* holds and the vault still keeps."""
    return frozenset(h for h in HANDLE.findall(text) if VAULT.value(h) is not None)
