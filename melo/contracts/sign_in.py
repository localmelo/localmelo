"""A password the person gives in their own message, kept out of everything but the page (#547).

melo signs in with the details the person writes in a message of this turn, so
it does not stop to ask for what it was handed. The value must never reach a
model, a record, a log or a memory, so it is cut out of the message the moment
the message arrives and a handle stands in its place: ``my password is
hunter2`` reaches the model, the history, the events and the stores as ``my
password is [password-1]``. :class:`Vault` keeps the value in this process only,
for the chat, and gives it back only to code that has checked who may use it
(``StepReviewer.sign_in_value``): a text box of a password, on a site the
person's own words name, in the turn the person gave it. The model types the
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
from typing import Final

# The words that say a value is a password.
CUES: Final = ("password", "passwd", "pwd", "pw", "密码", "口令")
# A value is 4 to 128 characters with no space (a quoted one may hold spaces).
MIN_CHARS: Final = 4
MAX_CHARS: Final = 128
# Passwords one vault keeps; one past that is hidden but cannot be typed.
MAX_HANDLES: Final = 8
UNUSABLE: Final = "[password-hidden]"

HANDLE: Final = re.compile(r"\[password-(?:[1-9]\d?|hidden)\]")

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
    """The passwords one chat's person gave, by handle; in this process's memory only.

    :meth:`hide` cuts them out of a message and keeps them; :meth:`value`
    gives one back by its handle, and :meth:`forget` drops them all. Nothing
    is written anywhere: a restart forgets them, and ``repr`` shows none.
    """

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def __repr__(self) -> str:
        return f"Vault({len(self._values)} password(s))"

    def hide(self, text: str) -> tuple[str, tuple[str, ...]]:
        """*text* with each password it gives replaced by its handle, and the handles in it.

        The same value always has the same handle, so hiding a text again
        (or the same message twice) changes nothing the second time.
        """
        spans = _values(text)
        if not spans:
            return text, ()
        handles = tuple(self._handle(value) for _, _, value in spans)
        out = text
        for (start, end, _), handle in reversed(list(zip(spans, handles, strict=True))):
            out = out[:start] + handle + out[end:]
        return out, handles

    def _handle(self, value: str) -> str:
        for handle, kept in self._values.items():
            if kept == value:
                return handle
        if len(self._values) >= MAX_HANDLES:
            return UNUSABLE
        handle = f"[password-{len(self._values) + 1}]"
        self._values[handle] = value
        return handle

    def value(self, handle: str) -> str | None:
        """The password *handle* stands for, or ``None``."""
        return self._values.get(handle)

    def forget(self) -> None:
        """Drop every password."""
        self._values.clear()
