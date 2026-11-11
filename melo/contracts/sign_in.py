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

A password in a text file the person attached may be typed too, when the file
holds exactly one (#626). Which handles a message's files gave is said by the
door that hid them, as data (:meth:`Vault.hide_with`, ``hide_message``) and
carried beside the message down to the run; it is never found by reading a text,
so a file that merely holds handle-shaped text gives nothing, and where no set
is carried (a turn tried again, a message read back from a store, a task, a note,
a turn melo began) no file's password can be typed.

What counts as given is decided here, by code, from the person's text: a
password word (:data:`CUES`) followed by a value, never by the model and never
by a page (a page's text never reaches :func:`hide`). When the text is not
plainly a password given, nothing is cut out and nothing can be typed: the page
is handed to the person, as before. A card number or a one-time code is never
cut out or typed: those stay the person's. A value is read whole or not at all:
one over :data:`MAX_CHARS` is cut out too, with :data:`SPENT` in its place and
nothing kept (melo could not type it), and :data:`SPENT` is never read as a value
where it stands. A shell's working folder is no password (``PWD=/tmp make``,
``$PWD``, ``pwd /tmp``: see ``_folder``), nor is a file block's tag. The reading
takes time in proportion to the text: it never reads a run of characters twice.

Text that is kept and never typed from is read by two functions that only
read: :func:`password_spans` says where a text gives a password, for memory to
refuse the text or leave it out, and :func:`without_passwords` cuts the values
out, for chats loaded from a file. Neither keeps a value or makes a handle.

This module imports only the standard library and :mod:`.attachments`.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from typing import Final

from localmelo.melo.contracts.attachments import is_file_tag

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
# next space, quote or backtick (or the end of the text): neither is ever cut short.
_QUOTES = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("「", "」"))
_OPENING = "".join(left for left, _ in _QUOTES)
_QUOTE_CHARS = "".join(left + right for left, right in _QUOTES)
_TRAILING = ".,;，。；)）"
_TAIL = f"[{re.escape(_TRAILING)}]*+"
# A password word (not inside a longer word: "keypassword"), a joint that says
# a value follows, and the start of the value, no more: the mark of a value cut
# out (read where it stands, whatever joint comes before it, so a cut text is
# never read twice), an opening quote, or the first MIN_CHARS characters of a bare
# run. The end of a quoted or a bare value is found in code (see _next), so no
# match reads a run to its end. After ":" or "=" any value counts; after "is" or
# only a space a bare value counts only when it is no plain word (it holds a digit
# or a symbol), so "the password manager" gives nothing.
_GIVEN = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<cue>{_CUE})(?![A-Za-z0-9_])"
    r"(?P<joint>(?P<strong>\s*[:：=]\s*)|\s+(?:is|are)\s+|\s*[为是]\s*|\s+)"
    rf"(?:(?P<mark>{re.escape(SPENT)}{_TAIL}(?![^\s{_QUOTE_CHARS}`]))"
    rf"|(?P<open>[{_OPENING}])"
    rf"|[^\s{_QUOTE_CHARS}`]{{{MIN_CHARS}}})",
    re.IGNORECASE,
)
_CLOSING = {left: re.compile(re.escape(right)) for left, right in _QUOTES}
_LINE = re.compile("\n")
_RUN_END = re.compile(rf"[\s{_QUOTE_CHARS}`]")

# A shell line says ``pwd`` too, and the folder it gives is no password. Read as
# the shell reads it, in these shapes only (each narrower than a shell line would
# need, so that a password that looks like a shell word is still hidden):
#   a variable   ``$PWD``, ``${PWD:-/tmp}``: upper case, straight after ``$`` or
#                ``${``, no ``=`` in the joint; nothing after it is read;
#   an assignment  ``PWD=/some/folder make``: upper case, at the start of the text
#                or after a blank, ``=`` with no blank on either side, a bare
#                folder value after it;
#   the command  ``pwd /some/folder``, ``pwd --version``: lower case, only blanks
#                between, a bare folder value of a path or a ``--word`` option.
# A folder value is read whole: it begins with ``/``, ``~/``, ``./``, ``../`` or a
# whole variable reference (``$NAME``, ``${NAME}`` in upper case, ending at ``/``
# or the end, or ``$(word)``) and holds only ``/``, letters, digits and ``. _ - ~
# +`` and whole variable references. Its trailing punctuation does not count.
_VARIABLE = (
    r"\$(?:[A-Z_][A-Z0-9_]*+|\{[A-Z_][A-Z0-9_]*+\}|\([a-z]+\))" rf"(?=/|{_TAIL}\Z)"
)
_FOLDER = re.compile(
    rf"(?:/|~/|\.{{1,2}}/|(?=\$))(?:[A-Za-z0-9._~+/-]|{_VARIABLE})*+{_TAIL}\Z"
)
_OPTION = re.compile(rf"--[a-z][a-z-]*+{_TAIL}\Z")


def _variable(match: re.Match[str], text: str) -> bool:
    """Whether the cue is the shell's ``$PWD``: it reads nothing of the value."""
    return (
        match.group("cue") == "PWD"
        and text.endswith(("$", "${"), 0, match.start("cue"))
        and "=" not in match.group("joint")
    )


def _folder(match: re.Match[str], text: str, start: int, stop: int) -> bool:
    """Whether the cue and the bare run ``text[start:stop]`` after it are the
    shell's assignment or command (see above).

    It reads the run's characters, so it must not run for many cues of one run:
    an assignment's cue stands at the start of its run, and a command's value is
    the whole run after the blank that follows its cue, so each run is read for
    one cue at most.
    """
    cue, at, joint = match.group("cue"), match.start("cue"), match.group("joint")
    if cue == "PWD":
        return (
            joint == "="
            and (at == 0 or text[at - 1].isspace())
            and _FOLDER.match(text, start, stop) is not None
        )
    if (
        cue == "pwd"
        and not joint.strip(" \t")
        and not text.endswith(("$", "${"), 0, at)
    ):
        if text[start] == "$":
            return False
        return (
            _FOLDER.match(text, start, stop) is not None
            or _OPTION.match(text, start, stop) is not None
        )
    return False


def _next(
    text: str, find: re.Pattern[str], start: int, memo: dict[re.Pattern[str], int]
) -> int:
    """Where *find* first matches in *text* at or after *start*, or ``len(text)``.

    *start* only grows from call to call (a search goes on from where the last
    one read), so an answer found before still holds while it is not behind
    *start*, and "nowhere" holds for good: a line of opening quotes with no
    closing one, or one long run with many password words in it, is read once,
    not once per word.
    """
    at = memo.get(find, -1)
    if at < start:
        found = find.search(text, start)
        memo[find] = at = len(text) if found is None else found.start()
    return at


def _values(text: str) -> list[tuple[int, int, str]]:
    """The passwords *text* gives, as ``(start, end, value)`` of the value, in order.

    A value is a whole run: a bare one to the next space, quote or backtick (or
    the end), a quoted one to its closing quote on the same line; never a part of
    one. A run over :data:`MAX_CHARS` is a value too (the caller keeps none of it).
    """
    found: list[tuple[int, int, str]] = []
    memo: dict[re.Pattern[str], int] = {}
    pos = 0
    while (match := _GIVEN.search(text, pos)) is not None:
        # Where the next look starts when nothing was read: after the word.
        pos = match.end("cue")
        if match.group("mark") is not None:
            pos = match.end()
            continue
        if _variable(match, text):
            continue
        opening = match.group("open")
        if opening is not None:
            start = match.end()
            end = _next(text, _CLOSING[opening], start, memo)
            if (
                end == len(text)
                or _next(text, _LINE, start, memo) < end
                or end - start < MIN_CHARS
            ):
                continue
            value, after = text[start:end], end + 1
        else:
            start = match.end("joint")
            stop = _next(text, _RUN_END, match.end(), memo)
            if is_file_tag(text, start, stop) or _folder(match, text, start, stop):
                continue
            value = text[start:stop].rstrip(_TRAILING)
            end, after = start + len(value), stop
            # A bare word after "is" or a space is no password by its shape.
            if len(value) < MIN_CHARS or (
                match.group("strong") is None and value.isalpha()
            ):
                pos = after
                continue
        pos = after
        if not HANDLE.fullmatch(value):
            found.append((start, end, value))
    return found


def _splice(text: str, spans: Sequence[tuple[int, int]], marks: Sequence[str]) -> str:
    """*text* with each span, in order, replaced by its mark: one join, not a copy
    of the text per span."""
    parts: list[str] = []
    at = 0
    for (start, end), mark in zip(spans, marks, strict=True):
        parts += (text[at:start], mark)
        at = end
    parts.append(text[at:])
    return "".join(parts)


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
        return self.hide_with(text)[0]

    def hide_with(self, text: str) -> tuple[str, frozenset[str]]:
        """:meth:`hide`, and the handles this call made from values in *text*.

        The set is what a door says of the text it read: a handle for a value it
        replaced. A handle-shaped text *text* already held is no value (it is left
        as it is), so it is never in the set, and a text hidden a second time has
        an empty one.
        """
        spans = _values(text)
        if not spans:
            return text, frozenset()
        with self._lock:
            handles = [
                self._handle(value) if len(value) <= MAX_CHARS else SPENT
                for _, _, value in spans
            ]
        hidden = _splice(text, [(start, end) for start, end, _ in spans], handles)
        return hidden, frozenset(handles) - {SPENT}

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
    spans = password_spans(text)
    return _splice(text, spans, [SPENT] * len(spans))


def spent(text: str) -> str:
    """*text* with its handles gone: it reads the same, and nothing in it can be typed."""
    return HANDLE.sub(SPENT, text)


def handles_in(text: str) -> frozenset[str]:
    """The handles of passwords that *text* holds and the vault still keeps."""
    return frozenset(h for h in HANDLE.findall(text) if VAULT.value(h) is not None)
