"""A password the person gives in their own message, kept out of everything but the page (#547).

melo signs in with the details the person writes in a message of this turn, so
it does not stop to ask for what it was handed. The value must never reach a
model, a record, a log or a memory, so it is cut out of the message the moment
the message arrives and a handle stands in its place: ``my password is
hunter2`` reaches the model, the history, the events and the stores as ``my
password is [password-482913075]``. :func:`hide_passwords` is the one function that does
it, called where the person's text comes in (a gateway route, the terminal's
prompt, a session's runner, an agent's run): before the text is routed,
queued, titled, stored or read by a model, and running it again on its own
output changes nothing. :data:`VAULT` keeps the values in this process only and
gives one back only to code that has checked who may use it
(``StepReviewer.sign_in_value``): a text box of a password, on a site the
person's own words name, in a turn the person began. The model types the
handle; the browser tool types the value.

A password in a text file the person attached may be typed too, when the file
holds exactly one and the message's words say to sign in (#626,
:func:`says_sign_in`). Which handles a message's files gave is said by the door
that hid them, as data (:meth:`Vault.hide_with`, ``hide_message``) and carried
beside the message down to the run; it is never found by reading a text, so a
file that merely holds handle-shaped text gives nothing, and where no set is
carried (a turn tried again, a message read back from a store, a task, a note, a
turn melo began) no file's password can be typed.

What counts as given is decided here, by code, from the person's text: a
password word (:data:`CUES`) followed by a value, never by the model and never
by a page (a page's text never reaches :func:`hide`). When the text is not
plainly a password given, nothing is cut out and nothing can be typed: the page
is handed to the person, as before. A card number or a one-time code is never
cut out or typed: those stay the person's. A value is read whole or not at all:
a bare one is the whole run to the next blank, whatever quote characters or
backticks it holds, without the punctuation, quotes, backticks and closing
brackets after a quote at its end; a quoted one (a backtick pair is a quote pair)
runs to its first closing quote on its line, and an opening quote with no closing
one on its line is no quote: the bare run after it is the value. A plain word
with a question mark after it is a question, not a password. The password word
may be wrapped in quotes, backticks or emphasis (``"password": "x"``,
``**Password:** x``) and may end a longer name (``DB_PASSWORD=x``,
``dbPassword: x``), with a ``:`` or ``=`` joint only. One over :data:`MAX_CHARS`
is cut out too, with :data:`SPENT` in its place and nothing kept (melo could not
type it), and :data:`SPENT` is never read as a value where it stands. A shell's
working folder is no password (``PWD=/tmp make``, ``$PWD``, ``pwd /tmp``: see
``_folder``), nor is a file block's tag. The reading takes time in proportion to
the text: where a run or a quote ends is found once and kept (see :func:`_next`).

A password given next to another secret hides both: when cutting the passwords
takes the keyword or the first line of another secret (a key, a bearer token, a key
block, a one-time code: what the redaction's ``secret_spans`` finds), so that the
redaction no longer finds it in the cut text, every secret of the text given is cut
with them, in one cut from that text (:func:`_cut_secrets`). A text with no
password is not read for secrets, and a secret the redaction still finds after the
cut stays readable.

Text that is kept and never typed from is read by two functions that only
read: :func:`password_spans` says where a text gives a password, for memory to
refuse the text or leave it out, and :func:`without_passwords` cuts the values
out, for chats loaded from a file. Neither keeps a value or makes a handle.
A reader that cuts by place joins places with :func:`merged_spans` (spans that overlap
or touch become one).

This module imports only the standard library and :mod:`.attachments`, and the
redaction's reader when :func:`_cut_secrets` first runs (the checker's package
reaches the agent, which imports this module).
"""

from __future__ import annotations

import re
import secrets
import threading
from collections import Counter
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
# A password word straight after a secret: none after a letter or a digit, but one once
# the mark stands before it (see :func:`_cut_secrets`).
_WORDS_AFTER = re.compile(_CUE, re.IGNORECASE)
# The first characters of the words: a look-ahead for them lets the search skip the
# text between words at once.
_FIRST = "[" + "".join(dict.fromkeys(word[0] for word in CUES)) + "]"
# A message is about signing in when its words hold a password word or one of
# these (not inside a longer word; "sign up" is none); the one reading of a text
# for it. The Chinese ones need no boundary: a space is not written there.
_SIGN_IN = re.compile(
    rf"(?<![A-Za-z0-9_])(?:{_CUE}|(?:sign|log)(?: me)?[ -]?in(?:to)?|log[ -]?on)"
    r"(?![A-Za-z0-9_])|登[录陆入]",
    re.IGNORECASE,
)
# "the log in the folder" says where a log is: "log in" between two determiners is
# no sign-in word.
_DET = "the|a|an|this|that|my|your|our|its|their"
_PLACE = re.compile(rf"\b(?:{_DET})\s+log\s+in\s+(?:{_DET}|these|those)\b", re.I)
# A quoted value runs to the first closing quote on its line; a bare one is the whole
# run to the next blank (or the end of the text), a quote or a backtick inside it
# part of it. A backtick pair is a quote pair.
_QUOTES = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("「", "」"),
    ("`", "`"),
)
_OPENING = "".join(left for left, _ in _QUOTES)
# A closing quote that stands first after the joint belongs to what came before it
# (hiding it would change how an earlier quote pairs), so it is no part of the value.
_SHUT = "".join(right for left, right in _QUOTES if right != left)
_QUOTE_CHARS = "".join(left + right for left, right in _QUOTES)
# The marks that do not count at the end of a bare run: one set, read by the value,
# the folder and option shapes below and the mark of a cut value.
_TRAILING = ".,;，。；)）" + _QUOTE_CHARS
_TAIL = f"[{re.escape(_TRAILING)}]*+"
# What may stand between a password word and its ``:`` (a quoted or emphasised
# word: ``"password":``, ``**Password:**``), and the emphasis after the ``:``.
_WRAP = re.escape(_QUOTE_CHARS + "*")
# A password word (not inside a longer word: "keypassword"; but the last part of a
# name counts for a ":" or "=" joint: after "_" or written in camel case, as
# "DB_PASSWORD", "dbPassword"), a joint that says a value follows, and the start of
# the value, no more: the mark of a value cut out (read where it stands, whatever
# joint comes before it, so a cut text is never read twice), an opening quote, or
# the first MIN_CHARS characters of a bare run. The end of a quoted or a bare value
# is found in code (see _next), so no match reads a run to its end. After ":" or "="
# any value counts; after "is" or only a space a bare value counts only when it is
# no plain word (it holds a digit or a symbol), so "the password manager" gives
# nothing. Every repeat is bounded and every lookbehind has one width.
_GIVEN = re.compile(
    rf"(?={_FIRST})"
    r"(?:(?<![A-Za-z0-9_])|(?P<glued>(?<=_)|(?-i:(?<=[a-z0-9])(?=[A-Z]))))"
    rf"(?P<cue>{_CUE})(?![A-Za-z0-9_])"
    rf"(?P<joint>(?P<strong>[{_WRAP}]{{0,3}}\s*[:：=](?:\*{{1,3}}(?=\s))?\s*)"
    r"|\s+(?:is|are)\s+|\s*[为是]\s*|\s+)"
    rf"(?P<shut>[{_SHUT}]{{0,3}})"
    rf"(?:(?P<mark>{re.escape(SPENT)}{_TAIL}(?!\S))"
    rf"|(?P<open>[{_OPENING}])"
    rf"|\S{{{MIN_CHARS}}})",
    re.IGNORECASE,
)
_CLOSING = {left: re.compile(re.escape(right)) for left, right in _QUOTES}
_LINE = re.compile("\n")
_RUN_END = re.compile(r"\s")
_QUOTE = re.compile(f"[{re.escape(_QUOTE_CHARS)}]")
# Quote characters and backticks are no symbols for the plain-word test: "isn't".
_NO_QUOTES = str.maketrans("", "", _QUOTE_CHARS)

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


def _trim(text: str, start: int, stop: int) -> int:
    """*stop* moved back over the marks at the end of the run ``text[start:stop]``
    that do not count (:data:`_TRAILING`, and closing brackets after a quote:
    ``hunter22"}``); it reads only those marks."""
    tail = stop
    while tail > start and text[tail - 1] in _TRAILING + "}]":
        tail -= 1
    if (quote := _QUOTE.search(text, tail, stop)) is not None:
        stop = quote.start()  # the brackets after it close what the quote began
    while stop > start and text[stop - 1] in _TRAILING:
        stop -= 1
    return stop


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


def _plain(text: str, start: int, end: int) -> bool:
    """Whether ``text[start:end]`` is a plain word: letters only, where quote
    characters ("isn't") and one question mark at its end ("again?") are no symbols."""
    return text[start : end - (text[end - 1] in "?？")].translate(_NO_QUOTES).isalpha()


def _values(text: str) -> list[tuple[int, int]]:
    """The passwords *text* gives, as ``(start, end)`` of each value, in order.

    A value is a whole run: a bare one to the next blank (or the end), whatever
    quotes or backticks it holds, without the marks at its end; a quoted one to its
    closing quote on the same line; never a part of one. An opening quote with no
    closing one on its line is no quote: the bare run after it is the value. A run
    over :data:`MAX_CHARS` is a value too (the caller keeps none of it).

    A password word inside a run (or a quoted value) is read for what follows it. A
    value that starts inside the last run and reaches past it is one value with it,
    from the run's first character (a quoted run starts at its quote, so no quote is
    left to pair another way) to the end of the run that holds its end. So is one
    inside a run that is no value by its shape (a plain word, or a quoted pair too
    short to count): read again with a handle in the quote, such a run would look
    like a value, or leave a quote with nothing to pair with.
    """
    found: list[tuple[int, int]] = []
    memo: dict[re.Pattern[str], int] = {}
    pos = covered = run = 0  # the last run read is text[run:covered]
    hidden = False  # it is cut out (a value, or a handle that stands for one)
    while (match := _GIVEN.search(text, pos)) is not None:
        # Where the next look starts when nothing was read: after the word.
        pos = match.end("cue")
        if match.group("mark") is not None:
            pos = match.end()
            continue
        # The last part of a longer name counts for a ":" or "=" joint only.
        if (
            match.group("glued") is not None and match.group("strong") is None
        ) or _variable(match, text):
            continue
        start, closing = match.end("shut"), len(text)
        if (opening := match.group("open")) is not None:
            start += 1
            closing = _next(text, _CLOSING[opening], start, memo)
            if _next(text, _LINE, start, memo) < closing:
                closing = len(text)  # not on the line: no quote
        quoted = closing < len(text)
        first = start - quoted  # a quoted value is read from its quote
        after = start  # the next look: a word inside a value is read for its own value
        inside = first < covered  # it starts in the last run
        if quoted:
            if hidden and inside and closing < covered:
                continue  # in a value that is cut out
            end, stop = closing, closing + 1
            no_value = closing - start < MIN_CHARS
            if inside and not no_value:
                stop = _next(text, _RUN_END, closing, memo)
                end = _trim(text, closing, stop)
        else:
            stop = _next(text, _RUN_END, start, memo)
            if (
                stop <= covered  # inside the last run, which hides it or is no value
                or stop - start < MIN_CHARS
                or is_file_tag(text, start, stop)
                or _folder(match, text, start, stop)
            ):
                continue
            end = _trim(text, start, stop)
            # A bare word after "is" or a space is no password by its shape.
            no_value = end - start < MIN_CHARS or (
                match.group("strong") is None and _plain(text, start, end)
            )
        if no_value:
            # Still a run: a word outside the last one, or past a line break, starts a
            # new one (a run never grows over a line break); a word inside makes that
            # one longer unless it is cut out; a value may start in it.
            if match.start("cue") >= covered or text.find("\n", covered, stop) >= 0:
                run, covered, hidden, pos = first, stop, False, after
            elif not hidden:
                covered = max(covered, stop)
            continue
        if inside:
            if found and found[-1][0] >= run:
                found.pop()
            start = run
        else:
            run = first
        covered, hidden, pos = stop, True, after
        if not HANDLE.fullmatch(text, start, end):
            found.append((start, end))
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


def merged_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """*spans* in order, each run of overlapping or touching ones as one span."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _cut_secrets(
    text: str, hidden: str, spans: list[tuple[int, int]], handles: list[str]
) -> str:
    """*hidden* (*text* with each password at *spans* replaced by its handle) when the
    redaction still finds every secret of *text* in it (the same texts, counted, one
    per place); otherwise *text* cut once: the places of the passwords and of every
    secret of *text* joined where they overlap or touch, each joined place becoming
    the handles of the passwords in it, or the redaction's mark when it holds none. A
    password word straight after a secret would become a cue once the secret is cut:
    it takes its line with it, so nothing that might be its value stays."""
    # Imported here: the checker's package reaches the agent, which imports this module.
    from localmelo.melo.checker.redaction import REDACTED, secret_spans

    found = {place for place in secret_spans(text) if place[0] < place[1]}
    if not found or not (
        Counter(text[start:end] for start, end in found)
        - Counter(hidden[start:end] for start, end in set(secret_spans(hidden)))
    ):
        return hidden
    memo: dict[re.Pattern[str], int] = {}  # by end, so each line end is found once
    cut = [
        (a, _next(text, _LINE, b, memo) if _WORDS_AFTER.match(text, b) else b)
        for a, b in sorted(found, key=lambda place: place[1])
    ]
    places = merged_spans(spans + cut)
    marks: list[str] = []
    seen = 0
    for start, end in places:
        first = seen
        while seen < len(spans) and spans[seen][0] < end:
            seen += 1
        mark = " ".join(handles[first:seen]) or REDACTED
        # Quotes are paired by line: a place that held a line break keeps one.
        marks.append(f"\n{mark}\n" if text.find("\n", start, end) >= 0 else mark)
    return _splice(text, places, marks)


class Vault:
    """The passwords the person gave, by handle; in this process's memory only.

    :meth:`hide` cuts them out of a message and keeps them; :meth:`value` gives
    one back by its handle, and :meth:`forget` drops them all. Nothing is
    written anywhere: a restart forgets them, and ``repr`` shows none. Safe to
    call from any thread.
    """

    def __init__(self) -> None:
        """An empty vault whose first handle is a random nine-digit number.

        A handle's text outlives the process that made it: it stays in a stored
        or queued message, which a later process reads as the person's message
        again. Numbering from 1 in every process would let the newer process's
        first password answer to the older message's handle, so each vault
        starts at a number drawn at random from 100000000 to 899999999
        (:mod:`secrets`; nothing is written to disk). The handles that two
        vaults hold meet by chance only: a vault holds at most :data:`MAX_KEPT`
        consecutive numbers, so an older handle is one of them with a chance of
        at most ``MAX_KEPT / 800000000``, about 1 in 50,000,000 (1 in
        800,000,000 for a vault that has made one handle). Nine digits, never
        fewer: a run of 4 to 8 digits next to a code word is read as a one-time
        code (``redact_secrets``), and a handle must pass that reader unchanged;
        :data:`HANDLE` allows nine, and 100,000,000 numbers remain above the
        highest base. Only :meth:`forget`, for tests, restarts at 1.
        """
        self._values: dict[str, str] = {}
        # The last number made: the first handle is one of 100000000 to 899999999.
        self._made = 99_999_999 + secrets.randbelow(800_000_000)
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
        """:meth:`hide`, and what this call put in place of values in *text*.

        The set is what a door says of the text it read: a handle for each value
        it replaced, and :data:`SPENT` when one was too long to keep. A
        handle-shaped text *text* already held is no value (it is left as it is),
        so it is never in the set, and a text hidden a second time has an empty
        one. A secret beside a password goes with it when the cut would hide it from
        the redaction (:func:`_cut_secrets`).
        """
        spans = _values(text)
        if not spans:
            return text, frozenset()
        with self._lock:
            handles = [
                self._handle(text[start:end]) if end - start <= MAX_CHARS else SPENT
                for start, end in spans
            ]
        hidden = _cut_secrets(text, _splice(text, spans, handles), spans, handles)
        return hidden, frozenset(handles)

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

        Nothing in melo calls it: a restart forgets them all, and a new vault
        starts at a random number (see ``__init__``). A test does, to start from
        an empty vault whose first handle is ``[password-1]``.
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
    return _values(text)


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


def says_sign_in(words: str) -> bool:
    """Whether the person's *words* are about signing in: a password word
    (:data:`CUES`) or a sign-in word (``sign in``, ``log me in``, ``login``, ``登录``)."""
    return _SIGN_IN.search(_PLACE.sub(" ", words)) is not None


def handles_in(text: str) -> frozenset[str]:
    """The handles of passwords that *text* holds and the vault still keeps."""
    return frozenset(h for h in HANDLE.findall(text) if VAULT.value(h) is not None)
