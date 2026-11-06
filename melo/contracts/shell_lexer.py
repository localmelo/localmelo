"""The standard library's shell lexer, reading a long word in linear time.

:class:`Lexer` is :class:`shlex.shlex` with one method replaced: ``read_token``
is copied from CPython's ``Lib/shlex.py`` (identical in 3.11 to 3.14; the
project pins 3.13) with the same states, punctuation and escape handling and
the same posix quirks. The only difference is how a token's characters are
kept. The original appends each to a string (``self.token += nextchar``),
which copies the token every time, so reading one word of ``n`` characters
costs ``O(n^2)``; here they collect in a list that is joined once. The
original's diagnostic printing (``debug``) is left out, and the text of the
token in progress is not kept in ``self.token``. The copied method is by Eric
S. Raymond, Gustavo Niemeyer and Vinay Sajip, under the Python Software
Foundation License.

:func:`split` is :func:`shlex.split` on it. The review reads every command
before it runs, so a command near the length limit must not cost seconds.
``tests/contracts/test_shell_lexer.py`` compares both with the standard library
token for token.

:func:`unwrap` finds the program a simple command runs, past shell keywords,
``NAME=value`` words and the wrappers of :data:`WRAPPERS` (``sudo``, ``env``,
``timeout``…). The review rules (``melo.checker.review.shell``) and the keys
of a step (``melo.contracts.review``) both read a command with it, so a
verdict and a key never name two different programs. Each wrapper lists all
the options of its manual; one it does not list makes the command unread,
and every program a reading of it may run is named (at most
:data:`MAX_READINGS`, each a place in the words), never just one. A
``NAME=value`` word after a wrapper is where its manual puts it: an assignment
for ``env`` and ``sudo``, else the wrapper's next word (its program,
``timeout``'s duration).

This module imports only the standard library.
"""

from __future__ import annotations

import re
import shlex
from collections import deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, NamedTuple, overload


class Lexer(shlex.shlex):
    """:class:`shlex.shlex` whose ``read_token`` is linear in the token's length."""

    # Attributes the standard library sets that its type stub leaves out.
    posix: bool
    state: str | None
    pushback: deque[str]
    _pushback_chars: deque[str]

    def read_token(self) -> str | None:
        quoted = False
        escapedstate = " "
        token: list[str] = []
        while True:
            if self.punctuation_chars and self._pushback_chars:
                nextchar = self._pushback_chars.pop()
            else:
                nextchar = self.instream.read(1)
            if nextchar == "\n":
                self.lineno += 1
            if self.state is None:
                token = []  # past end of file
                break
            elif self.state == " ":
                if not nextchar:
                    self.state = None  # end of file
                    break
                elif nextchar in self.whitespace:
                    if token or (self.posix and quoted):
                        break  # emit current token
                    else:
                        continue
                elif nextchar in self.commenters:
                    self.instream.readline()
                    self.lineno += 1
                elif self.posix and nextchar in self.escape:
                    escapedstate = "a"
                    self.state = nextchar
                elif nextchar in self.wordchars:
                    token = [nextchar]
                    self.state = "a"
                elif nextchar in self.punctuation_chars:
                    token = [nextchar]
                    self.state = "c"
                elif nextchar in self.quotes:
                    if not self.posix:
                        token = [nextchar]
                    self.state = nextchar
                elif self.whitespace_split:
                    token = [nextchar]
                    self.state = "a"
                else:
                    token = [nextchar]
                    if token or (self.posix and quoted):
                        break  # emit current token
                    else:
                        continue
            elif self.state in self.quotes:
                quoted = True
                if not nextchar:  # end of file
                    raise ValueError("No closing quotation")
                if nextchar == self.state:
                    if not self.posix:
                        token.append(nextchar)
                        self.state = " "
                        break
                    else:
                        self.state = "a"
                elif (
                    self.posix
                    and nextchar in self.escape
                    and self.state in self.escapedquotes
                ):
                    escapedstate = self.state
                    self.state = nextchar
                else:
                    token.append(nextchar)
            elif self.state in self.escape:
                if not nextchar:  # end of file
                    raise ValueError("No escaped character")
                # In posix shells, only the quote itself or the escape
                # character may be escaped within quotes.
                if (
                    escapedstate in self.quotes
                    and nextchar != self.state
                    and nextchar != escapedstate
                ):
                    token.append(self.state)
                token.append(nextchar)
                self.state = escapedstate
            elif self.state in ("a", "c"):
                if not nextchar:
                    self.state = None  # end of file
                    break
                elif nextchar in self.whitespace:
                    self.state = " "
                    if token or (self.posix and quoted):
                        break  # emit current token
                    else:
                        continue
                elif nextchar in self.commenters:
                    self.instream.readline()
                    self.lineno += 1
                    if self.posix:
                        self.state = " "
                        if token or (self.posix and quoted):
                            break  # emit current token
                        else:
                            continue
                elif self.state == "c":
                    if nextchar in self.punctuation_chars:
                        token.append(nextchar)
                    else:
                        if nextchar not in self.whitespace:
                            self._pushback_chars.append(nextchar)
                        self.state = " "
                        break
                elif self.posix and nextchar in self.quotes:
                    self.state = nextchar
                elif self.posix and nextchar in self.escape:
                    escapedstate = "a"
                    self.state = nextchar
                elif (
                    nextchar in self.wordchars
                    or nextchar in self.quotes
                    or (
                        self.whitespace_split and nextchar not in self.punctuation_chars
                    )
                ):
                    token.append(nextchar)
                else:
                    if self.punctuation_chars:
                        self._pushback_chars.append(nextchar)
                    else:
                        self.pushback.appendleft(nextchar)
                    self.state = " "
                    if token or (self.posix and quoted):
                        break  # emit current token
                    else:
                        continue
        result: str | None = "".join(token)
        if self.posix and not quoted and result == "":
            result = None
        return result


def split(s: str, comments: bool = False, posix: bool = True) -> list[str]:
    """Split *s* with shell-like syntax: :func:`shlex.split`, in linear time."""
    if s is None:
        raise ValueError("s argument must not be None")
    lexer = Lexer(s, posix=posix)
    lexer.whitespace_split = True
    if not comments:
        lexer.commenters = ""
    return list(lexer)


# ── Words and operators, simple commands ──

# The operators of a shell text, longest first: ``2>&1`` holds ``>&``.
OPERATOR: Final = re.compile(
    r"&&|\|\||;;|\|&|&>>|&>|>>|>&|<&|<<<|<<|<\(|>\(|>\||[;&|()<>\n]"
)
_OPERATOR_OR_BACKQUOTE: Final = re.compile("`|" + OPERATOR.pattern)
# The operators that end a simple command; the others redirect one.
CUTS: Final = frozenset(
    {";", "&&", "||", "|", "|&", "&", "(", ")", "<(", ">(", ";;", "\n"}
)
# A simple command: the operator before it, its words, its redirections.
Command = tuple[str, list[str], list[tuple[str, str]]]


def tokens(text: str, backquotes: bool = False) -> list[str]:
    """*text* as words and operators (``;``, ``&&``, ``(``, ``>``…), quotes kept together.

    With *backquotes*, a backquote is an operator too. A quote that does not
    close raises :class:`ValueError`.
    """
    chars = "();<>|&\n" + ("`" if backquotes else "")
    split = (_OPERATOR_OR_BACKQUOTE if backquotes else OPERATOR).findall
    lexer = Lexer(text, posix=True, punctuation_chars=chars)
    lexer.whitespace, lexer.whitespace_split, lexer.commenters = " \t\r", True, ""
    found: list[str] = []
    for token in lexer:
        found += split(token) if token and all(c in chars for c in token) else [token]
    return found


def commands(words: Sequence[str], cuts: Collection[str] = CUTS) -> list[Command]:
    """The simple commands of the tokens *words* (see :func:`tokens`).

    A redirection's operator and target are set apart, without the number
    of a file descriptor before it (``2>``).
    """
    argv: list[str] = []
    redirs: list[tuple[str, str]] = []
    out: list[Command] = []
    op, at = "", 0
    while at < len(words):
        if words[at] in cuts:
            out.append((op, argv, redirs))
            op, argv, redirs = words[at], [], []
        elif OPERATOR.fullmatch(words[at]):
            if argv and argv[-1].isdigit():
                argv.pop()
            redirs.append((words[at], words[at + 1] if at + 1 < len(words) else ""))
            at += 1
        else:
            argv.append(words[at])
        at += 1
    return [c for c in [*out, (op, argv, redirs)] if c[1] or c[2]]


# ── The program a command runs ──


# Where a program takes NAME=value words (see Options).
Assignments = Literal["", "among", "after"]


@dataclass(frozen=True)
class Options:
    """A program's options from its manual: those that take no value, those that take one.

    ``--`` among the ``flags`` ends the options; a program that does not
    list it does not take it. ``splits`` are options whose value the program
    splits into arguments of its own, read in its place (``env -S``).
    ``assignments`` is where it takes ``NAME=value`` words: ``"among"`` its
    options (sudo), ``"after"`` them, where the first ends them (env), or
    ``""``: nowhere, such a word is its next word (its program, timeout's
    duration, flock's file).
    """

    flags: frozenset[str]
    valued: frozenset[str]
    splits: frozenset[str] = frozenset()
    assignments: Assignments = ""

    @classmethod
    def of(
        cls,
        flags: str,
        valued: str = "",
        splits: str = "",
        assignments: Assignments = "",
    ) -> Options:
        """The table from its options as words, each kind apart."""
        flag, value, split = (
            frozenset(names.split()) for names in (flags, valued, splits)
        )
        return cls(flag, value, split, assignments)


# The most ways one command is read: past them melo reads no more, and asks.
MAX_READINGS: Final = 8


_TIMEOUT: Final = "-f --foreground -p --preserve-status -v --verbose"
# Programs that run the command after them, each with all the options of its
# manual. An option that is not listed makes the command unread (unwrap).
WRAPPERS: Final[Mapping[str, Options]] = MappingProxyType(
    {
        # macOS arch(1); its architectures are options too
        "arch": Options.of(
            "-32 -64 -arm64 -arm64e -i386 -x86_64 -x86_64h -c -h", "-arch -d -e"
        ),
        # bash(1), the shell sh runs on macOS: builtin takes no options
        "builtin": Options.of(""),
        # macOS caffeinate(8)
        "caffeinate": Options.of("-d -i -m -s -u", "-t -w"),
        # moreutils chronic(1)
        "chronic": Options.of("-e -v"),
        # bash(1)
        "command": Options.of("-p -V -v --"),
        # bash(1)
        "disown": Options.of("-a -h -r --"),
        # OpenBSD doas(1)
        "doas": Options.of("-L -n -s", "-a -C -u"),
        # macOS env(1); - is its old form of -i
        "env": Options.of("- -0 -i -v", "-C -P -S -u", "-S", "after"),
        # bash(1)
        "exec": Options.of("-c -l --", "-a"),
        # util-linux flock(1)
        "flock": Options.of(
            "-F --no-fork -e -x --exclusive -n --nb --nonblocking -o --close"
            " -s --shared -u --unlock --fcntl --verbose -h --help -V --version",
            "-c --command -E --conflict-exit-code -w --wait --timeout --start --length",
        ),
        # GNU coreutils timeout(1); -- from the manual's common options
        "gtimeout": Options.of(
            _TIMEOUT + " --help --version --",
            "-k --kill-after -s --signal",
        ),
        # util-linux ionice(1)
        "ionice": Options.of(
            "-t --ignore -h --help -V --version",
            "-c --class -n --classdata -p --pid -P --pgid -u --uid",
        ),
        # macOS nice(1); -0 to -9 spell its old -increment form
        "nice": Options.of(" ".join(f"-{n}" for n in range(10)), "-n"),
        # macOS nohup(1)
        "nohup": Options.of("--"),
        # util-linux setsid(1)
        "setsid": Options.of("-c --ctty -f --fork -w --wait -h --help -V --version"),
        # macOS stdbuf(1)
        "stdbuf": Options.of("", "-e -i -o"),
        # macOS sudo(8)
        "sudo": Options.of(
            "-A --askpass -B --bell -b --background -E --preserve-env -e --edit"
            " -H --set-home --help -i --login -K --remove-timestamp"
            " -k --reset-timestamp -l --list -N --no-update -n --non-interactive"
            " -P --preserve-groups -S --stdin -s --shell -V --version"
            " -v --validate --",
            "-C --close-from -D --chdir -g --group -h --host -p --prompt"
            " -R --chroot -T --command-timeout -U --other-user -u --user",
            assignments="among",
        ),
        # macOS taskpolicy(8)
        "taskpolicy": Options.of("-b -B -a -s -x -X", "-d -g -c -p -m -j -P -S -t -l"),
        # macOS time(1); the shell's own time takes -p only
        "time": Options.of("-a -h -l -p", "-o"),
        # macOS timeout(1)
        "timeout": Options.of(_TIMEOUT, "-k --kill-after -s --signal"),
        # expect unbuffer(1)
        "unbuffer": Options.of("-p"),
        # procps-ng watch(1)
        "watch": Options.of(
            "-b --beep -c --color -C --no-color -d --differences -e --errexit"
            " -f --follow -g --chgexit -p --precise -r --no-rerun -t --no-title"
            " -w --no-wrap -x --exec -h --help -v --version",
            "-n --interval -q --equexit -s --shotsdir",
        ),
        # macOS xargs(1)
        "xargs": Options.of(
            "-0 --null -o -p --interactive -r --no-run-if-empty -t --verbose"
            " -x --exit",
            "-E -I -J -L -n --max-args -P --max-procs -R -S -s --max-chars",
        ),
        # macOS xcrun(1)
        "xcrun": Options.of(
            "-v --verbose -n --no-cache -k --kill-cache -l --log -f --find"
            " -r --run --show-sdk-path --show-sdk-version --show-sdk-build-version"
            " --show-sdk-platform-path --show-sdk-platform-version"
            " --show-toolchain-path",
            "--sdk --toolchain",
        ),
    }
)
# Wrappers that take one more word before the command: timeout's duration,
# flock's lock file.
OPERAND: Final = frozenset({"timeout", "gtimeout", "flock"})
# Options whose value a shell runs: ``env -S`` (and GNU's long form, which
# an env melo could not read may be), ``flock -c``.
_CODE: Final = {"env": {"-S", "--split-string"}, "flock": {"-c", "--command"}}
# Wrappers that hand the words after their options, joined by spaces, to
# ``sh -c``, unless given one of these options: procps-ng watch(1).
_JOINS: Final = {"watch": frozenset({"-x", "--exec"})}
# Wrappers a shell also has as a keyword of its own (bash's ``time``): the
# command after one may be the shell's, which takes ``NAME=value`` words
# first, or the program time(1) runs.
_KEYWORDS_TOO: Final = frozenset({"time"})
# Shell words that open or close a compound command or a part of it. The
# command follows an opening word; a closing word ends a group, whose output
# is its last command's. ``function NAME`` and ``coproc NAME {`` are read in
# :func:`unwrap`.
KEYWORDS: Final = frozenset(
    {"!", "{", "}", "do", "done", "elif", "else", "fi", "if", "then", "until", "while"}
)
# A word a shell reads as an assignment: plain, an append, an array element.
ASSIGNMENT: Final = re.compile(r"([A-Za-z_]\w*)(?:\[[^\]]*\])?\+?=(.*)", re.S)


def program(word: str) -> str:
    """The name of the program *word* runs: lower case, without its folder."""
    return word.replace("\\", "/").rsplit("/", 1)[-1].lower()


class Spliced(Sequence[str]):
    """The words ``env -S`` makes: env's word and the words it splits out
    (``head``), then the words of ``rest`` from ``end`` on, not copied.

    ``rest`` is the innermost list the rest starts in, so a chain of splits
    (each read again past ``busybox``) neither copies nor deepens.
    """

    head: list[str]
    rest: Sequence[str]
    end: int

    def __init__(self, head: list[str], rest: Sequence[str], end: int) -> None:
        while isinstance(rest, Spliced) and end >= len(rest.head):
            end += rest.end - len(rest.head)
            rest = rest.rest
        self.head, self.rest, self.end = head, rest, end
        self._len = len(head) + len(rest) - end

    def __len__(self) -> int:
        return self._len

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[at] for at in range(*index.indices(self._len))]
        at = index + self._len if index < 0 else index
        if not 0 <= at < self._len:
            raise IndexError(index)
        head = len(self.head)
        return self.head[at] if at < head else self.rest[self.end + at - head]


# Where a reading's program starts: in the command's words, or in the words
# ``env -S`` splits out followed by the rest.
Start = tuple[Sequence[str], int]


# Words :func:`past_options` has read: each with whether an option that changes
# what the program does came before it, and whether it was read as env's
# assignment.
Read = set[tuple[int, bool, bool]]


class Walked:
    """What :func:`past_options` has read, so that it reads a word once per program.

    A word can be both an option and a place to read from (one that starts
    with ``-`` and names a wrapper by its last path part): read afresh from
    each such place, a command would cost the square of its length. Where an
    earlier walk of the same program read a word in the same state, the ends
    that word leads to are already known.
    """

    def __init__(self) -> None:
        self._read: dict[tuple[int, str, bool], tuple[object, Read]] = {}

    def of(self, words: Sequence[str], program: str, anywhere: bool) -> Read:
        """What was read in *words* for *program*, with assignments read anywhere."""
        key = id(words), program, anywhere
        return self._read.setdefault(key, (words, set()))[1]  # keeps the id theirs


@dataclass(frozen=True)
class Unwrapped:
    """A simple command from the program it runs, and what came before it.

    ``starts`` are where each reading's program starts: one, or one for each
    way to read an option a wrapper's table does not list (``unread``), at
    most :data:`MAX_READINGS`, the first with each such option taking no
    value and no lone ``-`` or ``--`` read as getopt(3) reads it. ``added``
    follows the words of each (what ``xargs`` adds).
    ``assignments`` are the ``NAME=value`` words before the program (and
    among a wrapper's options), in order; ``wrappers`` the wrappers it runs
    under by name, outermost first; ``codes`` the command lines a wrapper
    hands to a shell (``env -S``, ``flock FILE -c``): of all the readings.
    ``joins`` are where the words a wrapper joins into a command line for a
    shell start (``watch`` without ``-x``; see :meth:`line`), a place each:
    one for each reading, the first reading's first, at most
    :data:`MAX_READINGS`.
    """

    starts: list[Start]
    assignments: list[tuple[str, str]]
    wrappers: list[str]
    codes: list[str]
    unread: bool = False
    added: tuple[str, ...] = ()
    joins: tuple[Start, ...] = ()

    @property
    def readings(self) -> list[list[str]]:
        """The words of each reading, from its program on."""
        return [[*words[at:], *self.added] for words, at in self.starts]

    def line(self, join: Start) -> str:
        """The command line a shell runs for one of ``joins``: its words, joined."""
        words, at = join
        return " ".join([*words[at:], *self.added])


def unwrap(
    words: Sequence[str],
    added: Sequence[str] = (),
    start: int = 0,
    shell: bool = True,
    assignments_anywhere: bool = False,
    walked: Walked | None = None,
) -> Unwrapped:
    """*words* from the program they run, from *start*: past keywords,
    assignments and wrappers.

    A wrapper's options are read with its table (:data:`WRAPPERS`); one it
    does not list makes the command unread (see :func:`past_options`), and
    so does a command read more ways than :data:`MAX_READINGS`. ``env -S``
    is read as env(1) runs it: the words of its value, then the rest; a
    value with a form env(1) reads its own way makes the command unread
    (see :func:`_split`). The words ``watch`` hands to ``sh -c`` are a
    command line too (``joins``), and still a program and its arguments.

    A ``NAME=value`` word is an assignment where a shell reads one: at
    *start* when *shell* (not in the words a program runs: ``busybox``'s
    applet, ``uv run``'s command), and after a keyword or an assignment.
    After a wrapper it is where the wrapper's manual puts its next word
    (see :class:`Options`); after ``time``, which a shell runs as its
    keyword, it is read both ways and the command is unread. With
    *assignments_anywhere* it is an assignment wherever it stands before
    the program: the keys of a step read a command that way too, so that
    they never narrow.

    *added* stands for what ``xargs`` adds to the command it runs, from its
    input. *walked* carries what was read across calls on the same words.
    """
    walked = Walked() if walked is None else walked
    # The command's words, then each list env -S makes; a place is (list,
    # word, whether a NAME=value word there is an assignment).
    sources: list[Sequence[str]] = [words]
    starts: list[Start] = []
    assignments: list[tuple[str, str]] = []
    wrappers: list[str] = []
    codes: list[str] = []
    joins: dict[tuple[int, int], Start] = {}
    todo, seen = [(0, start, shell)], set[tuple[int, int, bool]]()
    unread = xargs = False
    while todo:  # from each word a reading reaches, once, the first reading first
        if len(starts) == MAX_READINGS:
            unread = True
            break
        place = todo.pop()
        if place in seen:
            continue
        seen.add(place)
        source, at, assigns = place
        words = sources[source]
        head = words[at : at + 3]
        assignment = ASSIGNMENT.fullmatch(head[0]) if head else None
        keyword = _keyword(head)
        name = program(head[0]) if head else ""
        if keyword:
            todo.append((source, at + keyword, True))
        elif assignment is not None and (assigns or assignments_anywhere):
            assignments.append((assignment[1], assignment[2]))
            todo.append((source, at + 1, True))
        elif name in WRAPPERS:
            wrappers.append(name)
            execs = _JOINS.get(name, frozenset())
            passed = past_options(
                words,
                WRAPPERS[name],
                at + 1,
                assignments_anywhere,
                walked.of(words, name, assignments_anywhere),
                execs,
            )
            assignments += passed.assignments
            code = _CODE.get(name, set())
            codes += [
                value for option, value in passed.options if option in code and value
            ]
            unread = unread or passed.unread
            if name in _JOINS and (
                passed.unread or not execs & {option for option, _ in passed.options}
            ):
                # Its words from each end are a line for sh -c.
                joins.update({(source, end): (words, end) for end, _ in passed.ends})
            for end, split in reversed(passed.ends):
                if split is not None:  # its words in its place, then the rest
                    spliced, own = _split(split)
                    unread = unread or own or len(sources) > MAX_READINGS
                    if len(sources) <= MAX_READINGS:
                        sources.append(Spliced([words[at], *spliced], words, end))
                        todo.append((len(sources) - 1, 0, False))
                    continue
                if name in OPERAND and end < len(words):
                    end += 1
                    command = words[end : end + 1] in (["-c"], ["--command"])
                    if name == "flock" and command and end + 1 < len(words):
                        codes.append(words[end + 1])
                        end += 2
                xargs = xargs or name == "xargs" and end < len(words)
                todo.append((source, end, False))
                both = name in _KEYWORDS_TOO and not assignments_anywhere
                if both and ASSIGNMENT.fullmatch("".join(words[end : end + 1])):
                    unread = True  # the shell's time, or time(1): melo cannot tell
                    todo.append((source, end, True))
        else:
            starts.append((words, at))
    more = tuple(added) if xargs else ()
    lines = tuple(joins.values())
    unread = unread or len(lines) > MAX_READINGS
    return Unwrapped(
        starts, assignments, wrappers, codes, unread, more, lines[:MAX_READINGS]
    )


class Passed(NamedTuple):
    """The options at the front of some words (see :func:`past_options`).

    ``ends`` are where they may end, each with the value of an option the
    program splits into arguments in its place (else ``None``), the first
    where no unlisted option takes a value; those of getopt(3)'s reading of
    an unlisted ``-`` or ``--`` come last.
    """

    ends: list[tuple[int, str | None]]
    options: list[tuple[str, str]]
    assignments: list[tuple[str, str]]
    unread: bool


def past_options(
    words: Sequence[str],
    options: Options,
    start: int = 0,
    assignments_anywhere: bool = False,
    seen: Read | None = None,
    marks: Collection[str] = (),
) -> Passed:
    """Where the options of *words* from *start* may end, and what they pass.

    Each option is passed with its value (``""`` for none), a cluster
    (``-xvf``) as its letters, and a ``NAME=value`` word as an assignment
    where *options* takes one (:attr:`Options.assignments`), or anywhere
    with *assignments_anywhere*; elsewhere such a word ends them, as the
    program's next word. ``--`` ends them when *options* lists it, and so
    does a split option (``env -S``). An option *options* does not list
    makes them ``unread``: it may take the next word as its value or not,
    so they may end where either way leads; getopt(3)'s reading of an
    unlisted ``--`` (their end) or ``-`` (an operand) is one more, after
    those. At most :data:`MAX_READINGS` ends, the first ones. A word *seen*
    holds was read before: where it leads is known, and is not repeated.
    One of *marks* changes what the program does with the rest, so a word is
    read apart before and after one.
    """
    ends: list[tuple[int, str | None]] = []
    operands: list[tuple[int, str | None]] = []
    passed: list[tuple[str, str]] = []
    seen = set() if seen is None else seen
    todo, unread = [(start, False)], False
    while todo:
        if len(ends) >= MAX_READINGS:
            unread = True
            break
        at, marked = todo.pop()
        while (at, marked, False) not in seen:
            seen.add((at, marked, False))
            word = words[at] if at < len(words) else ""
            if word == "--" and "--" in options.flags:
                ends.append((at + 1, None))
                break
            if not word.startswith("-") and (
                "=" not in word or not (options.assignments or assignments_anywhere)
            ):
                ends.append((at, None))  # the program, its next word, or none left
                break
            after = options.assignments == "after" and not assignments_anywhere
            if not word.startswith("-") and after:
                # env(1): they end its options, and its program follows them
                end = _past_assignments(words, at, seen, passed)
                if end is not None:
                    ends.append((end, None))
                break
            name, _, value = word.partition("=")
            found = (
                _held(word, options) if word.startswith("-") else ([(name, value)], 0)
            )
            if found is None:
                unread = True
                # getopt(3): an operand, the options' end
                if word in ("-", "--") and len(operands) < MAX_READINGS:
                    operands.append((at if word == "-" else at + 1, None))
                todo.append((min(at + 2, len(words)), marked))  # the way with a value
                if not (word.startswith("--") and "=" in word):
                    name, value = word, "".join(words[at + 1 : at + 2])
                found = [(name, value)], 0
            held, takes = found
            if takes:
                held[-1] = (held[-1][0], "".join(words[at + 1 : at + 2]))
            passed += held
            marked = marked or any(option in marks for option, _ in held)
            at = min(at + 1 + takes, len(words))
            if held[-1][0] in options.splits:
                ends.append((at, held[-1][1]))
                break
    found_ends = list(dict.fromkeys([*ends, *operands]))
    if len(found_ends) > MAX_READINGS:
        unread, found_ends = True, found_ends[:MAX_READINGS]
    names = [option for option in passed if not option[0].startswith("-")]
    return Passed(found_ends, passed, names, unread)


def _past_assignments(
    words: Sequence[str], at: int, seen: Read, passed: list[tuple[str, str]]
) -> int | None:
    """Past the ``NAME=value`` words env(1) reads after its options, from *at*:
    every word with ``=``. Where its program is, or ``None`` when another
    way read them already (*seen*); each is added to *passed*.
    """
    while at < len(words) and "=" in words[at]:
        if (at, False, True) in seen:
            return None
        seen.add((at, False, True))
        name, _, value = words[at].partition("=")
        passed.append((name, value))
        at += 1
    return at


def _held(word: str, options: Options) -> tuple[list[tuple[str, str]], int] | None:
    """The options *word* holds, each with its value when attached, and how many
    more words the last takes (0 or 1); ``None`` when *options* does not list one.

    A cluster (``-xvf``) is its letters: options that take no value, then at
    most one that takes one, with its value attached or in the next word.
    """
    if word in options.flags or word in options.valued:
        return [(word, "")], int(word in options.valued)
    name, equals, value = word.partition("=")
    if word.startswith("--"):
        listed = equals and (name in options.flags or name in options.valued)
        return ([(name, value)], 0) if listed else None
    held: list[tuple[str, str]] = []
    for at, letter in enumerate(word[1:], 2):
        option = "-" + letter
        if option in options.valued:
            return [*held, (option, word[at:])], int(at == len(word))
        if letter == "-" or option not in options.flags:
            return None
        held.append((option, ""))
    return (held, 0) if held else None


def past_keywords(words: Sequence[str]) -> list[str]:
    """*words* without the shell keywords in front of the command (``do``, ``if``…)."""
    rest = list(words)
    while keyword := _keyword(rest):
        rest = rest[keyword:]
    return rest


def _keyword(words: Sequence[str]) -> int:
    """How many words at the front of *words* are one keyword, 0 for none.

    ``function NAME`` and ``coproc NAME {`` (whose command is inside) count
    with the name; ``coproc`` alone is followed by the command itself.
    """
    if not words:
        return 0
    if words[0] in KEYWORDS:
        return 1
    if words[0] == "function" and words[1:]:
        return 2
    if words[0] == "coproc":
        return 2 if words[2:3] == ["{"] else 1
    return 0


# What env(1) reads its own way in the value of -S ("Details of -S"): a
# backslash (its escapes; ``\c`` ends the value, ``\_`` separates), ``$``
# (``${NAME}`` from the environment), and a vertical tab or a form feed
# (which separate).
_ENV_OWN: Final = re.compile(r"[\\$\v\f]")


def _split(value: str) -> tuple[list[str], bool]:
    """The arguments ``env -S`` splits *value* into: at spaces, past quotes;
    and whether env(1) may read it another way (:data:`_ENV_OWN`, or a word
    that starts with ``#``, which ends the value), which melo does not do.
    """
    try:
        words = split(value)
    except ValueError:
        words = value.split()
    own = bool(_ENV_OWN.search(value)) or any(w.startswith("#") for w in words)
    return words, own
