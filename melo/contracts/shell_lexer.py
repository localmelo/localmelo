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
and every program a reading of it may run is named, never just one.

This module imports only the standard library.
"""

from __future__ import annotations

import re
import shlex
from collections import deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, NamedTuple


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


@dataclass(frozen=True)
class Options:
    """A program's options from its manual: those that take no value, those that take one.

    ``--`` among the ``flags`` ends the options; a program that does not
    list it does not take it.
    """

    flags: frozenset[str]
    valued: frozenset[str]


def _options(flags: str, valued: str = "") -> Options:
    return Options(frozenset(flags.split()), frozenset(valued.split()))


_TIMEOUT: Final = "-f --foreground -p --preserve-status -v --verbose"
# Programs that run the command after them, each with all the options of its
# manual. An option that is not listed makes the command unread (unwrap).
WRAPPERS: Final[Mapping[str, Options]] = MappingProxyType(
    {
        # macOS arch(1); its architectures are options too
        "arch": _options(
            "-32 -64 -arm64 -arm64e -i386 -x86_64 -x86_64h -c -h", "-arch -d -e"
        ),
        # bash(1), the shell sh runs on macOS: builtin takes no options
        "builtin": _options(""),
        # macOS caffeinate(8)
        "caffeinate": _options("-d -i -m -s -u", "-t -w"),
        # moreutils chronic(1)
        "chronic": _options("-e -v"),
        # bash(1)
        "command": _options("-p -V -v --"),
        # bash(1)
        "disown": _options("-a -h -r --"),
        # OpenBSD doas(1)
        "doas": _options("-L -n -s", "-a -C -u"),
        # macOS env(1); - is its old form of -i
        "env": _options("- -0 -i -v", "-C -P -S -u"),
        # bash(1)
        "exec": _options("-c -l --", "-a"),
        # util-linux flock(1)
        "flock": _options(
            "-F --no-fork -e -x --exclusive -n --nb --nonblocking -o --close"
            " -s --shared -u --unlock --fcntl --verbose -h --help -V --version",
            "-c --command -E --conflict-exit-code -w --wait --timeout --start --length",
        ),
        # GNU coreutils timeout(1); -- from the manual's common options
        "gtimeout": _options(
            _TIMEOUT + " --help --version --",
            "-k --kill-after -s --signal",
        ),
        # util-linux ionice(1)
        "ionice": _options(
            "-t --ignore -h --help -V --version",
            "-c --class -n --classdata -p --pid -P --pgid -u --uid",
        ),
        # macOS nice(1); -0 to -9 spell its old -increment form
        "nice": _options(" ".join(f"-{n}" for n in range(10)), "-n"),
        # macOS nohup(1)
        "nohup": _options("--"),
        # util-linux setsid(1)
        "setsid": _options("-c --ctty -f --fork -w --wait -h --help -V --version"),
        # macOS stdbuf(1)
        "stdbuf": _options("", "-e -i -o"),
        # macOS sudo(8)
        "sudo": _options(
            "-A --askpass -B --bell -b --background -E --preserve-env -e --edit"
            " -H --set-home --help -i --login -K --remove-timestamp"
            " -k --reset-timestamp -l --list -N --no-update -n --non-interactive"
            " -P --preserve-groups -S --stdin -s --shell -V --version"
            " -v --validate --",
            "-C --close-from -D --chdir -g --group -h --host -p --prompt"
            " -R --chroot -T --command-timeout -U --other-user -u --user",
        ),
        # macOS taskpolicy(8)
        "taskpolicy": _options("-b -B -a -s -x -X", "-d -g -c -p -m -j -P -S -t -l"),
        # macOS time(1); the shell's own time takes -p only
        "time": _options("-a -h -l -p", "-o"),
        # macOS timeout(1)
        "timeout": _options(_TIMEOUT, "-k --kill-after -s --signal"),
        # expect unbuffer(1)
        "unbuffer": _options("-p"),
        # procps-ng watch(1)
        "watch": _options(
            "-b --beep -c --color -C --no-color -d --differences -e --errexit"
            " -f --follow -g --chgexit -p --precise -r --no-rerun -t --no-title"
            " -w --no-wrap -x --exec -h --help -v --version",
            "-n --interval -q --equexit -s --shotsdir",
        ),
        # macOS xargs(1)
        "xargs": _options(
            "-0 --null -o -p --interactive -r --no-run-if-empty -t --verbose"
            " -x --exit",
            "-E -I -J -L -n --max-args -P --max-procs -R -S -s --max-chars",
        ),
        # macOS xcrun(1)
        "xcrun": _options(
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
_OPERAND: Final = frozenset({"timeout", "gtimeout", "flock"})
# Options whose value a shell runs: ``env -S`` (and GNU's long form, which
# an env melo could not read may be), ``flock -c``.
_CODE: Final = {"env": {"-S", "--split-string"}, "flock": {"-c", "--command"}}
# Shell words that open or close a compound command or a part of it. The
# command follows an opening word; a closing word ends a group, whose output
# is its last command's. ``function NAME`` and ``coproc NAME {`` are read in
# :func:`unwrap`.
KEYWORDS: Final = frozenset(
    {"!", "{", "}", "do", "done", "elif", "else", "fi", "if", "then", "until", "while"}
)
_ASSIGNMENT: Final = re.compile(r"([A-Za-z_]\w*)=(.*)", re.S)


def program(word: str) -> str:
    """The name of the program *word* runs: lower case, without its folder."""
    return word.replace("\\", "/").rsplit("/", 1)[-1].lower()


@dataclass(frozen=True)
class Unwrapped:
    """A simple command from the program it runs, and what came before it.

    ``readings`` are the words from the program on: one, or one for each way
    to read an option a wrapper's table does not list (``unread``), the
    first with each such option taking no value. ``assignments`` are the
    ``NAME=value`` words before the program (and among a wrapper's
    options), in order; ``wrappers`` the wrappers it runs under by name,
    outermost first; ``codes`` the command lines a wrapper hands to a shell
    (``env -S``, ``flock FILE -c``): of all the readings.
    """

    readings: list[list[str]]
    assignments: list[tuple[str, str]]
    wrappers: list[str]
    codes: list[str]
    unread: bool = False


def unwrap(words: Sequence[str], added: Sequence[str] = ()) -> Unwrapped:
    """*words* from the program they run: past keywords, assignments and wrappers.

    A wrapper's options are read with its table (:data:`WRAPPERS`); one it
    does not list makes the command unread (see :func:`past_options`).
    *added* stands for what ``xargs`` adds to the command it runs, from its
    input.
    """
    words = list(words)
    readings: list[list[str]] = []
    assignments: list[tuple[str, str]] = []
    wrappers: list[str] = []
    codes: list[str] = []
    todo, seen, unread, xargs = [0], set[int](), False, False
    while todo:  # from each word a reading reaches, once, the first reading first
        at = todo.pop()
        if at in seen:
            continue
        seen.add(at)
        head = words[at : at + 3]
        assignment = _ASSIGNMENT.fullmatch(head[0]) if head else None
        keyword = _keyword(head)
        name = program(head[0]) if head else ""
        if keyword:
            todo.append(at + keyword)
        elif assignment is not None:
            assignments.append((assignment[1], assignment[2]))
            todo.append(at + 1)
        elif name in WRAPPERS:
            wrappers.append(name)
            passed = past_options(words[at + 1 :], WRAPPERS[name])
            assignments += passed.assignments
            code = _CODE.get(name, set())
            codes += [
                value for option, value in passed.options if option in code and value
            ]
            unread = unread or passed.unread
            for end in reversed([at + 1 + end for end in passed.ends]):
                if name in _OPERAND and end < len(words):
                    end += 1
                    command = words[end : end + 1] in (["-c"], ["--command"])
                    if name == "flock" and command and end + 1 < len(words):
                        codes.append(words[end + 1])
                        end += 2
                xargs = xargs or name == "xargs" and end < len(words)
                todo.append(end)
        else:
            readings.append(words[at:])
    if xargs:
        readings = [[*reading, *added] for reading in readings]
    return Unwrapped(readings, assignments, wrappers, codes, unread)


class Passed(NamedTuple):
    """The options at the front of some words (see :func:`past_options`)."""

    ends: list[int]
    options: list[tuple[str, str]]
    assignments: list[tuple[str, str]]
    unread: bool


def past_options(words: Sequence[str], options: Options) -> Passed:
    """Where the options at the front of *words* may end, and what they pass.

    Each option is passed with its value (``""`` for none), a cluster
    (``-xvf``) as its letters, and a ``NAME=value`` word as an assignment
    (``env`` and ``sudo`` set them). ``--`` ends them when *options* lists
    it. An option *options* does not list makes them ``unread``: it may take
    the next word as its value or not, so they may end where either way
    leads; the first end is where no such option takes one.
    """
    ends: list[int] = []
    passed: list[tuple[str, str]] = []
    todo, seen, unread = [0], set[int](), False
    while todo:
        at = todo.pop()
        while at not in seen:
            seen.add(at)
            word = words[at] if at < len(words) else ""
            if word == "--" and "--" in options.flags:
                ends.append(at + 1)
                break
            if not word.startswith("-") and "=" not in word:
                ends.append(at)  # the program, or no word left
                break
            name, _, value = word.partition("=")
            found = (
                _held(word, options) if word.startswith("-") else ([(name, value)], 0)
            )
            if found is None:
                unread = True
                todo.append(min(at + 2, len(words)))  # the way with a value
                if not (word.startswith("--") and "=" in word):
                    name, value = word, "".join(words[at + 1 : at + 2])
                found = [(name, value)], 0
            held, takes = found
            if takes:
                held[-1] = (held[-1][0], "".join(words[at + 1 : at + 2]))
            passed += held
            at = min(at + 1 + takes, len(words))
    assigned = [option for option in passed if not option[0].startswith("-")]
    return Passed(list(dict.fromkeys(ends)), passed, assigned, unread)


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
