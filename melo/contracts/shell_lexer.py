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
verdict and a key never name two different programs.

This module imports only the standard library.
"""

from __future__ import annotations

import re
import shlex
from collections import deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


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

_TIMEOUT: Final = frozenset({"-k", "-s", "--kill-after", "--signal"})
# Programs that run the command after them, each with its options that take a
# value (so the value is not read as the program), from each program's manual.
WRAPPERS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "arch": frozenset({"-arch", "-d", "-e"}),
        "builtin": frozenset(),
        "caffeinate": frozenset({"-t", "-w"}),
        "chronic": frozenset(),
        "command": frozenset(),
        "disown": frozenset(),
        "doas": frozenset({"-a", "-C", "-u"}),
        "env": frozenset(
            {"-C", "-L", "-P", "-S", "-u", "--chdir", "--split-string", "--unset"}
        ),
        "exec": frozenset({"-a"}),
        "flock": frozenset({"-E", "-w"}),
        "gtimeout": _TIMEOUT,
        "ionice": frozenset({"-c", "-n", "-p", "-P", "-u"}),
        "nice": frozenset({"-n"}),
        "nohup": frozenset(),
        "setsid": frozenset(),
        "stdbuf": frozenset({"-e", "-i", "-o"}),
        "sudo": frozenset(
            {"-C", "-D", "-g", "-h", "-p", "-R", "-r", "-T", "-t", "-U", "-u"}
        ),
        "taskpolicy": frozenset({"-c", "-d", "-g", "-j", "-l", "-m", "-p", "-S", "-t"}),
        "time": frozenset({"-f", "-o"}),
        "timeout": _TIMEOUT,
        "unbuffer": frozenset(),
        "watch": frozenset({"-n"}),
        "xargs": frozenset(
            {"-a", "-d", "-E", "-I", "-J", "-L", "-n", "-P", "-R", "-S", "-s"}
        ),
        "xcrun": frozenset({"--sdk", "-sdk", "--toolchain", "-toolchain"}),
    }
)
# Wrappers that take one more word before the command: timeout's duration,
# flock's lock file.
_OPERAND: Final = frozenset({"timeout", "gtimeout", "flock"})
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

    ``assignments`` are the ``NAME=value`` words before the program (and
    among a wrapper's options), in order; ``wrappers`` the wrappers it runs
    under by name, outermost first; ``codes`` the command lines a wrapper
    hands to a shell (``env -S``, ``flock FILE -c``).
    """

    words: list[str]
    assignments: list[tuple[str, str]]
    wrappers: list[str]
    codes: list[str]


def unwrap(words: Sequence[str], added: Sequence[str] = ()) -> Unwrapped:
    """*words* from the program they run: past keywords, assignments and wrappers.

    A wrapper's options are passed with their values (:data:`WRAPPERS`).
    *added* stands for what ``xargs`` adds to the command it runs, from its
    input.
    """
    rest = list(words)
    assignments: list[tuple[str, str]] = []
    wrappers: list[str] = []
    codes: list[str] = []
    while rest:
        name = program(rest[0])
        assignment = _ASSIGNMENT.fullmatch(rest[0])
        keyword = _keyword(rest)
        if keyword:
            rest = rest[keyword:]
        elif assignment is not None:
            assignments.append((assignment[1], assignment[2]))
            rest = rest[1:]
        elif name in WRAPPERS:
            wrappers.append(name)
            rest, options, more = past_options(rest[1:], WRAPPERS[name])
            assignments += more
            if name == "env":
                codes += [
                    code for option in options if (code := _split_string(*option))
                ]
            if name in _OPERAND and rest:
                rest = rest[1:]
                if name == "flock" and rest[:1] in (["-c"], ["--command"]) and rest[1:]:
                    codes.append(rest[1])
                    rest = rest[2:]
            if name == "xargs" and rest:
                rest = [*rest, *added]
        else:
            break
    return Unwrapped(rest, assignments, wrappers, codes)


def past_options(
    words: Sequence[str], valued: Collection[str]
) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    """*words* past the options at their front.

    Also each option passed with its value (``""`` for none; one of *valued*
    takes the next word), and the ``NAME=value`` words among them (``env``
    and ``sudo`` set them).
    """
    rest = list(words)
    options: list[tuple[str, str]] = []
    assignments: list[tuple[str, str]] = []
    while rest and (rest[0].startswith("-") or "=" in rest[0]):
        word = rest[0]
        if not word.startswith("-"):
            name, _, value = word.partition("=")
            assignments.append((name, value))
        options.append((word, rest[1] if word in valued and rest[1:] else ""))
        rest = rest[2:] if word in valued else rest[1:]
    return rest, options, assignments


def past_keywords(words: Sequence[str]) -> list[str]:
    """*words* without the shell keywords in front of the command (``do``, ``if``…)."""
    rest = list(words)
    while keyword := _keyword(rest):
        rest = rest[keyword:]
    return rest


def _keyword(words: list[str]) -> int:
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


def _split_string(option: str, value: str) -> str:
    """The command line an ``env`` option makes it split and run (``-S``), else ``""``."""
    if option in ("-S", "--split-string"):
        return value
    if option.startswith("--split-string="):
        return option.partition("=")[2]
    return option[2:] if option.startswith("-S") else ""
