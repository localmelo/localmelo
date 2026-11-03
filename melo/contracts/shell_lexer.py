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

This module imports only the standard library.
"""

from __future__ import annotations

import shlex
from collections import deque


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
