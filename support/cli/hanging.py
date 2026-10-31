"""Output that hangs under the line it belongs to, behind a quiet connector.

One shape for every place the terminal does this (a command's output under
its echo, a tool call's output under its header, a ``!`` result)::

    › /approvals auto
      └ Approvals: auto · grants: none
        later lines line up under the first

The first line starts with two spaces, the connector in the muted style and a
space; later lines start with four spaces; blank lines stay empty. The body is
laid out in the terminal's width minus those four cells, so a line that wraps
continues under itself. The connector is the symbol set's, so a terminal
without Unicode gets ``|``; with ``NO_COLOR`` it is drawn without a colour.

:class:`Hung` is the renderable. :func:`under_command` makes everything a
task prints meanwhile hang as one body (the connector once, on the first line
that has text): the output classes of :mod:`localmelo.support.cli.output`
ask :func:`wrap` on every print and :func:`narrowed` for their width, so
handlers neither know nor care.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
from collections.abc import Coroutine, Iterator
from typing import Any, Final

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.segment import Segment

from localmelo.support.cli.theme import Symbols

# Cells before the body: "  └ " on the first line, four spaces on the others.
INDENT: Final = 4


def prefix(symbols: Symbols, *, connect: bool = True) -> str:
    """``  └ `` for the first line of a body, four spaces for the others."""
    return f"  {symbols.connector} " if connect else " " * INDENT


class Hang:
    """One body hung in several pieces: the connector is drawn once, on the first text."""

    def __init__(self, symbols: Symbols) -> None:
        self.symbols = symbols
        self.pending = True  # the connector is not drawn yet

    def hung(self, body: RenderableType) -> Hung:
        """*body* as a piece of this hang; a body that is hung already stays as it is."""
        return body if isinstance(body, Hung) else Hung(body, self.symbols, self)


class Hung:
    """*body* hung under the connector (see the module's text).

    A body of its own draws the connector on its first line. Pieces of one
    :class:`Hang` share it: only the first piece with text draws the connector,
    the rest line up under it.
    """

    def __init__(
        self, body: RenderableType, symbols: Symbols, hang: Hang | None = None
    ) -> None:
        self.body = body
        self.hang = hang or Hang(symbols)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        inner = options.update_width(max(1, options.max_width - INDENT))
        muted = console.get_style("muted", default="none")
        hang = self.hang
        for line in console.render_lines(self.body, inner, pad=False):
            if not any(segment.text.strip() for segment in line):
                yield Segment.line()  # a blank line stays empty
                continue
            if hang.pending:
                hang.pending = False
                yield Segment(prefix(hang.symbols), muted)
            else:
                yield Segment(prefix(hang.symbols, connect=False))
            yield from line
            yield Segment.line()


# The body being hung by the task that is running a command, if any. A task
# started meanwhile copies it, so :func:`spawn_detached` clears it.
_CURRENT: contextvars.ContextVar[Hang | None] = contextvars.ContextVar(
    "localmelo_hang", default=None
)


@contextlib.contextmanager
def under_command(symbols: Symbols) -> Iterator[Hang]:
    """Hang what this task prints meanwhile as one body under the command's echo."""
    hang = Hang(symbols)
    token = _CURRENT.set(hang)
    try:
        yield hang
    finally:
        _CURRENT.reset(token)


@contextlib.contextmanager
def full_width() -> Iterator[None]:
    """Print at the left edge, as a rule or the welcome does, inside a command."""
    token = _CURRENT.set(None)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def active() -> bool:
    return _CURRENT.get() is not None


def wrap(renderables: tuple[RenderableType, ...]) -> tuple[RenderableType, ...]:
    """*renderables* hung when a command is printing, else as they are."""
    hang = _CURRENT.get()
    if hang is None:
        return renderables
    return tuple(hang.hung(renderable) for renderable in renderables)


def narrowed(width: int) -> int:
    """The width a body has: the terminal's, less the indent while a command prints.

    What a command computes from the output's width (wrapped rows, the choice
    of a narrow table) fits the hung body.
    """
    return max(1, width - INDENT) if _CURRENT.get() is not None else width


def spawn_detached(coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """A task for *coroutine* that does not inherit the command's hanging.

    What a command starts (a turn, a stream that follows a chat) prints on its
    own, at the left edge.
    """
    context = contextvars.copy_context()
    context.run(_CURRENT.set, None)
    return asyncio.create_task(coroutine, context=context)
