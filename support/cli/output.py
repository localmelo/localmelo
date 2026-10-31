"""Where the terminal UI prints: into scrollback, above the live area.

:class:`TerminalOut` renders rich renderables to ANSI text and prints it
above the running app (prompt_toolkit erases the live area, the input box
and the status bar, prints, and draws them again). What is printed in one
pass of the event loop is printed in one go, so a burst of events costs one
redraw. :class:`RecordingOut` keeps plain text, for tests.

While a slash command runs (:func:`~localmelo.support.cli.hanging.under_command`)
both hang what is printed under the command's echo and report a width less
the indent; ``width`` is what a body has, not the terminal's.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
from collections.abc import Iterator
from typing import Final, Protocol

from prompt_toolkit.application import get_app_or_none
from prompt_toolkit.output import Output
from rich.console import Console, RenderableType

from localmelo.support.cli import hanging
from localmelo.support.cli.theme import active

# Cursor home, erase the screen, erase the scrollback.
CLEAR_SCREEN: Final = "\x1b[H\x1b[2J\x1b[3J"


class Out(Protocol):
    """Prints renderables into scrollback."""

    @property
    def width(self) -> int: ...

    def print(self, *renderables: RenderableType) -> None: ...

    def write_control(self, sequence: str) -> bool: ...

    def clear_screen(self) -> bool: ...


def _console(file: io.StringIO, width: int, *, no_color: bool) -> Console:
    palette = active()
    return Console(
        file=file,
        force_terminal=True,
        color_system=palette.rich_system,
        no_color=no_color,
        width=width,
        markup=False,
        highlight=False,
        emoji=False,
        theme=palette.rich_theme,
        legacy_windows=False,
    )


class TerminalOut:
    """Prints to *output* above the running app, or straight when none runs."""

    def __init__(self, output: Output, *, no_color: bool) -> None:
        self._output = output
        self._no_color = no_color
        self._pending: list[str] = []
        self._scheduled = False
        self._detached = False
        self._held = False
        self._escapes = not no_color and os.environ.get("TERM") != "dumb"
        try:
            self._escapes = self._escapes and os.isatty(self._output.fileno())
        except (OSError, ValueError, NotImplementedError):
            self._escapes = False

    @property
    def width(self) -> int:
        """The width a printed body has: the terminal's, less a command's hang indent."""
        return hanging.narrowed(self._columns())

    def _columns(self) -> int:
        return max(1, self._output.get_size().columns)

    def render(self, *renderables: RenderableType) -> str:
        """*renderables* as ANSI text at the terminal's width."""
        buffer = io.StringIO()
        console = _console(buffer, self._columns(), no_color=self._no_color)
        for renderable in renderables:
            console.print(renderable)
        return buffer.getvalue()

    def print(self, *renderables: RenderableType) -> None:
        if self._detached:
            return
        self._enqueue(self.render(*hanging.wrap(renderables)))

    def clear_screen(self) -> bool:
        """Wipe the screen and its scrollback before what is printed next.

        Queued with the text, so the app is erased, the screen cleared and the
        text written in one pass, then the app is drawn again from the top.
        False, and nothing queued, where escape codes are not wanted: no
        colour (``NO_COLOR``), ``TERM=dumb``, output that is not a terminal.
        """
        if self._detached or not self._escapes:
            return False
        self._enqueue(CLEAR_SCREEN)
        return True

    def _enqueue(self, text: str) -> None:
        self._pending.append(text)
        app = get_app_or_none()
        if app is None or not app.is_running:
            self.flush()
            return
        if not self._scheduled and not self._held:
            self._scheduled = True
            asyncio.get_running_loop().call_soon(self._print_above)

    @contextlib.contextmanager
    def held(self) -> Iterator[None]:
        """Keep what is printed meanwhile for after: a child owns the terminal."""
        self._held = True
        try:
            yield
        finally:
            self._held = False
            if self._pending and not self._scheduled:
                self._scheduled = True
                with contextlib.suppress(RuntimeError):
                    asyncio.get_running_loop().call_soon(self._print_above)

    def _print_above(self) -> None:
        """Write what is pending above the app, the terminal kept raw.

        prompt_toolkit's ``run_in_terminal`` switches the terminal to cooked
        mode while it runs: keys typed meanwhile were echoed into the
        scrollback, and Enter arrived as Ctrl+J, a new line, so a message
        typed then was not sent. Only text is written here, so the app is
        erased, the text written, and the app drawn again below it.
        """
        self._scheduled = False
        if not self._pending or self._held:
            return
        app = get_app_or_none()
        if app is None or not app.is_running:
            self.flush()
            return
        app.renderer.erase()
        self.flush()
        app.renderer.reset()
        app.renderer.request_absolute_cursor_position()
        app.invalidate()

    def flush(self) -> None:
        """Write what is pending; the newlines go out as CR LF."""
        text, self._pending = "".join(self._pending), []
        if not text or self._detached:
            return
        with contextlib.suppress(OSError, ValueError):
            self._output.write_raw(text.replace("\r\n", "\n").replace("\n", "\r\n"))
            self._output.flush()

    def write_control(self, sequence: str) -> bool:
        """Write a trusted zero-width control request, never model-provided escapes."""
        if self._detached:
            return False
        try:
            self._output.write_raw(sequence)
            self._output.flush()
        except (OSError, ValueError):
            return False
        return True

    def detach(self) -> None:
        """Drop all output from now on: the terminal is gone."""
        self._detached = True
        self._pending.clear()


class RecordingOut:
    """Keeps what is printed as plain text, at a fixed width (tests)."""

    def __init__(self, width: int = 80) -> None:
        self._width = width
        self.controls: list[str] = []
        self.cleared = 0  # screen clears asked for
        self.can_clear = True  # False: a terminal with no escape codes
        self.console = Console(
            file=io.StringIO(),
            record=True,
            width=width,
            color_system=None,
            force_terminal=False,
            markup=False,
            highlight=False,
            emoji=False,
            theme=active().rich_theme,
        )

    @property
    def width(self) -> int:
        return hanging.narrowed(self._width)

    def print(self, *renderables: RenderableType) -> None:
        for renderable in hanging.wrap(renderables):
            self.console.print(renderable)

    def write_control(self, sequence: str) -> bool:
        self.controls.append(sequence)
        return True

    def clear_screen(self) -> bool:
        """What was printed before is gone from :meth:`text`, as from a screen."""
        if not self.can_clear:
            return False
        self.cleared += 1
        self.console.export_text(clear=True)
        return True

    def held(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def text(self) -> str:
        """What was printed, with each line's trailing spaces removed."""
        exported = self.console.export_text(clear=False)
        return "\n".join(line.rstrip() for line in exported.split("\n"))
