"""Where the terminal UI prints: into scrollback, above the live area.

:class:`TerminalOut` renders rich renderables to ANSI text and prints it
above the running app (prompt_toolkit erases the live area, the input box
and the status bar, prints, and draws them again). What is printed in one
pass of the event loop is printed in one go, so a burst of events costs one
redraw. :class:`RecordingOut` keeps plain text, for tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
from typing import Protocol

from prompt_toolkit.application import get_app_or_none
from prompt_toolkit.output import Output
from rich.console import Console, RenderableType

from localmelo.support.cli.theme import RICH_THEME


class Out(Protocol):
    """Prints renderables into scrollback."""

    @property
    def width(self) -> int: ...

    def print(self, *renderables: RenderableType) -> None: ...


def _console(file: io.StringIO, width: int, *, no_color: bool) -> Console:
    return Console(
        file=file,
        force_terminal=True,
        color_system="standard",
        no_color=no_color,
        width=width,
        markup=False,
        highlight=False,
        emoji=False,
        theme=RICH_THEME,
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

    @property
    def width(self) -> int:
        return max(1, self._output.get_size().columns)

    def render(self, *renderables: RenderableType) -> str:
        """*renderables* as ANSI text at the current width."""
        buffer = io.StringIO()
        console = _console(buffer, self.width, no_color=self._no_color)
        for renderable in renderables:
            console.print(renderable)
        return buffer.getvalue()

    def print(self, *renderables: RenderableType) -> None:
        if self._detached:
            return
        self._pending.append(self.render(*renderables))
        app = get_app_or_none()
        if app is None or not app.is_running:
            self.flush()
            return
        if not self._scheduled:
            self._scheduled = True
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
        if not self._pending:
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

    def detach(self) -> None:
        """Drop all output from now on: the terminal is gone."""
        self._detached = True
        self._pending.clear()


class RecordingOut:
    """Keeps what is printed as plain text, at a fixed width (tests)."""

    def __init__(self, width: int = 80) -> None:
        self._width = width
        self.console = Console(
            file=io.StringIO(),
            record=True,
            width=width,
            color_system=None,
            force_terminal=False,
            markup=False,
            highlight=False,
            emoji=False,
            theme=RICH_THEME,
        )

    @property
    def width(self) -> int:
        return self._width

    def print(self, *renderables: RenderableType) -> None:
        for renderable in renderables:
            self.console.print(renderable)

    def text(self) -> str:
        """What was printed, with each line's trailing spaces removed."""
        exported = self.console.export_text(clear=False)
        return "\n".join(line.rstrip() for line in exported.split("\n"))
