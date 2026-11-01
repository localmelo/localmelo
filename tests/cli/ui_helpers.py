"""Drive the terminal UI's app in tests: a pipe for keys, pyte for the screen."""

from __future__ import annotations

import asyncio
import contextlib
import io
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import pyte
from prompt_toolkit.completion import Completer
from prompt_toolkit.data_structures import Size
from prompt_toolkit.history import History
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

from localmelo.support.cli.app import InteractiveApp, StatusBar
from localmelo.support.cli.approvals import TerminalApprover
from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.render import TurnRenderer
from localmelo.support.cli.theme import UNICODE
from localmelo.support.cli.views import BackendInfo

BACKSPACE = "\x7f"
ESC = "\x1b"
CTRL_C = "\x03"
CTRL_D = "\x04"
CTRL_J = "\n"
ENTER = "\r"
UP = "\x1b[A"
SHIFT_TAB = "\x1b[Z"


@dataclass
class FakeController:
    """A session for the app: it records what the app asks of it."""

    state: str = "idle"
    queued: str | None = None
    activity: str | None = None
    entered: list[str] = field(default_factory=list)
    stops: int = 0
    title: str = "New session"
    # Shift+Tab: how many times it was pressed.
    mode_presses: int = 0

    def enter(self, text: str) -> str | None:
        if self.state == "idle":
            self.entered.append(text)
            return None
        if self.state == "running":
            if self.queued is not None:
                return "One message is already queued (↑ edits it)."
            self.queued = text
            return None
        return "Stopping the turn; enter sends once it has stopped."

    def stop(self) -> bool:
        self.stops += 1
        if self.state == "running":
            self.state = "stopping"
        return True

    def take_queued(self) -> str | None:
        text, self.queued = self.queued, None
        return text

    def clear_queued(self) -> bool:
        had = self.queued is not None
        self.queued = None
        return had

    def tick(self) -> None:
        pass

    def cycle_mode(self) -> None:
        self.mode_presses += 1

    def status_bar(self) -> StatusBar:
        return StatusBar(self.title, "ask")


def backend() -> BackendInfo:
    return BackendInfo(
        key="openai-compatible",
        label="OpenAI-compatible",
        model="stub-model",
        base_url="http://127.0.0.1:9/v1",
    )


class Driver:
    """A running app: type into it, read its screen, wait for a condition."""

    def __init__(
        self,
        ui: InteractiveApp,
        pipe: PipeInput,
        written: io.StringIO,
        columns: int,
        rows: int,
    ) -> None:
        self.ui = ui
        self.pipe = pipe
        self.written = written
        self.columns = columns
        self.rows = rows

    def type(self, text: str) -> None:
        self.pipe.send_text(text)

    def screen(self) -> list[str]:
        screen = pyte.Screen(self.columns, self.rows)
        pyte.Stream(screen).feed(self.written.getvalue())
        return [line.rstrip() for line in screen.display]

    def text(self) -> str:
        return "\n".join(self.screen())

    def cursor_hidden(self) -> bool:
        screen = pyte.Screen(self.columns, self.rows)
        pyte.Stream(screen).feed(self.written.getvalue())
        return bool(screen.cursor.hidden)

    async def wait(self, check: Callable[[], bool], timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while not check():
            if time.monotonic() > deadline:
                raise AssertionError(f"timed out; screen:\n{self.text()}")
            await asyncio.sleep(0.02)

    async def settle(self, seconds: float = 0.15) -> None:
        """Let the app read what was typed and redraw."""
        await asyncio.sleep(seconds)


@contextlib.asynccontextmanager
async def running_app(
    controller: FakeController | None = None,
    *,
    columns: int = 80,
    rows: int = 24,
    history: History | None = None,
    completer: Completer | None = None,
    approver: TerminalApprover | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[Driver]:
    written = io.StringIO()
    output = Vt100_Output(
        written, lambda: Size(rows=rows, columns=columns), term="xterm-256color"
    )
    with create_pipe_input() as pipe:
        renderer = TurnRenderer(
            RecordingOut(columns), symbols=UNICODE, backend=backend, clock=clock
        )
        ui = InteractiveApp(
            controller or FakeController(),
            renderer,
            symbols=UNICODE,
            history=history,
            completer=completer,
            approver=approver,
            input=pipe,
            output=output,
            clock=clock,
        )
        task = asyncio.create_task(ui.run())
        driver = Driver(ui, pipe, written, columns, rows)
        try:
            await driver.wait(lambda: ui.app.is_running and "─" in written.getvalue())
            yield driver
        finally:
            ui.exit()
            with contextlib.suppress(asyncio.CancelledError, EOFError):
                await asyncio.wait_for(task, 5)
