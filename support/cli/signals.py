"""SIGTERM, SIGHUP and a closed terminal in the terminal UI.

The first such signal bounds every live Agent's clean-up
(``Agent.begin_exit``), stops the session with cause ``shutdown`` and ends
the app; the process then exits by the signal. If the clean-up still runs
after :data:`SIGNAL_CLEANUP_S`, the tool providers are killed, the
terminal's settings are restored and the process exits there and then.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from collections.abc import Callable
from typing import Any, Final, NoReturn

from localmelo.support.cli.agents import AgentFactory

# As direct mode: the benchmark's SIGKILL comes 10 s after its SIGTERM.
SIGNAL_CLEANUP_S: Final = 8.0
_EXIT_MARGIN_S: Final = 1.0

EXIT_SIGNALS: Final = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)
)

# Bracketed paste off, cursor shown, attributes reset.
TERMINAL_RESET: Final = "\x1b[?2004l\x1b[?25h\x1b[0m"


def exit_by_signal(signum: int) -> NoReturn:
    """End the process by *signum*, as its default action would."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(OSError, ValueError, AttributeError):
            stream.flush()
    signal.signal(signum, signal.SIG_DFL)
    signal.raise_signal(signum)
    os._exit(128 + signum)  # only if the signal is blocked


def save_terminal() -> Callable[[], None]:
    """Save stdin's terminal settings; the function returned restores them."""
    try:
        import termios

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
    except (ImportError, OSError, ValueError, AttributeError):
        return lambda: None

    def restore() -> None:
        with contextlib.suppress(OSError, ValueError):
            termios.tcsetattr(fd, termios.TCSANOW, saved)
            os.write(sys.stdout.fileno(), TERMINAL_RESET.encode())

    return restore


class ExitSignals:
    """Handles SIGTERM and SIGHUP in the running loop (see the module docstring).

    *shutdown* starts the session's shutdown with a reason; *end_app* ends
    the app when it still runs (the caller then awaits
    :attr:`shutting_down`); *on_hangup* stops all drawing, since the
    terminal is gone.
    """

    def __init__(
        self,
        factory: AgentFactory,
        *,
        shutdown: Callable[[str], Any],
        end_app: Callable[[], None],
        on_hangup: Callable[[], None],
        restore_terminal: Callable[[], None],
    ) -> None:
        self._factory = factory
        self._shutdown = shutdown
        self._end_app = end_app
        self._on_hangup = on_hangup
        self._restore_terminal = restore_terminal
        self._installed: list[int] = []
        self.received: int | None = None
        self.shutting_down: asyncio.Future[Any] | None = None
        # Set when the first signal (or a hangup) arrives.
        self.arrived = asyncio.Event()

    def install(self) -> None:
        loop = asyncio.get_running_loop()
        for signum in EXIT_SIGNALS:
            if signal.getsignal(signum) == signal.SIG_IGN:
                continue  # nohup: stays ignored
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(signum, self.receive, signum)
                self._installed.append(signum)

    def uninstall(self) -> None:
        loop = asyncio.get_running_loop()
        for signum in self._installed:
            loop.remove_signal_handler(signum)
        self._installed.clear()

    def hangup(self) -> None:
        """The terminal closed: the SIGHUP path, with or without the signal."""
        if hasattr(signal, "SIGHUP"):
            self.receive(signal.SIGHUP)

    def receive(self, signum: int) -> None:
        if self.received is not None:
            return
        self.received = signum
        self.arrived.set()
        name = signal.Signals(signum).name
        if signum == getattr(signal, "SIGHUP", None):
            self._on_hangup()
        reason = f"Cancelled by {name}"
        for agent in self._factory.live_agents():
            agent.begin_exit(reason, within_s=SIGNAL_CLEANUP_S - _EXIT_MARGIN_S)
        self.shutting_down = asyncio.ensure_future(self._shutdown(reason))
        self._end_app()
        asyncio.get_running_loop().call_later(SIGNAL_CLEANUP_S, self._overdue)

    def _overdue(self) -> None:
        for agent in self._factory.live_agents():
            for provider in agent._tool_providers:
                with contextlib.suppress(Exception):
                    provider.kill()
        if self.received != getattr(signal, "SIGHUP", None):
            self._restore_terminal()
        assert self.received is not None
        exit_by_signal(self.received)
