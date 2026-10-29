"""``! command``: the person's own shell, run from the terminal UI.

This is not a model tool and not a sandbox. The command runs through the
person's ``$SHELL`` (``/bin/sh`` when that is unusable) with their account's
full permissions, in the workspace as its working directory, and it can leave
the workspace. No approval, checker or policy applies to it, and the model is
never told that it ran.

The child gets a pseudo-terminal of its own, in its own session and process
group, and melo relays bytes between that and the real terminal while the
prompt is paused. So a program that reads input, draws a screen or reacts to a
resize works, Ctrl+C and Esc reach the child (never melo's agent), and melo can
keep a bounded tail of the output for an explicit ``/attach !``. Only the
child's group is ever signalled; a descendant that detaches itself from that
group (``setsid``, ``nohup``) is out of reach, and melo does not claim
otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import os
import re
import select
import signal
import subprocess
import sys
import termios
import time
import tty
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.support.cli.attachments import MAX_FILE_BYTES, Attachment, _clean

# The tail of the output kept for a snapshot: what one attachment may hold.
MAX_OUTPUT_BYTES: Final = MAX_FILE_BYTES
# After SIGTERM (or SIGHUP) the child's group has this long before SIGKILL.
KILL_AFTER_S: Final = 2.0
# After the child exits, how long to wait for the last of its output.
DRAIN_S: Final = 0.25
RESIZE_POLL_S: Final = 0.1

# Leaves the cursor shown, attributes reset, mouse reporting and bracketed paste
# off: what a program killed in mid-screen may have left on.
TERMINAL_RESET: Final = (
    "\x1b[0m\x1b[?25h\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?2004l"
)

# Writes the escape and the command as arguments, so nothing is quoted twice.
# The bootstrap makes the child a session leader with the pty as its
# controlling terminal, and gives the shell back the default SIGPIPE that
# Python ignores.
_BOOTSTRAP: Final = (
    "import fcntl, os, signal, sys, termios\n"
    "os.setsid()\n"
    "fcntl.ioctl(0, termios.TIOCSCTTY, 0)\n"
    "signal.signal(signal.SIGPIPE, signal.SIG_DFL)\n"
    "os.execv(sys.argv[1], [sys.argv[1], '-c', sys.argv[2]])\n"
)

ESCAPE: Final = "\\!"

HELP: Final = (
    "! COMMAND runs COMMAND in your shell ($SHELL, else /bin/sh), in the workspace, as you.",
    "It is not a sandbox: no approval, no checker, your account's full permissions, and it can leave the workspace.",
    "Its output stays in the terminal. /attach ! attaches a redacted snapshot of the last output to your next message.",
    "melo must be idle. Ctrl+C and esc go to the command. Shell commands are not saved in the input history.",
    "\\! text sends a message that starts with ! (the backslash is dropped).",
)


@dataclass(frozen=True)
class ShellInput:
    kind: Literal["run", "help", "literal"]
    text: str  # the command, or the message for "literal"


def parse(text: str) -> ShellInput | None:
    """What a draft means as shell input; None for an ordinary message.

    ``! cmd`` runs, ``!`` alone shows help, ``\\! text`` is the message
    ``! text``. A draft that starts with a space or ``!word`` is a message.
    """
    if text.startswith(ESCAPE) and text[len(ESCAPE) :][:1] in ("", " ", "\t"):
        return ShellInput("literal", "!" + text[len(ESCAPE) :])
    if not text.startswith("!") or text[1:2] not in ("", " ", "\t", "\n"):
        return None
    command = text[1:].strip()
    return ShellInput("run", command) if command else ShellInput("help", "")


def user_shell(environ: dict[str, str] | None = None) -> str:
    """``$SHELL`` when it is an executable absolute path, else ``/bin/sh``."""
    shell = (os.environ if environ is None else environ).get("SHELL", "")
    if os.path.isabs(shell) and os.access(shell, os.X_OK) and not os.path.isdir(shell):
        return shell
    return "/bin/sh"


@dataclass(frozen=True)
class ShellResult:
    command: str
    cwd: str
    elapsed_s: float
    returncode: int | None  # negative: killed by that signal; None: never ran
    output: bytes  # the last MAX_OUTPUT_BYTES of what the child wrote
    total_bytes: int
    error: str = ""  # why it could not start
    hung_up: bool = False
    stopped: bool = False  # melo ended it

    @property
    def signal(self) -> str | None:
        if self.returncode is None or self.returncode >= 0:
            return None
        try:
            return signal.Signals(-self.returncode).name
        except ValueError:
            return f"signal {-self.returncode}"

    @property
    def truncated(self) -> bool:
        return self.total_bytes > len(self.output)

    def status(self) -> str:
        """``exit 0``, ``killed by SIGINT`` or why it did not run."""
        if self.error:
            return f"could not start: {self.error}"
        if self.signal:
            return f"killed by {self.signal}"
        return f"exit {self.returncode}"


def _window(fd: int) -> bytes | None:
    try:
        return fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8)  # type: ignore[no-any-return]
    except OSError:
        return None


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            view = view[os.write(fd, view) :]
        except BlockingIOError:
            select.select([], [fd], [], 1.0)
        except InterruptedError:
            continue


class ShellRun:
    """One ``!`` command: spawn, relay, restore (see the module docstring).

    *stdin_fd* and *stdout_fd* are the real terminal, or None for none
    (tests): without *stdin_fd* nothing is forwarded, without *stdout_fd* the
    output is only kept.
    """

    def __init__(
        self,
        command: str,
        cwd: str,
        *,
        stdin_fd: int | None = None,
        stdout_fd: int | None = None,
        shell: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.command = command
        self.cwd = cwd
        self._in = stdin_fd
        self._out = stdout_fd
        self._shell = shell or user_shell()
        self._clock = clock
        self._process: subprocess.Popen[bytes] | None = None
        self._tail = bytearray()
        self._total = 0
        self._last = b""
        self._restored = False
        self._saved: list[Any] | None = None
        self._hung_up = False
        self._stopped = False
        self.finished = False

    # ── Signals ──

    def _signal(self, number: int) -> None:
        """Signal the child's group; never one that melo or anyone else is in."""
        process = self._process
        if process is None or process.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            if os.getpgid(process.pid) == process.pid != os.getpgrp():
                os.killpg(process.pid, number)
            else:  # before the child's setsid: only the process itself
                os.kill(process.pid, number)

    def stop(self, number: int = signal.SIGTERM) -> None:
        """End the child's group now (melo exits); SIGKILL if it lingers."""
        self._stopped = True
        self._signal(number)
        with contextlib.suppress(RuntimeError):
            loop = asyncio.get_running_loop()
            loop.call_later(KILL_AFTER_S, self._signal, signal.SIGKILL)

    # ── The terminal ──

    def _enter_raw(self) -> None:
        fd = self._in
        if fd is None or not os.isatty(fd):
            return
        try:
            self._saved = termios.tcgetattr(fd)
            tty.setraw(fd, termios.TCSANOW)
        except termios.error:
            self._saved = None

    def _restore(self) -> None:
        """Give the terminal back, once: its mode, then what the child may have set."""
        if self._restored:
            return
        self._restored = True
        fd = self._in
        if fd is not None and self._saved is not None:
            with contextlib.suppress(termios.error, OSError):
                termios.tcsetattr(fd, termios.TCSANOW, self._saved)
                termios.tcflush(fd, termios.TCIFLUSH)  # keys the child never read
        if self._out is not None:
            ending = b"" if self._last in (b"", b"\n") else b"\r\n"
            with contextlib.suppress(OSError):
                _write_all(self._out, ending + TERMINAL_RESET.encode())

    # ── Running ──

    async def run(self) -> ShellResult:
        start = self._clock()
        try:
            master, slave = os.openpty()
        except OSError as exc:
            return self._result(start, None, str(exc))
        loop = asyncio.get_running_loop()
        try:
            self._size_pty(slave)
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    _BOOTSTRAP,
                    self._shell,
                    self.command,
                ],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                cwd=self.cwd,
                close_fds=True,
            )
        except (OSError, ValueError) as exc:
            os.close(master)
            os.close(slave)
            reason = (
                os.strerror(exc.errno)
                if isinstance(exc, OSError) and exc.errno
                else str(exc)
            )
            return self._result(start, None, reason)
        os.close(slave)
        os.set_blocking(master, False)
        eof = asyncio.Event()
        readers: list[int] = []

        def on_master() -> None:
            try:
                data = os.read(master, 65536)
            except BlockingIOError:
                return
            except OSError:
                data = b""  # EIO: nobody holds the child's end any more
            if not data:
                loop.remove_reader(master)
                eof.set()
                return
            self._keep(data)
            if self._out is not None:
                with contextlib.suppress(OSError):
                    _write_all(self._out, data)

        def on_input() -> None:
            assert self._in is not None
            try:
                data = os.read(self._in, 4096)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    return
                data = b""
            if not data:  # the terminal closed under us
                self._hung_up = True
                self.stop(signal.SIGHUP)
                loop.remove_reader(self._in)
                return
            with contextlib.suppress(OSError):
                os.write(master, data)

        try:
            self._enter_raw()
            loop.add_reader(master, on_master)
            readers.append(master)
            if self._in is not None:
                loop.add_reader(self._in, on_input)
                readers.append(self._in)
            resize = asyncio.ensure_future(self._follow_size(master))
            try:
                while self._process.poll() is None:
                    await asyncio.sleep(0.005)
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(DRAIN_S):
                        await eof.wait()
            finally:
                resize.cancel()
        finally:
            for fd in readers:
                with contextlib.suppress(ValueError, OSError):
                    loop.remove_reader(fd)
            os.close(master)  # hangs up on whatever still holds the pty
            self._restore()
            if self._process.poll() is None:  # cancelled: do not leave it running
                self.stop()
        self.finished = True
        return self._result(start, self._process.returncode)

    def _result(self, start: float, code: int | None, error: str = "") -> ShellResult:
        return ShellResult(
            self.command,
            self.cwd,
            max(0.0, self._clock() - start),
            code,
            bytes(self._tail),
            self._total,
            error,
            self._hung_up,
            self._stopped,
        )

    def _keep(self, data: bytes) -> None:
        self._total += len(data)
        self._last = data[-1:]
        self._tail.extend(data)
        if len(self._tail) > MAX_OUTPUT_BYTES:
            del self._tail[: len(self._tail) - MAX_OUTPUT_BYTES]

    def _size_pty(self, pty_fd: int) -> None:
        """Give the child's pty the terminal's settings and size."""
        fd = self._in
        if fd is None or not os.isatty(fd):
            return
        with contextlib.suppress(termios.error, OSError):
            termios.tcsetattr(pty_fd, termios.TCSANOW, termios.tcgetattr(fd))
        size = _window(fd)
        if size is not None:
            with contextlib.suppress(OSError):
                fcntl.ioctl(pty_fd, termios.TIOCSWINSZ, size)

    async def _follow_size(self, master: int) -> None:
        """Pass a resize on; the pty's kernel sends SIGWINCH to the child's group."""
        fd = self._in
        if fd is None or not os.isatty(fd):
            return
        last = _window(fd)
        while True:
            await asyncio.sleep(RESIZE_POLL_S)
            now = _window(fd)
            if now is not None and now != last:
                last = now
                with contextlib.suppress(OSError):
                    fcntl.ioctl(master, termios.TIOCSWINSZ, now)


# ── Snapshots ──

_ESCAPES: Final = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC
    r"|\x1b[PX^_][^\x1b]*(?:\x1b\\)?"  # DCS, SOS, PM, APC
    r"|\x1b\[[0-?]*[ -/]*[@-~]"  # CSI
    r"|\x1b[ -/]*[0-~]"  # the rest, two to three bytes
)
_OVERSTRIKE: Final = re.compile(r"[^\n]\x08")


def readable(output: bytes) -> str:
    """The text a person saw: escapes dropped, ``\\r`` redraws and backspaces applied."""
    text = output.decode("utf-8", errors="replace").lstrip("�")
    text = _ESCAPES.sub("", text).replace("\r\n", "\n")
    text = "\n".join(line.rsplit("\r", 1)[-1] for line in text.split("\n"))
    while _OVERSTRIKE.search(text):
        text = _OVERSTRIKE.sub("", text)
    return _clean(text)


def snapshot(result: ShellResult) -> Attachment | None:
    """The offer: the output's tail, readable and redacted; None when empty.

    ``changed`` says a secret was redacted, so that the caller shows the exact
    snapshot and asks, as it does for a file.
    """
    readable_text = readable(result.output)
    text = readable_text
    if not text.strip():
        return None
    redacted, _ = redact_secrets(text)
    text = redacted
    if result.truncated:
        text = (
            f"[earlier output left out: the last {len(result.output):,} of "
            f"{result.total_bytes:,} bytes]\n{text}"
        )
    name = " ".join(_clean(result.command).split())
    if len(name) > 60:
        name = name[:59] + "…"
    label, _ = redact_secrets(f"shell output: {name}")
    return Attachment(
        label, text, len(result.output), len(text), redacted != readable_text
    )
