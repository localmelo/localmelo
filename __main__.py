from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import importlib.metadata
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, TextIO, cast

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent.options import (
    NoteWriter,
    ReasoningMode,
    reasoning_refusal,
)
from localmelo.melo.checker import builtin_set_for
from localmelo.melo.executor import builtin_names
from localmelo.melo.memory.recall import (
    RECALL_K_MAX,
    parse_kinds,
    parse_triggers,
)
from localmelo.melo.schema import RunResult

if TYPE_CHECKING:
    from localmelo.melo.contracts.providers import BaseEmbeddingProvider
    from localmelo.melo.contracts.tools import BaseToolProvider
    from localmelo.support.config import Config
    from localmelo.support.progress_line import ProgressLine

# Exit codes of a direct-mode run: completed, crashed, and every other status.
EXIT_COMPLETED = 0
EXIT_CRASHED = 1
EXIT_NOT_COMPLETED = 2

# Seconds within which SIGTERM or SIGHUP ends a direct-mode run, clean-up
# included. The benchmark harness sends SIGKILL to melo's process group 10 s
# after its SIGTERM, and that SIGKILL does not reach shell commands or MCP
# servers: they run in sessions of their own, so only melo can stop them.
SIGNAL_CLEANUP_S = 8.0

# Part of SIGNAL_CLEANUP_S left over from the agent's own clean-up (see
# Agent.begin_exit), for the rest of close() and asyncio's shutdown.
_SIGNAL_EXIT_MARGIN_S = 1.0

# Signals that end a direct-mode run the way Ctrl+C does.
EXIT_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)
)

# Characters of the error that the text-mode status line shows at most.
STATUS_LINE_ERROR_CHARS = 500

# Flags that apply to direct mode (a query without --serve), by dest. The
# interactive UI also takes --workspace.
_DIRECT_FLAGS = {
    "workspace": "--workspace",
    "fixed_prefix": "--fixed-prefix",
    "captured_history": "--captured-history",
    "context_window": "--context-window",
    "output_reserve": "--output-reserve",
    "safety_margin": "--safety-margin",
    "max_steps": "--max-steps",
    "attempt_policy": "--attempt-policy",
    "reflection_gate": "--reflection-gate",
    "tool_exposure": "--tool-exposure",
    "builtins": "--builtins",
    "mcp_config": "--mcp-config",
    "mcp_only": "--mcp-only",
    "memory": "--memory",
    "memory_dir": "--memory-dir",
    "timeout": "--timeout",
    "request_timeout": "--request-timeout",
    "output_format": "--output-format",
    "embedding_base_url": "--embedding-base-url",
    "embedding_model": "--embedding-model",
    "progress": "--progress",
    "routine": "--routine",
    "reasoning": "--reasoning",
    "reasoning_field": "--reasoning-field",
    "replay_turns": "--replay-turns",
    "note_writer": "--note-writer",
    "retriever": "--retriever",
    "recall_at": "--recall-at",
    "recall_k": "--recall-k",
    "recall_kinds": "--recall-kinds",
    "recall_placement": "--recall-placement",
    "recall_min_score": "--recall-min-score",
    "retrieval_extra_body": "--retrieval-extra-body",
    "admit_on_outcome": "--admit-on-outcome",
    "keep_decision_text": "--keep-decision-text",
}

# Flags of the interactive UI (bare melo), by dest.
_UI_FLAGS = {
    "yolo": "--yolo",
    "continue_session": "-c",
    "resume": "-r",
}

# Direct-mode flags the interactive UI refuses: its Agents offer every tool,
# an approval's wait would count toward a timeout, it has no machine output,
# it draws its own progress. `--keep-decision-text` is an evaluation
# switch, so the UI never accepts it.
_UI_REFUSED = (
    "tool_exposure",
    "timeout",
    "output_format",
    "progress",
    "routine",
    "keep_decision_text",
)


def _register_backends() -> None:
    """Ensure built-in backend adapters are registered.

    Delegates to :func:`~localmelo.support.backends.ensure_defaults_registered`
    which is idempotent — safe to call multiple times and also called
    automatically by :func:`get_backend` / :func:`list_backends`.
    """
    from localmelo.support.backends import ensure_defaults_registered

    ensure_defaults_registered()


# ── Direct mode: output ──


def _exit_code(status: str) -> int:
    """0 for ``completed``, 1 for ``crashed``, 2 for any other status."""
    if status == "completed":
        return EXIT_COMPLETED
    if status == "crashed":
        return EXIT_CRASHED
    return EXIT_NOT_COMPLETED


def _one_line(text: str) -> str:
    """*text* as one line of at most :data:`STATUS_LINE_ERROR_CHARS` characters.

    Whitespace, newlines included (a proxy's HTML error page quoted in a
    provider error has them), is collapsed to single spaces; a longer text
    is cut and ends with ``...``.
    """
    line = " ".join(text.split())
    if len(line) > STATUS_LINE_ERROR_CHARS:
        line = line[: STATUS_LINE_ERROR_CHARS - 3] + "..."
    return line


def _exception_text(exc: BaseException) -> str:
    """``"<type>: <message>"``, or the type alone when the message is empty."""
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


class _StartupConfigError(Exception):
    """A config-file value that stops a direct worker from starting.

    It reaches the worker's ``startup_error`` result with its message, while
    the same failure without ``--brief`` stays a usage error or the old
    printed refusal.
    """


def _home_relative(text: str) -> str:
    """Show paths under the user's home directory as ``~/...``."""
    home = os.path.expanduser("~")
    if not home or home == os.sep:
        return text
    return re.sub(re.escape(home) + r"(?=/|$)", "~", text)


def _discard_stdout() -> None:
    """Point fd 1 at the null device, so shutdown cannot fail on a broken pipe.

    A failed ``print`` can leave the record in ``sys.stdout``'s buffer, and
    the interpreter's last flush would then fail again and exit 120 (see the
    worker startup path).
    """
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        return
    if fd != 1:
        return
    try:
        null = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return
    try:
        os.dup2(null, fd)
    except OSError:
        pass
    finally:
        os.close(null)


def _status_line(outcome: RunResult) -> str:
    """``melo: run ended with status <status>: <error>``, as one line.

    The error goes through :func:`_one_line`.
    """
    reason = _one_line(outcome.error) or "no error message"
    return f"melo: run ended with status {outcome.status}: {reason}"


def _localmelo_version() -> str:
    """``git describe --tags --always --dirty`` of the package's checkout.

    Only when the package directory is the top of a git checkout; otherwise
    (no git, not a checkout, or the package inside another repository) the
    installed package's version, or ``unknown``.
    """
    package_dir = Path(__file__).resolve().parent
    # Read-only: never take the index lock, which a concurrent git command
    # in the same checkout may hold.
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args],
            cwd=package_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return done.stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() == package_dir:
            described = git("describe", "--tags", "--always", "--dirty")
            if described:
                return described
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return importlib.metadata.version("localmelo")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _report(
    outcome: RunResult,
    output_format: str,
    version: str = "",
    out: TextIO | None = None,
) -> None:
    """Print a finished run in *output_format*.

    ``json``: exactly one line on stdout, the ``RunResult.to_dict()``
    object (fields in dataclass order) plus ``localmelo_version``, which
    is *version*. ``text``: the result, if any, on stdout, and the status
    line on stderr when the run did not complete. Stdout is *out*, by
    default ``sys.stdout`` (see :func:`_stdout_for_the_report`); it is
    flushed at once, so the report is out before the agent closes.
    """
    out = out if out is not None else sys.stdout
    if output_format == "json":
        record = {**outcome.to_dict(), "localmelo_version": version}
        print(json.dumps(record), file=out, flush=True)
        return
    if outcome.result:
        print(outcome.result, file=out, flush=True)
    if outcome.status != "completed":
        print(_status_line(outcome), file=sys.stderr)


@contextlib.contextmanager
def _stdout_for_the_report() -> Iterator[TextIO]:
    """Keep stdout for the report of a direct run, and yield its stream.

    When ``sys.stdout`` writes to file descriptor 1, as it does when melo
    runs as a program, fd 1 points at stderr until the block ends, and
    the stream yielded writes to the original stdout. Output that a tool
    writes to fd 1 itself, rather than through ``sys.stdout`` (a
    subprocess that ``python_exec`` code starts without capturing its
    output, ``os.write(1, ...)``, ``sys.__stdout__``), then goes to stderr
    and cannot mix with the result or the JSON line. When ``sys.stdout``
    is something else, such as a test's capture, nothing written to fd 1
    can reach it: it is yielded as it is, and fd 1 is left alone.
    """
    stdout = sys.stdout
    try:
        fd = stdout.fileno()
    except (AttributeError, OSError, ValueError):  # no file, or closed
        fd = -1
    if fd != 1:
        yield stdout
        return
    stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
    except OSError:  # no stderr to send it to: leave stdout as it is
        os.close(saved)
        yield stdout
        return
    try:
        with open(
            saved, "w", encoding=stdout.encoding, errors=stdout.errors, closefd=False
        ) as report:
            yield report
    finally:
        # What the run wrote to sys.stdout itself goes where fd 1 pointed
        # meanwhile: to stderr.
        with contextlib.suppress(OSError, ValueError):
            stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


# ── Direct mode: signals ──


def _exit_by_signal(signum: int) -> NoReturn:
    """End the process by *signum*, as its default action would."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(OSError, ValueError):
            stream.flush()
    signal.signal(signum, signal.SIG_DFL)
    signal.raise_signal(signum)
    # Reached only if the signal is blocked: exit as a shell reports it.
    os._exit(128 + signum)


class _ExitSignals:
    """Ends a direct-mode run on SIGTERM or SIGHUP the way Ctrl+C does.

    The first of these signals cancels the main task, as Ctrl+C does, once
    :meth:`Agent.begin_exit` has bounded the clean-up: a shell command is
    killed with its process group as the cancellation lands, the tool
    providers are closed (killed if they do not close in time), then the
    task record is saved, and reflections are not promoted. main() then
    re-raises the signal with its default handler, so the process exits by
    it, and prints no result. Should the clean-up still be running after
    :data:`SIGNAL_CLEANUP_S`, the tool providers are killed and the signal
    re-raised there and then. Later signals change nothing.

    A signal that melo was started with ignored (``nohup`` ignores SIGHUP)
    stays ignored.

    *before_exit*, when given, is called on the first signal before
    anything else, and again right before the process exits by it: the
    progress line uses it to erase itself (see :meth:`ProgressLine.close`).
    """

    def __init__(
        self,
        agent: Agent,
        tool_providers: Sequence[BaseToolProvider] = (),
        *,
        before_exit: Callable[[], object] | None = None,
    ) -> None:
        self._agent = agent
        self._tool_providers = tuple(tool_providers)
        self._before_exit = before_exit
        self._previous: list[tuple[int, Any]] = []
        # The signal that ended the run, if one did.
        self.received: int | None = None

    def install(self) -> None:
        """Handle the signals in the running loop, for its current task."""
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        for signum in EXIT_SIGNALS:
            previous = signal.getsignal(signum)
            if previous == signal.SIG_IGN:
                continue
            try:
                loop.add_signal_handler(signum, self._receive, signum, task)
            except (NotImplementedError, RuntimeError):
                continue  # not on this platform, or not the main thread
            self._previous.append((signum, previous))

    def uninstall(self) -> None:
        """Give the signals back to the handlers they had before install()."""
        loop = asyncio.get_running_loop()
        for signum, previous in self._previous:
            loop.remove_signal_handler(signum)
            if previous is not None:
                signal.signal(signum, previous)
        self._previous.clear()

    def _receive(self, signum: int, task: asyncio.Task[Any] | None) -> None:
        if self.received is not None:
            return  # the clean-up is under way already
        self.received = signum
        self._erase()
        self._agent.begin_exit(
            f"Cancelled by {signal.Signals(signum).name}",
            within_s=SIGNAL_CLEANUP_S - _SIGNAL_EXIT_MARGIN_S,
        )
        asyncio.get_running_loop().call_later(SIGNAL_CLEANUP_S, self._overdue)
        if task is not None:
            task.cancel()

    def _overdue(self) -> None:
        for provider in self._tool_providers:
            with contextlib.suppress(Exception):
                provider.kill()
        assert self.received is not None
        self._erase()
        _exit_by_signal(self.received)

    def _erase(self) -> None:
        if self._before_exit is not None:
            with contextlib.suppress(Exception):
                self._before_exit()


# ── Direct mode: run ──


def _enable_line_editing() -> None:
    """Edit the loop's prompt with readline when stdin is a terminal.

    Without it, the terminal's own Backspace removes one byte of a
    multi-byte character, such as one of 你好, so the screen shows a
    character that is gone, or input() fails to decode the line. Python
    builds without readline keep the terminal's editing.
    """
    if not sys.stdin.isatty():
        return
    with contextlib.suppress(ImportError):
        import readline  # noqa: F401  (input() uses it once imported)


async def _prompt() -> str:
    """Read a terminal line without blocking signal handling in the event loop.

    A daemon reader does not hold up process exit if a signal arrives while
    stdin is idle. The default asyncio executor would wait for a blocked
    input() thread during shutdown.
    """
    loop = asyncio.get_running_loop()
    line: asyncio.Future[str] = loop.create_future()

    def deliver(value: str | BaseException) -> None:
        if line.done():
            return
        if isinstance(value, BaseException):
            line.set_exception(value)
        else:
            line.set_result(value)

    def read() -> None:
        try:
            value: str | BaseException = input("\n❯ ")
        except BaseException as exc:
            value = exc
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(deliver, value)

    threading.Thread(target=read, daemon=True).start()
    return await line


async def _run_one(
    agent: Agent,
    query: str,
    *,
    progress: ProgressLine | None = None,
    routine: str | None = None,
) -> RunResult:
    """Run one task, with *progress* drawn while it runs, if given.

    Without *progress* the run starts as it always has: no sink and no
    progress events. *routine*, when given, names the run's routine,
    whether or not the line is drawn. The line is closed, and so erased,
    before this returns or raises.
    """
    kwargs: dict[str, Any] = {}
    if routine is not None:
        kwargs["routine"] = routine
    if progress is None:
        return await agent.run_task(query, **kwargs)
    progress.start()
    try:
        return await agent.run_task(
            query, observe=progress.observe, progress=True, **kwargs
        )
    finally:
        progress.close()


async def _run_worker_brief(
    agent: Agent,
    query: str,
    output_format: str,
    out: TextIO,
    save_session: str | None,
    parent_run: str | None,
    routine: str | None,
    inbox_path: str | None = None,
) -> int:
    """Run one opt-in worker and write its versioned report."""
    from localmelo.support.cli.worker_contract import (
        WorkerObserver,
        failure_record,
        result_record,
    )
    from localmelo.support.cli.worker_inbox import Inbox

    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None
    workspace = agent.workspace_root or os.getcwd()
    observer = WorkerObserver(
        out,
        workspace,
        stream=output_format == "stream-json",
        on_failure=lambda: loop.call_soon_threadsafe(task.cancel),
    )
    inbox_args: dict[str, Any] = (
        {"inbox": Inbox(inbox_path)} if inbox_path is not None else {}
    )
    try:
        if save_session is None:
            outcome = await agent.run_task(
                query, observe=observer, routine=routine, **inbox_args
            )
        else:
            from localmelo.support.sessions.worker import run_worker

            outcome = await run_worker(
                agent,
                query,
                title=save_session,
                parent_run=parent_run,
                observe=observer,
                routine=routine,
                **inbox_args,
            )
        if observer.error is not None:
            record = await asyncio.to_thread(
                failure_record,
                "transport_error",
                workspace,
                "Worker event output failed",
            )
            await asyncio.to_thread(observer.transport_result, record)
            await asyncio.to_thread(observer.close)
            print("melo: worker event output failed", file=sys.stderr)
            return EXIT_CRASHED
        code = _exit_code(outcome.status)
        record = await asyncio.to_thread(
            result_record, outcome, observer, workspace, code
        )
        if output_format == "stream-json":
            observer._enqueue({"v": 1, "type": "result", "result": record})
        else:
            print(
                json.dumps(record, ensure_ascii=True, allow_nan=False),
                file=out,
                flush=True,
            )
        await asyncio.to_thread(observer.close)
        if observer.error is not None:
            print(
                f"melo: worker output failed: {_one_line(str(observer.error))}",
                file=sys.stderr,
            )
            return EXIT_CRASHED
        return code
    except asyncio.CancelledError:
        if observer.error is None:
            raise
        record = await asyncio.to_thread(
            failure_record, "transport_error", workspace, "Worker event output failed"
        )
        await asyncio.to_thread(observer.transport_result, record)
        print(
            f"melo: worker output failed: {_one_line(str(observer.error))}",
            file=sys.stderr,
        )
        return EXIT_CRASHED
    finally:
        if observer.thread is not None and observer.thread.is_alive():
            await asyncio.to_thread(observer.close)


async def _run(
    agent: Agent,
    query: str | None,
    *,
    output_format: str = "text",
    version: str = "",
    exit_signals: _ExitSignals | None = None,
    out: TextIO | None = None,
    progress: ProgressLine | None = None,
    routine: str | None = None,
    save_session: str | None = None,
    parent_run: str | None = None,
    worker_brief: bool = False,
    inbox_path: str | None = None,
) -> int:
    """Run *query*, or with no query the interactive loop; return the exit code.

    One query is reported in *output_format* on *out* (see
    :func:`_report`), unless one of *exit_signals* arrived during the
    run; *progress* and *routine* apply to it (see :func:`_run_one`).
    The interactive loop prints text: each turn's result on stdout
    and, for a turn that did not complete, the status line on stderr,
    and goes on to the next turn.
    *exit_signals* is installed for the whole run, closing the agent
    included.
    """
    if exit_signals is not None:
        exit_signals.install()
    try:
        if query:
            if worker_brief:
                return await _run_worker_brief(
                    agent,
                    query,
                    output_format,
                    out or sys.stdout,
                    save_session,
                    parent_run,
                    routine,
                    inbox_path,
                )
            if save_session is None:
                outcome = await _run_one(
                    agent, query, progress=progress, routine=routine
                )
            else:
                from localmelo.support.sessions.worker import run_worker

                if progress is not None:
                    progress.start()
                try:
                    outcome = await run_worker(
                        agent,
                        query,
                        title=save_session,
                        parent_run=parent_run,
                        observe=None if progress is None else progress.observe,
                        routine=routine,
                    )
                finally:
                    if progress is not None:
                        progress.close()
            if exit_signals is not None and exit_signals.received is not None:
                return EXIT_CRASHED  # not used: the process exits by the signal
            _report(outcome, output_format, version, out)
            return _exit_code(outcome.status)
        _enable_line_editing()
        print("localmelo · local agent")
        print("Enter a task to begin. Type /help for commands or /exit to quit.")
        if "shell_exec" not in builtin_set_for(
            agent.options.mode, agent.options.builtins
        ):
            print(
                'shell_exec is off in this loop (direct mode); run melo "" '
                "--builtins all to turn it on."
            )
        while True:
            try:
                q = await _prompt()
            except (EOFError, KeyboardInterrupt):
                break
            except UnicodeDecodeError:
                print(
                    "melo: that line was not valid UTF-8; type it again.",
                    file=sys.stderr,
                )
                continue
            command = q.strip().lower()
            if command in ("exit", "quit", "/exit", "/quit"):
                break
            if command == "/help":
                print(
                    "Send a task to the configured model. /exit quits. Ctrl-D also exits."
                )
                print("Start the web interface separately with melo --serve.")
                continue
            if not command:
                continue
            outcome = await agent.run_task(q)
            if exit_signals is not None and exit_signals.received is not None:
                return EXIT_CRASHED  # the process exits by the signal
            if outcome.result:
                print(f"\n{outcome.result}")
            if outcome.status != "completed":
                print(_status_line(outcome), file=sys.stderr)
        return EXIT_COMPLETED
    finally:
        try:
            await agent.close()
        finally:
            if exit_signals is not None:
                exit_signals.uninstall()


def _progress_line(
    mode: str | None, query: str, output_format: str
) -> ProgressLine | None:
    """The progress line *mode* asks for (``--progress``), or None.

    Only with ``auto`` (the default) for a task with text output, while
    stderr is a terminal (not ``TERM=dumb``); see
    :func:`~localmelo.support.progress_line.wants_line`.
    """
    if mode == "off" or not query or output_format != "text":
        return None
    from localmelo.support.progress_line import (
        ProgressLine,
        wants_color,
        wants_line,
        wants_wide_ambiguous,
    )

    stream = sys.stderr
    if not wants_line(mode, query=query, output_format=output_format, stream=stream):
        return None
    return ProgressLine(
        stream, color=wants_color(), wide_ambiguous=wants_wide_ambiguous()
    )


def _run_direct(
    agent: Agent,
    query: str,
    output_format: str,
    tool_providers: Sequence[BaseToolProvider] = (),
    *,
    progress: str | None = None,
    routine: str | None = None,
    save_session: str | None = None,
    parent_run: str | None = None,
    worker_brief: bool = False,
    inbox_path: str | None = None,
) -> int:
    """Run *query*, or for an empty one the interactive loop; return the exit code.

    Stdout carries the report alone: what tools write to it during the
    run goes to stderr (see :func:`_stdout_for_the_report`).

    *progress* is ``--progress`` (None for its default, ``auto``): a
    progress line is drawn on stderr as :func:`_progress_line` says, and
    erased before the report, on SIGTERM or SIGHUP, and on any exception.
    *routine* is ``--routine``.

    A run ended by SIGTERM or SIGHUP does not return: the process exits by
    that signal (see :class:`_ExitSignals`), also when closing the agent
    then fails, which is reported in one line on stderr.
    """
    # Taken before the run, so the record names the code that was loaded.
    version = _localmelo_version() if output_format == "json" else ""
    line = _progress_line(progress, query, output_format)
    exit_signals = _ExitSignals(
        agent, tool_providers, before_exit=None if line is None else line.close
    )
    reported = False
    try:
        output = (
            _stdout_for_the_report() if query else contextlib.nullcontext(sys.stdout)
        )
        with output as out:
            worker_args: dict[str, Any] = {}
            if worker_brief:
                worker_args["worker_brief"] = True
            if inbox_path is not None:
                worker_args["inbox_path"] = inbox_path
            code = asyncio.run(
                _run(
                    agent,
                    query,
                    output_format=output_format,
                    version=version,
                    exit_signals=exit_signals,
                    out=out,
                    progress=line,
                    routine=routine,
                    save_session=save_session,
                    parent_run=parent_run,
                    **worker_args,
                )
            )
            reported = code == EXIT_CRASHED
    except BaseException as exc:
        if line is not None:
            line.close()  # before a traceback or the line below
        if (
            isinstance(exc, BrokenPipeError)
            and worker_brief
            and exit_signals.received is None
        ):
            if not reported:
                print("melo: worker output failed: broken pipe", file=sys.stderr)
            return EXIT_CRASHED
        if exit_signals.received is None:
            if not query and isinstance(exc, KeyboardInterrupt):
                _exit_by_signal(signal.SIGINT)
            raise
        if not isinstance(exc, asyncio.CancelledError):
            text = _exception_text(exc)
            print(f"melo: error while stopping: {_one_line(text)}", file=sys.stderr)
        code = EXIT_CRASHED  # not used: the signal ends the process below
    if exit_signals.received is not None:
        if line is not None:
            line.close()
        _exit_by_signal(exit_signals.received)
    return code


# ── Direct mode: flags ──


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(
            f"must be a finite number of seconds > 0, got {text}"
        )
    return value


def _non_empty(text: str) -> str:
    if not text:
        raise argparse.ArgumentTypeError("must not be empty")
    return text


def _routine_name(text: str) -> str:
    if not text.strip():
        raise argparse.ArgumentTypeError("must not be blank")
    return text


def _parse_builtins(parser: argparse.ArgumentParser, text: str) -> frozenset[str]:
    """``all`` (every built-in, named explicitly), ``none`` or a name list."""
    if text == "all":
        return frozenset(builtin_names())
    if text == "none":
        return frozenset()
    names = [name.strip() for name in text.split(",")]
    if not all(names):
        parser.error(
            "argument --builtins: expected all, none or built-in tool names "
            f"separated by commas, got {text!r}"
        )
    known = builtin_names() | {"recall"}
    unknown = sorted(set(names) - known)
    if unknown:
        parser.error(
            f"argument --builtins: unknown built-in tools {', '.join(unknown)}; "
            f"known: {', '.join(sorted(known))}"
        )
    return frozenset(names)


def _recall_triggers(text: str) -> frozenset[str]:
    try:
        return parse_triggers(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _recall_kinds(text: str) -> frozenset[str]:
    try:
        return parse_kinds(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _recall_k(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if not 1 <= value <= RECALL_K_MAX:
        raise argparse.ArgumentTypeError(
            f"must be an integer from 1 to {RECALL_K_MAX}, got {value}"
        )
    return value


def _unit_score(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"must be a number from 0 to 1, got {text}")
    return value


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not JSON: {exc}") from None
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    # Validate here, not in the provider: a reserved field is a usage
    # error (exit 2, one line), never a traceback from OpenAICompatLLM.
    from localmelo.support.providers.llm._protocol import prepare_extra_body

    try:
        prepare_extra_body(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return value


def _recall_flag_given(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, dest, None) is not None
        for dest in (
            "retriever",
            "recall_at",
            "recall_k",
            "recall_kinds",
            "recall_placement",
            "recall_min_score",
            "retrieval_extra_body",
        )
    )


def _check_recall_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject R5's flag combinations before anything runs."""
    effective_legacy = getattr(args, "reasoning", None) == "legacy" or (
        args.base_url and not (args.serve or args.daemon) and args.reasoning is None
    )
    if effective_legacy and _recall_flag_given(args):
        parser.error(
            "argument --retriever/--recall-*/--retrieval-*: retrieval is not "
            "available in legacy; use --reasoning compress, compress+replay, "
            "drop or replay"
        )
    recall_flags = [
        flag
        for dest, flag in _DIRECT_FLAGS.items()
        if dest
        in (
            "recall_at",
            "recall_k",
            "recall_kinds",
            "recall_placement",
            "recall_min_score",
        )
        and getattr(args, dest) is not None
    ]
    retriever = getattr(args, "retriever", None)
    if retriever == "off" and recall_flags:
        parser.error("argument --retriever off: do not give " + ", ".join(recall_flags))
    recall_at = getattr(args, "recall_at", None)
    if recall_at and "tool" in recall_at:
        if args.mcp_only:
            parser.error("argument --recall-at: tool is not available with --mcp-only")
        if args.builtins is not None and args.builtins != "all":
            names = _parse_builtins(parser, args.builtins)
            if "recall" not in names:
                parser.error(
                    "argument --recall-at tool: --builtins must include recall"
                )
        if retriever == "off":
            parser.error(
                "argument --recall-at tool: it needs a retriever other than off"
            )
    if getattr(args, "recall_placement", None) == "message" and args.reasoning in (
        "replay",
        "compress+replay",
    ):
        parser.error(
            "argument --recall-placement: message is not available in replay modes"
        )
    if getattr(args, "retrieval_extra_body", None) is not None and retriever in (
        "off",
        "bm25",
        "embedding",
    ):
        parser.error("argument --retrieval-extra-body: it needs --retriever llm")


def _check_effective_recall(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    options: AgentOptions,
) -> None:
    """Reject a recall combination that flags or the config table built."""
    if options.reasoning in ("replay", "compress+replay") and (
        options.recall_placement == "message"
    ):
        parser.error(
            "argument --recall-placement: message is not available in replay modes"
        )
    if options.retriever == "off" and (
        options.recall_at
        or options.recall_k != 3
        or options.recall_kinds is not None
        or options.recall_placement != "tool"
        or options.recall_min_score != 0.35
    ):
        parser.error("argument --retriever off: do not give the --recall-* flags")
    if (
        options.builtins is not None
        and "recall" in options.builtins
        and "tool" not in (options.recall_at or ())
    ):
        parser.error("argument --builtins: recall needs --recall-at tool")
    if options.recall_at and "tool" in options.recall_at:
        if getattr(args, "mcp_only", False):
            parser.error("argument --recall-at: tool is not available with --mcp-only")
        if (
            options.builtins is not None
            and "recall" not in options.builtins
            and getattr(args, "builtins", None) != "all"
        ):
            parser.error("argument --recall-at tool: --builtins must include recall")
        if options.retriever == "off":
            parser.error(
                "argument --recall-at tool: it needs a retriever other than off"
            )
    if getattr(
        args, "retrieval_extra_body", None
    ) is not None and options.retriever in ("off", "bm25", "embedding"):
        parser.error("argument --retrieval-extra-body: it needs --retriever llm")


def _check_ui_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    options: AgentOptions,
) -> None:
    """Apply the direct recall refusals after the UI resolves its config defaults."""
    effective = argparse.Namespace(**vars(args))
    effective.reasoning = options.reasoning
    _check_recall_args(parser, effective)
    _check_effective_recall(parser, args, options)


def _check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject flag combinations before anything runs (argparse exit code 2)."""
    if args.import_session is not None:
        if args.import_session[0] != "dsh-jsonl":
            parser.error("--import-session format must be dsh-jsonl")
        if (
            args.query
            or args.serve
            or args.daemon
            or args.reconfigure
            or args.save_session is not None
            or args.parent_run
        ):
            parser.error(
                "--import-session cannot be combined with a task or another mode"
            )
    elif args.title is not None:
        parser.error("--title requires --import-session")
    if args.parent_run is not None and args.save_session is None:
        parser.error("--parent-run requires --save-session")
    if args.save_session is not None and (
        not args.query
        or not " ".join(args.query).strip()
        or args.serve
        or args.daemon
        or args.reconfigure
    ):
        parser.error("--save-session is only for a direct task")
    if args.mcp_only and args.mcp_config is None:
        parser.error("--mcp-only requires --mcp-config")
    if args.mcp_only:
        for dest in ("builtins", "tool_exposure"):
            if getattr(args, dest) is not None:
                parser.error(f"--mcp-only sets {_DIRECT_FLAGS[dest]}; do not give both")
    _check_recall_args(parser, args)
    if args.embedding_model is not None and args.embedding_base_url is None:
        parser.error("--embedding-model requires --embedding-base-url")
    if args.embedding_base_url is not None and not args.base_url:
        parser.error("--embedding-base-url requires --base-url")
    reasoning = getattr(args, "reasoning", None)
    reasoning_field = getattr(args, "reasoning_field", None)
    if (
        args.base_url
        and not (args.serve or args.daemon)
        and reasoning in ("replay", "compress+replay")
        and reasoning_field is None
    ):
        parser.error(
            "argument --reasoning: replay and compress+replay need a reasoning "
            "field: give --reasoning-field (auto, reasoning_content or "
            "reasoning) with --base-url, or set reasoning_field in the "
            "backend's config"
        )
    if (args.serve or args.daemon) and (args.base_url or args.chat_model):
        parser.error(
            "--base-url and --chat-model are for a task or the interactive UI; "
            "--serve uses the config file (melo --reconfigure)"
        )
    if args.reconfigure and args.base_url and not args.query:
        parser.error(
            "--reconfigure sets up the config file, which --base-url does not use; "
            "give one of them"
        )
    if getattr(args, "enable_playground", False) and (
        not args.serve or args.daemon or args.query
    ):
        parser.error("--enable-playground is only for --serve")
    if getattr(args, "allow_unconfigured", False):
        if not args.serve:
            parser.error("--allow-unconfigured is only for --serve")
        if args.reconfigure:
            parser.error("--allow-unconfigured cannot be used with --reconfigure")
    # Bare melo, which opens the interactive UI; `melo ""` is direct mode.
    interactive = not args.query and not args.serve and not args.daemon
    ui_given = [
        flag
        for dest, flag in _UI_FLAGS.items()
        if getattr(args, dest, None) not in (None, False)
    ]
    if ui_given and not interactive:
        parser.error(
            "only for the interactive UI (melo without a query, --serve or --daemon): "
            + ", ".join(ui_given)
        )
    if (
        getattr(args, "continue_session", False)
        and getattr(args, "resume", None) is not None
    ):
        parser.error("-c and -r cannot be combined")
    auth_command = getattr(args, "gateway_login", False) or getattr(
        args, "gateway_rotate_token", False
    )
    if (
        interactive
        and not auth_command
        and (args.host is not None or args.port is not None)
    ):
        parser.error(
            "--host and --port are for --serve and --daemon; "
            "bare melo opens the interactive UI"
        )
    # main() joins the words into the query, and an empty one (`melo ""`)
    # runs no task, so the direct-mode flags would have nothing to apply to.
    query = " ".join(args.query)
    direct = bool(query) and not args.serve
    # `melo ""` is the loop: it runs turns, so --builtins applies to them
    # even though the other direct-mode flags have no task. --serve and
    # --daemon are not the loop and keep refusing it, as on main.
    loop = bool(args.query) and not args.serve and not args.daemon and not query
    if not direct:
        given = [
            flag
            for dest, flag in _DIRECT_FLAGS.items()
            if getattr(args, dest, None) not in (None, False)
            and (not interactive or dest in _UI_REFUSED)
            and not (dest == "builtins" and loop)
        ]
        if given:
            parser.error(
                "only for direct mode (a query without --serve): " + ", ".join(given)
            )


def _direct_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> AgentOptions:
    """The AgentOptions the direct-mode flags ask for; defaults otherwise."""
    fields: dict[str, Any] = {}
    if args.workspace is not None and args.query:
        # Direct mode only: bare melo resolves the folder itself (launch.py),
        # and `melo ""` refuses the flag in _check_args.
        root = os.path.realpath(os.path.expanduser(args.workspace))
        if not os.path.isdir(root):
            parser.exit(
                EXIT_NOT_COMPLETED,
                f"{parser.prog}: --workspace: not a directory: {args.workspace}\n",
            )
        fields["workspace_root"] = root
    if args.max_steps is not None:
        fields["max_steps"] = args.max_steps
    if args.attempt_policy is not None:
        fields["attempt_policy"] = args.attempt_policy
    if args.reflection_gate is not None:
        fields["reflection_gate"] = args.reflection_gate
    if args.tool_exposure is not None:
        fields["tool_exposure"] = args.tool_exposure
    if args.builtins is not None:
        fields["builtins"] = _parse_builtins(parser, args.builtins)
    if args.mcp_only:
        fields["builtins"] = frozenset()
        fields["tool_exposure"] = "all"
    if args.memory is not None:
        fields["memory"] = args.memory == "on"
    if args.memory_dir is not None:
        fields["memory_dir"] = args.memory_dir
    if args.timeout is not None:
        fields["timeout_s"] = args.timeout
    if getattr(args, "fixed_prefix", None) is not None:
        fields["fixed_prefix"] = args.fixed_prefix == "on"
    if getattr(args, "captured_history", None) is not None:
        fields["captured_history"] = args.captured_history == "on"
    reasoning = getattr(args, "reasoning", None)
    if reasoning is not None:
        fields["reasoning"] = reasoning
    replay_turns = getattr(args, "replay_turns", None)
    if replay_turns is not None:
        fields["replay_turns"] = replay_turns
    note_writer = getattr(args, "note_writer", None)
    if note_writer is not None:
        fields["note_writer"] = note_writer
    retriever = getattr(args, "retriever", None)
    if retriever is not None:
        fields["retriever"] = retriever
    recall_at = getattr(args, "recall_at", None)
    if recall_at is not None:
        fields["recall_at"] = recall_at
    recall_k = getattr(args, "recall_k", None)
    if recall_k is not None:
        fields["recall_k"] = recall_k
    recall_kinds = getattr(args, "recall_kinds", None)
    if recall_kinds is not None:
        fields["recall_kinds"] = recall_kinds
    recall_placement = getattr(args, "recall_placement", None)
    if recall_placement is not None:
        fields["recall_placement"] = recall_placement
    recall_min_score = getattr(args, "recall_min_score", None)
    if recall_min_score is not None:
        fields["recall_min_score"] = recall_min_score
    admit_on_outcome = getattr(args, "admit_on_outcome", None)
    if admit_on_outcome is not None:
        fields["admit_on_outcome"] = admit_on_outcome == "on"
    if getattr(args, "keep_decision_text", False):
        fields["keep_decision_text"] = True
    try:
        return AgentOptions(**fields)
    except ValueError as exc:  # the flag types should have caught it
        parser.error(str(exc))


def _direct_tool_providers(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[BaseToolProvider]:
    """One McpToolProvider for --mcp-config; a bad file ends melo (exit 2)."""
    if args.mcp_config is None:
        return []
    from localmelo.support.mcp import McpConfigError, McpToolProvider, load_mcp_config

    # load_mcp_config reports a file it cannot read, or one that is not
    # valid JSON or not a valid config, as McpConfigError, but not text
    # that is not UTF-8 (a UTF-16 file, say), an integer too long to
    # convert, or JSON nested deeper than the decoder can follow.
    path = args.mcp_config
    try:
        servers = load_mcp_config(path)
    except McpConfigError as exc:
        problem = str(exc)
    except UnicodeDecodeError as exc:
        problem = f"Cannot read MCP config {path}: {exc}"
    except (ValueError, RecursionError) as exc:
        problem = f"Cannot parse MCP config {path}: {_exception_text(exc)}"
    else:
        return [McpToolProvider(servers)]
    parser.exit(EXIT_NOT_COMPLETED, f"{parser.prog}: error: {problem}\n")


class _WorkerFlagParser(argparse.ArgumentParser):
    """Keep old abbreviations unambiguous; new options require full names."""

    def _check_value(self, action: argparse.Action, value: str) -> None:
        if action.dest == "output_format" and value == "stream-json":
            return
        super()._check_value(action, value)

    def _get_option_tuples(self, option_string: str) -> Any:
        full_name_only = {
            "--save-session",
            "--brief",
            "--inbox",
            "--parent-run",
            "--import-session",
            "--title",
            "--fixed-prefix",
            "--captured-history",
            "--context-window",
            "--output-reserve",
            "--safety-margin",
        }
        return [
            item
            for item in super()._get_option_tuples(option_string)
            if item[1] not in full_name_only
        ]


class _DirectParser(_WorkerFlagParser):
    """Probe legacy argv without printing usage or exiting the process."""

    validation_error: str | None = None

    def error(self, message: str) -> NoReturn:
        original = sys.exception()
        if isinstance(original, argparse.ArgumentError) and original.args:
            action = original.args[0]
            if isinstance(action, argparse.Action) and (
                action.type is not None or action.choices is not None
            ):
                self.validation_error = message
        raise argparse.ArgumentError(None, message)


class _InteractiveParser(_WorkerFlagParser):
    """Report a legacy value error without reinterpreting an abbreviated option."""

    validation_error: str | None = None

    def error(self, message: str) -> NoReturn:
        super().error(self.validation_error or message)


def _build_parser(*, direct_only: bool = False) -> argparse.ArgumentParser:
    parser_type = _DirectParser if direct_only else _InteractiveParser
    parser = parser_type(
        prog="melo", description="localmelo agent", add_help=not direct_only
    )
    parser.add_argument("query", nargs="*", help="Task to execute (direct mode)")
    parser.add_argument("--base-url", default=None, help="LLM API base URL")
    parser.add_argument("--chat-model", default=None, help="Chat model name")

    direct = parser.add_argument_group(
        "direct mode", "Options for a query run without --serve."
    )
    for name in ("context-window", "output-reserve", "safety-margin"):
        direct.add_argument(
            f"--{name}", type=_positive_int, metavar="TOKENS", default=argparse.SUPPRESS
        )
    direct.add_argument(
        "--inbox", metavar="FILE", help="Read appended worker instructions from FILE"
    )
    direct.add_argument(
        "--brief", metavar="FILE", help="Read a direct worker task from a UTF-8 file"
    )
    direct.add_argument(
        "--save-session",
        nargs="?",
        const="",
        metavar="TITLE",
        help="Save this direct run as a worker session",
    )
    direct.add_argument(
        "--parent-run", metavar="ID", help="Launcher run ID (with --save-session)"
    )
    direct.add_argument(
        "--workspace",
        metavar="DIR",
        help="The folder melo works in (default: the current directory)",
    )
    direct.add_argument(
        "--max-steps",
        type=_positive_int,
        metavar="N",
        help="Cap on planning calls for the run (default 30)",
    )
    direct.add_argument(
        "--attempt-policy",
        choices=["reflect", "none"],
        help="Reflect between attempts (default), or run one attempt",
    )
    direct.add_argument(
        "--reflection-gate",
        choices=["flags", "utility"],
        help="After a reflection, start the next attempt when it chooses continue "
        "and its three checks hold (flags, default), or also require its utility "
        "score to reach 0.1 (utility, the behavior before this option)",
    )
    direct.add_argument(
        "--captured-history",
        choices=["on", "off"],
        help="Store bounded, redacted captured tool results (default off)",
    )
    direct.add_argument(
        "--fixed-prefix",
        choices=["on", "off"],
        help="Use a versioned run-frozen system prefix and local date (default off)",
    )
    direct.add_argument(
        "--tool-exposure",
        choices=["retrieve", "all"],
        help="Offer every built-in plus the extra tools retrieved for each step "
        "(default), or all tools",
    )
    direct.add_argument(
        "--builtins",
        metavar="all|none|NAME,...",
        help="Built-in tools to register (default: the mode's set)",
    )
    direct.add_argument(
        "--mcp-config",
        metavar="PATH",
        help="MCP servers to use, in the mcpServers JSON format",
    )
    direct.add_argument(
        "--mcp-only",
        action="store_true",
        help="Only the MCP tools: --builtins none --tool-exposure all",
    )
    direct.add_argument(
        "--memory", choices=["on", "off"], help="Long-term memory (default on)"
    )
    direct.add_argument(
        "--memory-dir",
        type=_non_empty,
        metavar="DIR",
        help="Directory for SQLite memory (history.db, long_term.db, procedures.db)",
    )
    direct.add_argument(
        "--timeout",
        type=_seconds,
        metavar="SECONDS",
        help="Wall-clock limit of the run (default none)",
    )
    direct.add_argument(
        "--request-timeout",
        type=_seconds,
        metavar="SECONDS",
        help="HTTP timeout of each chat request, in seconds (default: "
        "request_timeout_s from the config file, else 120; 300 for the ollama "
        "backend)",
    )
    direct.add_argument(
        "--output-format",
        choices=["text", "json", "stream-json"],
        help="text (default), one JSON line, or stream-json with --brief",
    )
    direct.add_argument(
        "--embedding-base-url",
        metavar="URL",
        help="OpenAI-compatible embedding endpoint (with --base-url)",
    )
    direct.add_argument(
        "--embedding-model",
        metavar="NAME",
        help="Embedding model name (with --embedding-base-url)",
    )
    direct.add_argument(
        "--progress",
        choices=["auto", "off"],
        help="Draw a progress line on stderr while the task runs: auto (default) "
        "when stderr is a terminal and the output is text, or off",
    )
    direct.add_argument(
        "--routine",
        type=_routine_name,
        metavar="NAME",
        help="Learn and recall the task's steps as the routine NAME, in any "
        "folder and whatever the task's wording",
    )
    direct.add_argument(
        "--reasoning",
        choices=["legacy", "drop", "replay", "compress", "compress+replay"],
        help="How a reply's reasoning is sent back: legacy (default, the "
        "provider's own behavior), drop (never), replay (on assistant turns "
        "in scope, under one key), compress (no reasoning; each reply becomes "
        "a short note, with its drafts, in working memory), or "
        "compress+replay (compress and replay); the compress modes recall "
        "nothing yet",
    )
    direct.add_argument(
        "--reasoning-field",
        choices=["auto", "reasoning_content", "reasoning"],
        help="The request key replay and compress+replay use (with --base-url, "
        "or a backend's reasoning_field); auto follows the key the last reply "
        "came from",
    )
    direct.add_argument(
        "--replay-turns",
        type=_positive_int,
        metavar="N",
        help="With replay, keep the newest N assistant turns' reasoning "
        "(default: every turn in scope)",
    )
    direct.add_argument(
        "--note-writer",
        choices=["auto", "heuristic"],
        help="With compress or compress+replay, who writes a reply's note: "
        "auto (default: read the model's <note>, else the heuristic) or "
        "heuristic (always from the reply's text and thinking). Accepted and "
        "inert in the other modes",
    )
    direct.add_argument(
        "--retriever",
        choices=["auto", "off", "bm25", "llm", "embedding"],
        help="How recall ranks documents: auto (default: embedding when "
        "configured; else llm at start/attempt and bm25 at step/tool), or "
        "one named retriever for every trigger",
    )
    direct.add_argument(
        "--recall-at",
        type=_recall_triggers,
        metavar="LIST",
        help="Comma list of recall triggers: start, attempt, step, tool "
        "(default start,attempt,step; tool is never on unless named)",
    )
    direct.add_argument(
        "--recall-k",
        type=_recall_k,
        metavar="N",
        help=f"Task-local hits per recall, 1-{RECALL_K_MAX} (default 3)",
    )
    direct.add_argument(
        "--recall-kinds",
        type=_recall_kinds,
        metavar="LIST",
        help="Comma list of document kinds: note, draft, thought, digest, "
        "reflection, long (default all but thought)",
    )
    direct.add_argument(
        "--recall-placement",
        choices=["tool", "message"],
        help="Where a step block goes: the last tool result (tool, default), "
        "or its own user message (message; not in replay modes)",
    )
    direct.add_argument(
        "--recall-min-score",
        type=_unit_score,
        metavar="F",
        help="Cosine floor for the long-term leg and embedding hits, 0-1 "
        "(default 0.35; inert only for the task-local bm25/llm ranking)",
    )
    direct.add_argument(
        "--admit-on-outcome",
        choices=["on", "off"],
        help="R6-lite: with a reasoning mode other than legacy, admit the final "
        "attempt digest to long-term memory at the end of the run (default off); "
        "inert in legacy",
    )
    direct.add_argument(
        "--keep-decision-text",
        action="store_true",
        help="R6-lite: also store the checked text of each admission decision "
        "(the hash is always kept); an evaluation switch, refused in the UI",
    )
    direct.add_argument(
        "--retrieval-extra-body",
        type=_json_object,
        metavar="JSON",
        help="Extra top-level fields for retrieve requests only (llm; a JSON "
        "object, in memory only)",
    )

    if not direct_only:
        interactive = parser.add_argument_group(
            "interactive mode",
            "Options for bare melo, which opens the interactive UI on a terminal. "
            "It also takes --base-url, --chat-model, --workspace, --max-steps, "
            "--attempt-policy, --reflection-gate, --builtins, --memory, "
            "--memory-dir, --request-timeout, --reasoning, --reasoning-field, "
            "--replay-turns, --note-writer, --retriever, --recall-*, "
            "--retrieval-extra-body and the --embedding-* options.",
        )
        interactive.add_argument(
            "--yolo",
            action="store_true",
            help="Run every tool without asking for approval",
        )
        interactive.add_argument(
            "-c",
            "--continue",
            dest="continue_session",
            action="store_true",
            help="Resume the newest session in this workspace",
        )
        interactive.add_argument(
            "-r",
            "--resume",
            nargs="?",
            const="",
            metavar="ID|TITLE",
            help="Resume a session, or open the session picker without a value",
        )

    if not direct_only:
        parser.add_argument(
            "--gateway-login",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Print a one-use console sign-in link to the terminal",
        )
        parser.add_argument(
            "--gateway-rotate-token",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Rotate the local gateway access token",
        )
        parser.add_argument(
            "--enable-playground",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Enable authenticated development playground routes",
        )
    # gateway mode
    parser.add_argument("--serve", action="store_true", help="Run as gateway server")
    parser.add_argument("--host", default=None, help="Gateway listen host")
    parser.add_argument("--port", type=int, default=None, help="Gateway listen port")
    parser.add_argument(
        "--allow-unconfigured",
        action="store_true",
        help="Start the gateway even when no backend is configured or the config has errors",
    )

    # setup
    parser.add_argument(
        "--import-session",
        nargs=2,
        metavar=("FORMAT", "FILE"),
        help="Import a worker history (format: dsh-jsonl)",
    )
    parser.add_argument("--title", help="Title for an imported worker history")
    parser.add_argument(
        "--reconfigure", action="store_true", help="Re-run backend setup"
    )
    parser.add_argument(
        "--daemon",
        choices=["install", "uninstall", "status"],
        help="Manage launchd daemon",
    )
    return parser


def main() -> None:
    _register_backends()

    legacy = _build_parser(direct_only=True)
    try:
        args, unknown = legacy.parse_known_args()
    except argparse.ArgumentError:
        args, unknown = None, []
    # Keep the legacy namespace for direct runs (even query == [""]). The
    # full parser reports errors, but must not reinterpret --c or task text.
    parser = _build_parser()
    if isinstance(parser, _InteractiveParser) and isinstance(legacy, _DirectParser):
        parser.validation_error = legacy.validation_error
    if args is None or unknown or not args.query or args.serve or args.daemon:
        args = parser.parse_args()
    if args.inbox is not None and args.brief is None:
        parser.error("--inbox requires --brief")
    if args.brief is not None:
        if (
            args.query
            or args.serve
            or args.daemon
            or args.reconfigure
            or args.import_session
        ):
            parser.error("--brief cannot be combined with a task or another mode")
        if args.output_format not in (None, "json", "stream-json"):
            parser.error("--brief requires json or stream-json output")
        from localmelo.support.cli.worker_contract import read_brief

        try:
            args.query = [read_brief(args.brief)]
        except ValueError as exc:
            parser.error(f"--brief: {_one_line(str(exc))}")
        if args.output_format is None:
            args.output_format = "json"
    elif args.output_format == "stream-json":
        parser.error("--output-format stream-json requires --brief")
    _check_args(parser, args)

    if getattr(args, "gateway_login", False) or getattr(
        args, "gateway_rotate_token", False
    ):
        if (
            args.query
            or args.serve
            or args.daemon
            or (
                getattr(args, "gateway_login", False)
                and getattr(args, "gateway_rotate_token", False)
            )
        ):
            parser.error(
                "gateway authentication commands cannot be combined with a task or --serve"
            )
        from localmelo.support import config
        from localmelo.support.gateway.auth import GatewayAuth

        cfg = config.load()
        auth_port = args.port or cfg.gateway.port
        host = args.host or cfg.gateway.host
        if host in ("::1", "[::1]"):
            host = "[::1]"
        elif host.lower() in ("localhost", "localhost."):
            host = "localhost"
        elif host != "127.0.0.1":
            parser.error("gateway authentication requires a supported loopback host")
        if getattr(args, "gateway_login", False) and not sys.stdout.isatty():
            parser.exit(2, "melo: sign-in links are printed only to a terminal.\n")
        auth = GatewayAuth(allow_corrupt=getattr(args, "gateway_rotate_token", False))
        try:
            if getattr(args, "gateway_login", False):
                login_code = auth.issue_login(auth_port)
                print(
                    f"Open http://{host}:{auth_port}/#login={login_code} within 60 seconds. Do not share this link."
                )
            else:
                auth.rotate()
                print("Gateway token rotated; browser sessions were signed out.")
        finally:
            auth.close()
        return

    if args.import_session is not None:
        from localmelo.support.sessions.dsh_import import import_dsh

        try:
            imported = asyncio.run(import_dsh(args.import_session[1], title=args.title))
        except (OSError, ValueError) as exc:
            parser.exit(
                EXIT_NOT_COMPLETED,
                f"{parser.prog}: import failed: {_one_line(str(exc))}\n",
            )
        print(
            f"Worker session {imported.session_id}: {'imported' if imported.imported else 'already imported'}; {imported.skipped} records skipped"
        )
        return

    # daemon commands
    if args.daemon:
        from localmelo.support.gateway import daemon

        if args.daemon == "install":
            kwargs: dict = {}
            if args.port is not None:
                kwargs["port"] = args.port
            path = daemon.install(**kwargs)
            print(f"Daemon installed: {path}")
        elif args.daemon == "uninstall":
            if daemon.uninstall():
                print("Daemon uninstalled")
            else:
                print("Daemon not installed")
        elif args.daemon == "status":
            info = daemon.status()
            for k, v in info.items():
                print(f"  {k}: {v}")
        return

    # Checked for direct mode, the `melo ""` loop and the gateway alike,
    # whether or not memory is kept, so a bad value is one line, not a
    # traceback.
    from localmelo.melo.memory._sqlite import thinking_retention_days

    try:
        thinking_retention_days()
    except ValueError as exc:
        parser.exit(EXIT_NOT_COMPLETED, f"{parser.prog}: error: {exc}\n")

    # direct mode (no gateway): exit code 0 completed, 1 crashed, 2 otherwise
    if args.query and not args.serve:
        query = " ".join(args.query)
        try:
            options = _direct_options(parser, args)
            # The surface is known here: direct mode has no approver, so its
            # default set leaves shell_exec out (--builtins all names it).
            options = dataclasses.replace(options, mode="direct")
            if not query:
                # The loop keeps its answers, so each turn sees the earlier ones; a
                # single task's options stay as they were.
                options = dataclasses.replace(options, record_answers=True)
            tool_providers = _direct_tool_providers(parser, args)
            agent = _build_direct_mode_agent(
                args, options=options, tool_providers=tool_providers
            )
        except (Exception, SystemExit) as exc:
            if args.brief is None or (isinstance(exc, SystemExit) and exc.code != 1):
                raise
            from localmelo.support.cli.worker_contract import failure_record

            error = (
                "Worker startup failed"
                if isinstance(exc, SystemExit)
                else _one_line(_home_relative(str(exc)))
            )
            workspace = os.path.realpath(
                os.path.expanduser(args.workspace or os.getcwd())
            )
            record = failure_record("startup_error", workspace, error)
            print(f"melo: worker startup failed: {record['error']}", file=sys.stderr)
            if args.output_format == "stream-json":
                record = {"v": 1, "type": "result", "result": record}
            try:
                print(
                    json.dumps(record, ensure_ascii=True, allow_nan=False), flush=True
                )
            except BrokenPipeError:
                print("melo: worker output failed: broken pipe", file=sys.stderr)
                _discard_stdout()
            raise SystemExit(EXIT_CRASHED) from None
        worker_args: dict[str, Any] = (
            {"worker_brief": True} if args.brief is not None else {}
        )
        if args.inbox is not None:
            worker_args["inbox_path"] = args.inbox
        code = _run_direct(
            agent,
            query,
            args.output_format or "text",
            tool_providers,
            progress=args.progress,
            routine=args.routine,
            **worker_args,
            **(
                {"save_session": args.save_session, "parent_run": args.parent_run}
                if args.save_session is not None
                else {}
            ),
        )
        if code != EXIT_COMPLETED:
            raise SystemExit(code)
        return

    # interactive mode: bare melo; --daemon has returned above. The UI's
    # libraries are imported here only, never by direct mode.
    if not args.serve:
        options = _direct_options(parser, args)
        # The UI asks before an exec call, so it keeps shell_exec.
        options = dataclasses.replace(options, mode="interactive")
        from localmelo.support import cli as ui

        # Without a terminal the UI refuses: no folder is made for it.
        if options.memory_dir is not None and ui.ui_terminal():
            _make_memory_dir(options.memory_dir)
        code = ui.run_interactive(
            args,
            parser=parser,
            options=options,
            validate_options=lambda resolved: _check_ui_options(parser, args, resolved),
        )
        if code != EXIT_COMPLETED:
            raise SystemExit(code)
        return

    # gateway mode (--serve): load or create config
    from localmelo.support import config
    from localmelo.support.onboard import run_backend_setup

    cfg = config.load()
    allow_unconfigured = bool(getattr(args, "allow_unconfigured", False))

    if args.reconfigure or (not cfg.is_configured and not allow_unconfigured):
        if not args.reconfigure and not _on_a_terminal():
            _exit_without_setup()
        with _setup_questions():
            setup_cfg = run_backend_setup()
        if setup_cfg is None:
            return
        cfg = setup_cfg

    # apply CLI overrides
    if args.host is not None:
        cfg.gateway.host = args.host
    if args.port is not None:
        cfg.gateway.port = args.port

    # validate before anything else — fail fast with actionable errors
    from localmelo.support.config import ConfigError

    if allow_unconfigured:
        if cfg.gateway.port <= 0 or cfg.gateway.port > 65535:
            print(
                "\n  Configuration errors:\n"
                f"  - [gateway] port={cfg.gateway.port} is out of range (1-65535)\n",
                file=sys.stderr,
            )
            raise SystemExit(1)
        cfg._allow_unconfigured = True
    else:
        try:
            cfg.validate_or_raise()
        except ConfigError as e:
            print(f"\n  {e}\n", file=sys.stderr)
            raise SystemExit(1) from e

    # start gateway
    if getattr(args, "enable_playground", False):
        cfg._enable_playground = True
    _start_gateway(cfg)


def _on_a_terminal() -> bool:
    """Whether stdin and stdout are a terminal, where melo can ask questions.

    A closed stdin or stdout (None) is not one. ``TERM=dumb`` is: the
    questions read plain lines. (The interactive UI needs more; see
    ``localmelo.support.cli.launch.ui_terminal``.)
    """
    return (
        sys.stdin is not None
        and sys.stdout is not None
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )


@contextlib.contextmanager
def _setup_questions() -> Iterator[None]:
    """End melo with ``Setup cancelled.`` on Ctrl-D (exit 0) or Ctrl+C (130)."""
    try:
        yield
    except EOFError:
        print("\nSetup cancelled.")
        raise SystemExit(EXIT_COMPLETED) from None
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        raise SystemExit(130) from None


def _exit_without_setup() -> NoReturn:
    """End melo with one line: no backend, and no terminal to ask (exit code 1).

    Under launchd stdin is /dev/null, so the wizard would end at once,
    again every few seconds. ``--reconfigure`` still runs the wizard
    without a terminal, reading the answers from stdin.
    """
    print(
        "melo: no backend is configured; run melo --reconfigure in a terminal",
        file=sys.stderr,
    )
    raise SystemExit(EXIT_CRASHED)


def _make_memory_dir(memory_dir: str) -> None:
    """Create --memory-dir, as the Agent would, or end melo with exit code 2.

    The Agent creates the directory too (with ``~`` expanded the same
    way, and mode 0700 when it is new), so this changes nothing for a path
    it can create. A path that
    cannot be a directory, such as an existing file, a path through a
    file, or a new directory where there is no permission to create one,
    is a flag value that is not valid: it is reported in one line before
    the run, as ``melo: error: ...``, rather than as the Agent's
    traceback.
    """
    from localmelo.melo.memory._sqlite import make_private_dir

    try:
        make_private_dir(os.path.expanduser(memory_dir))
    except OSError as exc:
        if isinstance(exc, FileExistsError):
            reason = "not a directory"
        else:
            reason = exc.strerror or _exception_text(exc)
        print(
            f"melo: error: argument --memory-dir: cannot use {memory_dir}: {reason}",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_NOT_COMPLETED) from None


def _direct_embedding(args: argparse.Namespace) -> BaseEmbeddingProvider | None:
    """The embedding provider --embedding-base-url names, or None (no embedding)."""
    base_url = getattr(args, "embedding_base_url", None)
    if not base_url:
        return None
    from localmelo.support.providers.embedding.openai_compat import (
        OpenAICompatEmbedding,
    )

    model = getattr(args, "embedding_model", None) or "default"
    return OpenAICompatEmbedding(base_url=base_url, model=model)


def _build_direct_mode_agent(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
    options: AgentOptions | None = None,
    tool_providers: Sequence[BaseToolProvider] = (),
) -> Agent:
    """Build an Agent for direct CLI mode from CLI flags.

    Two cases:
    1. --base-url provided: generic OpenAI-compatible endpoint injection
       (preserves the exact URL as given). There is no embedding unless
       --embedding-base-url names an OpenAI-compatible embedding endpoint.
       --request-timeout, when given, is the chat provider's HTTP timeout.
    2. No --base-url: use configured backends from the config file. The
       config is validated first, as for gateway startup, so a bad value
       (e.g. an invalid extra_body) is reported instead of a traceback.
       --request-timeout, when given, replaces the configured
       request_timeout_s for this run, and --chat-model the configured
       chat model; the file is not written.

    *options* and *tool_providers* go to the Agent in both cases. The
    memory directory that *options* name is created just before the Agent is
    built, after every refusal, so a refused run leaves no directory behind
    (see :func:`_make_memory_dir`).
    """
    if parser is None:
        parser = _build_parser()
    base_url = args.base_url
    request_timeout = getattr(args, "request_timeout", None)
    from localmelo.support import config
    from localmelo.support.sessions.agents import configured_context_options

    def config_value_error(message: str) -> NoReturn:
        """Refuse a bad config value as a usage error, or a worker result.

        With ``--brief`` a config-file value is a startup failure, not an
        argv misuse, so it becomes a ``startup_error`` record instead of
        ``parser.error``'s resultless exit 2.
        """
        if getattr(args, "brief", None) is None:
            parser.error(message)
        raise _StartupConfigError(message)

    try:
        budget_cfg = config.load()
    except (OSError, ValueError):
        if not base_url:
            raise
        budget_cfg = config.Config()
    try:
        options = configured_context_options(
            budget_cfg,
            options or AgentOptions(),
            {
                name: getattr(args, name, None)
                for name in ("context_window", "output_reserve", "safety_margin")
            },
        )
    except ValueError as exc:
        config_value_error(str(exc))
    if not isinstance(budget_cfg.agent.fixed_prefix, bool):
        config_value_error("[agent] fixed_prefix must be a boolean")
    if getattr(args, "fixed_prefix", None) is None and budget_cfg.agent.fixed_prefix:
        options = dataclasses.replace(options, fixed_prefix=True)

    if not isinstance(budget_cfg.agent.captured_history, bool):
        config_value_error("[agent] captured_history must be a boolean")
    if (
        getattr(args, "captured_history", None) is None
        and budget_cfg.agent.captured_history
    ):
        options = dataclasses.replace(options, captured_history=True)

    if base_url:
        # Generic OpenAI-compatible endpoint
        from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

        model = args.chat_model or "default"
        # Without --request-timeout the provider keeps its default (120 s);
        # httpx would read timeout=None as no timeout at all.
        llm_options: dict[str, Any] = (
            {} if request_timeout is None else {"timeout": request_timeout}
        )
        reasoning_field = getattr(args, "reasoning_field", None)
        mode = options.reasoning if options is not None else "legacy"
        if reasoning_field is not None and mode in ("replay", "compress+replay"):
            llm_options["reasoning_field"] = reasoning_field
        extra_body = getattr(args, "retrieval_extra_body", None)
        if extra_body is not None:
            llm_options["extra_body_by_kind"] = {"retrieve": extra_body}
        if options is not None:
            _check_effective_recall(parser, args, options)
            if options.retriever == "embedding" and not getattr(
                args, "embedding_base_url", None
            ):
                parser.error(
                    "argument --retriever: embedding needs an embedding provider "
                    "(--embedding-base-url with --base-url)"
                )
            if options.retriever == "embedding" and not options.memory:
                parser.error("argument --retriever: embedding needs --memory on")
        llm = OpenAICompatLLM(base_url=base_url, model=model, **llm_options)
        if options is not None and options.memory_dir is not None:
            _make_memory_dir(options.memory_dir)
        return Agent(
            llm=llm,
            embedding=_direct_embedding(args),
            options=options,
            tool_providers=tool_providers,
        )

    # No --base-url: use configured backends from config file
    from localmelo.support import config

    cfg = budget_cfg
    for name in ("context_window", "output_reserve", "safety_margin"):
        setattr(cfg.agent, name, getattr(options, name))
    if not cfg.is_configured:
        if getattr(args, "brief", None) is not None:
            raise _StartupConfigError(
                "No backend configured. Run 'melo --serve --reconfigure' to set up, "
                "or use --base-url to connect to an OpenAI-compatible endpoint."
            )
        print(
            "No backend configured. Run 'melo --serve --reconfigure' to set up,\n"
            "or use --base-url to connect to an OpenAI-compatible endpoint.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if request_timeout is not None:
        # For this run only, like the gateway's --host and --port: the
        # loaded config is changed in memory and never saved.
        cfg.request_timeout_s = request_timeout

    if args.chat_model:
        section = getattr(cfg, cfg.chat_backend, None)
        if section is not None:
            section.chat_model = args.chat_model

    try:
        cfg.validate_or_raise()
    except config.ConfigError as e:
        if getattr(args, "brief", None) is not None:
            raise _StartupConfigError(str(e)) from e
        print(f"\n  {e}\n", file=sys.stderr)
        raise SystemExit(1) from e

    section = getattr(cfg, cfg.chat_backend, None)
    if section is None:
        parser.error(f"chat_backend {cfg.chat_backend!r}: no registered adapter found")
    reasoning_field = getattr(args, "reasoning_field", None)
    if reasoning_field is not None and cfg.chat_backend == "ollama":
        parser.error(
            "argument --reasoning-field: the ollama backend always reads "
            "and sends the thinking field"
        )

    if options is None:
        options = AgentOptions()
    if inherited := cfg.tool_environment():
        options = dataclasses.replace(
            options, tool_env={**inherited, **(options.tool_env or {})}
        )
    if getattr(args, "reasoning", None) is None and cfg.agent.reasoning != "legacy":
        options = dataclasses.replace(
            options, reasoning=cast(ReasoningMode, cfg.agent.reasoning)
        )
    if (
        getattr(args, "replay_turns", None) is None
        and cfg.agent.replay_turns is not None
    ):
        options = dataclasses.replace(options, replay_turns=cfg.agent.replay_turns)
    if getattr(args, "note_writer", None) is None and cfg.agent.note_writer != "auto":
        options = dataclasses.replace(
            options, note_writer=cast(NoteWriter, cfg.agent.note_writer)
        )
    if getattr(args, "admit_on_outcome", None) is None and cfg.memory.admit_on_outcome:
        options = dataclasses.replace(options, admit_on_outcome=True)

    # R5's [retrieval]: a value applies only when its flag is absent, and
    # the table is inert in legacy (decision 10).
    if options.reasoning != "legacy":
        retrieval = cfg.retrieval
        if getattr(args, "retriever", None) is None and retrieval.retriever != "auto":
            options = dataclasses.replace(
                options, retriever=cast(Any, retrieval.retriever)
            )
        if getattr(args, "recall_at", None) is None and retrieval.recall_at:
            options = dataclasses.replace(
                options, recall_at=frozenset(retrieval.recall_at)
            )
        if getattr(args, "recall_k", None) is None and retrieval.k != 3:
            options = dataclasses.replace(options, recall_k=retrieval.k)
        if getattr(args, "recall_kinds", None) is None and retrieval.kinds:
            options = dataclasses.replace(
                options, recall_kinds=frozenset(retrieval.kinds)
            )
        if (
            getattr(args, "recall_placement", None) is None
            and retrieval.placement != "tool"
        ):
            options = dataclasses.replace(
                options, recall_placement=cast(Any, retrieval.placement)
            )
        if (
            getattr(args, "recall_min_score", None) is None
            and retrieval.min_score != 0.35
        ):
            options = dataclasses.replace(options, recall_min_score=retrieval.min_score)
    elif _recall_flag_given(args):
        parser.error(
            "argument --retriever/--recall-*/--retrieval-*: retrieval is not "
            "available in legacy; use --reasoning compress, compress+replay, "
            "drop or replay"
        )

    if options.retriever == "embedding":
        if not cfg.has_embedding:
            parser.error(
                "argument --retriever: embedding needs an embedding provider "
                "in the config"
            )
        if not options.memory:
            parser.error("argument --retriever: embedding needs --memory on")
    requires_replay = bool(getattr(section, "requires_reasoning_replay", False))
    if options.reasoning == "compress" and requires_replay:
        # A provider that requires replay runs compress as compress+replay;
        # resolve it here too, so the field and the refusal agree.
        options = dataclasses.replace(
            options,
            reasoning="compress+replay",
            replay_turns=None,
            replay_scope="task",
        )
    # After the conversion above: a recall placement that the effective
    # mode refuses must be a usage error, not a traceback from Agent.
    _check_effective_recall(parser, args, options)

    # The field only has meaning in replay; legacy keeps the provider's own
    # behavior (OpenAI-compatible sends none) even if the file names one.
    if options.reasoning in ("replay", "compress+replay"):
        if reasoning_field is not None:
            section.reasoning_field = reasoning_field
    else:
        section.reasoning_field = ""

    field = getattr(section, "reasoning_field", "") or (
        "thinking" if cfg.chat_backend == "ollama" else None
    )
    if options.reasoning in ("replay", "compress+replay") and not field:
        parser.error(
            "argument --reasoning: replay and compress+replay need a reasoning "
            "field: set reasoning_field in the chat backend's config section"
        )
    refusal = reasoning_refusal(options.reasoning, field, requires_replay)
    if refusal is not None:
        parser.error(f"argument --reasoning: {refusal}")

    if options.memory_dir is not None:
        _make_memory_dir(options.memory_dir)
    agent = Agent(config=cfg, options=options, tool_providers=tool_providers)
    extra_body = getattr(args, "retrieval_extra_body", None)
    if extra_body is not None:
        _set_retrieval_extra_body(parser, agent, extra_body)
    return agent


def _set_retrieval_extra_body(
    parser: argparse.ArgumentParser, agent: Agent, body: dict[str, Any]
) -> None:
    """Put *body* on the agent's provider for retrieve requests only."""
    from localmelo.support.providers.llm._protocol import prepare_extra_body

    provider = getattr(agent, "_llm", None)
    if provider is None or not hasattr(provider, "extra_body_by_kind"):
        parser.error(
            "argument --retrieval-extra-body: this backend does not support "
            "per-kind extra body fields"
        )
    provider.extra_body_by_kind = {"retrieve": prepare_extra_body(body)}


def _start_gateway(cfg: Config) -> None:
    """Thin seam for gateway startup — keeps the lazy import local."""
    from localmelo.support.gateway import start_gateway

    host = cfg.gateway.host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # an IPv6 address, which a URL writes in brackets
    print("\n  Starting localmelo gateway ...")
    print(f"  Chat: {cfg.chat_backend}")
    print(f"  Embedding: {cfg.embedding_backend}")
    print(f"  Gateway: http://{host}:{cfg.gateway.port}")
    print(f"  Web UI:  http://{host}:{cfg.gateway.port}/")
    print()

    if getattr(cfg, "_allow_unconfigured", False):
        start_gateway(cfg, allow_unconfigured=True)
    else:
        start_gateway(cfg)


if __name__ == "__main__":
    main()
