from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, TextIO, cast

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent.options import ReasoningMode, reasoning_refusal
from localmelo.melo.executor import builtin_names
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

# Flags that apply to direct mode only (a query without --serve), by dest.
_DIRECT_FLAGS = {
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
}

# Flags of the interactive UI (bare melo), by dest.
_UI_FLAGS = {"workspace": "--workspace", "yolo": "--yolo"}

# Direct-mode flags the interactive UI refuses: its Agents offer every tool,
# an approval's wait would count toward a timeout, it has no machine output,
# it draws its own progress, and it does not wire the reasoning field yet
# (which has its own message; R2b wires it).
_UI_REFUSED = (
    "tool_exposure",
    "timeout",
    "output_format",
    "progress",
    "routine",
    "reasoning_field",
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
            outcome = await _run_one(agent, query, progress=progress, routine=routine)
            if exit_signals is not None and exit_signals.received is not None:
                return EXIT_CRASHED  # not used: the process exits by the signal
            _report(outcome, output_format, version, out)
            return _exit_code(outcome.status)
        _enable_line_editing()
        print("localmelo · local agent")
        print("Enter a task to begin. Type /help for commands or /exit to quit.")
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
    try:
        output = (
            _stdout_for_the_report() if query else contextlib.nullcontext(sys.stdout)
        )
        with output as out:
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
                )
            )
    except BaseException as exc:
        if line is not None:
            line.close()  # before a traceback or the line below
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


def _parse_builtins(
    parser: argparse.ArgumentParser, text: str
) -> frozenset[str] | None:
    """``all`` (None: every built-in), ``none`` or comma-separated names."""
    if text == "all":
        return None
    if text == "none":
        return frozenset()
    names = [name.strip() for name in text.split(",")]
    if not all(names):
        parser.error(
            "argument --builtins: expected all, none or built-in tool names "
            f"separated by commas, got {text!r}"
        )
    known = builtin_names()
    unknown = sorted(set(names) - known)
    if unknown:
        parser.error(
            f"argument --builtins: unknown built-in tools {', '.join(unknown)}; "
            f"known: {', '.join(sorted(known))}"
        )
    return frozenset(names)


def _check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject flag combinations before anything runs (argparse exit code 2)."""
    if args.mcp_only and args.mcp_config is None:
        parser.error("--mcp-only requires --mcp-config")
    if args.mcp_only:
        for dest in ("builtins", "tool_exposure"):
            if getattr(args, dest) is not None:
                parser.error(f"--mcp-only sets {_DIRECT_FLAGS[dest]}; do not give both")
    if args.embedding_model is not None and args.embedding_base_url is None:
        parser.error("--embedding-model requires --embedding-base-url")
    if args.embedding_base_url is not None and not args.base_url:
        parser.error("--embedding-base-url requires --base-url")
    reasoning = getattr(args, "reasoning", None)
    reasoning_field = getattr(args, "reasoning_field", None)
    if (
        args.base_url
        and not (args.serve or args.daemon)
        and reasoning == "replay"
        and reasoning_field is None
    ):
        parser.error(
            "argument --reasoning: replay needs a reasoning field: give "
            "--reasoning-field (auto, reasoning_content or reasoning) with "
            "--base-url, or set reasoning_field in the backend's config"
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
    # Bare melo, which opens the interactive UI; `melo ""` is direct mode.
    interactive = not args.query and not args.serve and not args.daemon
    ui_given = [
        flag
        for dest, flag in _UI_FLAGS.items()
        if getattr(args, dest) not in (None, False)
    ]
    if ui_given and not interactive:
        parser.error(
            "only for the interactive UI (melo without a query, --serve or --daemon): "
            + ", ".join(ui_given)
        )
    if interactive and (args.host is not None or args.port is not None):
        parser.error(
            "--host and --port are for --serve and --daemon; "
            "bare melo opens the interactive UI"
        )
    if interactive and args.reasoning_field is not None:
        parser.error(
            "--reasoning-field is not in the interactive UI yet; "
            'give it with a task (melo -- "<task>")'
        )
    # main() joins the words into the query, and an empty one (`melo ""`)
    # runs no task, so the direct-mode flags would have nothing to apply to.
    direct = bool(" ".join(args.query)) and not args.serve
    if not direct:
        given = [
            flag
            for dest, flag in _DIRECT_FLAGS.items()
            if getattr(args, dest) not in (None, False)
            and (not interactive or dest in _UI_REFUSED)
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
    reasoning = getattr(args, "reasoning", None)
    if reasoning is not None:
        fields["reasoning"] = reasoning
    replay_turns = getattr(args, "replay_turns", None)
    if replay_turns is not None:
        fields["replay_turns"] = replay_turns
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="melo", description="localmelo agent")
    parser.add_argument("query", nargs="*", help="Task to execute (direct mode)")
    parser.add_argument("--base-url", default=None, help="LLM API base URL")
    parser.add_argument("--chat-model", default=None, help="Chat model name")

    direct = parser.add_argument_group(
        "direct mode", "Options for a query run without --serve."
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
        "--tool-exposure",
        choices=["retrieve", "all"],
        help="Offer every built-in plus the extra tools retrieved for each step "
        "(default), or all tools",
    )
    direct.add_argument(
        "--builtins",
        metavar="all|none|NAME,...",
        help="Built-in tools to register (default all)",
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
        choices=["text", "json"],
        help="text (default), or one JSON line with the run's record",
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
        choices=["legacy", "drop", "replay"],
        help="How a reply's reasoning is sent back: legacy (default, the "
        "provider's own behavior), drop (never), or replay (on assistant "
        "turns in scope, under one key)",
    )
    direct.add_argument(
        "--reasoning-field",
        choices=["auto", "reasoning_content", "reasoning"],
        help="The request key replay uses (with --base-url, or a backend's "
        "reasoning_field); auto follows the key the last reply came from",
    )
    direct.add_argument(
        "--replay-turns",
        type=_positive_int,
        metavar="N",
        help="With replay, keep the newest N assistant turns' reasoning "
        "(default: every turn in scope)",
    )

    interactive = parser.add_argument_group(
        "interactive mode",
        "Options for bare melo, which opens the interactive UI on a terminal. "
        "It also takes --base-url, --chat-model, --max-steps, --attempt-policy, "
        "--reflection-gate, --builtins, --memory, --memory-dir, "
        "--request-timeout, --reasoning, --replay-turns "
        "and the --embedding-* options. It refuses --reasoning-field until a "
        "later release wires it.",
    )
    interactive.add_argument(
        "--workspace",
        metavar="DIR",
        help="The folder melo works in (default: the current directory)",
    )
    interactive.add_argument(
        "--yolo",
        action="store_true",
        help="Run every tool without asking for approval",
    )

    # gateway mode
    parser.add_argument("--serve", action="store_true", help="Run as gateway server")
    parser.add_argument("--host", default=None, help="Gateway listen host")
    parser.add_argument("--port", type=int, default=None, help="Gateway listen port")

    # setup
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

    parser = _build_parser()
    args = parser.parse_args()
    _check_args(parser, args)

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
        options = _direct_options(parser, args)
        if not query:
            # The loop keeps its answers, so each turn sees the earlier ones; a
            # single task's options stay as they were.
            options = dataclasses.replace(options, record_answers=True)
        tool_providers = _direct_tool_providers(parser, args)
        agent = _build_direct_mode_agent(
            args, options=options, tool_providers=tool_providers
        )
        code = _run_direct(
            agent,
            query,
            args.output_format or "text",
            tool_providers,
            progress=args.progress,
            routine=args.routine,
        )
        if code != EXIT_COMPLETED:
            raise SystemExit(code)
        return

    # interactive mode: bare melo; --daemon has returned above. The UI's
    # libraries are imported here only, never by direct mode.
    if not args.serve:
        options = _direct_options(parser, args)
        from localmelo.support import cli as ui

        # Without a terminal the UI refuses: no folder is made for it.
        if options.memory_dir is not None and ui.ui_terminal():
            _make_memory_dir(options.memory_dir)
        code = ui.run_interactive(args, parser=parser, options=options)
        if code != EXIT_COMPLETED:
            raise SystemExit(code)
        return

    # gateway mode (--serve): load or create config
    from localmelo.support import config
    from localmelo.support.onboard import run_backend_setup

    cfg = config.load()

    if args.reconfigure or not cfg.is_configured:
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

    try:
        cfg.validate_or_raise()
    except ConfigError as e:
        print(f"\n  {e}\n", file=sys.stderr)
        raise SystemExit(1) from e

    # start gateway
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
        if reasoning_field is not None and mode == "replay":
            llm_options["reasoning_field"] = reasoning_field
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

    cfg = config.load()
    if not cfg.is_configured:
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
    if getattr(args, "reasoning", None) is None and cfg.agent.reasoning != "legacy":
        options = dataclasses.replace(
            options, reasoning=cast(ReasoningMode, cfg.agent.reasoning)
        )
    if (
        getattr(args, "replay_turns", None) is None
        and cfg.agent.replay_turns is not None
    ):
        options = dataclasses.replace(options, replay_turns=cfg.agent.replay_turns)

    # The field only has meaning in replay; legacy keeps the provider's own
    # behavior (OpenAI-compatible sends none) even if the file names one.
    if options.reasoning == "replay":
        if reasoning_field is not None:
            section.reasoning_field = reasoning_field
    else:
        section.reasoning_field = ""

    field = getattr(section, "reasoning_field", "") or (
        "thinking" if cfg.chat_backend == "ollama" else None
    )
    requires_replay = bool(getattr(section, "requires_reasoning_replay", False))
    if options.reasoning == "replay" and not field:
        parser.error(
            "argument --reasoning: replay needs a reasoning field: set "
            "reasoning_field in the chat backend's config section"
        )
    refusal = reasoning_refusal(options.reasoning, field, requires_replay)
    if refusal is not None:
        parser.error(f"argument --reasoning: {refusal}")

    if options.memory_dir is not None:
        _make_memory_dir(options.memory_dir)
    return Agent(config=cfg, options=options, tool_providers=tool_providers)


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

    start_gateway(cfg)


if __name__ == "__main__":
    main()
