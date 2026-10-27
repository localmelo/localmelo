"""--progress and --routine in direct mode, in process.

stderr is replaced by a stream that says it is a terminal, so the line is
drawn without a pseudo-terminal; tests/cli/test_progress_process.py runs
the real CLI on one. The agent is a mock, except where a test says so.
"""

from __future__ import annotations

import asyncio
import io
import logging
import signal
import sys
from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.schema import RunResult
from localmelo.support.progress_line import DIM, ERASE
from localmelo.tests.agent.procedural_helpers import (
    ScriptedLLM,
    fix_run,
    make_agent,
    passing_tools,
)
from localmelo.tests.cli.test_direct_mode import (
    _agent,
    _ExitedBySignalError,
    _main,
    _outcome,
    _raise_exited,
)

ONLY_DIRECT = "only for direct mode (a query without --serve): "


class TtyStream(io.StringIO):
    """A stderr that says it is a terminal (of unknown width: 80 columns)."""

    def isatty(self) -> bool:
        return True

    def installed(self) -> Any:
        """A context in which this stream is sys.stderr.

        Within the test itself: pytest's capture replaces sys.stderr as
        the test starts, after its fixtures.
        """
        return mock.patch.object(sys, "stderr", self)


@pytest.fixture
def tty(monkeypatch: pytest.MonkeyPatch) -> Iterator[TtyStream]:
    """A terminal for stderr (see TtyStream.installed), TERM set, NO_COLOR on."""
    stream = TtyStream()
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("NO_COLOR", "1")
    root_handlers = list(logging.getLogger().handlers)
    yield stream
    # No handler is left behind on the root logger.
    assert logging.getLogger().handlers == root_handlers


def _main_on(stream: TtyStream, *argv: str, **kwargs: Any) -> Any:
    """_main with *stream* as stderr."""
    with stream.installed():
        return _main(*argv, **kwargs)


# ── Flags ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--progress", "off", ""], ONLY_DIRECT + "--progress"),
        (["--progress", "auto", ""], ONLY_DIRECT + "--progress"),
        (["--routine", "daily", ""], ONLY_DIRECT + "--routine"),
        (["--serve", "--progress", "auto", "q"], ONLY_DIRECT + "--progress"),
        (["--serve", "--routine", "daily"], ONLY_DIRECT + "--routine"),
        (
            ["--serve", "--routine", "daily", "--progress", "off", "q"],
            ONLY_DIRECT + "--progress, --routine",
        ),
        (["--routine", "  ", "q"], "argument --routine: must not be blank"),
        (["--routine", "", "q"], "argument --routine: must not be blank"),
        (["--progress", "on", "q"], "argument --progress: invalid choice: 'on'"),
    ],
)
def test_rejected_before_the_run(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    ran = _main(*argv)
    assert ran.code == 2
    ran.build.assert_not_called()
    out, err = capsys.readouterr()
    assert out == ""
    assert f"melo: error: {message}" in err


def test_the_routine_reaches_the_run() -> None:
    ran = _main("--routine", "daily", "q")
    assert ran.code == 0
    ran.agent.run_task.assert_awaited_once_with("q", routine="daily")
    assert ran.options == AgentOptions()


@pytest.mark.parametrize("argv", [[], ["--progress", "auto"], ["--progress", "off"]])
def test_without_a_terminal_the_run_starts_as_before(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    ran = _main(*argv, "q")
    ran.agent.run_task.assert_awaited_once_with("q")
    assert capsys.readouterr() == ("42\n", "")


def test_the_help_names_both_flags(capsys: pytest.CaptureFixture[str]) -> None:
    assert _main("--help").code == 0
    out = capsys.readouterr().out
    assert "--progress {auto,off}" in out
    assert "--routine NAME" in out


# ── On a terminal ───────────────────────────────────────────────────────────


def test_text_output_on_a_terminal_draws_the_line(
    tty: TtyStream, capsys: pytest.CaptureFixture[str]
) -> None:
    ran = _main_on(tty, "q")
    assert ran.code == 0
    call = ran.agent.run_task.await_args
    assert call.args == ("q",)
    assert set(call.kwargs) == {"observe", "progress"}
    assert call.kwargs["progress"] is True
    assert callable(call.kwargs["observe"])
    # Drawn, then erased; the result alone on stdout.
    assert tty.getvalue() == ERASE + "starting · 0s" + ERASE
    assert capsys.readouterr().out == "42\n"


def test_the_routine_with_the_line(tty: TtyStream) -> None:
    ran = _main_on(tty, "--routine", "daily-mail", "q")
    call = ran.agent.run_task.await_args
    assert call.args == ("q",)
    assert set(call.kwargs) == {"observe", "progress", "routine"}
    assert call.kwargs["routine"] == "daily-mail"


@pytest.mark.parametrize(
    "argv", [["--output-format", "json"], ["--progress", "off"]], ids=["json", "off"]
)
def test_no_line_and_no_observer(
    tty: TtyStream, argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    ran = _main_on(tty, *argv, "q")
    ran.agent.run_task.assert_awaited_once_with("q")
    assert "\x1b" not in tty.getvalue()


def test_no_line_on_a_dumb_terminal(
    tty: TtyStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "dumb")
    _main_on(tty, "q").agent.run_task.assert_awaited_once_with("q")
    assert tty.getvalue() == ""


def test_dim_unless_no_color(tty: TtyStream, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR")
    _main_on(tty, "q")
    assert tty.getvalue().startswith(ERASE + DIM + "starting · 0s")


def test_the_line_is_erased_before_the_status_line(
    tty: TtyStream, capsys: pytest.CaptureFixture[str]
) -> None:
    outcome = _outcome("budget_exhausted", result="best effort", error="Max steps")
    assert _main_on(tty, "q", outcome=outcome).code == 2
    assert tty.getvalue() == (
        ERASE
        + "starting · 0s"
        + ERASE
        + "melo: run ended with status budget_exhausted: Max steps\n"
    )
    assert capsys.readouterr().out == "best effort\n"


def test_a_real_run_on_a_terminal(
    tty: TtyStream, capsys: pytest.CaptureFixture[str]
) -> None:
    """An Agent with scripted replies: the line follows its events."""

    def build(args: Any, *, options: AgentOptions, tool_providers: Any) -> Agent:
        return make_agent(ScriptedLLM(fix_run(2)), tools=passing_tools())

    with (
        mock.patch.object(cli, "_register_backends"),
        mock.patch.object(cli, "_build_direct_mode_agent", build),
        mock.patch("sys.argv", ["melo", "fix the tests"]),
        tty.installed(),
    ):
        cli.main()  # completed: no SystemExit
    text = tty.getvalue()
    draws = [part for part in text.split(ERASE) if part]
    assert draws[0] == "starting · 0s"
    assert "step 1 · Read test_parser.py 0:00 · 0s" in draws
    assert "step 3 · Verify (cycle 1) · tests 7/8 · 0s" in draws
    assert "step 5 · Finishing (cycle 2) · tests 8/8 · 0s" in draws
    assert "\n" not in text
    assert text.endswith(ERASE)
    assert capsys.readouterr().out == "Fixed.\n"


def test_ctrl_c_erases_the_line_before_the_traceback(tty: TtyStream) -> None:
    agent = _agent()
    agent.run_task.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt), tty.installed():
        cli._run_direct(agent, "q", "text")
    assert tty.getvalue() == ERASE + "starting · 0s" + ERASE


def test_an_exception_erases_the_line(tty: TtyStream) -> None:
    agent = _agent()
    agent.close.side_effect = RuntimeError("cannot close")
    with pytest.raises(RuntimeError), tty.installed():
        cli._run_direct(agent, "q", "text")
    assert tty.getvalue().endswith(ERASE)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_erases_the_line_before_the_clean_up(
    tty: TtyStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_exit_by_signal", _raise_exited)
    agent = _agent()
    seen: list[str] = []
    agent.begin_exit.side_effect = lambda *a, **k: seen.append(tty.getvalue())

    async def run_task(query: str, **kwargs: Any) -> RunResult:
        signal.raise_signal(signal.SIGTERM)
        await asyncio.Event().wait()
        raise AssertionError("the run was not cancelled")

    agent.run_task.side_effect = run_task
    with pytest.raises(_ExitedBySignalError), tty.installed():
        cli._run_direct(agent, "q", "text")
    assert seen == [ERASE + "starting · 0s" + ERASE]
    assert tty.getvalue() == ERASE + "starting · 0s" + ERASE


# ── _ExitSignals(before_exit=...) ───────────────────────────────────────────


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_before_exit_runs_on_the_signal_and_before_the_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(cli, "_exit_by_signal", lambda signum: order.append("exit"))
    monkeypatch.setattr(cli, "SIGNAL_CLEANUP_S", 0.01)
    agent = _agent()
    agent.begin_exit.side_effect = lambda *a, **k: order.append("begin_exit")
    signals = cli._ExitSignals(agent, before_exit=lambda: order.append("erase"))

    async def scenario() -> None:
        signals._receive(signal.SIGTERM, None)
        signals._receive(signal.SIGHUP, None)  # changes nothing
        await asyncio.sleep(0.2)

    asyncio.run(scenario())
    assert order == ["erase", "begin_exit", "erase", "exit"]


def test_a_before_exit_that_fails_does_not_stop_the_exit() -> None:
    agent = _agent()
    failing = mock.Mock(side_effect=OSError(5, "Input/output error"))
    signals = cli._ExitSignals(agent, before_exit=failing)
    with mock.patch("asyncio.get_running_loop"):
        signals._receive(signal.SIGTERM, None)
    failing.assert_called_once_with()
    agent.begin_exit.assert_called_once()
