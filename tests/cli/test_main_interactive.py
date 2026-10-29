"""melo without a task: the refusals, and what needs a terminal.

Which model server the interactive UI uses is tests/cli/test_launch.py's
(the `melo ""` loop's rules of U2.0, moved there). The subprocess tests
need no model server at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.melo.schema import RunResult
from localmelo.support.config import Config, LocalBackendConfig

URL = "http://127.0.0.1:8500/v1"


def _agent() -> mock.MagicMock:
    agent = mock.MagicMock()
    agent.run_task = mock.AsyncMock(
        return_value=RunResult(task_id="turn", status="completed", result="42")
    )
    agent.close = mock.AsyncMock()
    return agent


def _configured() -> Config:
    return Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(chat_url=URL, chat_model="configured"),
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--serve", "--base-url", URL],
        ["--serve", "--chat-model", "M"],
        ["--daemon", "status", "--base-url", URL],
        ["--daemon", "install", "--chat-model", "M"],
    ],
)
def test_serve_and_daemon_refuse_a_base_url_or_a_chat_model(
    capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    with (
        mock.patch("sys.argv", ["melo", *argv]),
        mock.patch.object(cli, "_start_gateway") as gateway,
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    gateway.assert_not_called()
    assert capsys.readouterr().err.endswith(
        "melo: error: --base-url and --chat-model are for a task or the interactive "
        "UI; --serve uses the config file (melo --reconfigure)\n"
    )


@pytest.mark.parametrize("argv", [["--port", "9000"], ["--host", "0.0.0.0"]])
def test_host_and_port_are_refused_where_melo_opens_the_ui(
    capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    with (
        mock.patch("sys.argv", ["melo", *argv]),
        mock.patch.object(cli, "_on_a_terminal", return_value=True),
        mock.patch.object(cli, "_start_gateway") as gateway,
        mock.patch.object(cli, "_build_direct_mode_agent") as build,
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    gateway.assert_not_called()
    build.assert_not_called()
    assert capsys.readouterr().err.endswith(
        "melo: error: --host and --port are for --serve and --daemon; "
        "bare melo opens the interactive UI\n"
    )


def test_an_empty_task_stays_direct_mode_without_a_model_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`melo ""` is unchanged: no listing, no setup, no Model line."""
    with (
        mock.patch("sys.argv", ["melo", "--base-url", URL, ""]),
        mock.patch.object(cli, "_on_a_terminal", return_value=True),
        mock.patch("localmelo.support.backends.discovery.list_models") as listing,
        mock.patch.object(
            cli, "_build_direct_mode_agent", return_value=_agent()
        ) as build,
        mock.patch("builtins.input", side_effect=["/exit"]),
    ):
        cli.main()
    listing.assert_not_called()
    assert build.call_args.args[0].chat_model is None
    assert capsys.readouterr().out.startswith("localmelo · local agent\n")


def test_serve_with_no_config_and_no_terminal_exits_with_one_line(
    tmp_path: Path,
) -> None:
    """Under launchd stdin is /dev/null: one line, exit 1, no wizard, no traceback."""
    env = {**os.environ, "HOME": str(tmp_path)}
    env.pop("LOCALMELO_DETECT_URLS", None)
    result = subprocess.run(
        [sys.executable, "-m", "localmelo", "--serve"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 1
    assert result.stderr == (
        "melo: no backend is configured; run melo --reconfigure in a terminal\n"
    )
    assert "Welcome" not in result.stdout
    assert not (tmp_path / ".cache" / "localmelo" / "config.toml").exists()


# ── Review fixes ──


def test_off_a_terminal_a_base_url_never_reads_the_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without a terminal the UI cannot open: exit 2 before the config is read."""
    with (
        mock.patch("sys.argv", ["melo", "--base-url", URL, "--chat-model", "M"]),
        mock.patch("localmelo.support.cli.launch.ui_terminal", return_value=False),
        mock.patch("localmelo.support.config.load", side_effect=AssertionError("read")),
        mock.patch("localmelo.support.backends.discovery.list_models") as listing,
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    listing.assert_not_called()
    assert "the interactive UI needs a terminal" in capsys.readouterr().err


def test_reconfigure_with_a_base_url_and_no_task_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        mock.patch("sys.argv", ["melo", "--reconfigure", "--base-url", URL]),
        mock.patch("localmelo.support.onboard.run_backend_setup") as setup,
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    setup.assert_not_called()
    assert capsys.readouterr().err.endswith(
        "melo: error: --reconfigure sets up the config file, which --base-url "
        "does not use; give one of them\n"
    )


def test_reconfigure_with_a_base_url_and_an_empty_task_stays_direct_mode() -> None:
    with (
        mock.patch("sys.argv", ["melo", "--reconfigure", "--base-url", URL, ""]),
        mock.patch("localmelo.support.onboard.run_backend_setup") as setup,
        mock.patch.object(cli, "_build_direct_mode_agent", return_value=_agent()),
        mock.patch("builtins.input", side_effect=["/exit"]),
    ):
        cli.main()
    setup.assert_not_called()


def test_reconfigure_off_a_terminal_runs_the_wizard_then_exits() -> None:
    with (
        mock.patch("sys.argv", ["melo", "--reconfigure", "--chat-model", "M"]),
        mock.patch("localmelo.support.cli.launch.ui_terminal", return_value=False),
        mock.patch(
            "localmelo.support.onboard.run_backend_setup", return_value=_configured()
        ) as setup,
        mock.patch.object(cli, "_start_gateway") as gateway,
        mock.patch.object(cli, "_build_direct_mode_agent") as build,
    ):
        cli.main()
    setup.assert_called_once_with(guided=True)
    gateway.assert_not_called()
    build.assert_not_called()


class _Stream:
    def __init__(self, tty: bool) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


@pytest.mark.parametrize(
    ("stdin", "stdout", "term", "expected"),
    [
        (True, True, "xterm-256color", True),
        (True, True, "dumb", True),  # plain lines work on a dumb terminal
        (True, False, "xterm-256color", False),  # `melo > log`
        (False, True, "xterm-256color", False),  # `melo < /dev/null`
        (None, True, "xterm-256color", False),  # stdin closed: `melo <&-`
        (True, None, "xterm-256color", False),
    ],
)
def test_on_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
    stdin: bool | None,
    stdout: bool | None,
    term: str,
    expected: bool,
) -> None:
    monkeypatch.setattr(sys, "stdin", None if stdin is None else _Stream(stdin))
    monkeypatch.setattr(sys, "stdout", None if stdout is None else _Stream(stdout))
    monkeypatch.setenv("TERM", term)
    assert cli._on_a_terminal() is expected


def _config_home(home: Path) -> None:
    cache = home / ".cache" / "localmelo"
    cache.mkdir(parents=True)
    (cache / "config.toml").write_text(
        'chat_backend = "vllm"\nembedding_backend = "none"\n\n'
        f'[vllm]\nchat_url = "{URL}"\nchat_model = "m"\n',
        encoding="utf-8",
    )


def _child(home: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    """main() with *argv*, stdin /dev/null; the gateway itself is stubbed."""
    child = (
        "import sys, localmelo.__main__ as m\n"
        "m._start_gateway = lambda cfg: print('gateway', cfg.chat_backend)\n"
        f"sys.argv = {argv!r}\n"
        "m.main()\n"
    )
    env = {**os.environ, "HOME": str(home)}
    env.pop("LOCALMELO_DETECT_URLS", None)
    return subprocess.run(
        [sys.executable, "-c", child],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_serve_with_a_config_and_no_terminal_starts_the_gateway(
    tmp_path: Path,
) -> None:
    """The gateway as on main, for a script or launchd."""
    _config_home(tmp_path)
    result = _child(tmp_path, ["melo", "--serve"])
    assert (result.returncode, result.stdout, result.stderr) == (
        0,
        "gateway vllm\n",
        "",
    )


def test_bare_melo_without_a_terminal_exits_2_and_says_why(tmp_path: Path) -> None:
    """Bare melo opens the UI, which needs a terminal; it no longer starts the gateway."""
    _config_home(tmp_path)
    result = _child(tmp_path, ["melo"])
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr == (
        "melo: the interactive UI needs a terminal (stdin and stdout must be TTYs).\n"
        '  Run one task:      melo "your task"\n'
        "  Start the gateway: melo --serve\n"
    )
    assert "Traceback" not in result.stderr


# ── The interactive UI's flags (U2a.1) ──


@pytest.mark.parametrize(
    ("argv", "given"),
    [
        (["--yolo", "task"], "--yolo"),
        (["--yolo", ""], "--yolo"),  # `melo ""` is direct mode
        (["--serve", "--yolo", "--workspace", "/tmp"], "--yolo"),
        (["--daemon", "status", "--yolo"], "--yolo"),
    ],
)
def test_the_ui_flags_are_refused_outside_the_ui(
    capsys: pytest.CaptureFixture[str], argv: list[str], given: str
) -> None:
    with (
        mock.patch("sys.argv", ["melo", *argv]),
        mock.patch("localmelo.support.cli.run_interactive") as ui,
        mock.patch.object(cli, "_build_direct_mode_agent") as build,
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    ui.assert_not_called()
    build.assert_not_called()
    assert capsys.readouterr().err.endswith(
        "melo: error: only for the interactive UI (melo without a query, --serve "
        f"or --daemon): {given}\n"
    )


@pytest.mark.parametrize(
    ("argv", "flags"),
    [
        (["--timeout", "5"], "--timeout"),
        (["--output-format", "json"], "--output-format"),
        (["--tool-exposure", "all"], "--tool-exposure"),
        (["--progress", "off"], "--progress"),
        (["--routine", "daily"], "--routine"),
        (
            ["--max-steps", "4", "--timeout", "5", "--progress", "off"],
            "--timeout, --progress",
        ),
    ],
)
def test_the_ui_refuses_the_direct_flags_it_cannot_honor(
    capsys: pytest.CaptureFixture[str], argv: list[str], flags: str
) -> None:
    with (
        mock.patch("sys.argv", ["melo", *argv]),
        mock.patch("localmelo.support.cli.run_interactive") as ui,
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    ui.assert_not_called()
    assert capsys.readouterr().err.endswith(
        f"melo: error: only for direct mode (a query without --serve): {flags}\n"
    )


@pytest.mark.parametrize(
    "argv", [["--mcp-config", "x.json"], ["--mcp-config", "x.json", "--mcp-only"]]
)
def test_the_mcp_flags_reach_the_ui(argv: list[str]) -> None:
    with (
        mock.patch("sys.argv", ["melo", *argv]),
        mock.patch("localmelo.support.cli.run_interactive", return_value=0) as ui,
    ):
        try:
            cli.main()
        except SystemExit as exc:  # the UI's own exit code
            assert exc.code == 0
    ui.assert_called_once()


def test_a_daemon_with_a_port_is_not_refused() -> None:
    with (
        mock.patch("sys.argv", ["melo", "--daemon", "install", "--port", "8401"]),
        mock.patch(
            "localmelo.support.gateway.daemon.install", return_value="p"
        ) as inst,
    ):
        cli.main()
    inst.assert_called_once_with(port=8401)


def test_an_abbreviation_without_a_task_reaches_the_ui() -> None:
    """The user's `--chat-mode X` reads as --chat-model X, as they meant."""
    with (
        mock.patch("sys.argv", ["melo", "--chat-mode", "X"]),
        mock.patch("localmelo.support.cli.run_interactive", return_value=0) as ui,
    ):
        cli.main()
    assert ui.call_args.args[0].chat_model == "X"


def test_an_abbreviation_with_a_task_stays_direct_mode() -> None:
    agent = mock.MagicMock()
    agent.run_task = mock.AsyncMock(
        return_value=RunResult(task_id="t", status="completed", result="ok")
    )
    agent.close = mock.AsyncMock()
    with (
        mock.patch("sys.argv", ["melo", "--chat-mode", "X", "--", "task"]),
        mock.patch("localmelo.support.cli.run_interactive") as ui,
        mock.patch.object(cli, "_build_direct_mode_agent", return_value=agent) as build,
    ):
        cli.main()
    ui.assert_not_called()
    assert build.call_args.args[0].chat_model == "X"
    agent.run_task.assert_awaited_once_with("task")


def test_a_memory_dir_that_cannot_be_one_exits_2_before_the_ui(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    afile = tmp_path / "file"
    afile.write_text("x")
    with (
        mock.patch("sys.argv", ["melo", "--memory-dir", str(afile)]),
        mock.patch("localmelo.support.cli.ui_terminal", return_value=True),
        mock.patch("localmelo.support.cli.run_interactive") as ui,
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    ui.assert_not_called()
    assert "argument --memory-dir: cannot use" in capsys.readouterr().err


def test_without_a_terminal_the_memory_dir_is_not_made(tmp_path: Path) -> None:
    folder = tmp_path / "memory"
    with (
        mock.patch("sys.argv", ["melo", "--memory-dir", str(folder)]),
        mock.patch("localmelo.support.cli.ui_terminal", return_value=False),
        mock.patch("localmelo.support.cli.run_interactive", return_value=2) as ui,
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    ui.assert_called_once()
    assert not folder.exists()
