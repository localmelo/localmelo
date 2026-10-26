"""The plain loop behind `melo ""`, and what bare `melo` starts.

`melo "<task>"` runs the task (direct mode), `melo ""` opens the plain
loop, and bare `melo` and `melo --reconfigure` set up a backend when
needed, then start the gateway. The direct-mode flags apply to a task
only.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.melo.schema import RunResult
from localmelo.support.config import Config, LocalBackendConfig


def _agent() -> mock.MagicMock:
    agent = mock.MagicMock()
    agent.run_task = mock.AsyncMock(
        return_value=RunResult(task_id="turn", status="completed", result="42")
    )
    agent.close = mock.AsyncMock()
    return agent


def _config(chat_model: str = "configured-model") -> Config:
    return Config(
        chat_backend="mlc",
        embedding_backend="none",
        mlc=LocalBackendConfig(
            chat_url="http://127.0.0.1:8500/v1", chat_model=chat_model
        ),
    )


def test_an_empty_query_opens_the_loop(capsys: pytest.CaptureFixture[str]) -> None:
    agent = _agent()
    with (
        mock.patch("sys.argv", ["melo", "--base-url", "http://127.0.0.1:8500/v1", ""]),
        mock.patch.object(cli, "_build_direct_mode_agent", return_value=agent) as build,
        mock.patch.object(cli, "_start_gateway") as gateway,
        mock.patch("builtins.input", side_effect=["/help", "", "6*7", "/exit"]),
        mock.patch.object(cli, "_stdout_for_the_report") as redirect,
    ):
        cli.main()
    agent.run_task.assert_awaited_once_with("6*7")
    agent.close.assert_awaited_once()
    gateway.assert_not_called()
    redirect.assert_not_called()
    # The loop keeps its answers, so each turn sees the earlier ones.
    assert build.call_args.kwargs["options"] == AgentOptions(record_answers=True)
    out, err = capsys.readouterr()
    assert "/help" in out and "melo --serve" in out and "42" in out
    assert not err


@pytest.mark.parametrize(
    ("argv", "flags"),
    [
        (["--max-steps", "5", ""], "--max-steps"),
        (["--max-steps", "3"], "--max-steps"),
        (["--output-format", "json", ""], "--output-format"),
        (["--output-format", "json"], "--output-format"),
        (["--reconfigure", "--max-steps", "3"], "--max-steps"),
    ],
)
def test_direct_flags_without_a_task_are_rejected_before_anything_runs(
    capsys: pytest.CaptureFixture[str], argv: list[str], flags: str
) -> None:
    with (
        mock.patch("sys.argv", ["melo", *argv]),
        mock.patch.object(cli, "_build_direct_mode_agent") as build,
        mock.patch("localmelo.support.onboard.run_backend_setup") as setup,
        mock.patch.object(cli, "_start_gateway") as gateway,
        pytest.raises(SystemExit) as exited,
    ):
        cli.main()
    assert exited.value.code == cli.EXIT_NOT_COMPLETED
    build.assert_not_called()
    setup.assert_not_called()
    gateway.assert_not_called()
    out, err = capsys.readouterr()
    assert out == ""
    assert (
        f"melo: error: only for direct mode (a query without --serve): {flags}" in err
    )


def test_a_provider_error_keeps_the_loop_usable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _agent()
    agent.run_task.side_effect = [
        RunResult(task_id="first", status="provider_error", error="HTTP 503"),
        RunResult(task_id="second", status="completed", result="Recovered"),
    ]
    with (
        mock.patch("sys.argv", ["melo", ""]),
        mock.patch.object(cli, "_build_direct_mode_agent", return_value=agent),
        mock.patch.object(cli, "_start_gateway") as gateway,
        mock.patch("builtins.input", side_effect=["first", "retry", EOFError]),
    ):
        cli.main()
    assert [call.args for call in agent.run_task.await_args_list] == [
        ("first",),
        ("retry",),
    ]
    agent.close.assert_awaited_once()
    gateway.assert_not_called()
    out, err = capsys.readouterr()
    assert "Recovered" in out
    assert err == "melo: run ended with status provider_error: HTTP 503\n"


def test_bare_melo_starts_the_configured_gateway() -> None:
    cfg = _config()
    with (
        mock.patch("sys.argv", ["melo"]),
        mock.patch("localmelo.support.config.load", return_value=cfg),
        mock.patch("localmelo.support.onboard.run_backend_setup") as setup,
        mock.patch.object(cli, "_start_gateway") as gateway,
        mock.patch.object(cli, "_build_direct_mode_agent") as build,
    ):
        cli.main()
    setup.assert_not_called()
    gateway.assert_called_once_with(cfg)
    build.assert_not_called()


def test_bare_melo_sets_up_a_backend_first_when_none_is_configured() -> None:
    new_cfg = _config()
    with (
        mock.patch("sys.argv", ["melo"]),
        mock.patch("localmelo.support.config.load", return_value=Config()),
        mock.patch(
            "localmelo.support.onboard.run_backend_setup", return_value=new_cfg
        ) as setup,
        mock.patch.object(cli, "_start_gateway") as gateway,
        mock.patch.object(cli, "_build_direct_mode_agent") as build,
    ):
        cli.main()
    setup.assert_called_once_with()
    gateway.assert_called_once_with(new_cfg)
    build.assert_not_called()


def test_reconfigure_runs_the_wizard_then_starts_the_gateway() -> None:
    new_cfg = _config("new-model")
    with (
        mock.patch("sys.argv", ["melo", "--reconfigure"]),
        mock.patch("localmelo.support.config.load", return_value=_config()),
        mock.patch(
            "localmelo.support.onboard.run_backend_setup", return_value=new_cfg
        ) as setup,
        mock.patch.object(cli, "_start_gateway") as gateway,
        mock.patch.object(cli, "_build_direct_mode_agent") as build,
    ):
        cli.main()
    setup.assert_called_once_with()
    gateway.assert_called_once_with(new_cfg)
    build.assert_not_called()


def test_an_explicit_task_keeps_priority_over_reconfigure() -> None:
    agent = _agent()
    with (
        mock.patch("sys.argv", ["melo", "--reconfigure", "--max-steps", "3", "6*7"]),
        mock.patch.object(cli, "_build_direct_mode_agent", return_value=agent),
        mock.patch("localmelo.support.onboard.run_backend_setup") as setup,
        mock.patch.object(cli, "_start_gateway") as gateway,
    ):
        cli.main()
    agent.run_task.assert_awaited_once_with("6*7")
    agent.close.assert_awaited_once()
    setup.assert_not_called()
    gateway.assert_not_called()


def test_model_override_applies_to_config_without_saving() -> None:
    cfg = _config()
    with (
        mock.patch("localmelo.support.config.load", return_value=cfg),
        mock.patch("localmelo.support.config.save") as save,
        mock.patch.object(cli, "Agent") as constructor,
    ):
        cli._build_direct_mode_agent(
            argparse.Namespace(base_url=None, chat_model="selected-model")
        )
    assert constructor.call_args.kwargs["config"].mlc.chat_model == "selected-model"
    save.assert_not_called()


def test_a_task_with_chat_model_and_no_base_url_sends_that_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # In-memory history: nothing is written under ~/.cache/localmelo.
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)
    models: list[str] = []

    async def post_json(
        client: httpx.AsyncClient, path: str, payload: dict[str, Any], **kwargs: Any
    ) -> tuple[httpx.Response, Any]:
        models.append(payload["model"])
        body = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        return httpx.Response(200, json=body), body

    with (
        mock.patch("sys.argv", ["melo", "--chat-model", "selected-model", "task"]),
        mock.patch("localmelo.support.config.load", return_value=_config()),
        mock.patch("localmelo.support.config.save") as save,
        mock.patch(
            "localmelo.support.providers.llm.openai_compat.post_json", post_json
        ),
    ):
        cli.main()
    assert models
    assert set(models) == {"selected-model"}
    save.assert_not_called()
    assert capsys.readouterr().out == "done\n"


def _command() -> list[str]:
    return [
        sys.executable,
        "-I",
        "-m",
        "localmelo",
        "--base-url",
        "http://127.0.0.1:1/v1",
        "",
    ]


def _env(home: Path) -> dict[str, str]:
    """This environment without localmelo's settings, HOME at *home*.

    The loop takes no --memory-dir (a direct-mode flag), so its history
    stays in memory and nothing reaches the real ~/.cache/localmelo.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LOCALMELO_")
    }
    env["HOME"] = str(home)
    return env


def test_real_interactive_help_stays_on_stdout(tmp_path: Path) -> None:
    done = subprocess.run(
        _command(),
        input="/help\n/exit\n",
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=_env(tmp_path),
        timeout=10,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "localmelo · local agent" in done.stdout
    assert "melo --serve" in done.stdout
    assert not done.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_signal_stops_an_idle_interactive_prompt(tmp_path: Path, signum: int) -> None:
    if signal.getsignal(signum) == signal.SIG_IGN:
        pytest.skip("the invoking environment ignores this signal")
    process = subprocess.Popen(
        _command(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=_env(tmp_path),
        start_new_session=True,
    )
    try:
        assert process.stdout is not None
        output = b""
        deadline = time.monotonic() + 5
        while "❯".encode() not in output and time.monotonic() < deadline:
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break
                output += chunk
        assert "❯".encode() in output, output
        process.send_signal(signum)
        process.wait(timeout=cli.SIGNAL_CLEANUP_S + 2)
        _, err = process.communicate()
        assert process.returncode == -signum
        assert b"Traceback" not in err
    finally:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            process.communicate(timeout=5)
