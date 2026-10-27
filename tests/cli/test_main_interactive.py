"""melo without a task: which model server the `melo ""` loop uses, and the refusals.

Model listing is faked, so nothing reaches a model server, except in the
subprocess test, which needs no server at all.
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
from localmelo.support.backends.discovery import DiscoveryError
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


def _loop(
    argv: list[str],
    *,
    terminal: bool,
    cfg: Config | None = None,
    listed: list[str] | Exception | None = None,
) -> tuple[mock.MagicMock, mock.AsyncMock, mock.MagicMock]:
    """Run main() with *argv* up to the loop, which reads `/exit` at once.

    Returns the agent builder, the model listing and offer_to_save.
    """
    listing = mock.AsyncMock(
        side_effect=listed if isinstance(listed, Exception) else None,
        return_value=listed if isinstance(listed, list) else [],
    )
    with (
        mock.patch("sys.argv", ["melo", *argv]),
        mock.patch.object(cli, "_on_a_terminal", return_value=terminal),
        mock.patch("localmelo.support.config.load", return_value=cfg or Config()),
        mock.patch("localmelo.support.backends.discovery.list_models", listing),
        mock.patch("localmelo.support.onboard.offer_to_save") as offer,
        mock.patch("localmelo.support.onboard.run_first_setup") as first,
        mock.patch("localmelo.support.onboard.run_backend_setup") as setup,
        mock.patch.object(cli, "_start_gateway") as gateway,
        mock.patch.object(
            cli, "_build_direct_mode_agent", return_value=_agent()
        ) as build,
        mock.patch("builtins.input", side_effect=["/exit"]),
    ):
        cli.main()
    gateway.assert_not_called()
    setup.assert_not_called()
    first.assert_not_called()
    assert build.call_args.args[0].query == [""]
    return build, listing, offer


@pytest.mark.parametrize("terminal", [True, False])
def test_a_base_url_without_a_task_opens_the_loop_with_the_first_chat_model(
    capsys: pytest.CaptureFixture[str], terminal: bool
) -> None:
    build, listing, offer = _loop(
        ["--base-url", URL],
        terminal=terminal,
        cfg=_configured(),
        listed=["bge-embed", "Qwen3-8B", "Qwen3-32B"],
    )
    listing.assert_awaited_once_with("openai-compatible", URL)
    args = build.call_args.args[0]
    assert (args.base_url, args.chat_model) == (URL, "Qwen3-8B")
    offer.assert_not_called()  # a config exists
    out = capsys.readouterr().out
    assert out.startswith(f"Model: Qwen3-8B · {URL}\nlocalmelo · local agent\n")


def test_a_base_url_with_a_model_sends_no_listing_when_a_config_exists() -> None:
    build, listing, offer = _loop(
        ["--base-url", URL, "--chat-model", "M"], terminal=True, cfg=_configured()
    )
    listing.assert_not_called()
    assert build.call_args.args[0].chat_model == "M"
    offer.assert_not_called()


def test_a_server_that_cannot_list_its_models_gets_the_model_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    build, _, offer = _loop(
        ["--base-url", URL],
        terminal=True,
        listed=DiscoveryError("connection refused", reachable=False),
    )
    assert build.call_args.args[0].chat_model is None  # the builder sends "default"
    offer.assert_not_called()
    out, err = capsys.readouterr()
    assert err == (
        f"melo: could not list the models at {URL} (connection refused); "
        "asking for the model 'default' (--chat-model NAME picks one)\n"
    )
    assert out.startswith(f"Model: default · {URL}\n")


def test_with_no_config_a_terminal_offers_to_save_the_listed_model() -> None:
    _, _, offer = _loop(["--base-url", URL], terminal=True, listed=["m1", "m2"])
    offer.assert_called_once_with(URL, "m1")


def test_with_no_config_a_named_model_is_offered_when_the_server_lists_it() -> None:
    _, listing, offer = _loop(
        ["--base-url", URL, "--chat-model", "m2"], terminal=True, listed=["m1", "m2"]
    )
    listing.assert_awaited_once()
    offer.assert_called_once_with(URL, "m2")


def test_with_no_config_a_model_the_server_does_not_list_is_not_offered(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, offer = _loop(
        ["--base-url", URL, "--chat-model", "typo"], terminal=True, listed=["m1"]
    )
    offer.assert_not_called()
    # The listing was only for the offer, so its failure would not be reported.
    assert capsys.readouterr().err == ""


def test_without_a_terminal_nothing_is_offered() -> None:
    _, _, offer = _loop(["--base-url", URL], terminal=False, listed=["m1"])
    offer.assert_not_called()


def test_a_chat_model_without_a_task_opens_the_loop_on_the_configured_backend(
    capsys: pytest.CaptureFixture[str],
) -> None:
    build, listing, _ = _loop(
        ["--chat-model", "other"], terminal=False, cfg=_configured()
    )
    listing.assert_not_called()
    args = build.call_args.args[0]
    assert (args.base_url, args.chat_model) == (None, "other")
    assert capsys.readouterr().out.startswith(
        f"Model: other · {URL} (melo --reconfigure changes it)\n"
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
        "melo: error: --base-url and --chat-model are for a task or the chat loop; "
        "--serve uses the config file (melo --reconfigure)\n"
    )


@pytest.mark.parametrize("argv", [["--port", "9000"], ["--host", "0.0.0.0"]])
def test_host_and_port_are_refused_where_melo_opens_the_loop(
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
        "melo without a task opens the chat loop\n"
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


def test_a_server_that_lists_no_models_is_reported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    build, _, _ = _loop(["--base-url", URL], terminal=False, listed=[])
    assert build.call_args.args[0].chat_model is None
    assert capsys.readouterr().err == (
        f"melo: could not list the models at {URL} (it lists no models); "
        "asking for the model 'default' (--chat-model NAME picks one)\n"
    )


def test_a_url_without_v1_is_used_with_it_once_the_server_lists_there(
    capsys: pytest.CaptureFixture[str],
) -> None:
    build, listing, _ = _loop(
        ["--base-url", "http://127.0.0.1:8500/"], terminal=False, listed=["m"]
    )
    listing.assert_awaited_once_with("openai-compatible", "http://127.0.0.1:8500/")
    args = build.call_args.args[0]
    assert (args.base_url, args.chat_model) == (URL, "m")
    assert capsys.readouterr().out.startswith(f"Model: m · {URL}\n")


def test_a_url_that_does_not_list_is_used_as_given() -> None:
    build, _, _ = _loop(
        ["--base-url", "http://127.0.0.1:8500/api"],
        terminal=False,
        listed=DiscoveryError("HTTP 404", reachable=True, status_code=404),
    )
    assert build.call_args.args[0].base_url == "http://127.0.0.1:8500/api"


def test_off_a_terminal_a_base_url_never_reads_the_config() -> None:
    """A broken config file does not stop `melo --base-url X --chat-model Y`."""
    with (
        mock.patch("sys.argv", ["melo", "--base-url", URL, "--chat-model", "M"]),
        mock.patch.object(cli, "_on_a_terminal", return_value=False),
        mock.patch("localmelo.support.config.load", side_effect=AssertionError("read")),
        mock.patch("localmelo.support.backends.discovery.list_models") as listing,
        mock.patch.object(cli, "_build_direct_mode_agent", return_value=_agent()),
        mock.patch("builtins.input", side_effect=["/exit"]),
    ):
        cli.main()
    listing.assert_not_called()


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


def test_reconfigure_with_a_chat_model_runs_the_wizard_then_the_loop() -> None:
    with (
        mock.patch("sys.argv", ["melo", "--reconfigure", "--chat-model", "M"]),
        mock.patch.object(cli, "_on_a_terminal", return_value=False),
        mock.patch("localmelo.support.config.load", return_value=_configured()),
        mock.patch(
            "localmelo.support.onboard.run_backend_setup", return_value=_configured()
        ) as setup,
        mock.patch.object(cli, "_start_gateway") as gateway,
        mock.patch.object(
            cli, "_build_direct_mode_agent", return_value=_agent()
        ) as build,
        mock.patch("builtins.input", side_effect=["/exit"]),
    ):
        cli.main()
    setup.assert_called_once_with()
    gateway.assert_not_called()
    assert build.call_args.args[0].chat_model == "M"


def test_a_chat_model_wins_over_the_model_of_a_server_started_anyway() -> None:
    from localmelo.support.onboard import FirstSetup

    unsaved = FirstSetup(url=URL, model="typed", saved=False)
    with (
        mock.patch("sys.argv", ["melo", "--chat-model", "M"]),
        mock.patch.object(cli, "_on_a_terminal", return_value=True),
        mock.patch("localmelo.support.config.load", return_value=Config()),
        mock.patch("localmelo.support.onboard.run_first_setup", return_value=unsaved),
        mock.patch.object(
            cli, "_build_direct_mode_agent", return_value=_agent()
        ) as build,
        mock.patch("builtins.input", side_effect=["/exit"]),
    ):
        cli.main()
    args = build.call_args.args[0]
    assert (args.base_url, args.chat_model) == (URL, "M")


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


def test_bare_melo_with_a_config_and_no_terminal_starts_the_gateway(
    tmp_path: Path,
) -> None:
    """The gateway as on main, for a script or a pipe; the gateway itself is stubbed."""
    cache = tmp_path / ".cache" / "localmelo"
    cache.mkdir(parents=True)
    (cache / "config.toml").write_text(
        'chat_backend = "vllm"\nembedding_backend = "none"\n\n'
        f'[vllm]\nchat_url = "{URL}"\nchat_model = "m"\n',
        encoding="utf-8",
    )
    child = (
        "import sys, localmelo.__main__ as m\n"
        "m._start_gateway = lambda cfg: print('gateway', cfg.chat_backend)\n"
        "sys.argv = ['melo']\n"
        "m.main()\n"
    )
    env = {**os.environ, "HOME": str(tmp_path)}
    env.pop("LOCALMELO_DETECT_URLS", None)
    result = subprocess.run(
        [sys.executable, "-c", child],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert (result.returncode, result.stdout, result.stderr) == (
        0,
        "gateway vllm\n",
        "",
    )
