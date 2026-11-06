"""``melo serve`` and ``melo serve uninstall`` with the gateway's login service.

The service, the endpoint file, the console and the gateway are fakes: nothing
here runs launchctl, opens a browser or listens on a port.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from localmelo.support import console_link, gateway_endpoint, melo_serve
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.gateway import daemon
from localmelo.tests.fixtures.melo_serve import role, write_state

URL = "http://127.0.0.1:8321/v1"
LIVE = gateway_endpoint.Endpoint(1, "127.0.0.1", 8401, "", "id", 0.0)


class World:
    """The fakes a ``melo serve`` run meets, and a record of what it did."""

    def __init__(self) -> None:
        self.installed = True
        self.live: list[gateway_endpoint.Endpoint | None] = [None]
        self.started = 0
        self.opened: list[tuple[str, int]] = []
        self.foreground = mock.MagicMock()
        self.removed: list[str] = []

    def endpoint(self, *args: Any, **kwargs: Any) -> gateway_endpoint.Endpoint | None:
        return self.live.pop(0) if len(self.live) > 1 else self.live[0]

    def start(self) -> None:
        self.started += 1


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    fake = World()
    monkeypatch.setattr(daemon, "is_installed", lambda: fake.installed)
    monkeypatch.setattr(daemon, "start", fake.start)
    monkeypatch.setattr(daemon, "uninstall", lambda: fake.removed.append("gateway"))
    monkeypatch.setattr(daemon, "START_WAIT_S", 0.5)
    monkeypatch.setattr(gateway_endpoint, "live", fake.endpoint)
    monkeypatch.setattr(
        console_link,
        "open_console",
        lambda host, port: fake.opened.append((host, port)),
    )
    monkeypatch.setattr("localmelo.__main__._start_gateway", fake.foreground)
    monkeypatch.setattr("localmelo.__main__._register_backends", lambda: None)
    return fake


CHOOSE = "localmelo.support.cli.picker.choose_inline"


def _config(chat_backend: str = "vllm") -> Config:
    return Config(
        chat_backend=chat_backend,
        embedding_backend="none",
        vllm=LocalBackendConfig(chat_url="http://127.0.0.1:9/v1", chat_model="m"),
        melo_serve=LocalBackendConfig(chat_url=URL),
    )


def _run(*argv: str, cfg: Config | None = None) -> int:
    from localmelo.__main__ import main

    with (
        mock.patch("sys.argv", ["melo", *argv]),
        mock.patch("localmelo.support.config.load", return_value=cfg or _config()),
    ):
        try:
            main()
        except SystemExit as stop:
            return int(stop.code or 0)
    return 0


def test_serve_with_a_running_gateway_opens_the_console_and_runs_no_second_one(
    world: World, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    world.live = [LIVE]
    monkeypatch.setattr(console_link, "can_open", lambda: True)
    assert _run("serve") == 0
    assert capsys.readouterr().out == "✓ gateway   running\n"
    assert world.opened == [("127.0.0.1", 8401)]
    assert world.started == 0 and not world.foreground.called


def test_serve_starts_the_gateway_service_and_melo_serve_when_they_are_not_running(
    world: World,
    capsys: pytest.CaptureFixture[str],
    melo_serve_home: Path,
    melo_serve_socket: Path,
) -> None:
    write_state(melo_serve_home, chat=role("model-a"))
    world.live = [None, None, LIVE]  # it comes up after two looks
    assert _run("serve", cfg=_config("melo_serve")) == 0
    assert capsys.readouterr().out == (
        "✓ melo_serve   running\n"
        "✓ gateway   started\n"
        "  Web UI:  http://127.0.0.1:8401/\n"
        "  Sign in: run melo console -p 8401\n"
    )
    assert (
        world.started == 1 and world.opened == []
    )  # no screen: the address, not a tab


def test_serve_says_where_the_log_is_when_the_gateway_does_not_come_up(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run("serve") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"✗ gateway   did not start in 0.5 s; log: {daemon.LOG_HINT}\n"
    )
    assert world.opened == []


def test_serve_without_the_service_runs_the_gateway_in_the_foreground(
    world: World,
) -> None:
    world.installed = False
    assert _run("serve") == 0
    assert world.foreground.call_count == 1 and world.started == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["serve", "--port", "8401"],  # what the service itself runs
        ["--serve", "--port", "8401"],
        ["serve", "--host", "127.0.0.1"],
        ["serve", "--force"],
        ["serve", "-X", "ui.mode=default"],
    ],
    ids=["port", "service-argv", "host", "force", "setting"],
)
def test_serve_with_a_flag_that_asks_for_a_gateway_stays_in_the_foreground(
    world: World, argv: list[str]
) -> None:
    assert _run(*argv) == 0
    assert world.foreground.call_count == 1 and world.started == 0


def _melo_serve_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    marker = tmp_path / "melo-serve-ran"
    script = tmp_path / "melo-serve"
    script.write_text(f'#!/bin/sh\necho "$@" >> {marker}\necho melo_serve removed\n')
    script.chmod(0o755)
    monkeypatch.setattr(melo_serve, "command", lambda: str(script))
    return marker


def test_uninstall_names_both_services_asks_once_and_removes_both(
    world: World,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = _melo_serve_command(tmp_path, monkeypatch)
    monkeypatch.setattr("localmelo.__main__._on_a_terminal", lambda: True)
    with mock.patch(CHOOSE, return_value=0) as asked:
        assert _run("serve", "uninstall") == 0
    asked.assert_called_once_with(
        "Remove them?", ["Remove them", "Keep them"], default=1, escape=1
    )
    assert capsys.readouterr().out == (
        "This removes the login services and stops them:\n"
        "  ai.localmelo.gateway   the gateway\n"
        "  ai.localmelo.melo_serve   melo_serve\n"
        "Removed ai.localmelo.gateway.\n"
        "melo_serve removed\n"
    )
    assert world.removed == ["gateway"] and marker.read_text() == "uninstall\n"


@pytest.mark.parametrize(
    "answer", [{"return_value": 1}, {"side_effect": EOFError}], ids=["keep", "ctrl-d"]
)
def test_uninstall_removes_nothing_when_the_answer_is_to_keep_them(
    world: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answer: dict[str, Any],
) -> None:
    marker = _melo_serve_command(tmp_path, monkeypatch)
    monkeypatch.setattr("localmelo.__main__._on_a_terminal", lambda: True)
    with mock.patch(CHOOSE, **answer):
        assert _run("serve", "uninstall") == 0
    assert world.removed == [] and not marker.exists()


def test_uninstall_from_a_pipe_asks_nothing_and_removes_nothing(
    world: World,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = _melo_serve_command(tmp_path, monkeypatch)
    monkeypatch.setattr("localmelo.__main__._on_a_terminal", lambda: False)
    assert _run("serve", "uninstall") == 2
    assert "run it in a terminal" in capsys.readouterr().err
    assert world.removed == [] and not marker.exists()
