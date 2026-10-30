"""``melo console`` and ``melo serve`` open the console in the browser, or print its link."""

from __future__ import annotations

import io
import re
import socket
import sys
import webbrowser
from pathlib import Path
from unittest import mock

import pytest

from localmelo import __main__ as cli
from localmelo.support import console_link
from localmelo.support.gateway.auth import GatewayAuth

# The root conftest replaces the module's name with "never"; this keeps the real one.
REAL_CAN_OPEN = console_link.can_open
LINK = re.compile(r"http://127\.0\.0\.1:5000/#login=(\S+)")


class Stream(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


def fake_browser(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> mock.Mock:
    browser = mock.Mock(**kwargs)
    monkeypatch.setattr(webbrowser, "open", browser)
    return browser


def run_console(
    monkeypatch: pytest.MonkeyPatch, *, listening: bool = True, can_open: bool = True
) -> tuple[str, str, int | None]:
    """``melo console -p 5000`` at a terminal: what it printed, its error, its exit code."""
    out, err = Stream(tty=True), io.StringIO()
    monkeypatch.setattr(sys, "argv", ["melo", "console", "-p", "5000"])
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(cli, "_register_backends", lambda: None)
    monkeypatch.setattr(console_link, "listening", lambda host, port: listening)
    monkeypatch.setattr(console_link, "can_open", lambda: can_open)
    code: int | None = None
    try:
        cli.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    return out.getvalue(), err.getvalue(), code


def signs_in(link: str) -> bool:
    """The link's code works for a gateway on port 5000."""
    match = LINK.fullmatch(link)
    auth = GatewayAuth()
    try:
        return match is not None and auth.consume_login(match[1], 5000) is not None
    finally:
        auth.close()


@pytest.mark.parametrize(
    ("stdin", "stdout", "env", "expected"),
    [
        (True, True, None, True),
        (False, True, None, False),
        (True, False, None, False),
        *[(True, True, name, False) for name in console_link._NOT_AT_THE_SCREEN],
    ],
)
def test_a_browser_opens_only_for_a_person_at_this_computers_terminal(
    monkeypatch: pytest.MonkeyPatch,
    stdin: bool,
    stdout: bool,
    env: str | None,
    expected: bool,
) -> None:
    for name in console_link._NOT_AT_THE_SCREEN:
        monkeypatch.delenv(name, raising=False)
    if env is not None:
        monkeypatch.setenv(env, "1")
    monkeypatch.setattr(sys, "stdin", Stream(stdin))
    monkeypatch.setattr(sys, "stdout", Stream(stdout))
    assert REAL_CAN_OPEN() is expected


def test_listening_says_whether_something_accepts_connections() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert console_link.listening("127.0.0.1", port)
    assert not console_link.listening("127.0.0.1", port)


def test_console_opens_the_browser_signed_in_and_prints_no_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = fake_browser(monkeypatch, return_value=True)
    out, _, code = run_console(monkeypatch)
    assert (code, out) == (None, "Opened the web console in your browser.\n")
    browser.assert_called_once()
    assert signs_in(browser.call_args.args[0])


@pytest.mark.parametrize("case", ["no browser starts", "the launch fails", "over ssh"])
def test_console_prints_the_link_when_no_browser_can_open_it(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    browser = fake_browser(
        monkeypatch,
        return_value=case == "over ssh",
        side_effect=(
            webbrowser.Error("no runnable browser")
            if case == "the launch fails"
            else None
        ),
    )
    out, _, code = run_console(monkeypatch, can_open=case != "over ssh")
    first, second = out.splitlines()
    assert (code, first) == (None, "Open this link within 60 seconds. Do not share it:")
    assert signs_in(second)
    assert browser.called is (case != "over ssh")


def test_console_with_no_gateway_says_so_and_opens_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = fake_browser(monkeypatch, return_value=True)
    out, err, code = run_console(monkeypatch, listening=False)
    assert (code, out, browser.called) == (1, "", False)
    assert "no gateway is running on port 5000" in err and "melo serve" in err
    assert not (Path.home() / ".cache/localmelo/gateway/auth/token").exists()


def test_the_serve_callback_opens_the_console_signed_in(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    browser = fake_browser(monkeypatch, return_value=True)
    monkeypatch.setattr(console_link, "can_open", lambda: True)
    console_link.open_console("127.0.0.1", 5000)
    assert capsys.readouterr().out == (
        "  Web console: opened in your browser (melo console opens it again).\n"
    )
    assert signs_in(browser.call_args.args[0])
