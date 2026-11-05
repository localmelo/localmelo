"""melo_serve's snapshot (support/melo_serve.py): what the state and the socket say."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.support import melo_serve
from localmelo.support.backends import discovery
from localmelo.support.backends.registry import in_use
from localmelo.support.cli.agents import config_backend
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.gateway.console.routes_settings import model_json
from localmelo.tests.fixtures.melo_serve import URL, role, write_state


def _cfg() -> Config:
    return Config(
        chat_backend="melo_serve",
        embedding_backend="melo_serve",
        melo_serve=LocalBackendConfig(chat_url=URL, embedding_url=URL),
    )


def test_a_state_that_does_not_answer_serves_nothing(
    melo_serve_home: Path, melo_serve_socket: Path
) -> None:
    cfg = _cfg()
    write_state(melo_serve_home, chat=role("Qwen3.8-27B-Q4KM-splash"))
    assert in_use(cfg, "melo_serve") == (
        "Splash via melo_serve",
        "Qwen3.8-27B-Q4KM-splash",
    )
    # The same state, left by a stop: the file still names a model.
    write_state(melo_serve_home, chat=role("Qwen3.8-27B-Q4KM-splash"), status="stopped")
    assert melo_serve.snapshot() == melo_serve.Snapshot()
    # The same state again, with nothing behind the socket (a crash).
    write_state(melo_serve_home, chat=role("Qwen3.8-27B-Q4KM-splash"))
    melo_serve_socket.unlink()
    assert melo_serve.snapshot() == melo_serve.Snapshot()
    assert in_use(cfg, "melo_serve") == ("melo_serve", "")
    row = config_backend(cfg, None)  # what the welcome row is drawn from
    assert (row.label, row.model, row.note) == (
        "melo_serve",
        "none served",
        "melo_serve",
    )
    assert model_json(cfg)["name"] == ""


def _huge_pid(home: Path) -> None:
    write_state(home)
    state = json.loads((home / "state.json").read_text())
    state["front_door"]["pid"] = 2**100
    (home / "state.json").write_text(json.dumps(state))


def _edit(change: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    def write(home: Path) -> None:
        write_state(home, chat=role("m"))
        state = json.loads((home / "state.json").read_text())
        change(state)
        (home / "state.json").write_text(json.dumps(state))

    return write


def _nested(home: Path) -> None:
    deep = "[" * 100_000 + "]" * 100_000
    (home / "state.json").write_text(f'{{"schema": "melo_serve.state/1", "x": {deep}}}')


def _oversized(home: Path) -> None:
    write_state(home, chat=role("m"))
    (home / "state.json").write_text(
        (home / "state.json").read_text() + " " * melo_serve.MAX_STATE_BYTES
    )


NOT_RUNNING = melo_serve.Snapshot()
RUNNING_NO_ROLES = melo_serve.Snapshot(
    running=True, url=URL, status="ready", login=True
)
RUNNING_LOGINLESS = melo_serve.Snapshot(
    running=True,
    url=URL,
    status="ready",
    login=False,
    roles={"chat": melo_serve.Role("m", "m", "Splash", "ready")},
)


def _front_door(url: str) -> Callable[[Path], None]:
    return _edit(lambda state: state["front_door"].update(url=url))


# Each case is written to a state whose control socket accepts connections, so
# the field checks run. The second item is what the snapshot must be.
MALFORMED = {
    "a huge pid": (_huge_pid, RUNNING_NO_ROLES),  # the pid is not read
    "json nested past the recursion limit": (_nested, NOT_RUNNING),
    "an oversized file": (_oversized, NOT_RUNNING),
    "a model that is not a string": (
        _edit(lambda state: state["roles"]["chat"].update(model=["m"])),
        RUNNING_NO_ROLES,
    ),
    "a service that is not a dict": (
        _edit(lambda state: state.update(service="x")),
        RUNNING_LOGINLESS,
    ),
    "a front door url that is not loopback": (
        _front_door("http://example.com:8321/v1"),
        NOT_RUNNING,
    ),
    "a front door url with a control character": (
        _front_door("http://127.0.0.1:8321/v1\n\x1b[2J"),
        NOT_RUNNING,
    ),
    "a front door url with user info": (
        _front_door("http://user:pw@127.0.0.1:8321/v1"),
        NOT_RUNNING,
    ),
    "a front door url with a space": (
        _front_door("http://127.0.0.1:8321/v1 x"),
        NOT_RUNNING,
    ),
}


@pytest.mark.parametrize(
    ("write", "expected"), MALFORMED.values(), ids=MALFORMED.keys()
)
async def test_a_malformed_state_is_not_running_and_breaks_nothing(
    write: Callable[[Path], None],
    expected: melo_serve.Snapshot,
    melo_serve_home: Path,
    melo_serve_socket: Path,
) -> None:
    write(melo_serve_home)
    assert melo_serve.snapshot() == expected
    status = "\n".join(melo_serve.status_lines(expected))
    assert ("running: False" in status) == (not expected.running)
    chat = expected.roles.get("chat")
    assert model_json(_cfg())["name"] == (chat.model if chat else "")
    if not expected.running:
        assert melo_serve.ensure_running() == melo_serve.NOT_INSTALLED
        assert await discovery.detect_local_server([]) is None


def test_a_url_is_kept_rebuilt_from_its_parts(
    melo_serve_home: Path, melo_serve_socket: Path
) -> None:
    _front_door("http://LOCALHOST:8321/v1?a=1#frag")(melo_serve_home)
    assert melo_serve.snapshot().url == "http://localhost:8321/v1?a=1"


def test_a_live_socket_does_not_make_a_bad_state_good(
    melo_serve_home: Path, melo_serve_socket: Path
) -> None:
    MALFORMED["a front door url that is not loopback"][0](melo_serve_home)
    assert not melo_serve.snapshot().running  # its address is never offered
    MALFORMED["a model that is not a string"][0](melo_serve_home)
    assert melo_serve.snapshot() == melo_serve.Snapshot(
        running=True, url=URL, status="ready", login=True
    )  # running, but with no model to serve
    assert in_use(_cfg(), "melo_serve") == ("melo_serve", "")


def test_ensure_running_has_one_budget_for_up_and_the_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(melo_serve, "command", lambda: "/bin/melo-serve")
    timeouts: list[float] = []

    def too_slow(args: list[str], **kwargs: Any) -> None:
        timeouts.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", too_slow)
    line = melo_serve.ensure_running()
    assert 0 < timeouts[0] <= 15
    assert line == "melo_serve did not start in 15 s; try `melo-serve up`"
    assert "/bin/" not in line  # not the command's path


def test_validation_reads_no_state_so_melo_serve_can_be_started() -> None:
    cfg = _cfg()
    cfg.memory.retriever = "embedding"  # nothing is served and melo_serve is down
    assert cfg.has_embedding and cfg.validate() == []


def _fake_command(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "melo-serve"
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def real_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The real lookup of ``melo-serve`` (the suite stubs ``command``), with an empty PATH."""
    monkeypatch.setattr(melo_serve, "command", melo_serve._find_command)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    for name in ("UV_TOOL_BIN_DIR", "XDG_BIN_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def test_melo_serve_is_found_in_the_uv_tool_folder_when_path_lacks_it(
    real_command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert melo_serve.command() is None
    fake = _fake_command(real_command / "home" / ".local" / "bin")
    assert melo_serve.command() == str(fake)  # the launchd login item's case
    xdg = _fake_command(real_command / "xdg")
    monkeypatch.setenv("XDG_BIN_HOME", str(xdg.parent))
    assert melo_serve.command() == str(xdg)
    uv = _fake_command(real_command / "uv")
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(uv.parent))
    assert melo_serve.command() == str(uv)
    uv.chmod(0o644)  # not executable
    assert melo_serve.command() is None


def test_a_melo_serve_found_by_a_relative_path_entry_is_refused(
    real_command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_command(real_command / "bin")
    monkeypatch.chdir(real_command)
    monkeypatch.setenv("PATH", "bin")  # `which` answers bin/melo-serve
    assert melo_serve.command() is None


def test_a_melo_serve_another_user_owns_is_refused(
    real_command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_command(real_command / "home" / ".local" / "bin")
    assert melo_serve.command() == str(fake)
    monkeypatch.setattr(os, "getuid", lambda: os.stat(fake).st_uid + 1)
    assert melo_serve.command() is None
    monkeypatch.setenv("PATH", str(fake.parent))  # on PATH it is refused too
    assert melo_serve.command() is None


def _symlink(sock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    link = sock.with_name("link.sock")
    link.symlink_to(sock)
    monkeypatch.setenv(melo_serve.SOCKET_ENV, str(link))


def _regular_file(sock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plain = sock.with_name("plain.sock")
    plain.write_text("")
    monkeypatch.setenv(melo_serve.SOCKET_ENV, str(plain))


def _other_owner(sock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = os.lstat

    def lstat(path: Any, **kwargs: Any) -> os.stat_result:
        info = real(path, **kwargs)
        if str(path) != str(sock):
            return info
        return os.stat_result((*info[:4], info.st_uid + 1, *info[5:]))

    monkeypatch.setattr(os, "lstat", lstat)


def _shared_folder(sock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sock.parent.chmod(0o777)


REFUSED = {
    "a symlink to a real socket": _symlink,
    "a regular file": _regular_file,
    "a socket another user owns": _other_owner,
    "a folder others can write": _shared_folder,
}


@pytest.mark.parametrize("spoil", REFUSED.values(), ids=REFUSED.keys())
def test_a_control_socket_that_is_not_the_users_own_is_not_connected_to(
    spoil: Callable[[Path, pytest.MonkeyPatch], None],
    melo_serve_home: Path,
    melo_serve_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_state(melo_serve_home, chat=role("model-a"))
    assert melo_serve.snapshot().running  # the fixture's socket passes the checks
    spoil(melo_serve_socket, monkeypatch)
    assert melo_serve.snapshot() == melo_serve.Snapshot()
