"""melo_serve's snapshot (support/melo_serve.py): what the state and the socket say."""

from __future__ import annotations

import json
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


MALFORMED = {
    "a huge pid": _huge_pid,
    "json nested past the recursion limit": _nested,
    "an oversized file": _oversized,
    "a model that is not a string": _edit(
        lambda state: state["roles"]["chat"].update(model=["m"])
    ),
    "a service that is not a dict": _edit(lambda state: state.update(service="x")),
    "a front door url that is not loopback": _edit(
        lambda state: state["front_door"].update(url="http://example.com:8321/v1")
    ),
}


@pytest.mark.parametrize("write", MALFORMED.values(), ids=MALFORMED.keys())
async def test_a_malformed_state_is_not_running_and_breaks_nothing(
    write: Callable[[Path], None], melo_serve_home: Path
) -> None:
    write(melo_serve_home)
    assert melo_serve.snapshot() == melo_serve.Snapshot()
    assert "running: False" in "\n".join(melo_serve.status_lines(melo_serve.snapshot()))
    assert melo_serve.ensure_running() == melo_serve.NOT_INSTALLED
    assert await discovery.detect_local_server([]) is None
    assert model_json(_cfg())["name"] == ""


def test_a_live_socket_does_not_make_a_bad_state_good(
    melo_serve_home: Path, melo_serve_socket: Path
) -> None:
    MALFORMED["a front door url that is not loopback"](melo_serve_home)
    assert not melo_serve.snapshot().running  # its address is never offered
    MALFORMED["a model that is not a string"](melo_serve_home)
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
