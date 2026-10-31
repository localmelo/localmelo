"""The gateway's endpoint file: written once it listens, removed as it stops.

Whichever starts first, the terminal or the macOS app, the other finds the
running gateway through this file instead of starting a second one on the
same main chat. Only loopback ports the system picks are bound, under a
temporary HOME.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from localmelo.support import config, gateway_endpoint
from localmelo.support.cli import main_chat
from localmelo.support.gateway.auth import GatewayAuth
from localmelo.support.gateway_endpoint import Endpoint

ROOT = Path(__file__).resolve().parents[2]
INSTANCE = "0123456789abcdef" * 2


def _endpoint(pid: int, port: int = 8123, gateway_id: str = "f" * 32) -> Endpoint:
    return Endpoint(
        pid=pid, host="127.0.0.1", port=port, instance="", id=gateway_id, started=1.0
    )


def _file(home: Path) -> Path:
    return home / ".cache" / "localmelo" / "gateway" / "endpoint.json"


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=10)
    return process.pid


def _reap(process: subprocess.Popen[bytes]) -> None:
    """Reap *process* when it ends, as the terminal reaps the gateway it starts:
    an ended child nobody waited for still answers ``kill(pid, 0)``."""
    threading.Thread(target=process.wait, daemon=True).start()


@contextlib.contextmanager
def _sleeper() -> Iterator[subprocess.Popen[bytes]]:
    """A process of this test's own, standing in for a gateway's pid."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


def test_the_file_is_written_whole_and_private(home: Path) -> None:
    found = _endpoint(os.getpid())
    gateway_endpoint.write(found)
    path = _file(home)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())["port"] == 8123
    assert gateway_endpoint.read() == found
    assert [p.name for p in path.parent.iterdir()] == ["endpoint.json"]


def test_a_damaged_or_missing_file_names_no_gateway(home: Path) -> None:
    assert gateway_endpoint.read() is None
    _file(home).parent.mkdir(parents=True)
    _file(home).write_text("{not json")
    assert gateway_endpoint.read() is None
    _file(home).write_text(json.dumps({"pid": 0, "host": "h", "port": 1, "id": "x"}))
    assert gateway_endpoint.read() is None


def test_a_file_whose_process_ended_is_stale(home: Path) -> None:
    gateway_endpoint.write(_endpoint(_dead_pid()))
    assert gateway_endpoint.live(confirm=lambda found: True) is None


def test_a_port_that_answers_without_the_files_id_is_stale(home: Path) -> None:
    gateway_endpoint.write(_endpoint(os.getpid(), gateway_id="a" * 32))
    assert gateway_endpoint.live(confirm=lambda found: False) is None
    assert gateway_endpoint.live(confirm=lambda found: True) is not None


def test_a_gateway_removes_only_its_own_file(home: Path) -> None:
    gateway_endpoint.write(_endpoint(os.getpid(), gateway_id="b" * 32))
    gateway_endpoint.remove("a" * 32)  # an older gateway shutting down
    assert gateway_endpoint.read() is not None
    gateway_endpoint.remove("b" * 32)
    assert not _file(home).exists()


def test_serve_stop_with_nothing_running(home: Path) -> None:
    assert gateway_endpoint.stop() == (0, "No gateway is running.")
    gateway_endpoint.write(_endpoint(_dead_pid()))
    assert gateway_endpoint.stop() == (0, "No gateway is running.")


def test_serve_stop_ends_the_gateway_the_file_names(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gateway_endpoint, "confirmed", lambda found: True)
    with _sleeper() as process:
        gateway_endpoint.write(_endpoint(process.pid, port=8124))
        _reap(process)
        assert gateway_endpoint.stop() == (0, "Stopped the gateway on port 8124.")
        assert process.poll() is not None


# ── A real gateway, as the macOS app starts it ──


def _wait_for(path: Path, child: subprocess.Popen[bytes], present: bool) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if path.exists() == present:
            return
        if present and child.poll() is not None:
            raise AssertionError(f"the gateway exited with status {child.returncode}")
        time.sleep(0.05)
    raise AssertionError(f"endpoint file {'never written' if present else 'kept'}")


def _other_port() -> int:
    for _ in range(20):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        if port not in range(8400, 8800) and port not in (8955, 11434, 11500):
            return port
    raise RuntimeError("no free test port")


def test_the_terminal_attaches_to_the_apps_gateway(home: Path) -> None:
    """The app's gateway runs first (on a port of its own); the terminal uses it."""
    auth = GatewayAuth(home / ".cache/localmelo/gateway/auth")
    try:
        auth.read_token()
    finally:
        auth.close()
    stale = _endpoint(_dead_pid(), port=9)  # left by a gateway that crashed
    gateway_endpoint.write(stale)
    env = {
        **os.environ,
        "HOME": str(home),
        "LOCALMELO_DETECT_URLS": "",
        "LOCALMELO_APP_INSTANCE": INSTANCE,
    }
    child = subprocess.Popen(
        [sys.executable, "-m", "localmelo", "--serve", "--allow-unconfigured"]
        + ["--host", "127.0.0.1", "--port", "0"],
        cwd=ROOT.parent,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    path = _file(home)
    try:
        deadline = time.monotonic() + 30
        while gateway_endpoint.read() in (None, stale):
            assert child.poll() is None, f"gateway exited ({child.returncode})"
            assert time.monotonic() < deadline, "endpoint file never written"
            time.sleep(0.05)
        found = gateway_endpoint.read()
        assert found is not None
        assert (found.pid, found.host, found.instance) == (
            child.pid,
            "127.0.0.1",
            INSTANCE,
        )
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        page = httpx.get(found.url + "/", trust_env=False, timeout=5)
        assert page.headers[gateway_endpoint.GATEWAY_HEADER] == found.id
        assert gateway_endpoint.live() == found

        cfg = config.Config()
        cfg.gateway.port = _other_port()  # not the app's port

        def no_spawn(host: str, port: int) -> main_chat.Child:
            raise AssertionError("the terminal may not start a second gateway")

        link = main_chat.connect(cfg, spawn=no_spawn)
        assert link == main_chat.Link(found.url, found.port)
        assert link.line() == f"Connected to the gateway on port {found.port}."

        # `melo serve stop` ends it, and it removes its file as it goes.
        _reap(child)
        assert gateway_endpoint.stop(wait_s=20) == (
            0,
            f"Stopped the gateway on port {found.port}.",
        )
        _wait_for(path, child, present=False)
    finally:
        if child.poll() is None:
            child.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            child.wait(timeout=10)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
