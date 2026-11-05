"""The control-socket client of melo_serve (support/melo_serve.py), against a stub socket."""

from __future__ import annotations

import json
import shutil
import socketserver
import tempfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from localmelo.support.melo_serve import Control, MeloServeError


class _Handler(BaseHTTPRequestHandler):
    """A melo_serve that has one pick and refuses an unknown one."""

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("pick") == "27b":
            code, reply = 200, {"changed": True, "generation": 4, "pick": "27b"}
        else:
            error = {
                "type": "unknown_pick",
                "message": "no such pick",
                "suggestions": ["27b"],
            }
            code, reply = 404, {"error": error}
        data = json.dumps(reply).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        """No request log in the test output."""


@pytest.fixture
def control_socket(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    # macOS allows 104 bytes for a socket path, which pytest's tmp_path exceeds.
    folder = Path(tempfile.mkdtemp(dir="/tmp"))
    path = folder / "control.sock"
    server = socketserver.ThreadingUnixStreamServer(str(path), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("MELO_SERVE_SOCKET", str(path))
    yield path
    server.shutdown()
    server.server_close()
    shutil.rmtree(folder, ignore_errors=True)


async def test_the_control_client_switches_and_maps_errors(
    control_socket: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = Control()
    reply = await control.switch("27b", expected_generation=3, drain_s=0)
    assert reply == {"changed": True, "generation": 4, "pick": "27b"}
    with pytest.raises(MeloServeError) as refused:
        await control.switch("nope")
    assert (refused.value.status, refused.value.type) == (404, "unknown_pick")
    assert refused.value.message == "no such pick"
    assert refused.value.data == {"suggestions": ["27b"]}

    # No socket to connect to: melo_serve does not run.
    monkeypatch.setenv("MELO_SERVE_SOCKET", str(control_socket.parent / "gone.sock"))
    with pytest.raises(MeloServeError, match="melo-serve up") as down:
        await control.picks()
    assert (down.value.status, down.value.type) == (0, "not_running")
