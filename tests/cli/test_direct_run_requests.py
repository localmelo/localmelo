"""A direct run sends the same requests whatever kind of server it talks to.

``melo --base-url URL --chat-model MODEL "task"`` is what the benchmark runs.
Naming a server (``oMLX`` in a banner, a status row or the web console) reads
what a model listing already says, so a direct run, which lists nothing,
sends the chat requests and nothing else. The server here answers like oMLX
does (``owned_by``, a status route), on a free port of this machine.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from localmelo.tests.cli.test_direct_mode_process import _env

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX processes")

_LISTING = {
    "object": "list",
    "data": [{"id": "Qwen3-8B-4bit", "object": "model", "owned_by": "omlx"}],
}
_STATUS = {"models": [{"id": "Qwen3-8B-4bit", "model_alias": "q", "loaded": True}]}


class RecordingServer:
    """An oMLX-shaped server that records every request it gets."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, bool, bytes]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self, payload: Any, body: bytes = b"") -> None:
                server.requests.append(
                    (
                        self.command,
                        self.path,
                        "Authorization" in self.headers,
                        body,
                    )
                )
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._answer(_STATUS if self.path.endswith("/status") else _LISTING)

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                message = {"role": "assistant", "content": "done"}
                self._answer({"choices": [{"index": 0, "message": message}]}, body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}/v1"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def server() -> Iterator[RecordingServer]:
    recording = RecordingServer()
    try:
        yield recording
    finally:
        recording.close()


def test_a_direct_run_sends_only_its_chat_request(
    server: RecordingServer, tmp_path: Path
) -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "localmelo",
            "--base-url",
            server.base_url,
            "--chat-model",
            "Qwen3-8B-4bit",
            "say hello",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        env=_env(tmp_path),
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == "done\n"
    # No listing, no status lookup, no version check, no key: one chat request.
    [(method, path, has_key, body)] = server.requests
    assert (method, path, has_key) == ("POST", "/v1/chat/completions", False)
    sent = json.loads(body)
    assert sent["model"] == "Qwen3-8B-4bit"
    assert [m["role"] for m in sent["messages"]][-1] == "user"
    assert sent["messages"][-1]["content"] == "say hello"
