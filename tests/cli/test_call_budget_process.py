"""The real CLI in a subprocess: ``-X agent.max_calls`` and a refused request.

A stub chat server on a free port counts the requests a direct run sends. Every
process runs with a temporary HOME, so nothing reaches the real cache.
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

from localmelo.tests.cli.test_direct_mode_process import _env, _tool_call

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX processes")

TASK = "run the code"


def _call(n: int) -> dict[str, Any]:
    """A call to ``python_exec`` that differs by *n*: repeats would read as stuck."""
    return _tool_call("python_exec", {"code": f"print({n})"})


BUDGET_BODY = {
    "error": {
        "code": "budget_exhausted",
        "message": "budget_exhausted: the run has used its 30 steps",
        "type": "invalid_request_error",
    }
}


class CountingServer:
    """A chat server that answers request n with ``script[n]``, else a tool call.

    An entry is a message (HTTP 200) or ``(status, body)`` for an error answer. Past
    the script, every reply is a call to ``python_exec`` that no earlier one repeats,
    so a run never finishes on its own. ``bodies`` keeps the body of every chat request, in order.
    """

    def __init__(self, script: dict[int, Any]) -> None:
        self.script = script
        self.bodies: list[dict[str, Any]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                server.bodies.append(json.loads(body))
                entry = server.script.get(len(server.bodies)) or _call(
                    len(server.bodies)
                )
                status, payload = 200, None
                if isinstance(entry, tuple):
                    status, payload = entry
                else:
                    payload = {"choices": [{"index": 0, "message": entry}]}
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

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
def serve(tmp_path: Path) -> Iterator[Any]:
    servers: list[CountingServer] = []

    def make(script: dict[int, Any]) -> CountingServer:
        servers.append(CountingServer(script))
        return servers[-1]

    try:
        yield make
    finally:
        for server in servers:
            server.close()


def _run(
    server: CountingServer, tmp_path: Path, *settings: str
) -> tuple[int, Any, str]:
    argv = [
        sys.executable,
        "-I",
        "-m",
        "localmelo",
        "--base-url",
        server.base_url,
        "--chat-model",
        "m",
        "--tool-exposure",
        "all",
        "--output-format",
        "json",
    ]
    for setting in settings:
        argv += ["-X", setting]
    done = subprocess.run(
        [*argv, TASK],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        env=_env(tmp_path),
        cwd=tmp_path,
        check=False,
    )
    assert "Traceback" not in done.stderr, done.stderr
    return done.returncode, json.loads(done.stdout), done.stdout


def _answer(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def test_five_calls_stop_after_four_and_answer_on_the_fifth(
    serve: Any, tmp_path: Path
) -> None:
    server = serve({5: _answer("FINAL ANSWER: 42")})
    code, record, _ = _run(
        server, tmp_path, "agent.max_calls=5", "agent.answer_on_stop=on"
    )
    assert len(server.bodies) == 5
    assert [bool(body.get("tools")) for body in server.bodies] == [True] * 4 + [False]
    assert (code, record["status"]) == (
        2,
        "budget_exhausted",
    )  # as for any run out of budget
    assert (record["result"], record["answered_on_stop"]) == ("FINAL ANSWER: 42", True)
    assert record["error"].startswith("Call budget reached: 4 model requests sent")
    assert (record["steps"], record["n_llm"], record["n_aux"]) == (4, 5, 1)
    assert record["config"]["max_calls"] == 5


def test_a_run_that_stops_without_answer_on_stop_sends_max_calls_requests(
    serve: Any, tmp_path: Path
) -> None:
    server = serve({})
    code, record, _ = _run(server, tmp_path, "agent.max_calls=3")
    assert len(server.bodies) == 3
    assert (code, record["status"], record["result"]) == (2, "budget_exhausted", "")
    assert "answered_on_stop" not in record


def test_a_refusal_for_the_budget_gets_one_answer_request(
    serve: Any, tmp_path: Path
) -> None:
    server = serve({3: (400, BUDGET_BODY), 4: _answer("FINAL ANSWER: 8")})
    code, record, _ = _run(server, tmp_path, "agent.answer_on_stop=on")
    assert len(server.bodies) == 4
    assert not server.bodies[3].get("tools")
    assert (code, record["status"]) == (2, "provider_error")
    assert "budget_exhausted" in record["error"]
    assert (record["result"], record["answered_on_stop"]) == ("FINAL ANSWER: 8", True)
    assert (record["llm_errors"], record["n_aux"]) == (1, 1)


def test_a_refusal_ends_as_before_when_answer_on_stop_is_off(
    serve: Any, tmp_path: Path
) -> None:
    server = serve({3: (400, BUDGET_BODY), 4: _answer("never asked")})
    code, record, _ = _run(server, tmp_path)
    assert len(server.bodies) == 3
    assert (code, record["status"], record["result"]) == (2, "provider_error", "")
    assert "answered_on_stop" not in record and "max_calls" not in record["config"]


def test_a_limit_that_is_not_reached_changes_no_request(
    serve: Any, tmp_path: Path
) -> None:
    script = {3: _answer("FINAL ANSWER: 1")}
    plain, limited = serve(script), serve(script)
    code, first, out = _run(plain, tmp_path)
    code_limited, second, out_limited = _run(limited, tmp_path, "agent.max_calls=40")
    assert (code, code_limited) == (0, 0)
    assert (first["status"], first["result"]) == ("completed", "FINAL ANSWER: 1")
    assert plain.bodies == limited.bodies and len(plain.bodies) == 3
    for record in (first, second):
        for name in ("duration_s", "task_id", "usage", "config"):
            record.pop(name)
    assert first == second
    assert out.count("\n") == out_limited.count("\n") == 1
