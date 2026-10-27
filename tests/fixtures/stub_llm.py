"""A stub OpenAI-compatible model server with scripted replies.

For the terminal UI's tests and for screenshots; nothing leaves this
machine and no model runs. Standard library only.

Usage: ``python stub_llm.py --port 0 [--scenarios DIR]... [--scenario NAME]``.
The first line printed is ``stub listening on http://127.0.0.1:<port>/v1``.
In a test, :class:`StubServer` runs it in a thread.

A scenario is a JSON file ``<name>.json``::

    {"models": [...], "match": [...], "replies": [...], "sequential": false}

Every key is optional but ``replies``. ``GET /v1/models`` lists the
scenario's models (``["stub-model"]`` by default). A request picks its
scenario by its path (``/s/<name>/v1/...``), else by its newest user
message (a scenario whose ``match`` holds that text), else the default one
(``--scenario``, or ``POST /__scenario {"name": ...}``). The reply is the
scenario's reply numbered by the assistant messages after that user
message, so a retried request gets the same reply; the last one repeats.
With ``"sequential": true`` the scenario counts its requests instead.

A reply has ``content``, ``reasoning`` (sent as ``reasoning_content``),
``tool_calls`` (``[{"name", "arguments"}]``), ``usage`` and ``delay_s``;
or ``status`` and ``body``, an HTTP error; or ``{"hold": true, "then":
{...}}``, which sends its headers at once, then a space every 20 s until
``POST /__release`` (at most 600 s), then the ``then`` reply.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SCENARIOS_DIR = Path(__file__).with_name("stub_scenarios")
DEFAULT_USAGE = {"prompt_tokens": 120, "completion_tokens": 12}
HOLD_TICK_S = 20.0
HOLD_MAX_S = 600.0


def load_scenarios(folders: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Every scenario in *folders*; the first folder that has a name wins."""
    scenarios: dict[str, dict[str, Any]] = {}
    for folder in folders:
        for path in sorted(folder.glob("*.json")):
            scenarios.setdefault(
                path.stem, json.loads(path.read_text(encoding="utf-8"))
            )
    return scenarios


def _user_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return str(content or "")


class StubServer:
    """The stub on 127.0.0.1 and a free port, in a thread."""

    def __init__(
        self,
        scenarios: dict[str, dict[str, Any]] | None = None,
        default: str = "answer",
        port: int = 0,
    ) -> None:
        self.scenarios = scenarios or load_scenarios([SCENARIOS_DIR])
        self.default = default
        self.requests: list[dict[str, Any]] = []
        self.released = threading.Event()
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}/v1"

    def start(self) -> StubServer:
        self._thread.start()
        return self

    def close(self) -> None:
        self.released.set()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    def _scenario_for(self, path: str, body: dict[str, Any] | None) -> str:
        if path.startswith("/s/"):
            name = path.split("/")[2]
            if name in self.scenarios:
                return name
        if body is not None:
            messages = body.get("messages") or []
            users = [
                m for m in messages if isinstance(m, dict) and m.get("role") == "user"
            ]
            if users:
                newest = _user_text(users[-1])
                for name, scenario in self.scenarios.items():
                    if newest in scenario.get("match", []):
                        return name
        return self.default

    def _reply(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        scenario = self.scenarios[name]
        replies = scenario["replies"]
        if scenario.get("sequential"):
            with self._lock:
                index = self._counts.get(name, 0)
                self._counts[name] = index + 1
        else:
            messages = body.get("messages") or []
            last_user = max(
                (i for i, m in enumerate(messages) if m.get("role") == "user"),
                default=-1,
            )
            index = sum(
                1 for m in messages[last_user + 1 :] if m.get("role") == "assistant"
            )
        return dict(replies[min(index, len(replies) - 1)])

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _json(self, status: int, payload: Any) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802
                if not self.path.rstrip("/").endswith("/models"):
                    self._json(404, {"error": "not found"})
                    return
                name = server._scenario_for(self.path, None)
                models = server.scenarios.get(name, {}).get("models", ["stub-model"])
                self._json(200, {"object": "list", "data": [{"id": m} for m in models]})

            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.loads(raw or b"{}")
                if self.path == "/__release":
                    server.released.set()
                    self._json(200, {})
                    return
                if self.path == "/__scenario":
                    server.default = body["name"]
                    server._counts.pop(server.default, None)
                    self._json(200, {})
                    return
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self._json(404, {"error": "not found"})
                    return
                server.requests.append(body)
                reply = server._reply(server._scenario_for(self.path, body), body)
                if reply.get("hold"):
                    self._hold(reply.get("then") or {})
                    return
                time.sleep(float(reply.get("delay_s") or 0))
                if "status" in reply:
                    data = str(reply.get("body", "")).encode()
                    self.send_response(int(reply["status"]))
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self._json(200, _completion(reply, body))

            def _hold(self, then: dict[str, Any]) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                started = time.monotonic()
                try:
                    while not server.released.wait(HOLD_TICK_S):
                        if time.monotonic() - started > HOLD_MAX_S:
                            break
                        self.wfile.write(b" ")
                        self.wfile.flush()
                    self.wfile.write(json.dumps(_completion(then, {})).encode())
                except OSError:
                    pass  # the client went away, as after a stop

        return Handler


def _completion(reply: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": reply.get("content", "")}
    if reply.get("reasoning"):
        message["reasoning_content"] = reply["reasoning"]
    calls = reply.get("tool_calls") or []
    if calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c.get("arguments", {})),
                },
            }
            for i, c in enumerate(calls, 1)
        ]
    usage = dict(reply.get("usage") or DEFAULT_USAGE)
    usage.setdefault(
        "total_tokens",
        usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
    )
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "stub-model"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ],
        "usage": usage,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--scenarios", type=Path, action="append", default=[])
    parser.add_argument("--scenario", default="answer")
    args = parser.parse_args(argv)
    folders = [*args.scenarios, SCENARIOS_DIR]
    server = StubServer(load_scenarios(folders), default=args.scenario, port=args.port)
    print(f"stub listening on {server.base_url}", flush=True)
    try:
        server._httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server._httpd.server_close()
    sys.exit(0)


if __name__ == "__main__":
    main()
