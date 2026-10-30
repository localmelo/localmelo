"""A stub OpenAI-compatible model server with scripted replies.

For the terminal UI's tests and for screenshots; nothing leaves this
machine and no model runs. Standard library only.

Usage: ``python stub_llm.py --port 0 [--scenarios DIR]... [--scenario NAME]``.
The first line printed is ``stub listening on http://127.0.0.1:<port>/v1``.
In a test, :class:`StubServer` runs it in a thread.

A scenario is a JSON file ``<name>.json``::

    {"models": [...], "match": [...], "replies": [...], "sequential": false}

Every key is optional but ``replies``. ``GET /v1/models`` lists the
scenario's models (``["stub-model"]`` by default). With ``"omlx": [{"id",
"model_alias", "loaded"}, ...]`` the server answers as oMLX does: the listing
names each model by its alias (else its id) with ``owned_by`` ``omlx``, and
``GET /v1/models/status`` reports the ids, aliases and loaded flags. A
request picks its scenario by its path (``/s/<name>/v1/...``), else by its
newest user message (a scenario whose ``match`` holds that text), else the
default one (``--scenario``, or ``POST /__scenario {"name": ...}``). The reply
is the scenario's reply numbered by the assistant messages after that user
message, so a retried request gets the same reply; the last one repeats.
With ``"sequential": true`` the scenario counts its requests instead.

A reply has ``content``, ``reasoning`` (sent as ``reasoning_content``),
``tool_calls`` (``[{"name", "arguments"}]``), ``usage`` and ``delay_s``;
or ``status`` and ``body``, an HTTP error; or ``{"hold": true, "then":
{...}}``, which sends its headers at once, then a space every 20 s until
``POST /__release`` (at most 600 s), then the ``then`` reply.

A reply with ``stream`` is sent as incremental SSE instead of one JSON
body: ``{"chunks": [...], "reasoning": [...], "delay_s": 0.02, "slice": 0,
"drop": false}``. ``chunks`` are content deltas (the reply's ``content``
when absent), ``reasoning`` deltas go to ``reasoning_content``, and
``tool_calls`` are sent as complete fragments before the finish reason.
``slice`` writes the encoded body in pieces of that many bytes, so a test
can split UTF-8 or a tag across transport chunks. ``drop`` closes the
connection after the deltas, before the finish frame and ``[DONE]``.
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
                path = self.path.rstrip("/")
                scenario = server.scenarios.get(
                    server._scenario_for(self.path, None), {}
                )
                omlx = scenario.get("omlx")
                if omlx is not None and path.endswith("/models/status"):
                    self._json(200, {"models": omlx})
                    return
                if not path.endswith("/models"):
                    self._json(404, {"error": "not found"})
                    return
                if omlx is not None:
                    data = [
                        {"id": m.get("model_alias") or m["id"], "owned_by": "omlx"}
                        for m in omlx
                    ]
                else:
                    data = [{"id": m} for m in scenario.get("models", ["stub-model"])]
                self._json(200, {"object": "list", "data": data})

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
                if "stream" in reply:
                    self._stream(reply)
                    return
                if "status" in reply:
                    data = str(reply.get("body", "")).encode()
                    self.send_response(int(reply["status"]))
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self._json(200, _completion(reply, body))

            def _stream(self, reply: dict[str, Any]) -> None:
                stream = reply["stream"]
                body = _stream_body(reply)
                delay = float(stream.get("delay_s") or 0)
                size = int(stream.get("slice") or 0)
                # Without a byte slice, one SSE frame is one write, so the
                # delay paces the deltas a screen test sees.
                pieces = (
                    [frame + b"\n\n" for frame in body.split(b"\n\n") if frame]
                    if size <= 0
                    else [body[i : i + size] for i in range(0, len(body), size)]
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for piece in pieces:
                        self.wfile.write(piece)
                        self.wfile.flush()
                        if delay:
                            time.sleep(delay)
                except OSError:
                    pass  # the client went away, as after a stop

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


def _sse(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"


def _stream_body(reply: dict[str, Any]) -> bytes:
    """The SSE body of a streamed reply, without the finish when it drops."""
    stream = reply["stream"]
    chunks: list[bytes] = []

    def delta(value: dict[str, Any]) -> None:
        chunks.append(
            _sse({"choices": [{"index": 0, "delta": value, "finish_reason": None}]})
        )

    for text in stream.get("reasoning") or []:
        delta({"reasoning_content": text})
    for text in stream.get("chunks", [reply.get("content", "")]):
        delta({"content": text})
    calls = reply.get("tool_calls") or []
    for index, call in enumerate(calls):
        delta(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": f"call_{index + 1}",
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("arguments", {})),
                        },
                    }
                ]
            }
        )
    if stream.get("drop"):
        return b"".join(chunks)
    usage = dict(reply.get("usage") or DEFAULT_USAGE)
    usage.setdefault(
        "total_tokens",
        usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
    )
    chunks.append(_sse({"choices": [], "usage": usage}))
    chunks.append(
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if calls else "stop",
                    }
                ]
            }
        )
    )
    chunks.append(b"data: [DONE]\n\n")
    return b"".join(chunks)


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
