"""Fake stdio MCP server for the localmelo tests (stdlib only).

It speaks newline-delimited JSON-RPC 2.0. Tool calls run on worker
threads, so the main loop keeps reading stdin while a call is in flight;
that is what lets the tests exercise cancellation and a server-to-client
ping. Every message received is appended (one JSON per line) to the file
named by FAKE_MCP_LOG. FAKE_MCP_MODE selects misbehaviour:

- ``hang_initialize``: never answer ``initialize``
- ``garbage_stdout``: write a non-JSON line before every response
- ``stubborn``: ignore SIGTERM and keep running after stdin closes
"""

from __future__ import annotations

import base64
import json
import os
import signal
import sys
import threading
import time
from typing import Any

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kg"
    "AAAABJRU5ErkJggg=="
)
PAGE_SIZE = 4
SLOW_CAP_S = 3.0  # hard cap, so a broken cancel path cannot hang the suite
MODE = os.environ.get("FAKE_MCP_MODE", "")
LOG_PATH = os.environ.get("FAKE_MCP_LOG", "")

_write_lock = threading.Lock()
_log_lock = threading.Lock()
_cancelled: set[Any] = set()
_client_replies: dict[str, dict[str, Any]] = {}
_reply_events: dict[str, threading.Event] = {}


def _schema(**props: str) -> dict[str, Any]:
    return {"type": "object", "properties": {k: {"type": v} for k, v in props.items()}}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "echo",
        "description": "Echo text.",
        "inputSchema": _schema(text="string"),
    },
    {
        "name": "screenshot",
        "description": "Capture the screen.",
        "inputSchema": _schema(),
    },
    {"name": "fail", "description": "Report a tool error.", "inputSchema": _schema()},
    {
        "name": "slow",
        "description": "Sleep first.",
        "inputSchema": _schema(seconds="number"),
    },
    {
        "name": "crash",
        "description": "Exit without answering.",
        "inputSchema": _schema(),
    },
    {
        "name": "ping_client",
        "description": "Ping the client.",
        "inputSchema": _schema(),
    },
    {
        "name": "env",
        "description": "Read an env var.",
        "inputSchema": _schema(name="string"),
    },
    {"name": "cwd", "description": "Report the cwd.", "inputSchema": _schema()},
    {
        "name": "big_image",
        "description": "Large image.",
        "inputSchema": _schema(size="integer"),
    },
    {"name": "structured", "description": "Structured only.", "inputSchema": _schema()},
    {"name": "read.file", "description": "Name with a dot.", "inputSchema": _schema()},
]


def send(message: dict[str, Any]) -> None:
    data = (json.dumps(message) + "\n").encode()
    with _write_lock:
        if MODE == "garbage_stdout":
            sys.stdout.buffer.write(b"this is not JSON\n")
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()


def log(message: dict[str, Any]) -> None:
    if LOG_PATH:
        with _log_lock, open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(message) + "\n")


def text_result(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def call_tool(request_id: Any, name: str, args: dict[str, Any]) -> None:
    if name == "echo":
        result = text_result(str(args.get("text", "")))
    elif name == "screenshot":
        result = {
            "content": [
                {"type": "text", "text": "screen captured"},
                {
                    "type": "image",
                    "data": base64.b64encode(PNG_1X1).decode(),
                    "mimeType": "image/png",
                },
            ]
        }
    elif name == "fail":
        result = {
            "content": [{"type": "text", "text": "element not found"}],
            "isError": True,
        }
    elif name == "slow":
        deadline = time.monotonic() + min(float(args.get("seconds", 1)), SLOW_CAP_S)
        while time.monotonic() < deadline:
            if request_id in _cancelled:
                return  # a cancelled request gets no response
            time.sleep(0.01)
        result = text_result("slept")
    elif name == "crash":
        os._exit(3)
    elif name == "ping_client":
        event = threading.Event()
        _reply_events["srv-1"] = event
        send({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})
        event.wait(SLOW_CAP_S)
        reply = _client_replies.get("srv-1", {})
        result = text_result("pong ok" if reply.get("result") == {} else "no pong")
    elif name == "env":
        result = text_result(os.environ.get(str(args.get("name")), "<unset>"))
    elif name == "cwd":
        result = text_result(os.getcwd())
    elif name == "big_image":
        blob = os.urandom(int(args.get("size", 1_000_000)))
        result = {
            "content": [
                {
                    "type": "image",
                    "data": base64.b64encode(blob).decode(),
                    "mimeType": "image/png",
                }
            ]
        }
    elif name == "structured":
        result = {"content": [], "structuredContent": {"answer": 42}}
    elif name == "read.file":
        result = text_result("dotted name ok")
    else:
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": f"Unknown tool: {name}"},
            }
        )
        return
    send({"jsonrpc": "2.0", "id": request_id, "result": result})


def handle(message: dict[str, Any]) -> None:
    method = message.get("method")
    if method is None:  # a response to our ping
        key = str(message.get("id"))
        _client_replies[key] = message
        event = _reply_events.get(key)
        if event is not None:
            event.set()
        return
    params = message.get("params") or {}
    if method == "initialize":
        if MODE == "hang_initialize":
            return
        send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-mcp", "version": "1.0"},
                },
            }
        )
    elif method == "notifications/cancelled":
        _cancelled.add(params.get("requestId"))
    elif method == "tools/list":
        start = int(params.get("cursor") or 0)
        page: dict[str, Any] = {"tools": TOOLS[start : start + PAGE_SIZE]}
        if start + PAGE_SIZE < len(TOOLS):
            page["nextCursor"] = str(start + PAGE_SIZE)
        send({"jsonrpc": "2.0", "id": message["id"], "result": page})
    elif method == "tools/call":
        threading.Thread(
            target=call_tool,
            args=(message["id"], params.get("name"), params.get("arguments") or {}),
            daemon=True,
        ).start()
    elif "id" in message:
        send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32601, "message": f"Unknown method {method}"},
            }
        )


def main() -> None:
    if MODE == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    # More than a pipe buffer of stderr before reading anything: the client
    # must drain stderr or the handshake deadlocks.
    sys.stderr.write("fake-mcp: starting\n" + "." * 200_000 + "\n")
    sys.stderr.flush()
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            break
        if not line.strip():
            continue
        message = json.loads(line)
        log(message)
        handle(message)
    if MODE == "stubborn":
        time.sleep(10)


if __name__ == "__main__":
    main()
