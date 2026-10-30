"""Fake stdio MCP server for the localmelo tests (stdlib only).

It speaks newline-delimited JSON-RPC 2.0. Tool calls run on worker
threads, so the main loop keeps reading stdin while a call is in flight;
that is what lets the tests exercise cancellation and a server-to-client
ping. Every message received is appended (one JSON per line) to the file
named by FAKE_MCP_LOG, and the server writes its pid to the file named by
FAKE_MCP_PIDFILE, if set. FAKE_MCP_MODE selects misbehaviour:

- ``hang_initialize``: never answer ``initialize``
- ``hang_tools_list``: never answer ``tools/list``
- ``prompts_only``: declare the ``prompts`` capability instead of
  ``tools`` and answer ``tools/list`` with METHOD_NOT_FOUND, like a
  server that offers no tools
- ``no_tools_list``: declare the ``tools`` capability but answer
  ``tools/list`` with METHOD_NOT_FOUND
- ``tools_list_error``: answer ``tools/list`` with an internal error
- ``garbage_stdout``: write a non-JSON line before every response
- ``malformed_stdout``: write a line that is not UTF-8 and a line of JSON
  nested too deep to parse before every response
- ``bom_stdout``: start stdout with a UTF-8 byte order mark, so the first
  message, the ``initialize`` response, begins with one
- ``stubborn``: ignore SIGTERM and keep running after stdin closes
- ``stubborn_helper``: as ``stubborn``, and first start a helper process
  that holds stdout and stderr open from a session of its own; its pid
  goes to the file named by FAKE_MCP_HELPER_PIDFILE
- ``stubborn_hang_initialize``: as ``stubborn``, and never answer
  ``initialize``
"""

from __future__ import annotations

import base64
import codecs
import json
import os
import signal
import subprocess
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
PID_PATH = os.environ.get("FAKE_MCP_PIDFILE", "")
HELPER_PID_PATH = os.environ.get("FAKE_MCP_HELPER_PIDFILE", "")
STUBBORN_MODES = ("stubborn", "stubborn_helper", "stubborn_hang_initialize")
HANG_INITIALIZE_MODES = ("hang_initialize", "stubborn_hang_initialize")

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
    {
        "name": "cwd",
        "description": "Report the cwd.",
        "inputSchema": _schema(),
        "annotations": {"readOnlyHint": True, "destructiveHint": False},
    },
    {
        "name": "big_image",
        "description": "Large image.",
        "inputSchema": _schema(size="integer"),
    },
    {"name": "structured", "description": "Structured only.", "inputSchema": _schema()},
    {"name": "read.file", "description": "Name with a dot.", "inputSchema": _schema()},
    {
        "name": "close_stdout",
        "description": "Close stdout and keep running.",
        "inputSchema": _schema(),
    },
    {
        "name": "latin1",
        "description": "Text that is not UTF-8.",
        "inputSchema": _schema(),
    },
]


def send(message: dict[str, Any]) -> None:
    data = (json.dumps(message) + "\n").encode()
    with _write_lock:
        if MODE == "garbage_stdout":
            sys.stdout.buffer.write(b"this is not JSON\n")
        elif MODE == "malformed_stdout":
            sys.stdout.buffer.write(b"caf\xe9 log line\n")  # Latin-1
            sys.stdout.buffer.write(b"[" * 200_000 + b"\n")
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
        shot = base64.b64encode(PNG_1X1).decode()
        image = {"type": "image", "data": shot, "mimeType": "image/png"}
        result = {"content": [{"type": "text", "text": "screen captured"}, image]}
        if args.get("structured"):
            # As browser servers do: no text, the base64 again in structuredContent.
            result = {"content": [image], "structuredContent": {"screenshot": shot}}
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
    elif name == "latin1":
        # The response is encoded as Latin-1, not UTF-8.
        message = {"jsonrpc": "2.0", "id": request_id, "result": text_result("caf\xe9")}
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("latin-1")
        with _write_lock:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        return
    elif name == "close_stdout":
        # stdout becomes /dev/null: the client reads EOF while this process
        # keeps running, and this call gets no response.
        with _write_lock:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            os.close(devnull)
        return
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
        if MODE in HANG_INITIALIZE_MODES:
            return
        capabilities: dict[str, Any] = (
            {"prompts": {}} if MODE == "prompts_only" else {"tools": {}}
        )
        send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                    "capabilities": capabilities,
                    "serverInfo": {"name": "fake-mcp", "version": "1.0"},
                },
            }
        )
    elif method == "notifications/cancelled":
        _cancelled.add(params.get("requestId"))
    elif method == "tools/list" and MODE not in ("prompts_only", "no_tools_list"):
        if MODE == "hang_tools_list":
            return
        if MODE == "tools_list_error":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32603, "message": "Internal error"},
                }
            )
            return
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
        # Also tools/list in prompts_only and no_tools_list mode.
        send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32601, "message": f"Unknown method {method}"},
            }
        )


def start_helper() -> None:
    # The helper inherits stdout and stderr, the pipes to the client, and
    # its own session keeps it out of the process group the client signals.
    helper = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(10)"],
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    with open(HELPER_PID_PATH, "w", encoding="utf-8") as fh:
        fh.write(str(helper.pid))


def main() -> None:
    if MODE == "stubborn_helper":
        start_helper()
    if MODE in STUBBORN_MODES:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if PID_PATH:
        with open(PID_PATH, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
    # 200 KB of stderr before reading anything. The client's asyncio reads
    # a pipe into a buffer of about twice its stream limit whether or not
    # the client reads that buffer, so this blocks the server only when a
    # test lowers the limit and the client does not drain stderr.
    sys.stderr.write("fake-mcp: starting\n" + "x" * 200_000 + "\n")
    sys.stderr.flush()
    if MODE == "bom_stdout":
        # No newline: the next write, the first message, follows the mark.
        sys.stdout.buffer.write(codecs.BOM_UTF8)
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            break
        if not line.strip():
            continue
        message = json.loads(line)
        log(message)
        handle(message)
    if MODE in STUBBORN_MODES:
        time.sleep(10)


if __name__ == "__main__":
    main()
