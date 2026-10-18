"""A small stdio MCP client: JSON-RPC 2.0, one message per line.

It covers what an agent needs from an MCP server: the ``initialize``
handshake, ``tools/list`` (with pagination) and ``tools/call``. While a
request is pending it answers the server's ``ping`` requests and declines
other server-to-client requests (sampling, roots, elicitation). A
cancelled request (for example by the executor's timeout) is announced
with ``notifications/cancelled``.

The server runs as a child process. It inherits this process's
environment plus the configured ``env``, so wrappers such as the
benchmark's stdio tap see the same PATH and variables as the harness
(the official MCP SDK passes only a minimal default environment instead).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Any

from localmelo.support.mcp.config import McpServerConfig

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
KNOWN_PROTOCOL_VERSIONS = frozenset({"2024-11-05", "2025-03-26", "2025-06-18"})

# Base64 screenshots arrive as one JSON line; asyncio's default line limit
# is 64 KiB.
STREAM_LIMIT = 64 * 1024 * 1024
STDERR_TAIL_LINES = 200
MAX_TOOL_PAGES = 100

METHOD_NOT_FOUND = -32601


class McpError(Exception):
    """A failed MCP request, a JSON-RPC error, or a dead server."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class McpTool:
    """A tool as listed by the server (original, un-namespaced name)."""

    name: str
    description: str
    input_schema: dict[str, Any]


class StdioMcpClient:
    """Client for one stdio MCP server process."""

    def __init__(
        self,
        config: McpServerConfig,
        *,
        client_name: str = "localmelo",
        client_version: str = "0.1.0",
        shutdown_grace_s: float = 2.0,
    ) -> None:
        self.config = config
        self._client_info = {"name": client_name, "version": client_version}
        self._shutdown_grace_s = shutdown_grace_s
        self._proc: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 1
        self._write_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._exit_error: McpError | None = None
        self._closed = False
        self.stderr_tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        self.protocol_version = ""
        self.server_info: dict[str, Any] = {}
        self.server_capabilities: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode if self._proc is not None else None

    # ── Lifecycle ──

    async def start(self) -> None:
        """Spawn the server and complete the ``initialize`` handshake."""
        if self._proc is not None:
            return
        env = {**os.environ, **self.config.env}
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.cwd,
                limit=STREAM_LIMIT,
            )
        except OSError as exc:
            raise McpError(
                f"Cannot start MCP server {self.name!r} "
                f"({self.config.command}): {exc}"
            ) from exc
        self._spawn(self._read_stdout())
        self._spawn(self._read_stderr())

        try:
            result = await asyncio.wait_for(
                self._request(
                    "initialize",
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": self._client_info,
                    },
                ),
                timeout=self.config.startup_timeout_s,
            )
            if not isinstance(result, dict):
                raise McpError(f"MCP server {self.name!r}: invalid initialize result")
            self.protocol_version = str(result.get("protocolVersion", ""))
            if self.protocol_version not in KNOWN_PROTOCOL_VERSIONS:
                logger.warning(
                    "MCP server %r uses protocol version %r; continuing",
                    self.name,
                    self.protocol_version,
                )
            info = result.get("serverInfo")
            self.server_info = info if isinstance(info, dict) else {}
            caps = result.get("capabilities")
            self.server_capabilities = caps if isinstance(caps, dict) else {}
            await self._notify("notifications/initialized")
        except TimeoutError:
            await self.close()
            raise McpError(
                f"MCP server {self.name!r} did not answer initialize within "
                f"{self.config.startup_timeout_s:g}s{self._stderr_hint()}"
            ) from None
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        """Shut the server down: close stdin, then terminate, then kill."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is not None and proc.returncode is None:
            if proc.stdin is not None:
                with contextlib.suppress(Exception):
                    proc.stdin.close()
            if not await self._wait_exit(proc):
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
                if not await self._wait_exit(proc):
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    await proc.wait()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._fail_pending(McpError(f"MCP client for {self.name!r} is closed"))

    async def _wait_exit(self, proc: asyncio.subprocess.Process) -> bool:
        try:
            await asyncio.wait_for(proc.wait(), self._shutdown_grace_s)
        except TimeoutError:
            return False
        return True

    # ── Requests ──

    async def list_tools(self) -> list[McpTool]:
        """All tools the server offers, following ``nextCursor`` pages."""
        tools: list[McpTool] = []
        cursor: str | None = None
        for _ in range(MAX_TOOL_PAGES):
            params = {"cursor": cursor} if cursor else {}
            result = await asyncio.wait_for(
                self._request("tools/list", params),
                timeout=self.config.startup_timeout_s,
            )
            if not isinstance(result, dict):
                raise McpError(f"MCP server {self.name!r}: invalid tools/list result")
            for item in result.get("tools") or []:
                tool = _parse_tool(item)
                if tool is not None:
                    tools.append(tool)
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                return tools
            cursor = next_cursor
        raise McpError(
            f"MCP server {self.name!r}: tools/list did not end after "
            f"{MAX_TOOL_PAGES} pages"
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run ``tools/call`` and return the raw result object.

        There is no timeout here: the executor bounds the call, and a
        cancelled call is announced to the server.
        """
        result = await self._request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        if not isinstance(result, dict):
            raise McpError(f"MCP server {self.name!r}: invalid tools/call result")
        return result

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        self._check_usable()
        request_id = self._next_id
        self._next_id += 1
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
            return await future
        except asyncio.CancelledError:
            # The spec forbids cancelling initialize.
            if method != "initialize":
                self._send_nowait(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/cancelled",
                        "params": {"requestId": request_id, "reason": "cancelled"},
                    }
                )
            raise
        finally:
            self._pending.pop(request_id, None)

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        await self._send(message)

    def _check_usable(self) -> None:
        if self._exit_error is not None:
            raise McpError(str(self._exit_error))
        if self._closed or self._proc is None:
            raise McpError(f"MCP server {self.name!r} is not running")

    # ── Wire ──

    @staticmethod
    def _encode(message: dict[str, Any]) -> bytes:
        # json.dumps escapes newlines inside strings, so one message is
        # exactly one line, as the stdio transport requires.
        return (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")

    async def _send(self, message: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise McpError(f"MCP server {self.name!r} is not running")
        try:
            async with self._write_lock:
                proc.stdin.write(self._encode(message))
                await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            reason = (
                str(self._exit_error)
                if self._exit_error is not None
                else f"MCP server {self.name!r} closed its input{self._stderr_hint()}"
            )
            raise McpError(reason) from exc

    def _send_nowait(self, message: dict[str, Any]) -> None:
        """Best-effort write that never waits (used while being cancelled)."""
        proc = self._proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            return
        with contextlib.suppress(Exception):
            proc.stdin.write(self._encode(message))

    def _spawn(self, coro: Any) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _read_stdout(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        error: McpError | None = None
        try:
            while True:
                try:
                    line = await proc.stdout.readline()
                except ValueError:
                    error = McpError(
                        f"MCP server {self.name!r} sent a message larger than "
                        f"{STREAM_LIMIT} bytes"
                    )
                    break
                if not line:
                    break
                self._dispatch(line)
        finally:
            if error is None:
                code = await self._returncode_soon(proc)
                error = McpError(
                    f"MCP server {self.name!r} exited (code {code})"
                    f"{self._stderr_hint()}"
                )
            self._exit_error = error
            self._fail_pending(error)

    async def _returncode_soon(self, proc: asyncio.subprocess.Process) -> int | None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), 1.0)
        return proc.returncode

    async def _read_stderr(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        while True:
            try:
                line = await proc.stderr.readline()
            except ValueError:
                continue  # an over-long log line; keep draining
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip()
            self.stderr_tail.append(text)
            logger.debug("[mcp %s] %s", self.name, text)

    def _dispatch(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("MCP server %r wrote a non-JSON line; ignored", self.name)
            return
        for item in message if isinstance(message, list) else [message]:
            if isinstance(item, dict):
                self._dispatch_one(item)

    def _dispatch_one(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if isinstance(method, str):
            if "id" in message:
                self._answer_server_request(message["id"], method)
            elif method == "notifications/message":
                logger.debug("[mcp %s] %s", self.name, message.get("params"))
            return
        request_id = message.get("id")
        future = self._pending.get(request_id) if isinstance(request_id, int) else None
        if future is None or future.done():
            return
        error = message.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            future.set_exception(
                McpError(
                    f"MCP server {self.name!r}: {error.get('message', 'error')}",
                    code=code if isinstance(code, int) else None,
                )
            )
        else:
            future.set_result(message.get("result"))

    def _answer_server_request(self, request_id: Any, method: str) -> None:
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
        if method == "ping":
            reply["result"] = {}
        else:
            reply["error"] = {
                "code": METHOD_NOT_FOUND,
                "message": f"Client does not support {method}",
            }
        # Sent from a separate task so the reader never blocks on stdin.
        self._spawn(self._send_quietly(reply))

    async def _send_quietly(self, message: dict[str, Any]) -> None:
        with contextlib.suppress(McpError):
            await self._send(message)

    def _fail_pending(self, error: McpError) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(McpError(str(error), code=error.code))

    def _stderr_hint(self) -> str:
        if not self.stderr_tail:
            return ""
        tail = " | ".join(list(self.stderr_tail)[-5:])
        return f"; stderr: {tail[-500:]}"


def _parse_tool(item: Any) -> McpTool | None:
    if not isinstance(item, dict) or not isinstance(item.get("name"), str):
        return None
    schema = item.get("inputSchema")
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    description = item.get("description")
    return McpTool(
        name=item["name"],
        description=description if isinstance(description, str) else "",
        input_schema=schema,
    )
