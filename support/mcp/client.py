"""A small stdio MCP client: JSON-RPC 2.0, one message per line.

It covers what an agent needs from an MCP server: the ``initialize``
handshake, ``tools/list`` (with pagination) and ``tools/call``. While a
request is pending it answers the server's ``ping`` requests and declines
other server-to-client requests (sampling, roots, elicitation). A
cancelled request (for example by the executor's timeout) is announced
with ``notifications/cancelled``.

The server runs as a child process with a minimal environment: the
variables named in ``DEFAULT_ENV_VARS`` that are set in this process
(``PATH``, ``HOME``, user, shell, terminal, locale and ``TMPDIR``), then
the server's configured ``env`` on top, as literal values. Nothing else
is inherited, so API keys and other secrets in the agent's environment
do not reach MCP servers; a server that needs another variable must
list it in its ``env``.

On POSIX the server runs in a session of its own, and close() signals
its whole process group, so a server started through a wrapper
(``sh -c``, a launcher script) is stopped together with the wrapper. A
Ctrl-C in the agent's terminal does not reach the server.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any

from localmelo.melo.checker.validators import (
    MAX_IMAGE_BYTES,
    MAX_IMAGES_PER_RESULT,
    MAX_OUTPUT_LEN,
)
from localmelo.support.mcp.config import McpServerConfig

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
KNOWN_PROTOCOL_VERSIONS = frozenset({"2024-11-05", "2025-03-26", "2025-06-18"})

# One message is one line, with any images in it as base64, and a longer
# line ends the client. The limit holds the largest tools/call result the
# executor's result check passes unchanged: MAX_IMAGES_PER_RESULT images
# of MAX_IMAGE_BYTES each, base64-encoded, and MAX_OUTPUT_LEN characters
# of text at up to 12 bytes each as JSON (two \u escapes), plus 1 MiB for
# the JSON around them. About 215 MiB; asyncio's default is 64 KiB.
STREAM_LIMIT = (
    MAX_IMAGES_PER_RESULT * 4 * ((MAX_IMAGE_BYTES + 2) // 3)
    + 12 * MAX_OUTPUT_LEN
    + 1024 * 1024
)
# Each line of the server's stderr is cut to its last STDERR_LINE_CHARS
# characters, after "..." if its start was dropped. That text goes to the
# debug log and to stderr_tail, which keeps the last STDERR_TAIL_LINES.
STDERR_TAIL_LINES = 200
STDERR_LINE_CHARS = 1000
STDERR_LINE_BYTES = 4 * STDERR_LINE_CHARS  # UTF-8: at most 4 bytes a character
MAX_TOOL_PAGES = 100
# How long the stdout reader waits for the exit code once the server's
# output has ended.
EXIT_WAIT_S = 1.0
# How often close() checks whether the server's process group is gone.
GROUP_POLL_S = 0.05

METHOD_NOT_FOUND = -32601

# Variables a server inherits from this process: the official MCP SDK's
# POSIX defaults plus the locale and TMPDIR. Anything else must be set in
# the server's ``env``.
DEFAULT_ENV_VARS = (
    "HOME",
    "LOGNAME",
    "PATH",
    "SHELL",
    "TERM",
    "USER",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
)


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
        # close() has begun: requests are refused, and start() refuses to
        # run, from then on.
        self._closing = False
        # close() has shut a process down (process gone, readers stopped):
        # later calls do nothing. A close() before there is a process only
        # sets _closing.
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
        """Spawn the server and complete the ``initialize`` handshake.

        Once close() has begun, start() raises McpError instead. A close()
        that comes while the process is being spawned finds no process to
        stop, so start() stops the new process itself, without the
        handshake, and raises McpError.
        """
        if self._closing:
            raise McpError(
                f"Cannot start MCP server {self.name!r}: the client is closed"
            )
        if self._proc is not None:
            return
        env = {k: os.environ[k] for k in DEFAULT_ENV_VARS if k in os.environ}
        env.update(self.config.env)
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
                # Its own process group, which close() signals as a whole.
                start_new_session=sys.platform != "win32",
            )
        except (OSError, ValueError) as exc:
            # ValueError: a string the OS cannot take (a NUL, "=" in an env
            # name, text that cannot be encoded) in a config built in code,
            # which parse_mcp_config did not check.
            raise McpError(
                f"Cannot start MCP server {self.name!r} "
                f"({self.config.command}): {exc}"
            ) from exc
        if self._closing:
            # close() ran during the spawn and had no process to stop. If
            # this shutdown is cancelled, _proc is set and _closed is not,
            # so the next close() finishes it.
            await self.close()
            raise McpError(
                f"Cannot start MCP server {self.name!r}: the client was closed "
                "during start-up"
            )
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
        """Shut the server down: close stdin, then terminate, then kill.

        If the server has not exited within the grace period after its
        stdin is closed, it gets SIGTERM, and if anything is left running a
        grace period later, SIGKILL. On POSIX both go to the server's
        process group, so a server behind a wrapper is stopped too; on
        Windows only the process is terminated. A process still running a
        grace period after that makes close() raise McpError, which says
        whether the signals were refused (permission denied, as for a
        server started through sudo) or SIGKILL had no effect.

        Requests are refused as soon as close() begins, but the client only
        counts as closed once the process has exited and the readers have
        stopped. A close() cancelled part-way therefore leaves the shutdown
        to the next call, which finishes it; after that, close() does
        nothing.

        Before there is a process, close() only marks the client as closing
        and returns: start() then refuses to run, and a start() that is
        spawning the server stops the new process itself.
        """
        if self._closed:
            return
        self._closing = True
        proc = self._proc
        if proc is None:
            return
        if proc.returncode is None:
            if proc.stdin is not None:
                with contextlib.suppress(Exception):
                    proc.stdin.close()
            if not await self._wait_exit(proc):
                refused = _signal_server(proc, kill=False)
                if not await self._wait_stopped(proc):
                    refused = _signal_server(proc, kill=True)
                # A process that left the group can hold the pipes open, and
                # before Python 3.13 Process.wait() waits for them to close.
                _close_pipes(proc)
                if not await self._wait_exit(proc):
                    state = (
                        "could not be signalled (permission denied) and is still "
                        "running"
                        if refused
                        else "is still running after SIGKILL"
                    )
                    raise McpError(
                        f"Cannot stop MCP server {self.name!r}: process "
                        f"{proc.pid} {state}"
                    )
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._fail_pending(McpError(f"MCP client for {self.name!r} is closed"))
        self._closed = True

    async def _wait_exit(self, proc: asyncio.subprocess.Process) -> bool:
        try:
            await asyncio.wait_for(proc.wait(), self._shutdown_grace_s)
        except TimeoutError:
            return False
        return True

    async def _wait_stopped(self, proc: asyncio.subprocess.Process) -> bool:
        """Wait up to the grace period for the server to stop.

        On POSIX that means every process in its group, not only the one
        started: a wrapper often dies of SIGTERM while the server goes on.
        """
        if sys.platform == "win32":
            return await self._wait_exit(proc)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._shutdown_grace_s
        while True:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass  # zombies not yet reaped (macOS), or a setuid process
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(GROUP_POLL_S)

    # ── Requests ──

    async def list_tools(self) -> list[McpTool]:
        """All tools the server offers, following ``nextCursor`` pages."""
        tools: list[McpTool] = []
        cursor: str | None = None
        for _ in range(MAX_TOOL_PAGES):
            params = {"cursor": cursor} if cursor else {}
            try:
                result = await asyncio.wait_for(
                    self._request("tools/list", params),
                    timeout=self.config.startup_timeout_s,
                )
            except TimeoutError:
                raise McpError(
                    f"MCP server {self.name!r} did not answer tools/list within "
                    f"{self.config.startup_timeout_s:g}s{self._stderr_hint()}"
                ) from None
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
        if self._closing or self._proc is None:
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
                try:
                    self._dispatch(line)
                except Exception:
                    # One message the client cannot handle must not end the
                    # reader: the server is still there.
                    logger.warning(
                        "MCP server %r: a message could not be handled; skipped",
                        self.name,
                        exc_info=True,
                    )
        finally:
            if error is None:
                code = await self._returncode_soon(proc)
                if code is None:
                    error = McpError(
                        f"MCP server {self.name!r} stopped responding: its output "
                        f"ended while the process is still running"
                        f"{self._stderr_hint()}"
                    )
                else:
                    error = McpError(
                        f"MCP server {self.name!r} exited (code {code})"
                        f"{self._stderr_hint()}"
                    )
            self._exit_error = error
            self._fail_pending(error)

    async def _returncode_soon(self, proc: asyncio.subprocess.Process) -> int | None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), EXIT_WAIT_S)
        return proc.returncode

    async def _read_stderr(self) -> None:
        # Chunks, not lines, so that a huge log line is never held whole:
        # only the end of each line is kept.
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        partial = b""  # the line still being read
        cut = False  # whether the start of that line was dropped
        while True:
            chunk = await proc.stderr.read(64 * 1024)
            if not chunk:
                if partial:
                    self._add_stderr_line(partial, cut)
                return
            *lines, rest = (partial + chunk).split(b"\n")
            for line in lines:
                self._add_stderr_line(line, cut)
                cut = False
            cut = cut or len(rest) > STDERR_LINE_BYTES
            partial = rest[-STDERR_LINE_BYTES:]

    def _add_stderr_line(self, line: bytes, cut: bool) -> None:
        """Log the end of a stderr line and keep it in stderr_tail.

        *cut* says the reader already dropped the start of the line.
        """
        cut = cut or len(line) > STDERR_LINE_BYTES
        text = line[-STDERR_LINE_BYTES:].decode("utf-8", errors="replace").rstrip()
        if cut or len(text) > STDERR_LINE_CHARS:
            text = "..." + text[-STDERR_LINE_CHARS:]
        self.stderr_tail.append(text)
        logger.debug("[mcp %s] %s", self.name, text)

    def _dispatch(self, line: bytes) -> None:
        # errors="replace": text that is not UTF-8 (a stray Latin-1 log line)
        # must not end the reader. utf-8-sig drops a byte order mark at the
        # start of the line, as json.loads does when given bytes.
        text = line.decode("utf-8-sig", errors="replace")
        try:
            message = json.loads(text)
        except (ValueError, RecursionError) as exc:
            # RecursionError: JSON nested deeper than the parser can follow.
            logger.debug(
                "MCP server %r wrote a line that is not a JSON message (%s); ignored",
                self.name,
                exc,
            )
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


def _signal_server(proc: asyncio.subprocess.Process, *, kill: bool) -> bool:
    """Send SIGTERM, or SIGKILL with *kill*, to the server's process group.

    The server leads a session of its own, so its pid is the group id; the
    group lives on while any of its processes does, even once the server
    has exited. On Windows the process is terminated instead. Returns
    whether the signal was refused (permission denied).
    """
    if sys.platform == "win32":
        with contextlib.suppress(ProcessLookupError):
            if kill:
                proc.kill()
            else:
                proc.terminate()
        return False
    try:
        os.killpg(proc.pid, signal.SIGKILL if kill else signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS refuses to signal a group of zombies, and a setuid process
        # may refuse the signal.
        return True
    return False


def _close_pipes(proc: asyncio.subprocess.Process) -> None:
    """Close the pipes to the server; asyncio has no public API for them."""
    transport = getattr(proc, "_transport", None)
    if isinstance(transport, asyncio.SubprocessTransport):
        for fd in (0, 1, 2):
            pipe = transport.get_pipe_transport(fd)
            if pipe is not None:
                pipe.close()


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
