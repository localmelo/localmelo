"""Expose stdio MCP servers to the agent as a tool provider."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool, ToolFunction
from localmelo.melo.executor.models import ToolOutput
from localmelo.melo.schema import ToolDef
from localmelo.support.mcp.client import (
    METHOD_NOT_FOUND,
    McpError,
    McpTool,
    StdioMcpClient,
)
from localmelo.support.mcp.config import McpServerConfig, sanitize_name
from localmelo.support.mcp.results import tool_result_to_output

logger = logging.getLogger(__name__)

# OpenAI function names are limited to 64 characters.
MAX_TOOL_NAME_LEN = 64


def exposed_tool_name(server: McpServerConfig, tool_name: str) -> str:
    """``mcp__<server>__<tool>``, sanitized and capped at 64 characters."""
    name = f"mcp__{server.namespace}__{sanitize_name(tool_name)}"
    if len(name) <= MAX_TOOL_NAME_LEN:
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    return f"{name[: MAX_TOOL_NAME_LEN - 9]}_{digest}"


def _requires_approval(server: McpServerConfig, tool_name: str) -> bool:
    """Whether calls of the server's MCP tool *tool_name* ask for approval."""
    if isinstance(server.require_approval, bool):
        return server.require_approval
    return tool_name in server.require_approval


def _bind(client: StdioMcpClient, tool_name: str) -> ToolFunction:
    async def call(**arguments: Any) -> ToolOutput:
        return tool_result_to_output(await client.call_tool(tool_name, arguments))

    return call


async def _list_tools(client: StdioMcpClient) -> list[McpTool]:
    """The tools of a started server; none for a server without tools."""
    if "tools" not in client.server_capabilities:
        logger.warning(
            "MCP server %r offers no tools: it does not declare the tools "
            "capability",
            client.name,
        )
        return []
    try:
        return await client.list_tools()
    except McpError as exc:
        if exc.code != METHOD_NOT_FOUND:
            raise
        logger.warning(
            "MCP server %r offers no tools: it declares the tools capability "
            "but has no tools/list method",
            client.name,
        )
        return []


class McpToolProvider(BaseToolProvider):
    """Tools of one or more stdio MCP servers.

    Each server's tools are registered as ``mcp__<server>__<tool>`` with the
    server's input schema and description. A call goes to ``tools/call``
    with the original tool name; the executor bounds it with the server's
    ``timeout``. ``include_tools`` restricts which tools are registered.
    ``require_approval`` marks the tools whose calls ask the run's approval
    policy, when it has one, before they reach the server
    (``ProvidedTool.requires_approval``): every tool of the server, or the
    ones it names. A name in either list that the server does not offer is
    logged as a warning.

    A server without tools adds none, with a warning: one that does not
    declare the ``tools`` capability is not asked for them, and one that
    answers ``tools/list`` with METHOD_NOT_FOUND has none. Any other error
    from a server fails start().
    """

    def __init__(
        self,
        servers: Sequence[McpServerConfig],
        *,
        client_factory: Callable[[McpServerConfig], StdioMcpClient] = StdioMcpClient,
    ) -> None:
        self._servers = list(servers)
        self._client_factory = client_factory
        self._clients: list[StdioMcpClient] = []
        # close() has begun: start() starts no further server from then on.
        self._closing = False

    @property
    def clients(self) -> list[StdioMcpClient]:
        return list(self._clients)

    async def start(self) -> list[ProvidedTool]:
        """Start every server and return the tools to register.

        Once close() has begun, start() starts no further server, closes
        the clients it started and raises McpError, also when close() came
        while the last server was starting or listing its tools.
        """
        tools: list[ProvidedTool] = []
        owners: dict[str, str] = {}
        started: list[StdioMcpClient] = []
        try:
            for server in self._servers:
                self._check_open()
                client = self._client_factory(server)
                self._clients.append(client)
                started.append(client)
                await client.start()
                listed = await _list_tools(client)
                offered = {tool.name for tool in listed}
                wanted = set(server.include_tools or [])
                missing = wanted - offered
                if missing:
                    logger.warning(
                        "MCP server %r does not offer %s",
                        server.name,
                        ", ".join(sorted(missing)),
                    )
                if isinstance(server.require_approval, list):
                    unknown = set(server.require_approval) - offered
                    if unknown:
                        logger.warning(
                            "MCP server %r does not offer %s (named in "
                            "require_approval)",
                            server.name,
                            ", ".join(sorted(unknown)),
                        )
                for tool in listed:
                    if server.include_tools is not None and tool.name not in wanted:
                        continue
                    name = exposed_tool_name(server, tool.name)
                    if name in owners:
                        raise McpError(
                            f"MCP tools {owners[name]!r} and {tool.name!r} of server "
                            f"{server.name!r} both map to {name!r}"
                        )
                    owners[name] = tool.name
                    tools.append(
                        ProvidedTool(
                            definition=ToolDef(
                                name=name,
                                description=tool.description
                                or f"{tool.name} (MCP server {server.name})",
                                parameters=tool.input_schema,
                                semantic_tags=[server.name, tool.name],
                            ),
                            function=_bind(client, tool.name),
                            timeout_ms=server.timeout_s * 1000,
                            requires_approval=_requires_approval(server, tool.name),
                        )
                    )
            self._check_open()
        except BaseException:
            # The clients still listed, as before, and every client this
            # start() created. close() drops a client that has no process
            # yet; that client's start() stops the process it then gets,
            # and if that is cancelled, only this call still reaches it.
            await self._close_clients([*self._clients, *started])
            raise
        return tools

    async def close(self) -> None:
        """Close every server's client; errors are logged, not raised.

        From the moment close() begins, start() starts no further server.
        A client is dropped only once its close() has finished, so if this
        close() is cancelled part-way, the next call finishes the shutdown.
        """
        self._closing = True
        await self._close_clients(self._clients)

    def kill(self) -> None:
        """Kill every server whose client has not finished closing.

        See :meth:`StdioMcpClient.kill`: each server's process group gets
        SIGKILL at once, and close() finishes the shutdown. A client that
        close() dropped before it had a process is out of reach; its own
        start() stops the process it gets.
        """
        for client in list(self._clients):
            client.kill()

    def _check_open(self) -> None:
        if self._closing:
            raise McpError("Cannot start MCP servers: the tool provider is closed")

    async def _close_clients(self, clients: Iterable[StdioMcpClient]) -> None:
        # dict.fromkeys: a copy, each client once. return_exceptions=True:
        # when cancelled, gather waits for every client's close() to unwind
        # before re-raising the cancellation.
        await asyncio.gather(
            *(self._close_client(client) for client in dict.fromkeys(clients)),
            return_exceptions=True,
        )

    async def _close_client(self, client: StdioMcpClient) -> None:
        try:
            await client.close()
        except McpError as exc:
            # A server that close() could not stop: worth a line on stderr,
            # not a traceback (the run did not crash because of it).
            logger.warning("%s", exc)
        except Exception as exc:
            logger.warning(
                "Could not close MCP server %r: %s: %s",
                client.name,
                type(exc).__name__,
                exc,
            )
        # Not reached when close() is cancelled: the client stays listed.
        if client in self._clients:
            self._clients.remove(client)
