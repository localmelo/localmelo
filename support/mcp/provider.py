"""Expose stdio MCP servers to the agent as a tool provider."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable, Sequence
from typing import Any

from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool, ToolFunction
from localmelo.melo.executor.models import ToolOutput
from localmelo.melo.schema import ToolDef
from localmelo.support.mcp.client import McpError, StdioMcpClient
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


def _bind(client: StdioMcpClient, tool_name: str) -> ToolFunction:
    async def call(**arguments: Any) -> ToolOutput:
        return tool_result_to_output(await client.call_tool(tool_name, arguments))

    return call


class McpToolProvider(BaseToolProvider):
    """Tools of one or more stdio MCP servers.

    Each server's tools are registered as ``mcp__<server>__<tool>`` with the
    server's input schema and description. A call goes to ``tools/call``
    with the original tool name; the executor bounds it with the server's
    ``timeout``. ``include_tools`` restricts which tools are registered.
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

    @property
    def clients(self) -> list[StdioMcpClient]:
        return list(self._clients)

    async def start(self) -> list[ProvidedTool]:
        tools: list[ProvidedTool] = []
        owners: dict[str, str] = {}
        try:
            for server in self._servers:
                client = self._client_factory(server)
                self._clients.append(client)
                await client.start()
                listed = await client.list_tools()
                wanted = set(server.include_tools or [])
                missing = wanted - {tool.name for tool in listed}
                if missing:
                    logger.warning(
                        "MCP server %r does not offer %s",
                        server.name,
                        ", ".join(sorted(missing)),
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
                        )
                    )
        except BaseException:
            await self.close()
            raise
        return tools

    async def close(self) -> None:
        clients, self._clients = self._clients, []
        await asyncio.gather(*(c.close() for c in clients), return_exceptions=True)
