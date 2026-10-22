"""Runs localmelo's CLI with one failure injected, for the stderr tests.

Usage: ``python -I direct_mode_child.py SCENARIO FORMAT MCP_CONFIG``

It calls ``localmelo.__main__.main()`` as ``melo --output-format FORMAT
"say done"`` with an agent whose model answers ``done`` at once, plus the
SCENARIO's failure. Logging stays unconfigured, as in the real CLI, so
whatever the agent logs reaches stderr through ``logging.lastResort``.
Scenarios:

- ``close_failure``: a tool provider's close() raises;
- ``save_failure``: saving the task record fails once the run has ended;
- ``unstoppable_mcp``: ``--mcp-config MCP_CONFIG``, with an MCP client
  whose close() reports that it could not stop its server;
- ``crashed``: the model raises, so the run crashes.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Sequence
from typing import Any
from unittest import mock

import localmelo.__main__ as cli
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.memory.history import History
from localmelo.melo.schema import Message, TaskRecord, ToolDef
from localmelo.support.mcp import (
    McpError,
    McpServerConfig,
    McpTool,
    McpToolProvider,
    StdioMcpClient,
)

SCENARIOS = ("close_failure", "save_failure", "unstoppable_mcp", "crashed")


class AnswerLLM(BaseLLMProvider):
    """Answers ``done`` at once, or raises when *crash* is set."""

    def __init__(self, *, crash: bool) -> None:
        self.crash = crash

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        if self.crash:
            raise RuntimeError("the model server exploded")
        return Message(role="assistant", content="done")

    async def close(self) -> None:
        pass


class FailingCloseProvider(BaseToolProvider):
    async def start(self) -> list[ProvidedTool]:
        return []

    async def close(self) -> None:
        raise RuntimeError("close failed")


class FailingSaveHistory(History):
    """In-memory history whose saves fail once the run has ended."""

    async def save_task(self, task: TaskRecord) -> None:
        if task.status != "running":
            raise sqlite3.OperationalError("database is locked")
        await super().save_task(task)


class UnstoppableClient(StdioMcpClient):
    """A client that starts no process and cannot stop its server."""

    async def start(self) -> None:
        self.server_capabilities = {"tools": {}}

    async def list_tools(self) -> list[McpTool]:
        return []

    async def close(self) -> None:
        raise McpError(
            f"Cannot stop MCP server {self.name!r}: process 4242 is still "
            "running after SIGKILL"
        )


class UnstoppableMcpProvider(McpToolProvider):
    def __init__(self, servers: Sequence[McpServerConfig]) -> None:
        super().__init__(servers, client_factory=UnstoppableClient)


def main() -> None:
    scenario, output_format, mcp_config = sys.argv[1:]
    if scenario not in SCENARIOS:
        raise SystemExit(f"unknown scenario {scenario!r}")

    def build(
        args: Any,
        *,
        options: AgentOptions | None = None,
        tool_providers: Sequence[BaseToolProvider] = (),
    ) -> Agent:
        providers = list(tool_providers)
        if scenario == "close_failure":
            providers.append(FailingCloseProvider())
        agent = Agent(
            llm=AnswerLLM(crash=scenario == "crashed"),
            options=options,
            tool_providers=providers,
        )
        if scenario == "save_failure":
            agent.hippo.history = FailingSaveHistory()
        return agent

    argv = ["melo", "--output-format", output_format]
    if scenario == "unstoppable_mcp":
        argv += ["--mcp-config", mcp_config]
    argv.append("say done")
    with (
        mock.patch.object(cli, "_build_direct_mode_agent", build),
        mock.patch("localmelo.support.mcp.McpToolProvider", UnstoppableMcpProvider),
        mock.patch.object(sys, "argv", argv),
    ):
        cli.main()


if __name__ == "__main__":
    main()
