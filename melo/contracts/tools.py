"""Contract for tool providers that live outside the core (e.g. MCP servers)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from localmelo.melo.executor.models import ToolOutput
from localmelo.melo.schema import ToolDef

# Called with the tool-call arguments as keyword arguments.
ToolFunction = Callable[..., Coroutine[Any, Any, "str | ToolOutput"]]


@dataclass(frozen=True)
class ProvidedTool:
    """A tool definition, the coroutine that runs it, and its timeout."""

    definition: ToolDef
    function: ToolFunction
    timeout_ms: float | None = None


class BaseToolProvider(ABC):
    """Supplies tools from outside ``melo/``.

    The agent calls :meth:`start` once, before its first run, and registers
    every returned tool with the tool registry and the executor. Calls to
    those tools therefore pass the same checker and executor boundaries
    (registry lookup, pre-execute check, timeout, result check) as the
    built-in tools. :meth:`close` releases the provider's resources.
    """

    @abstractmethod
    async def start(self) -> list[ProvidedTool]:
        """Connect and return the tools to register."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources (processes, connections). Must be idempotent."""
