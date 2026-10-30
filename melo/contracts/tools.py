"""Contract for tool providers that live outside the core (e.g. MCP servers)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any

from localmelo.melo.executor.models import ToolOutput
from localmelo.melo.schema import ToolDef

# Called with the tool-call arguments as keyword arguments.
ToolFunction = Callable[..., Coroutine[Any, Any, "str | ToolOutput"]]


@dataclass(frozen=True)
class ProvidedTool:
    """A tool definition, the coroutine that runs it, and how it is run.

    ``timeout_ms`` bounds each call (the executor's default when ``None``).
    ``requires_approval`` marks a tool whose calls ask the run's approval
    policy before they run, when the run has one (see
    ``Agent.run_task``); without a policy it has no effect. ``group``
    names a set of tools that are offered together: with
    ``tool_exposure="retrieve"``, every tool of a group is offered on every
    step, as the built-ins are, instead of waiting for hints or BM25 to
    pick it (``"all"`` offers every tool anyway).
    """

    definition: ToolDef
    function: ToolFunction
    timeout_ms: float | None = None
    requires_approval: bool = False
    group: str | None = None


class BaseToolProvider(ABC):
    """Supplies tools from outside ``melo/``.

    The provider's owner calls :meth:`start` once, before the first run,
    and registers every returned tool with the tool registry and the
    executor. Calls to those tools therefore pass the same checker and
    executor boundaries as the built-in tools: registry lookup, the
    allowlist (a tool runs only if the planning step that called it
    offered it), the raw-arguments check (arguments that are not a JSON
    object are refused before anything runs), the pre-execute check, the
    approval policy of a run that has one, for a tool marked
    ``requires_approval``, the timeout and the result check.

    :meth:`close` releases the provider's resources. It may be called
    while :meth:`start` is still running, and must then make start() give
    up: stop whatever it started and raise instead of returning tools.
    :meth:`kill` stops the provider's processes at once when there is no
    time left for close().
    """

    @abstractmethod
    async def start(self) -> list[ProvidedTool]:
        """Connect and return the tools to register."""

    def system_notes(self) -> Sequence[str]:
        """Text for the system prompt of every planning request, once started.

        The agent adds it after this provider's tools are registered, for a
        provider whose tools return content the model must treat as data (a
        web page, say). The default is none.
        """
        return ()

    @abstractmethod
    async def close(self) -> None:
        """Release resources (processes, connections). Must be idempotent.

        Also when start() is still running: start() must then give up,
        leave nothing running, and raise.
        """

    def kill(self) -> None:  # noqa: B027 – optional hook, empty by default
        """Stop the provider's processes at once, without waiting for them.

        Called when a close(), or the clean-up of a start() that was
        cancelled, has not finished in time, for example when the process
        must exit (see ``Agent.begin_exit``). kill() must then also reach
        the processes that clean-up is still stopping. close() or start()
        may be running when kill() is called, and close() may be called
        again after it. The default does nothing, which suits a provider
        that starts no processes.
        """
