"""Contract for tool providers that live outside the core (e.g. MCP servers)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from localmelo.melo.schema import ToolDef

if TYPE_CHECKING:
    from localmelo.melo.executor.models import ToolOutput

# Called with the tool-call arguments as keyword arguments.
ToolFunction = Callable[..., Coroutine[Any, Any, "str | ToolOutput"]]

# What a running tool says about its own call, for the call's ``tool.finished``
# event. Each call of a run with an event sink gets a new dict; a tool sets a
# key in it (never the variable, which a copy of the context would not carry
# back). The event takes three keys: ``allowed_local`` (true: web_fetch read, or
# a browser step opened or acted on, a test server on this computer that
# ``[tools] web_fetch_allow_local`` lists),
# ``phases`` (seconds by phase of the call, beside its ``duration_s``: where a
# browser step's time went) and ``effect`` (:data:`NO_EFFECT`: an act after which
# no change was seen). Unset (None) outside such a run, where there is no event
# to mark.
TOOL_MARKS: ContextVar[dict[str, Any] | None] = ContextVar(
    "localmelo_tool_marks", default=None
)
# The one value of ``effect``: the event copies this, never a tool's own string.
NO_EFFECT: Final = "none"

# The refusals of the call that runs now: a list the agent sets for each call
# (``None`` outside one). See :func:`refuse_call`.
CALL_REFUSED: ContextVar[list[str] | None] = ContextVar(
    "localmelo_call_refused", default=None
)


def refuse_call(text: str) -> str:
    """Mark the running call as refused by the tool's own policy check; return *text*.

    A tool whose check refuses the call it was given (a note that is not the
    user's words, a task the user did not ask for) returns
    ``refuse_call(why)``: the model reads *why* as the call's result, and the
    agent knows the call did nothing, so its step need not count toward the
    run's steps (``AgentOptions.refused_steps``). Outside a run it only
    returns *text*.
    """
    found = CALL_REFUSED.get()
    if found is not None:
        found.append(text)
    return text


# The hand-off of the call that runs now: a list the agent sets only for a step
# whose one call is a hand-off call (``ProvidedTool.hands_off``), ``None``
# otherwise. See :func:`hand_off`.
CALL_HANDS_OFF: ContextVar[list[str] | None] = ContextVar(
    "localmelo_call_hands_off", default=None
)


def can_hand_off() -> bool:
    """Whether the running call may hand the rest of the run's work on (:func:`hand_off`).

    True only inside a run's step whose one call is this hand-off call: a tool
    checks it before it moves any work, so work it moved never stays in a run
    that goes on.
    """
    return CALL_HANDS_OFF.get() is not None


def hand_off(text: str) -> str:
    """Mark the running call as the run's hand-off; return *text*, its receipt.

    The rest of the run's work moved elsewhere (a task now holds it). Once the
    call's step is recorded, the agent ends the run ``completed`` with *text*
    as its result and ``RunResult.handed_off`` true, and takes no other step.
    Where :func:`can_hand_off` is false it only returns *text*.
    """
    found = CALL_HANDS_OFF.get()
    if found is not None:
        found.append(text)
    return text


@dataclass(frozen=True)
class ProvidedTool:
    """A tool definition, the coroutine that runs it, and how it is run.

    ``timeout_ms`` bounds each call (the executor's default when ``None``).
    ``requires_approval`` marks a tool whose calls ask the run's approval
    policy before they run, when the run has one (see
    ``Agent.run_task``); without a policy it has no effect. ``group``
    names a set of tools that belong together; nothing reads it now that every
    step offers every tool. ``asks_itself`` is for
    melo's own browser tools, which ask their questions themselves: the agent
    honours it only for the few names melo lists
    (``contracts.approvals.SELF_ASKING_TOOLS``), so it does not let any other
    tool skip the approval policy. ``hands_off`` names a boolean argument that
    makes a call of the tool the run's hand-off when it is true (see
    :func:`hand_off`): the agent runs such a call only as the one call of its
    step, and refuses a step that mixes it with others before running any.
    """

    definition: ToolDef
    function: ToolFunction
    timeout_ms: float | None = None
    requires_approval: bool = False
    group: str | None = None
    asks_itself: bool = False
    hands_off: str = ""


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
