"""Record the session event streams the interactive CLI's tests render.

Run it from the package directory::

    uv run python tests/fixtures/record_events.py            # write every stream
    uv run python tests/fixtures/record_events.py --check    # compare, write nothing
    uv run python tests/fixtures/record_events.py --scenario tools --out DIR

The design is section 9 of ``.melo/u2_spec.md`` ("Recorded event streams"):
a real :class:`~localmelo.melo.agent.Agent` on a scripted LLM provider (the
``ScriptedLLM`` pattern of ``tests/agent/test_retries.py``), a real
:class:`~localmelo.support.sessions.store.SessionStore` in a temporary
directory, and a :class:`~localmelo.support.sessions.runner.SessionRunner`
built with ``progress=True``. The stub model server
(``tests/fixtures/stub_llm.py``) is not used here: it stays for the e2e
tests and the screenshots.

Every id (run, turn, request, call, routine), every timestamp, every
duration and every temporary path is normalized, so two runs of this script
write byte-identical files:

- ``ts`` becomes ``1_761_000_000.0 + 0.25 * <line>``, a stable increasing
  series, and every ``duration_s`` (and the ``past`` wall times) a stable
  series that starts at 0.1 s;
- the temporary workspace becomes ``/home/me/demo-project`` and the
  temporary home ``~``, in every string, both slash forms;
- ids become ``run-N``, ``turn-N``, ``request-N``, ``call-N`` and
  ``routine-N`` in the order they first appear.

Retries do not wait: ``Chat.sleep`` is replaced with a function that
returns at once, as the agent's tests do. The events are the ones a real
run records, the 1 s and 4 s delays in the notices included; only the
waiting is gone.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import shutil
import sys
import tempfile
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import (
    AllowAllPolicy,
    ApprovalDecision,
    ApprovalRequest,
    DenyAllPolicy,
)
from localmelo.melo.contracts.providers import ProviderError
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.executor import ToolOutput
from localmelo.melo.schema import (
    SESSION_TEXT_CHARS,
    ImagePart,
    Message,
    ToolCall,
    ToolDef,
)
from localmelo.support.cli.session import STOP_REASON
from localmelo.support.sessions.approvals import RememberingPolicy
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import (
    NewSession,
    SessionStore,
    StoredEvent,
    process_holder,
)
from localmelo.tests.sessions.helpers import (
    T0,
    Clock,
    ScriptedLLM,
    core_event,
    started_agent,
    workspace,
)

# The process's temporary root, made by main() before any scenario runs.
PROC_TMP: Path

FIXTURES = Path(__file__).resolve().parent
EVENTS_DIR = FIXTURES / "events"
DEFAULT_MODEL = "stub-model"
# The routine that `tools` and `tools_again` share, so that the second
# session's first turn finds the template the first session learned.
TOOLS_ROUTINE = "summarize-the-demo-project"
HOLD_TOOL = "hold"
SCREENSHOT_TOOL = "screenshot"
EMIT_TOOL = "emit"

# A tiny PNG, as tests/agent/test_events.py uses one.
PNG = ImagePart.from_bytes(b"\x89PNG\r\n\x1a\n a small photo", "image/png")

# The output of the `escapes` scenario: a CSI colour, an OSC 52 clipboard
# write, a carriage-return progress line, and CRLF line ends.
ESCAPES = (
    "\x1b[31mred\x1b[0m\n"
    "\x1b]52;c;SGVsbG8=\x07\n"
    "copying 1/3\r"
    "copying 2/3\r"
    "copying 3/3\r\n"
    "done\r\n"
)

PROSE = "I would rather say it in words."

AskFunction = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]


# ── Scripted replies ────────────────────────────────────────────────────────


def usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def say(
    content: str,
    *,
    thinking: str = "",
    prompt: int = 120,
    completion: int = 12,
) -> Message:
    return Message(
        role="assistant",
        content=content,
        thinking=thinking,
        usage=usage(prompt, completion),
    )


def call(name: str, arguments: Mapping[str, Any], call_id: str) -> ToolCall:
    return ToolCall(tool_name=name, arguments=dict(arguments), id=call_id)


def reply(
    *calls: ToolCall,
    content: str = "",
    thinking: str = "",
    prompt: int = 140,
    completion: int = 10,
) -> Message:
    return Message(
        role="assistant",
        content=content,
        thinking=thinking,
        tool_calls=list(calls),
        usage=usage(prompt, completion),
    )


def shell(command: str, call_id: str, **reply_fields: Any) -> Message:
    return reply(call("shell_exec", {"command": command}, call_id), **reply_fields)


def read(path: str, call_id: str, **reply_fields: Any) -> Message:
    return reply(call("file_read", {"path": path}, call_id), **reply_fields)


def write_call(path: str, content: str, call_id: str, **reply_fields: Any) -> Message:
    return reply(
        call("file_write", {"path": path, "content": content}, call_id),
        **reply_fields,
    )


def reflection(
    action: str, *, prompt: int = 220, completion: int = 32, **fields: Any
) -> Message:
    decision: dict[str, Any] = {
        "summary": "Ran the check once.",
        "failed_hypotheses": [],
        "useful_evidence": [],
        "recommended_avoids": [],
        "next_promising_directions": ["run the check again"],
        "progress_made": True,
        "task_still_feasible": True,
        "new_information_gained": True,
        "next_step_is_concrete": True,
        "next_step_is_novel": True,
        "recommended_action": action,
        "rationale": "",
        "best_effort_result": "",
        "estimated_info_gain": 0.8,
        "estimated_cost": 0.1,
        "repeat_risk": 0.1,
        "novelty": 0.8,
        "feasibility": 0.8,
    }
    decision.update(fields)
    return say(json.dumps(decision), prompt=prompt, completion=completion)


def provider_error(
    message: str, status_code: int | None, *, retryable: bool
) -> ProviderError:
    return ProviderError(message, status_code=status_code, retryable=retryable)


# ── Provider errors, as the HTTP provider words them ────────────────────────

CONNECTION_REFUSED = provider_error(
    "POST /chat/completions failed: ConnectError: [Errno 61] Connection refused",
    None,
    retryable=True,
)
READ_TIMEOUT = provider_error(
    "POST /chat/completions failed: ReadTimeout: The read operation timed out",
    None,
    retryable=True,
)
HTTP_404 = provider_error(
    'POST /chat/completions failed with HTTP 404: {"error": "model not found"}',
    404,
    retryable=False,
)
HTTP_400 = provider_error(
    'POST /chat/completions failed with HTTP 400: {"error": "the model does not '
    'support tools"}',
    400,
    retryable=False,
)
HTTP_401 = provider_error(
    'POST /chat/completions failed with HTTP 401: {"error": "invalid api key"}',
    401,
    retryable=False,
)
HTTP_500 = provider_error(
    'POST /chat/completions failed with HTTP 500: {"error": "internal error"}',
    500,
    retryable=False,
)


# ── The scripted provider ───────────────────────────────────────────────────


class Scripted(ScriptedLLM):
    """A ``ScriptedLLM`` whose queue may also hold exceptions (retries' pattern).

    Each request pops the next entry; an exception is raised instead of
    returned. A request with nothing left waits, as an unanswered approval
    stop expects.
    """

    model = DEFAULT_MODEL

    def __init__(self, replies: Iterable[Message | BaseException] = ()) -> None:
        super().__init__()
        self.script: list[Message | BaseException] = list(replies)

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(copy.deepcopy(messages))
        if not self.script:
            self.blocked.set()
            await asyncio.Event().wait()
            raise AssertionError("not reached")
        entry = self.script.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        return entry

    async def close(self) -> None:
        pass


async def _no_wait(delay: float) -> None:
    pass


# ── The demo tool provider ──────────────────────────────────────────────────


class DemoTools(BaseToolProvider):
    """Provides ``hold``, ``screenshot`` and ``emit`` for their scenarios.

    ``hold_started`` is set once ``hold`` runs, so a stop can land while
    the turn waits in it.
    """

    def __init__(self) -> None:
        self.hold_started = asyncio.Event()

    async def start(self) -> list[ProvidedTool]:
        return [
            ProvidedTool(
                definition=ToolDef(
                    name=HOLD_TOOL,
                    description="Wait until the turn is stopped",
                    parameters={"type": "object", "properties": {}},
                ),
                function=self._hold,
            ),
            ProvidedTool(
                definition=ToolDef(
                    name=SCREENSHOT_TOOL,
                    description="Take a screenshot of the page",
                    parameters={"type": "object", "properties": {}},
                ),
                function=self._screenshot,
            ),
            ProvidedTool(
                definition=ToolDef(
                    name=EMIT_TOOL,
                    description="Print a page's raw terminal output",
                    parameters={"type": "object", "properties": {}},
                ),
                function=self._emit,
            ),
        ]

    async def _hold(self) -> str:
        self.hold_started.set()
        await asyncio.Event().wait()
        raise AssertionError("not reached")

    async def _screenshot(self) -> ToolOutput:
        return ToolOutput(text="a screenshot of the demo page", images=[PNG])

    async def _emit(self) -> str:
        return ESCAPES

    async def close(self) -> None:
        pass


class ShellAutoPolicy:
    """Allows every call as auto mode does, but asks the person about file_write.

    The history stream shows shell_exec calls auto-allowed and one
    file_write approved once, as section 9's description of it reads.
    """

    def __init__(self, ask: AskFunction) -> None:
        self._ask = ask

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if request.tool_name != "file_write":
            return ApprovalDecision(True, origin="policy")
        return await self._ask(request)


# ── The session harness ─────────────────────────────────────────────────────


@dataclass
class Env:
    """One scenario's temporary store, workspace and home."""

    name: str
    root: Path
    ws: Path
    home: Path
    path: Path
    store: SessionStore

    @property
    def workspace(self) -> str:
        return os.path.realpath(self.ws)


async def new_env(name: str, files: Mapping[str, str] | None = None) -> Env:
    root = PROC_TMP / name
    (root / "ws").mkdir(parents=True)
    # The store warns when its folder is readable by others.
    os.chmod(root, 0o700)
    ws = root / "ws"
    for relative, text in (files or {}).items():
        target = ws / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    store = SessionStore(root / "sessions.db", writer="gateway", clock=Clock())
    await store.open()
    return Env(
        name=name,
        root=root,
        ws=ws,
        home=PROC_TMP / "home",
        path=root / "sessions.db",
        store=store,
    )


async def make_agent(
    env: Env,
    replies: Iterable[Message | BaseException],
    *,
    provider: BaseToolProvider | None = None,
    **options: Any,
) -> tuple[Agent, Scripted]:
    llm = Scripted(replies)
    if provider is None:
        agent = await started_agent(env.ws, llm, **options)
    else:
        agent = Agent(
            llm=llm,
            tool_providers=[provider],
            options=AgentOptions(
                workspace_root=workspace(env.ws), record_answers=True, **options
            ),
        )
        await agent.start()
    agent.chat.sleep = _no_wait
    return agent, llm


def make_runner(env: Env, agent: Agent) -> SessionRunner:
    return SessionRunner(
        agent,
        env.store,
        None,
        new_session=NewSession(model=DEFAULT_MODEL, workspace=env.workspace),
        progress=True,
    )


async def recorded(env: Env, runner: SessionRunner) -> list[StoredEvent]:
    assert runner.session_id is not None
    return await env.store.list_events(runner.session_id)


@asynccontextmanager
async def session(
    env: Env,
    replies: Iterable[Message | BaseException],
    *,
    provider: BaseToolProvider | None = None,
    **options: Any,
) -> AsyncIterator[tuple[Agent, SessionRunner, Scripted]]:
    agent, llm = await make_agent(env, replies, provider=provider, **options)
    runner = make_runner(env, agent)
    try:
        yield agent, runner, llm
    finally:
        await runner.close()
        await agent.close()


def tools_script() -> list[Message]:
    """The `tools` scenario: list, read, write, answer."""
    return [
        reply(
            call("shell_exec", {"command": "ls -1"}, "call_1"),
            content="I will look at the project first.",
            prompt=140,
            completion=18,
        ),
        read("notes.txt", "call_2", prompt=180),
        write_call("summary.md", "buy milk\n", "call_3", prompt=200),
        say(
            "summary.md now holds what notes.txt says: buy milk.",
            prompt=240,
            completion=20,
        ),
    ]


TOOLS_QUERY = "Summarize the demo project's notes."


# ── Scenarios ───────────────────────────────────────────────────────────────


async def scenario_answer(env: Env) -> list[StoredEvent]:
    async with session(
        env,
        [
            say(
                "The demo project is a small notes app. notes.txt holds the one "
                "note, and summary.md is written from it.",
                prompt=140,
                completion=28,
            )
        ],
    ) as (_agent, runner, _llm):
        await runner.run_turn("What does the demo project do?")
    return await recorded(env, runner)


async def scenario_tools(env: Env) -> list[StoredEvent]:
    async with session(env, tools_script()) as (_agent, runner, _llm):
        await runner.run_turn(TOOLS_QUERY, routine=TOOLS_ROUTINE)
    return await recorded(env, runner)


async def scenario_tools_again(env: Env) -> list[StoredEvent]:
    # The first `tools` run fills the store; the second session, whose
    # first turn is the same routine, shows level 1 (`past`, `items` and a
    # routine match). Only the second session is kept.
    agent, _llm = await make_agent(env, tools_script() + tools_script())
    try:
        first = make_runner(env, agent)
        await first.run_turn(TOOLS_QUERY, routine=TOOLS_ROUTINE)
        await first.close()
        second = make_runner(env, agent)
        await second.run_turn(TOOLS_QUERY, routine=TOOLS_ROUTINE)
        events = await recorded(env, second)
        await second.close()
    finally:
        await agent.close()
    return events


async def scenario_thinking(env: Env) -> list[StoredEvent]:
    async with session(
        env,
        [
            reply(
                call("shell_exec", {"command": "ls -1"}, "call_1"),
                thinking="The notes are in this folder.",
                prompt=130,
                completion=16,
            ),
            reply(
                call("file_read", {"path": "notes.txt"}, "call_2"),
                content="<think>I should read the notes.</think>Reading notes.txt.",
                prompt=170,
                completion=20,
            ),
            say(
                "<think>The notes say buy milk.</think>The notes say: buy milk.",
                thinking="The answer is ready.",
                prompt=210,
                completion=22,
            ),
        ],
    ) as (_agent, runner, _llm):
        await runner.run_turn("What do my notes say?")
    return await recorded(env, runner)


async def scenario_allowed(env: Env) -> list[StoredEvent]:
    asked: list[str] = []

    async def ask(request: Any) -> ApprovalDecision:
        asked.append(request.tool_name)
        if len(asked) == 1:
            return ApprovalDecision(True)
        return ApprovalDecision(True, remember=True)

    replies = [
        shell("echo one", "call_1", prompt=140),
        shell("echo two", "call_2", prompt=180),
        shell("echo three", "call_3", prompt=220),
        say("All three commands ran.", prompt=260, completion=12),
        shell("echo four", "call_4", prompt=300),
        say("The command ran without asking: approvals are set to auto.", prompt=340),
    ]
    async with session(env, replies) as (_agent, runner, _llm):
        policy = RememberingPolicy(runner.observe(ask))
        await runner.run_turn("Run the check once.", approval=policy)
        await runner.run_turn("Run it again in auto mode.", approval=AllowAllPolicy())
    return await recorded(env, runner)


async def scenario_denied(env: Env) -> list[StoredEvent]:
    async def ask(request: Any) -> ApprovalDecision:
        return ApprovalDecision(False, "It would delete the build directory.")

    replies = [
        shell("rm -rf build", "call_1", prompt=140),
        say("I did not delete the build directory.", prompt=190, completion=14),
        shell("true", "call_2", prompt=230),
        say("That command did not run either.", prompt=280, completion=12),
    ]
    async with session(env, replies) as (_agent, runner, _llm):
        policy = RememberingPolicy(runner.observe(ask))
        await runner.run_turn("Clean the build directory.", approval=policy)
        deny = DenyAllPolicy("approvals are set to deny")
        await runner.run_turn("Run the check.", approval=deny)
    return await recorded(env, runner)


async def scenario_withdrawn(env: Env) -> list[StoredEvent]:
    asked = asyncio.Event()

    async def ask(request: Any) -> ApprovalDecision:
        asked.set()
        await asyncio.Event().wait()
        raise AssertionError("not reached")

    async with session(env, [shell("true", "call_1")]) as (_agent, runner, _llm):
        policy = RememberingPolicy(runner.observe(ask))
        await runner.start_turn("Run the check.", approval=policy)
        await asked.wait()
        runner.cancel(STOP_REASON)
        await runner.wait()
    return await recorded(env, runner)


async def scenario_retry_then_error(env: Env) -> list[StoredEvent]:
    busy = provider_error(
        'POST /chat/completions failed with HTTP 503: {"error": "busy"}',
        503,
        retryable=True,
    )
    async with session(env, [busy, HTTP_400]) as (_agent, runner, _llm):
        await runner.run_turn("Check the demo project.")
    return await recorded(env, runner)


async def scenario_provider_errors(env: Env) -> list[StoredEvent]:
    turns: list[tuple[str, list[Message | BaseException]]] = [
        ("Check the demo project.", [CONNECTION_REFUSED] * 3),
        ("Check it once more.", [READ_TIMEOUT] * 3),
        ("Try the missing model.", [HTTP_404]),
        ("Send a request the server refuses.", [HTTP_400]),
        ("Send a request with no key.", [HTTP_401]),
        ("Send a request that fails.", [HTTP_500]),
    ]
    replies: list[Message | BaseException] = []
    for _text, script in turns:
        replies.extend(script)
    async with session(env, replies) as (_agent, runner, _llm):
        for text, _script in turns:
            await runner.run_turn(text)
    return await recorded(env, runner)


async def scenario_budget_reflect(env: Env) -> list[StoredEvent]:
    async with session(
        env,
        [
            shell("true", "call_1", prompt=140, completion=8),
            reflection(
                "continue",
                summary="Ran a no-op check.",
                rationale="Look at the files next.",
                prompt=180,
            ),
            say("Nothing to change: the check passes.", prompt=240, completion=16),
        ],
        steps_per_attempt=1,
    ) as (_agent, runner, _llm):
        await runner.run_turn("Check the demo project and fix what fails.")
    return await recorded(env, runner)


async def scenario_unreadable_reflection(env: Env) -> list[StoredEvent]:
    async with session(
        env,
        [
            shell("true", "call_1", prompt=140, completion=8),
            # Attempt 1's reflection comes back as a tool call, so it is asked
            # again (#31); the retry is readable and the next attempt starts.
            reply(
                call("shell_exec", {"command": "true"}, "call_2"),
                prompt=180,
                completion=10,
            ),
            reflection("continue", prompt=220),
            # Attempt 2: the reply and its retry hold no JSON object.
            shell("true", "call_3", prompt=260, completion=8),
            say(PROSE, prompt=300),
            say(PROSE, prompt=340),
            # The last attempt: the same, at the final boundary.
            shell(
                "true",
                "call_4",
                content="Probably the check passes.",
                prompt=380,
                completion=8,
            ),
            say(PROSE, prompt=420),
            say(PROSE, prompt=460),
        ],
        steps_per_attempt=1,
        max_attempts=3,
    ) as (_agent, runner, _llm):
        await runner.run_turn("Check the demo project.")
    return await recorded(env, runner)


async def scenario_budget_none(env: Env) -> list[StoredEvent]:
    replies: list[Message | BaseException] = [
        shell(f"echo {n}", f"call_{n}", prompt=120 + 20 * n, completion=8)
        for n in range(1, 10)
    ]
    async with session(env, replies, max_steps=9, attempt_policy="none") as (
        _agent,
        runner,
        _llm,
    ):
        await runner.run_turn("Echo one line at a time, nine times.")
    return await recorded(env, runner)


async def scenario_gate_stopped(env: Env) -> list[StoredEvent]:
    async with session(
        env,
        [
            shell("true", "call_1", prompt=140, completion=8),
            reflection(
                "continue",
                next_step_is_novel=False,
                rationale="Trying the same command again would tell us nothing new.",
                prompt=180,
            ),
        ],
        steps_per_attempt=1,
    ) as (_agent, runner, _llm):
        await runner.run_turn("Make the check pass.")
    return await recorded(env, runner)


async def scenario_failed(env: Env) -> list[StoredEvent]:
    replies: list[Message | BaseException] = [
        # Turn 1: a call with an empty tool name fails the plan check.
        reply(
            ToolCall(tool_name="", arguments={}, id="call_1"),
            prompt=120,
            completion=4,
        ),
        # Turn 2: two empty replies in a row fail the run.
        say("", prompt=180, completion=0),
        say("", thinking="hmm", prompt=200, completion=0),
    ]
    async with session(env, replies) as (_agent, runner, _llm):
        await runner.run_turn("Call the check.")
        await runner.run_turn("Say it again.")
    return await recorded(env, runner)


async def scenario_empty_reply(env: Env) -> list[StoredEvent]:
    async with session(
        env,
        [
            say("", prompt=110, completion=0),
            say("Here is the answer: buy milk.", prompt=150, completion=14),
        ],
    ) as (_agent, runner, _llm):
        await runner.run_turn("What do my notes say?")
    return await recorded(env, runner)


async def scenario_cancelled(env: Env) -> list[StoredEvent]:
    provider = DemoTools()
    replies = [
        reply(
            call("shell_exec", {"command": "true"}, "call_1"),
            call(HOLD_TOOL, {}, "call_2"),
            call("shell_exec", {"command": "false"}, "call_3"),
            prompt=140,
            completion=12,
        ),
        say("All three commands ran.", prompt=200),
    ]
    async with session(env, replies, provider=provider, tool_exposure="all") as (
        _agent,
        runner,
        _llm,
    ):
        await runner.start_turn("Run the three checks one by one.")
        await provider.hold_started.wait()
        runner.cancel(STOP_REASON)
        await runner.wait()
    return await recorded(env, runner)


async def scenario_images(env: Env) -> list[StoredEvent]:
    provider = DemoTools()
    async with session(
        env,
        [
            reply(call(SCREENSHOT_TOOL, {}, "call_1"), prompt=140, completion=10),
            say("I took a screenshot of the page.", prompt=190, completion=12),
        ],
        provider=provider,
        tool_exposure="all",
    ) as (_agent, runner, _llm):
        await runner.run_turn("Take a screenshot of the demo page.")
    return await recorded(env, runner)


async def scenario_long_output(env: Env) -> list[StoredEvent]:
    async with session(
        env,
        [
            shell("seq 1 3000", "call_1", prompt=140, completion=10),
            say("The sequence has 3000 lines.", prompt=190, completion=12),
        ],
    ) as (_agent, runner, _llm):
        await runner.run_turn("Print the numbers from 1 to 3000.")
    return await recorded(env, runner)


async def scenario_crashed(env: Env) -> list[StoredEvent]:
    async with session(env, [RuntimeError("boom")]) as (_agent, runner, _llm):
        await runner.run_turn("Check the demo project.")
    return await recorded(env, runner)


async def scenario_escapes(env: Env) -> list[StoredEvent]:
    provider = DemoTools()
    async with session(
        env,
        [
            reply(call(EMIT_TOOL, {}, "call_1"), prompt=140, completion=10),
            say("The page printed escape sequences.", prompt=190, completion=12),
        ],
        provider=provider,
        tool_exposure="all",
    ) as (_agent, runner, _llm):
        await runner.run_turn("Print the demo page's raw output.")
    return await recorded(env, runner)


@dataclass(frozen=True)
class Attachment:
    name: str
    text: str
    chars: int


def attachment_block(name: str, body: str, chars: int, kept: int | None = None) -> str:
    cut = "" if kept is None else f' cut="{kept}"'
    return f'<attachment name="{name}" chars="{chars}"{cut}>\n{body}\n</attachment>'


def kept_for(chars: int, share: int) -> int:
    """The characters of a *chars*-long file that fit *share*, marker included."""
    kept = share
    while kept > 0:
        marker = f"\n... [{chars - kept} more characters not shown]"
        if kept + len(marker) + len(f' cut="{kept}"') <= share:
            return kept
        kept -= 1
    return 0


def compose_user_text(typed: str, attachments: Sequence[Attachment]) -> str:
    """Compose *typed* with attachment blocks, as U2's `attachments.py` will.

    This follows "attachments.py" in section 9 of `.melo/u2_spec.md`; the
    streams that carry attachments are recorded before that helper lands.
    """
    typed = typed.rstrip()
    fixed = [
        len("\n\n")
        + len(f'<attachment name="{a.name}" chars="{a.chars}">')
        + len("\n")
        + len("\n</attachment>")
        for a in attachments
    ]
    room = SESSION_TEXT_CHARS - len(typed) - sum(fixed)
    blocks: list[str | None] = [None] * len(attachments)
    files_left = len(attachments)
    for index in sorted(
        range(len(attachments)), key=lambda i: (attachments[i].chars, i)
    ):
        attachment = attachments[index]
        share = room // files_left
        if attachment.chars <= share:
            blocks[index] = attachment_block(
                attachment.name, attachment.text, attachment.chars
            )
            room -= attachment.chars
        else:
            kept = kept_for(attachment.chars, share)
            marker = f"\n... [{attachment.chars - kept} more characters not shown]"
            blocks[index] = attachment_block(
                attachment.name,
                attachment.text[:kept] + marker,
                attachment.chars,
                kept,
            )
            room -= share
        files_left -= 1
    return typed + "".join(f"\n\n{block}" for block in blocks if block is not None)


async def scenario_attachments(env: Env) -> list[StoredEvent]:
    typed = "Compare these two files."
    transcript = "session transcript line\n" * 400
    user_text = compose_user_text(
        typed,
        [
            Attachment("notes.txt", "buy milk\n", len("buy milk\n")),
            Attachment("transcript.md", transcript, len(transcript)),
        ],
    )
    async with session(
        env, [say("Both files say the same thing.", prompt=2600, completion=12)]
    ) as (_agent, runner, _llm):
        await runner.run_turn(user_text, title=typed)
    return await recorded(env, runner)


async def scenario_file_write_overwrite(env: Env) -> list[StoredEvent]:
    async def ask(request: Any) -> ApprovalDecision:
        return ApprovalDecision(True)

    async with session(
        env,
        [
            write_call("app.py", "print('new')\n", "call_1", prompt=160),
            say("app.py now prints 'new'.", prompt=210, completion=12),
        ],
    ) as (_agent, runner, _llm):
        policy = RememberingPolicy(runner.observe(ask))
        await runner.run_turn("Update app.py.", approval=policy)
    return await recorded(env, runner)


async def scenario_yolo(env: Env) -> list[StoredEvent]:
    async with session(
        env,
        [
            shell("true", "call_1", prompt=140),
            say("The check ran without asking.", prompt=190, completion=12),
        ],
    ) as (_agent, runner, _llm):
        await runner.run_turn("Run the check.")
    return await recorded(env, runner)


async def scenario_stale(env: Env) -> list[StoredEvent]:
    # A gateway file recovers every running turn when it opens, as after a
    # process was killed: turn.finished, and no run.finished.
    holder = process_holder("gateway")
    turn = await env.store.begin_turn(
        None,
        "Run the long check.",
        holder=holder,
        new_session=NewSession(model=DEFAULT_MODEL, workspace=env.workspace),
    )
    await env.store.append_events(
        turn.session_id,
        turn.turn,
        [
            core_event(
                "run.started",
                run_id="task-1",
                query="Run the long check.",
                max_steps=30,
                workspace_root=env.workspace,
                approval=False,
            ),
            core_event("step.started", step=1, attempt=1),
        ],
        holder=holder,
    )
    await env.store.aclose()
    restarted = SessionStore(env.path, writer="gateway", clock=Clock())
    await restarted.open()
    try:
        events = await restarted.list_events(turn.session_id)
    finally:
        await restarted.aclose()
    return events


async def scenario_other_stop(env: Env) -> list[StoredEvent]:
    provider = DemoTools()
    async with session(
        env,
        [reply(call(HOLD_TOOL, {}, "call_1"), prompt=140)],
        provider=provider,
        tool_exposure="all",
    ) as (_agent, runner, _llm):
        await runner.start_turn("Run the long check.")
        await provider.hold_started.wait()
        runner.cancel("Cancelled by SIGTERM", cause="shutdown")
        await runner.wait()
    return await recorded(env, runner)


HISTORY_ANSWER = (
    "The project is in good shape.\n"
    "\n"
    "- notes.txt holds the one note\n"
    "- `make check` still fails\n"
    "\n"
    "```\n"
    "make check\n"
    "false\n"
    "make: *** [check] Error 1\n"
    "```\n"
    "\n"
    "| file | state |\n"
    "| --- | --- |\n"
    "| notes.txt | present |\n"
    "| notes.md | missing |\n"
)


async def scenario_history(env: Env) -> list[StoredEvent]:
    async def ask(request: Any) -> ApprovalDecision:
        return ApprovalDecision(True)

    replies: list[Message | BaseException] = [
        reply(
            call("shell_exec", {"command": "ls -1"}, "call_1"),
            thinking="Let me see what is here.",
            content="I will inspect the project.",
            prompt=150,
            completion=18,
        ),
        shell(
            "make check",
            "call_2",
            content="Checking the build.",
            prompt=200,
            completion=12,
        ),
        read("notes.md", "call_3", prompt=260),
        read("notes.txt", "call_4", prompt=300),
        write_call("summary.md", "buy milk\n", "call_5", prompt=340),
        say(HISTORY_ANSWER, prompt=420, completion=180),
        # Turn 2: the same failing command three times is stuck detection,
        # then a reflection that continues and an answer.
        shell("make check", "call_6", prompt=900),
        shell("make check", "call_7", prompt=920),
        shell("make check", "call_8", prompt=940),
        reflection(
            "continue",
            summary="make check still fails on the false recipe.",
            rationale="Run it once more, then report the result.",
            prompt=960,
        ),
        say(
            "The build still fails: `make check` exits 2.",
            prompt=1000,
            completion=14,
        ),
    ]
    async with session(env, replies) as (_agent, runner, _llm):
        policy = ShellAutoPolicy(runner.observe(ask))
        await runner.run_turn("What is in the demo project?", approval=policy)
        await runner.run_turn("Run the check again.", approval=policy)
    return await recorded(env, runner)


SCENARIOS: dict[str, Any] = {
    "answer": scenario_answer,
    "tools": scenario_tools,
    "tools_again": scenario_tools_again,
    "thinking": scenario_thinking,
    "allowed": scenario_allowed,
    "denied": scenario_denied,
    "withdrawn": scenario_withdrawn,
    "retry_then_error": scenario_retry_then_error,
    "provider_errors": scenario_provider_errors,
    "budget_reflect": scenario_budget_reflect,
    "unreadable_reflection": scenario_unreadable_reflection,
    "budget_none": scenario_budget_none,
    "gate_stopped": scenario_gate_stopped,
    "failed": scenario_failed,
    "empty_reply": scenario_empty_reply,
    "cancelled": scenario_cancelled,
    "images": scenario_images,
    "long_output": scenario_long_output,
    "crashed": scenario_crashed,
    "escapes": scenario_escapes,
    "attachments": scenario_attachments,
    "file_write_overwrite": scenario_file_write_overwrite,
    "yolo": scenario_yolo,
    "stale": scenario_stale,
    "other_stop": scenario_other_stop,
    "history": scenario_history,
}

# The pre-run files of the scenarios that read the workspace.
SCENARIO_FILES: dict[str, dict[str, str]] = {
    "file_write_overwrite": {"app.py": "print('old')\n"},
    "history": {"Makefile": "check:\n\tfalse\n"},
}


# ── Normalization ───────────────────────────────────────────────────────────

# Keys whose value is a wall-clock duration, replaced by a stable series.
DURATION_KEYS = frozenset({"duration_s", "wall_lo_s", "wall_hi_s", "last_wall_s"})
# The ids a stream carries, by the key that holds them.
TURN_ID_KEYS = frozenset({"turn_id"})
RUN_ID_KEYS = frozenset({"task_id", "run_id"})
REQUEST_ID_KEYS = frozenset({"request_id"})
CALL_ID_KEYS = frozenset({"call_id", "id"})
ROUTINE_ID_KEYS = frozenset({"routine_id"})


class Normalizer:
    """Stable ids, timestamps, durations and paths for one stream."""

    def __init__(self, workspace_path: str, home_path: str) -> None:
        self._paths: list[tuple[str, str]] = []
        for path, replacement in (
            (workspace_path, "/home/me/demo-project"),
            (os.path.realpath(workspace_path), "/home/me/demo-project"),
            (home_path, "~"),
            (os.path.realpath(home_path), "~"),
        ):
            if not path or path == "/":
                continue
            for form, replacement_form in (
                (path, replacement),
                (path.replace("/", "\\"), replacement.replace("/", "\\")),
            ):
                if (form, replacement_form) not in self._paths:
                    self._paths.append((form, replacement_form))
        # Longest first, so a parent never cuts a child's path short.
        self._paths.sort(key=lambda pair: len(pair[0]), reverse=True)
        self._runs: dict[str, str] = {}
        self._turns: dict[str, str] = {}
        self._requests: dict[str, str] = {}
        self._routines: dict[str, str] = {}
        self._calls: dict[tuple[str, str], str] = {}
        self._call_counts: dict[str, int] = {}
        self._durations = 0

    def stream(self, events: Sequence[StoredEvent]) -> list[dict[str, Any]]:
        lines: list[dict[str, Any]] = []
        # A turn's run, for the session events that carry no run_id (the
        # approvals, above all), so their call ids share the run's map.
        turn_runs: dict[int | None, str] = {}
        for index, stored in enumerate(events):
            event = copy.deepcopy(stored.event)
            original_run = event.get("run_id") or ""
            if original_run:
                run = self.run_id(original_run)
                turn_runs[stored.turn] = run
            else:
                run = turn_runs.get(stored.turn, "")
            event["run_id"] = run if original_run else ""
            event["ts"] = T0 + 0.25 * index
            event["data"] = self.walk(event["data"], run)
            lines.append({"turn": stored.turn, "event": event})
        return lines

    def run_id(self, value: str) -> str:
        if not value:
            return ""
        if value not in self._runs:
            self._runs[value] = f"run-{len(self._runs) + 1}"
        return self._runs[value]

    def _turn_id(self, value: str) -> str:
        if value not in self._turns:
            self._turns[value] = f"turn-{len(self._turns) + 1}"
        return self._turns[value]

    def _request_id(self, value: str) -> str:
        if value not in self._requests:
            self._requests[value] = f"request-{len(self._requests) + 1}"
        return self._requests[value]

    def _routine_id(self, value: str) -> str:
        if value not in self._routines:
            self._routines[value] = f"routine-{len(self._routines) + 1}"
        return self._routines[value]

    def _call_id(self, run: str, value: str) -> str:
        if not value:
            return value
        key = (run, value)
        if key not in self._calls:
            count = self._call_counts.get(run, 0) + 1
            self._call_counts[run] = count
            self._calls[key] = f"call-{count}"
        return self._calls[key]

    def _duration(self) -> float:
        self._durations += 1
        return round(0.1 + 0.02 * (self._durations - 1), 3)

    def walk(self, value: Any, run: str) -> Any:
        if isinstance(value, Mapping):
            walked: dict[str, Any] = {}
            for key, item in value.items():
                if key in DURATION_KEYS and isinstance(item, int | float):
                    walked[key] = self._duration()
                elif key in TURN_ID_KEYS and isinstance(item, str):
                    walked[key] = self._turn_id(item)
                elif key in RUN_ID_KEYS and isinstance(item, str):
                    walked[key] = self.run_id(item)
                elif key in REQUEST_ID_KEYS and isinstance(item, str):
                    walked[key] = self._request_id(item)
                elif key in CALL_ID_KEYS and isinstance(item, str):
                    walked[key] = self._call_id(run, item)
                elif key in ROUTINE_ID_KEYS and isinstance(item, str):
                    walked[key] = self._routine_id(item)
                elif key == "template_id" and isinstance(item, str):
                    walked[key] = self._template_id(item)
                else:
                    walked[key] = self.walk(item, run)
            return walked
        if isinstance(value, list):
            return [self.walk(item, run) for item in value]
        if isinstance(value, str):
            return self.rewrite(value)
        return value

    def _template_id(self, value: str) -> str:
        routine, _, digest = value.partition(":")
        if not digest:
            return self.rewrite(value)
        return f"{self._routine_id(routine)}:{digest}"

    def rewrite(self, text: str) -> str:
        for path, replacement in self._paths:
            text = text.replace(path, replacement)
        return text


# ── Recording and files ─────────────────────────────────────────────────────


async def record_scenario(name: str) -> list[dict[str, Any]]:
    env = await new_env(name, SCENARIO_FILES.get(name))
    try:
        events = await SCENARIOS[name](env)
    finally:
        await env.store.aclose()
    normalizer = Normalizer(env.workspace, str(env.home))
    return normalizer.stream(events)


def dump(lines: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(line), ensure_ascii=False) + "\n" for line in lines
    ).encode("utf-8")


def save_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS),
        help="record only this scenario (repeatable)",
    )
    parser.add_argument("--out", default=str(EVENTS_DIR), help="the stream folder")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare with the files and write nothing",
    )
    args = parser.parse_args(argv)
    global PROC_TMP
    PROC_TMP = Path(tempfile.mkdtemp(prefix="melo-record-"))
    os.environ["HOME"] = str(PROC_TMP / "home")
    (PROC_TMP / "home").mkdir()
    # Child processes (ls, make) print in one locale on every machine.
    os.environ["LC_ALL"] = "C"
    os.environ.pop("LOCALMELO_PERSIST_MEMORY", None)
    os.environ.pop("LOCALMELO_MEMORY_DIR", None)
    out = Path(args.out)
    names = args.scenario or list(SCENARIOS)
    different = 0
    try:
        for index, name in enumerate(names):
            if index:
                shutil.rmtree(PROC_TMP / names[index - 1], ignore_errors=True)
            lines = asyncio.run(record_scenario(name))
            data = dump(lines)
            path = out / f"{name}.jsonl"
            if args.check:
                if not path.exists() or path.read_bytes() != data:
                    different += 1
                    print(f"{path}: {'missing' if not path.exists() else 'differs'}")
            else:
                save_file(path, data)
                print(f"{path}: {len(lines)} events, {len(data)} bytes")
    finally:
        shutil.rmtree(PROC_TMP, ignore_errors=True)
    if args.check:
        return 1 if different else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
