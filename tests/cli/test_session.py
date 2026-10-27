"""A terminal session on a real session store: turns, stops, the queue, commands.

The Agent's model is scripted and the app is a fake that records hints;
the store is a real ``SessionStore`` in a temporary file.
"""

from __future__ import annotations

import ast
import asyncio
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.buffer import Buffer

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.schema import Message
from localmelo.support.backends.discovery import DiscoveryError
from localmelo.support.cli import commands
from localmelo.support.cli.agents import AgentFactory, url_backend
from localmelo.support.cli.approvals import TerminalApprover
from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.render import TurnRenderer
from localmelo.support.cli.session import STOP_REASON, CliSession
from localmelo.support.cli.theme import UNICODE
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.sessions.helpers import ScriptedLLM, answer, started_agent

URL = "http://127.0.0.1:9/v1"


class FakeFactory(AgentFactory):
    """Builds Agents on scripted models; lists and saves without a server."""

    def __init__(self, root: Path, scripts: dict[str, list[Message]]) -> None:
        super().__init__(
            backend=url_backend(URL, "m1", source="saved"),
            cfg=None,
            options=AgentOptions(),
            workspace=str(root),
        )
        self.root = root
        self.scripts = scripts
        self.listed: list[str] | Exception = ["m1", "m2"]
        self.saved: list[tuple[str, str]] = []
        self.fail: set[str] = set()

    async def build(self, model: str | None = None, *, url: str | None = None) -> Agent:
        model = model or self.backend.model
        if model in self.fail:
            raise RuntimeError("the model is not loaded")
        agent = await started_agent(self.root, ScriptedLLM(self.scripts.get(model, [])))
        agent._llm.model = model
        self._live.append(agent)
        return agent

    async def list_models(self, url: str | None = None) -> list[str]:
        if isinstance(self.listed, Exception):
            raise self.listed
        return list(self.listed)

    def save(self) -> str:
        self.saved.append((self.backend.base_url, self.backend.model))
        return ""


class FakeApp:
    """What the session asks of the app: hints, questions, the buffer, exit."""

    def __init__(self, answers: list[str | None] | None = None) -> None:
        self.buffer = Buffer(multiline=True)
        self.hints: list[str] = []
        self.answers = list(answers or [])
        self.exited = False
        self.bells = 0

    def hint(self, text: str, seconds: float = 2.0) -> None:
        self.hints.append(text)

    def invalidate(self) -> None:
        pass

    def bell(self) -> None:
        self.bells += 1

    def exit(self) -> None:
        self.exited = True

    async def ask(self, prompt: str, default: str = "") -> str | None:
        answer = self.answers.pop(0)
        return default if answer == "" else answer


class Setup:
    def __init__(
        self,
        session: CliSession,
        out: RecordingOut,
        app: FakeApp,
        store: SessionStore,
        factory: FakeFactory,
    ) -> None:
        self.session = session
        self.out = out
        self.app = app
        self.store = store
        self.factory = factory

    async def idle(self, timeout_s: float = 5.0) -> None:
        await _until(lambda: self.session.state == "idle", timeout_s)

    async def rows(self) -> list[Any]:
        sessions = await self.store.list_sessions()
        if not sessions:
            return []
        return await self.store.list_turns(sessions[0].id)


async def _until(check: Callable[[], bool], timeout_s: float = 5.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not check():
            await asyncio.sleep(0.01)


@pytest.fixture
async def setup(tmp_path: Path) -> Any:
    scripts = {"m1": [answer("Paris.")], "m2": [answer("Lyon.")]}
    factory = FakeFactory(tmp_path / "work", scripts)
    store = SessionStore(tmp_path / "sessions.db", writer="cli")
    await store.open()
    agent = await factory.build()
    out = RecordingOut(80)
    renderer = TurnRenderer(out, symbols=UNICODE, backend=lambda: factory.backend)
    session = CliSession(
        store=store,
        factory=factory,
        agent=agent,
        renderer=renderer,
        approver=TerminalApprover(),
        mode="ask",
        out=out,
    )
    app = FakeApp()
    session.attach(app)  # type: ignore[arg-type]
    yield Setup(session, out, app, store, factory)
    await session.shutdown()


async def test_a_message_runs_a_turn_in_a_session_made_by_its_first_turn(
    setup: Setup,
) -> None:
    session = setup.session
    assert await setup.store.list_sessions() == []  # nothing until a message
    assert session.enter("What is the capital of France?") is None
    assert session.state == "running"
    await setup.idle()
    sessions = await setup.store.list_sessions()
    assert len(sessions) == 1 and sessions[0].title == "What is the capital of France?"
    turns = await setup.rows()
    assert [(t.status, t.cause) for t in turns] == [("completed", "run")]
    assert "● Paris." in setup.out.text()
    assert session.status_bar().title == "What is the capital of France?"


async def test_esc_stops_through_the_runner_and_retries_until_it_lands(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = setup.session
    setup.factory.scripts["m1"] = []  # the model never answers
    session.agent = await setup.factory.build()
    session.enter("wait for it")
    await _until(lambda: session.runner is not None and session.runner.busy)
    runner = session.runner
    assert runner is not None
    real = runner.cancel
    calls: list[bool] = []

    def flaky(reason: str = "", **kwargs: Any) -> bool:
        if len(calls) < 2:
            calls.append(False)
            return False  # as while begin_turn still writes the row
        calls.append(real(reason, **kwargs))
        return calls[-1]

    monkeypatch.setattr(runner, "cancel", flaky)
    assert session.stop()
    assert session.state == "stopping"
    assert not session.stop()  # once per turn
    session.tick()
    session.tick()
    await setup.idle()
    assert calls == [False, False, True]
    [turn] = await setup.rows()
    assert (turn.status, turn.cause) == ("cancelled", "user_stop")
    assert turn.result is not None and turn.result["error"] == STOP_REASON
    assert "■ Stopped" in setup.out.text()


@pytest.mark.parametrize(
    ("reason", "cause"),
    [(STOP_REASON, "user_stop"), ("Cancelled by SIGTERM", "shutdown")],
)
async def test_shutdown_during_a_turn_records_its_cause(
    setup: Setup, reason: str, cause: Any
) -> None:
    session = setup.session
    setup.factory.scripts["m1"] = []
    session.agent = await setup.factory.build()
    session.enter("wait for it")
    await _until(
        lambda: session.runner is not None and session.runner.current_turn is not None
    )
    rows_before = setup.store
    await session.shutdown(reason, cause=cause)
    store = SessionStore(rows_before.path, writer="cli")
    sessions = await store.list_sessions()
    [turn] = await store.list_turns(sessions[0].id)
    await store.aclose()
    assert (turn.status, turn.cause) == ("cancelled", cause)
    assert turn.result is not None and turn.result["error"] == reason


async def test_a_message_during_a_turn_is_queued_and_sent_when_it_completes(
    setup: Setup,
) -> None:
    session = setup.session
    gate = asyncio.Event()
    llm = ScriptedLLM([answer("first done"), answer("second done")])
    real_chat = llm.chat

    async def slow(messages: list[Message], tools: Any = None) -> Message:
        await gate.wait()
        return await real_chat(messages, tools)

    llm.chat = slow  # type: ignore[method-assign]
    session.agent = Agent(llm=llm, options=setup.session.agent.options)
    await session.agent.start()
    session.enter("first")
    assert session.enter("then this") is None
    assert session.queued == "then this"
    assert "Sent when this turn completes" in setup.app.hints[-1]
    assert session.enter("and this") == "One message is already queued (↑ edits it)."
    gate.set()
    await _until(lambda: "second done" in setup.out.text())
    await setup.idle()
    assert [t.status for t in await setup.rows()] == ["completed", "completed"]
    assert session.queued is None


async def test_a_queued_message_comes_back_when_the_turn_is_stopped(
    setup: Setup,
) -> None:
    session = setup.session
    setup.factory.scripts["m1"] = []
    session.agent = await setup.factory.build()
    session.enter("wait for it")
    await _until(
        lambda: session.runner is not None and session.runner.current_turn is not None
    )
    session.enter("follow-up")
    setup.app.buffer.text = "typed since"
    session.stop()
    await setup.idle()
    assert setup.app.buffer.text == "follow-up\ntyped since"
    assert "Your queued message is back in the input box." in setup.out.text()


async def test_commands_and_messages(setup: Setup) -> None:
    session = setup.session
    assert session.enter("/etc/hosts is broken") is None  # a message
    await setup.idle()
    assert session.enter("/nope") == ""
    assert "Unknown command /nope. /help lists the commands." in setup.out.text()
    assert session.enter("/help") is None
    await _until(lambda: "/backend [url]" in setup.out.text())


async def test_commands_that_need_an_idle_session_are_refused_during_a_turn(
    setup: Setup,
) -> None:
    session = setup.session
    setup.factory.scripts["m1"] = []
    session.agent = await setup.factory.build()
    session.enter("wait")
    for name in ("new", "model m2", "backend"):
        assert session.enter(f"/{name}") == (
            f"/{name.split()[0]} is not available while a turn runs (esc stops it)."
        )
    assert session.enter("/help") is None


async def test_new_starts_a_new_session_with_no_grants(setup: Setup) -> None:
    session = setup.session
    session.enter("one")
    await setup.idle()
    session.approvals._policy._granted.add("shell_exec")
    await session.new()
    assert session.runner is None and session.approvals.allowed_tools == frozenset()
    assert session.agent.conversation() == []
    assert "─ New session ─" in setup.out.text()
    setup.factory.scripts["m1"] = [answer("again")]
    session.agent = await setup.factory.build()
    session.enter("two")
    await setup.idle()
    assert len(await setup.store.list_sessions()) == 2


async def test_model_lists_then_switches_and_saves(setup: Setup) -> None:
    session = setup.session
    session.enter("one")
    await setup.idle()
    old = session.agent
    await session.model("")
    assert "❯ 1. m1\n  2. m2" in setup.out.text()
    await session.model("2")
    assert session.agent is not old and session.factory.backend.model == "m2"
    assert old not in setup.factory.live_agents()
    # The conversation moved to the new Agent.
    assert [m.content for m in session.agent.conversation()] == ["one", "Paris."]
    assert setup.factory.saved == [(URL, "m2")]
    assert "Model: m2 (OpenAI-compatible). Saved as the default." in setup.out.text()
    sessions = await setup.store.list_sessions()
    assert sessions[0].model == "m2"


async def test_a_model_that_cannot_start_keeps_the_old_one(setup: Setup) -> None:
    session = setup.session
    old = session.agent
    setup.factory.fail.add("m2")
    await session.model("m2")
    assert session.agent is old and session.factory.backend.model == "m1"
    assert (
        "✗ Could not start m2: the model is not loaded. Still using m1."
        in setup.out.text()
    )
    assert session.state == "idle"


async def test_backend_asks_checks_switches_and_saves(setup: Setup) -> None:
    session = setup.session
    setup.app.answers = ["http://127.0.0.1:9", "2"]
    setup.factory.listed = ["embed-x", "m2"]
    await session.backend("")
    assert session.factory.backend.base_url == URL  # /v1 added once it listed
    assert session.factory.backend.model == "m2"
    assert setup.factory.saved == [(URL, "m2")]
    assert f"Backend: {URL} · m2. Saved as the default." in setup.out.text()


async def test_backend_that_cannot_be_reached_changes_nothing(setup: Setup) -> None:
    session = setup.session
    setup.factory.listed = DiscoveryError("connection refused", reachable=False)
    await session.backend("http://127.0.0.1:1/v1")
    assert session.factory.backend.base_url == URL
    assert (
        "✗ Could not reach http://127.0.0.1:1/v1: connection refused. "
        f"Still using {URL}." in " ".join(setup.out.text().split())
    )
    assert setup.factory.saved == [] and session.state == "idle"


async def test_exit_ends_the_app(setup: Setup) -> None:
    setup.session.enter("/exit")
    await _until(lambda: setup.app.exited)


# ── Every command a hint names exists ──


def _strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_every_command_a_hint_names_exists() -> None:
    folder = Path(commands.__file__).parent
    named = {
        match
        for path in folder.glob("*.py")
        for text in _strings(path)
        for match in re.findall(r"(?:^|[\s(])/([a-z][a-z-]*)\b", text)
    }
    assert named, "no command found: the scan is broken"
    assert named <= set(commands.names()), named - set(commands.names())
