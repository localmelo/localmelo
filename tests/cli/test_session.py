"""A terminal session on a real session store: turns, stops, the queue, commands.

The Agent's model is scripted and the app is a fake that records hints;
the store is a real ``SessionStore`` in a temporary file.
"""

from __future__ import annotations

import ast
import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.buffer import Buffer

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.schema import Message
from localmelo.support.backends import discovery
from localmelo.support.backends.discovery import DiscoveryError
from localmelo.support.cli import commands
from localmelo.support.cli.agents import AgentFactory, url_backend
from localmelo.support.cli.approvals import TerminalApprover
from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.render import TurnRenderer
from localmelo.support.cli.session import STOP_REASON, CliSession
from localmelo.support.cli.theme import UNICODE
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import (
    SessionBusyError,
    SessionStore,
    StoredEvent,
)
from localmelo.tests.sessions.helpers import (
    ScriptedLLM,
    answer,
    started_agent,
    tool_call,
)

URL = "http://127.0.0.1:9/v1"


@pytest.fixture(autouse=True)
def _server_is_not_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real factory asks each new server whether it is Ollama; these are not."""

    async def is_ollama(url: str, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr("localmelo.support.backends.discovery.is_ollama", is_ollama)


class FakeFactory(AgentFactory):
    """Builds Agents on scripted models; lists and saves without a server."""

    def __init__(
        self, root: Path, scripts: dict[str, list[Message]], *, memory: bool = True
    ) -> None:
        super().__init__(
            backend=url_backend(URL, "m1", source="saved"),
            cfg=None,
            # The UI's own surface, as launch.py sets it.
            options=AgentOptions(memory=memory, mode="interactive"),
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
        agent = await started_agent(
            self.root, ScriptedLLM(self.scripts.get(model, [])), mode="interactive"
        )
        agent._llm.model = model
        self._live.append(agent)
        return agent

    async def list_models(self, url: str | None = None) -> list[str]:
        if isinstance(self.listed, Exception):
            raise self.listed
        return list(self.listed)

    async def save(self) -> str:
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
        self.invalidated = 0

    def hint(self, text: str, seconds: float = 2.0) -> None:
        self.hints.append(text)

    def invalidate(self) -> None:
        self.invalidated += 1

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
async def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("HOME", str(tmp_path))  # the store writes its host id there
    scripts = {"m1": [answer("Paris.")], "m2": [answer("Lyon.")]}
    factory = FakeFactory(tmp_path / "work", scripts)
    store = SessionStore(tmp_path / "sessions.db", writer="cli")
    await store.open()
    agent = await factory.build()
    out = RecordingOut(80)
    approver = TerminalApprover()
    renderer = TurnRenderer(
        out,
        symbols=UNICODE,
        backend=lambda: factory.backend,
        snapshots=approver.snapshots,
    )
    session = CliSession(
        store=store,
        factory=factory,
        agent=agent,
        renderer=renderer,
        approver=approver,
        mode="ask",
        out=out,
    )
    app = FakeApp()
    session.attach(app)  # type: ignore[arg-type]
    yield Setup(session, out, app, store, factory)
    await session.shutdown()


async def test_text_attachment_snapshot_survives_retry(setup: Setup) -> None:
    path = setup.factory.root / "input.txt"
    path.write_text("café", encoding="utf-8")
    session = setup.session
    setup.app.buffer.text = "draft @missing"
    assert session.attach_path("missing") is None
    await setup.idle()
    assert setup.app.buffer.text == "draft @missing"
    assert session.enter("/attach input.txt") is None
    await setup.idle()
    assert "input.txt" in session.attachment_line
    assert "✓ Attached input.txt (5 bytes)." in setup.out.text()
    assert session.enter("Summarize it") is None
    await setup.idle()
    assert "attached input.txt (5 bytes)" in setup.out.text()
    [turn] = await setup.rows()
    assert turn.typed_text == "Summarize it"
    assert turn.attachments is not None
    assert [(a.name, a.text) for a in turn.attachments] == [("input.txt", "café")]
    path.write_text("changed on disk", encoding="utf-8")
    assert session.enter("/new") is None
    await setup.idle()
    assert session.enter(f"/resume {turn.session_id}") is None
    await setup.idle()
    assert setup.out.text().count("attached input.txt (5 bytes)") == 2
    model = session.agent._llm
    assert isinstance(model, ScriptedLLM)
    model.replies.append(answer("Again."))
    assert session.enter("/retry") is None
    await setup.idle()
    turns = await setup.rows()
    assert turns[-1].attachments == turn.attachments
    assert "changed on disk" not in turns[-1].user_text
    assert "café" in turns[-1].user_text


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
    assert (
        "Model: m2 (OpenAI-compatible server). Saved as the default."
        in setup.out.text()
    )
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


async def test_backend_with_several_models_and_none_known_loaded_has_no_default(
    setup: Setup,
) -> None:
    setup.app.answers = ["http://127.0.0.1:9", "", "1"]
    setup.factory.listed = ["big", "small"]
    await setup.session.backend("")
    text = setup.out.text()
    assert "1. big\n" in text and "(loaded)" not in text
    assert discovery.NO_DEFAULT_HINT in " ".join(text.split())
    assert setup.factory.saved == [(URL, "big")]


async def test_backend_marks_the_loaded_model_and_defaults_to_it(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def loaded_models(url: str, **kwargs: object) -> discovery.LoadedModels:
        return discovery.LoadedModels(frozenset({"small"}), True)

    monkeypatch.setattr(discovery, "loaded_models", loaded_models)
    setup.app.answers = ["http://127.0.0.1:9", ""]
    setup.factory.listed = ["big", "small"]
    await setup.session.backend("")
    assert "2. small  (loaded)\n" in setup.out.text()
    assert setup.factory.saved == [(URL, "small")]


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
    pattern = re.compile(r"(?:^|[\s(])/([a-z][a-z-]*)(?=$|[^a-z0-9_/-])")
    assert pattern.findall("/usr/bin/pbcopy /help /etc/hosts (/copy)") == [
        "help",
        "copy",
    ]
    named = {
        match
        for path in folder.glob("*.py")
        for text in _strings(path)
        for match in pattern.findall(text)
    }
    assert named, "no command found: the scan is broken"
    assert named <= set(commands.names()), named - set(commands.names())


async def test_model_ids_from_the_server_cannot_act_on_the_terminal(
    setup: Setup,
) -> None:
    setup.factory.listed = ["m1", "evil\x1b]0;PWNED\x1b\\model"]
    await setup.session.model("")
    await setup.session.model("missing\x1b[2J")
    text = setup.out.text()
    assert "\x1b" not in text and "PWNED" not in text
    assert "2. evilmodel" in text


# ── Review fixes ──


def _gated_llm(setup: Setup, *answers: str) -> asyncio.Event:
    """The session's Agent answers *answers* in turn, each once the gate opens."""
    gate = asyncio.Event()
    llm = ScriptedLLM([answer(text) for text in answers])
    real_chat = llm.chat

    async def slow(messages: list[Message], tools: Any = None) -> Message:
        await gate.wait()
        return await real_chat(messages, tools)

    llm.chat = slow  # type: ignore[method-assign]
    setup.session.agent = Agent(llm=llm, options=setup.session.agent.options)
    return gate


async def _reopened_rows(setup: Setup) -> list[Any]:
    store = SessionStore(setup.store.path, writer="cli")
    try:
        sessions = await store.list_sessions()
        return [] if not sessions else await store.list_turns(sessions[0].id)
    finally:
        await store.aclose()


async def test_a_queued_message_is_not_sent_once_the_session_shuts_down(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = setup.session
    gate = _gated_llm(setup, "first done", "second done")
    await session.agent.start()
    session.enter("first")
    session.enter("then this")
    # The turn's end waits here, as while the runner finishes it.
    held = [True]
    monkeypatch.setattr(
        SessionRunner,
        "busy",
        property(lambda self: held[0] or self._active is not None),
    )
    gate.set()
    await _until(
        lambda: session.renderer.live is not None and session.renderer.live.finished
    )
    closing = asyncio.ensure_future(session.shutdown())
    await asyncio.sleep(0.1)
    held[0] = False
    await closing
    assert [t.user_text for t in await _reopened_rows(setup)] == ["first"]
    assert "second done" not in setup.out.text()


async def test_model_n_lists_the_models_first_when_none_were_listed(
    setup: Setup,
) -> None:
    session = setup.session
    assert session.models == []  # startup listed them, or nothing did
    await session.model("2")
    assert session.factory.backend.model == "m2"
    assert setup.factory.saved == [(URL, "m2")]


async def test_model_n_switches_nothing_when_the_models_cannot_be_listed(
    setup: Setup,
) -> None:
    session = setup.session
    setup.factory.listed = DiscoveryError("connection refused", reachable=False)
    await session.model("3")
    assert session.factory.backend.model == "m1" and setup.factory.saved == []
    assert "✗ Could not list the models: connection refused" in setup.out.text()


async def test_the_next_turn_after_a_switch_runs_on_the_new_model(
    setup: Setup,
) -> None:
    session = setup.session
    session.enter("one")
    await setup.idle()
    await session.model("m2")
    session.enter("two")
    await setup.idle()
    assert "● Lyon." in setup.out.text()
    turns = await setup.rows()
    assert [t.status for t in turns] == ["completed", "completed"]


async def test_a_switch_stays_switching_until_it_is_complete(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = setup.session
    closing, gate = asyncio.Event(), asyncio.Event()
    real_close = setup.factory.close

    async def slow_close(agent: Agent) -> None:
        closing.set()
        await gate.wait()
        await real_close(agent)

    monkeypatch.setattr(setup.factory, "close", slow_close)
    switching = asyncio.ensure_future(session.model("m2"))
    await asyncio.wait_for(closing.wait(), 5)
    assert session.state == "switching"
    assert session.enter("What is the capital?") is not None  # a hint, not a turn
    assert not session.stop()  # the new Agent is in place: nothing to cancel
    gate.set()
    await switching
    assert session.state == "idle" and session.factory.backend.model == "m2"
    assert (
        "Model: m2 (OpenAI-compatible server). Saved as the default."
        in setup.out.text()
    )


async def test_a_late_check_of_the_old_server_does_not_replace_the_new_list(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = setup.session
    gate = asyncio.Event()

    async def listing(url: str | None = None) -> list[str]:
        if url is None and setup.factory.backend.base_url == URL:
            await gate.wait()
            return ["old-a", "old-b"]
        return ["new-a"]

    monkeypatch.setattr(setup.factory, "list_models", listing)
    check = asyncio.ensure_future(session.check_server())
    await asyncio.sleep(0.05)
    await session.backend("http://127.0.0.1:19/v1")
    assert session.models == ["new-a"]
    gate.set()
    await check
    assert session.models == ["new-a"]


async def test_check_server_suggests_a_listed_embedding_model(setup: Setup) -> None:
    setup.factory.backend = replace(setup.factory.backend, key="vllm", label="vLLM")
    setup.factory.listed = ["m1", "Qwen3-Embedding-0.6B"]
    await setup.session.check_server()
    out = " ".join(setup.out.text().split())
    assert (
        "This server lists Qwen3-Embedding-0.6B; melo setup adds it "
        "for long-term memory."
    ) in out


async def test_check_server_does_not_suggest_without_a_listed_candidate(
    setup: Setup,
) -> None:
    setup.factory.backend = replace(setup.factory.backend, key="vllm", label="vLLM")
    setup.factory.listed = ["m1", "m2"]
    await setup.session.check_server()
    assert "This server lists" not in setup.out.text()


@pytest.mark.parametrize(
    ("embedding", "memory", "key"),
    [
        ("vllm · E", True, "vllm"),
        ("", False, "vllm"),
        ("", True, "openai-compatible"),
    ],
    ids=["already-configured", "memory-off", "base-url"],
)
async def test_check_server_does_not_suggest_an_embedding_where_it_cannot_act(
    setup: Setup, embedding: str, memory: bool, key: str
) -> None:
    setup.factory.backend = replace(
        setup.factory.backend, key=key, label=key, embedding=embedding
    )
    setup.factory._options = replace(setup.factory._options, memory=memory)
    setup.factory.listed = ["m1", "Qwen3-Embedding-0.6B"]
    await setup.session.check_server()
    assert "melo setup adds it" not in setup.out.text()


async def test_esc_after_the_run_ended_does_not_stop_the_turn(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = setup.session
    gate = _gated_llm(setup, "done")
    await session.agent.start()
    renderer = session.renderer
    real_on_event = renderer.on_event

    def on_event(event: Any) -> None:  # run.finished has not reached it yet
        if event.type != "run.finished":
            real_on_event(event)

    monkeypatch.setattr(renderer, "on_event", on_event)
    session.enter("first")
    await _until(lambda: session.runner is not None and session.runner.busy)
    runner = session.runner
    assert runner is not None
    ending = asyncio.Event()
    real_record = runner._record_end

    async def held_record(*args: Any) -> None:
        await ending.wait()
        await real_record(*args)

    monkeypatch.setattr(runner, "_record_end", held_record)
    gate.set()
    await _until(lambda: runner.run_ended)
    assert not session.stop()
    assert session.state == "running"
    ending.set()
    await setup.idle()
    [turn] = await setup.rows()
    assert (turn.status, turn.cause) == ("completed", "run")


async def test_the_session_stays_stopping_while_the_runner_finishes(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = setup.session
    held = [True]
    monkeypatch.setattr(SessionRunner, "busy", property(lambda self: held[0]))

    async def returns_early(self: SessionRunner, *args: Any, **kwargs: Any) -> Any:
        raise SessionBusyError("a turn still finishes")

    monkeypatch.setattr(SessionRunner, "run_turn", returns_early)
    session.enter("one")
    await asyncio.sleep(0.2)
    assert session.state == "running"
    held[0] = False
    await setup.idle()


def _config_factory(tmp_path: Path) -> AgentFactory:
    from localmelo.support import config
    from localmelo.support.cli.agents import config_backend

    cfg = config.Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=config.LocalBackendConfig(chat_url=URL, chat_model="m1"),
    )
    config.save(cfg)
    return AgentFactory(
        backend=config_backend(cfg, None),
        cfg=cfg,
        options=AgentOptions(),
        workspace=str(tmp_path),
    )


@pytest.mark.parametrize(
    ("edited", "problem"),
    [
        ("chat_backend = \n", "not saved: "),
        (
            'chat_backend = "vllm"\nembedding_backend = "none"\n\n[vllm]\n'
            'chat_url = "http://127.0.0.1:19/v1"\nchat_model = "m1"\n',
            "not saved: the config file changed",
        ),
    ],
)
async def test_a_config_file_edited_during_the_session_is_not_written_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edited: str, problem: str
) -> None:
    path = tmp_path / "config.toml"
    monkeypatch.setattr("localmelo.support.config.CONFIG_PATH", str(path))
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    factory = _config_factory(tmp_path)
    path.write_text(edited, encoding="utf-8")
    factory.use_model("m2")
    assert (await factory.save()).startswith(problem)
    assert path.read_text(encoding="utf-8") == edited
    assert factory._cfg is not None and factory._cfg.vllm.chat_url == URL


# ── /model switches among the models the server lists ──


async def test_model_refuses_a_name_the_server_does_not_list(setup: Setup) -> None:
    session = setup.session
    await session.model("m3")
    assert session.factory.backend.model == "m1" and setup.factory.saved == []
    assert f"✗ {URL} lists no model named 'm3'. /model lists them." in setup.out.text()
    assert session.state == "idle"


@pytest.mark.parametrize("number", ["0", "3", "9"])
async def test_model_refuses_a_number_outside_the_list(
    setup: Setup, number: str
) -> None:
    session = setup.session
    await session.model(number)
    assert session.factory.backend.model == "m1" and setup.factory.saved == []
    assert "✗ Type a number from 1 to 2, or a model id." in setup.out.text()


async def test_model_with_the_server_down_warns_and_still_switches(
    setup: Setup,
) -> None:
    session = setup.session
    setup.factory.listed = DiscoveryError("connection refused", reachable=False)
    await session.model("m2")
    text = " ".join(setup.out.text().split())
    assert "⚠ Could not list the models (connection refused); switching anyway." in text
    assert session.factory.backend.model == "m2"
    assert setup.factory.saved == [(URL, "m2")]


async def test_model_on_a_server_that_cannot_list_switches_without_asking(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = setup.session
    monkeypatch.setattr(setup.factory, "listable", lambda: False)  # a cloud API
    await session.model("any-model")
    assert session.factory.backend.model == "any-model"
    assert "lists no model" not in setup.out.text()


# ── /backend in a configured session keeps the rest of the config ──


async def test_a_new_server_keeps_the_configs_embedding_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.support import config
    from localmelo.support.cli import agents as agents_module
    from localmelo.support.cli.agents import config_backend

    path = tmp_path / "config.toml"
    monkeypatch.setattr("localmelo.support.config.CONFIG_PATH", str(path))
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    cfg = config.Config(
        chat_backend="sglang",
        embedding_backend="sglang",
        request_timeout_s=2,
        sglang=config.LocalBackendConfig(
            chat_url=URL,
            chat_model="m1",
            embedding_url=URL,
            embedding_model="embed-x",
        ),
        vllm=config.LocalBackendConfig(extra_body={"top_k": 20}),
    )
    config.save(cfg)
    built: list[dict[str, Any]] = []

    async def build_session_agent(**kwargs: Any) -> Any:
        built.append(kwargs)
        return object()

    monkeypatch.setattr(agents_module, "build_session_agent", build_session_agent)
    factory = AgentFactory(
        backend=config_backend(cfg, None),
        cfg=cfg,
        options=AgentOptions(),
        workspace=str(tmp_path),
    )
    new_url = "http://127.0.0.1:19/v1"
    await factory.build("m2", url=new_url)
    await factory.use_url(new_url, "m2")
    assert await factory.save() == ""
    # The session's Agent runs on the config as it is saved.
    [kwargs] = built
    running = kwargs["cfg"]
    assert kwargs["base_url"] is None and running is not cfg
    assert (running.chat_backend, running.vllm.chat_url, running.vllm.chat_model) == (
        "vllm",
        new_url,
        "m2",
    )
    assert running.embedding_backend == "sglang" and running.request_timeout_s == 2
    assert running.vllm.extra_body == {"top_k": 20}
    saved = config.load()
    assert (saved.chat_backend, saved.vllm.chat_url, saved.vllm.chat_model) == (
        "vllm",
        new_url,
        "m2",
    )
    assert saved.embedding_backend == "sglang" and saved.request_timeout_s == 2
    assert cfg.chat_backend == "sglang"  # the startup config is not changed
    assert factory.backend.base_url == new_url and factory.backend.model == "m2"
    assert factory.backend.embedding == "SGLang · embed-x"
    # A later /model on the new server saves its model only.
    built.clear()
    await factory.build("m3")
    factory.use_model("m3")
    assert await factory.save() == ""
    assert built[0]["cfg"].vllm.chat_url == new_url
    assert config.load().vllm.chat_model == "m3"
    assert config.load().embedding_backend == "sglang"


async def test_a_model_switch_never_saves_an_x_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.support import config
    from localmelo.support.cli.agents import config_backend

    path = tmp_path / "config.toml"
    monkeypatch.setattr("localmelo.support.config.CONFIG_PATH", str(path))
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    config.save(
        config.Config(
            chat_backend="sglang",
            embedding_backend="none",
            sglang=config.LocalBackendConfig(chat_url=URL, chat_model="m1"),
        )
    )
    config.set_overrides({"cli.stream": True, "retrieval.k": 9})
    cfg = config.load()
    assert cfg.cli.stream is True  # this session runs with the setting
    factory = AgentFactory(
        backend=config_backend(cfg, None),
        cfg=cfg,
        options=AgentOptions(),
        workspace=str(tmp_path),
    )
    factory.use_model("m2")
    assert await factory.save() == ""
    text = path.read_text(encoding="utf-8")
    assert 'chat_model = "m2"' in text
    assert "stream" not in text and "[retrieval]" not in text
    assert config.load().cli.stream is True


async def test_backend_saves_an_ollama_server_as_the_ollama_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.support import config
    from localmelo.support.cli import agents as agents_module
    from localmelo.support.cli.agents import config_backend

    path = tmp_path / "config.toml"
    monkeypatch.setattr("localmelo.support.config.CONFIG_PATH", str(path))
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    cfg = config.Config(
        chat_backend="sglang",
        embedding_backend="sglang",
        sglang=config.LocalBackendConfig(
            chat_url=URL,
            chat_model="m1",
            embedding_url=URL,
            embedding_model="embed-x",
        ),
    )
    config.save(cfg)
    asked: list[str] = []

    async def is_ollama(url: str, **kwargs: object) -> bool:
        asked.append(url)
        return True

    monkeypatch.setattr("localmelo.support.backends.discovery.is_ollama", is_ollama)
    built: list[dict[str, Any]] = []

    async def build_session_agent(**kwargs: Any) -> Any:
        built.append(kwargs)
        return object()

    monkeypatch.setattr(agents_module, "build_session_agent", build_session_agent)
    factory = AgentFactory(
        backend=config_backend(cfg, None),
        cfg=cfg,
        options=AgentOptions(),
        workspace=str(tmp_path),
    )
    new_url = "http://127.0.0.1:11434/v1"
    await factory.build("qwen3:8b", url=new_url)
    await factory.use_url(new_url, "qwen3:8b")
    # The switch names the server Ollama, and the Agent runs on that backend.
    assert (factory.backend.key, factory.backend.label) == ("ollama", "Ollama")
    assert await factory.save() == ""
    [kwargs] = built
    running = kwargs["cfg"]
    assert (
        running.chat_backend,
        running.ollama.chat_url,
        running.ollama.chat_model,
    ) == ("ollama", new_url, "qwen3:8b")
    saved = config.load()
    assert (saved.chat_backend, saved.embedding_backend) == ("ollama", "sglang")
    assert (saved.ollama.chat_url, saved.ollama.chat_model) == (new_url, "qwen3:8b")
    assert saved.sglang.embedding_model == "embed-x" and saved.sglang.chat_model == "m1"
    # Asked once for the switch and the save, not again for each.
    assert asked == [new_url]


# ── Text given back to the input box ──


async def test_a_queued_message_comes_back_with_the_cursor_after_it(
    setup: Setup,
) -> None:
    session = setup.session
    setup.factory.scripts["m1"] = []
    session.agent = await setup.factory.build()
    session.enter("wait for it")
    await _until(
        lambda: session.runner is not None and session.runner.current_turn is not None
    )
    session.enter("queued one")
    session.stop()
    await setup.idle()
    buffer = setup.app.buffer
    assert (buffer.text, buffer.cursor_position) == ("queued one", len("queued one"))
    buffer.insert_text("ZZ")
    assert buffer.text == "queued oneZZ"


async def test_a_queued_message_comes_back_before_the_draft_where_it_was(
    setup: Setup,
) -> None:
    session = setup.session
    setup.factory.scripts["m1"] = []
    session.agent = await setup.factory.build()
    session.enter("wait for it")
    await _until(
        lambda: session.runner is not None and session.runner.current_turn is not None
    )
    session.enter("queued one")
    buffer = setup.app.buffer
    buffer.insert_text("hello")
    buffer.cursor_position = 2  # he|llo
    session.stop()
    await setup.idle()
    assert buffer.text == "queued one\nhello"
    buffer.insert_text("ZZ")
    assert buffer.text == "queued one\nheZZllo"


async def test_a_turn_that_could_not_run_gives_the_message_back(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging
    import sqlite3

    session = setup.session
    calls: list[str] = []
    real_run_turn = SessionRunner.run_turn

    async def locked(self: SessionRunner, text: str, **kwargs: Any) -> Any:
        calls.append(text)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return await real_run_turn(self, text, **kwargs)

    monkeypatch.setattr(SessionRunner, "run_turn", locked)
    caplog.set_level(logging.INFO)
    session.enter("my important message")
    assert session.title == "my important message"
    await setup.idle()
    buffer = setup.app.buffer
    assert buffer.text == "my important message"
    assert buffer.cursor_position == len("my important message")
    text = setup.out.text()
    assert "✗ The turn could not run: OperationalError: database is locked" in text
    assert "Your message is back in the input box." in text
    # Logged, but below the level that prints a ⚠ line on the screen.
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("could not run" in r.getMessage() for r in caplog.records)
    # The next message names the session: no row was written for the first.
    session.enter("second try")
    assert session.title == "second try"
    await setup.idle()
    sessions = await setup.store.list_sessions()
    assert [s.title for s in sessions] == ["second try"]


# ── A command that runs is echoed, then its output ──


async def test_a_command_that_runs_is_echoed_before_its_output(setup: Setup) -> None:
    session = setup.session
    session.enter("/model")
    await _until(lambda: "/model NAME or /model N" in setup.out.text())
    await setup.idle()
    session.enter("/model m2")
    await _until(lambda: "Saved as the default." in setup.out.text())
    await setup.idle()
    text = setup.out.text()
    assert "› /model\nModel: m1 · " in text
    assert "/model NAME or /model N switches and saves.\n\n› /model m2\n" in text
    assert (
        "› /model m2\nModel: m2 (OpenAI-compatible server). Saved as the default.\n\n"
        in text
    )


async def test_a_refused_command_and_exit_are_not_echoed(setup: Setup) -> None:
    session = setup.session
    assert session.enter("/nope") == ""
    setup.factory.scripts["m1"] = []
    session.agent = await setup.factory.build()
    session.enter("wait")
    assert session.enter("/new") is not None  # refused during a turn
    session.enter("/exit")
    await _until(lambda: setup.app.exited)
    text = setup.out.text()
    assert "› /nope" not in text and "› /new" not in text and "› /exit" not in text


# ── U2a.2's commands ──


async def _tool_turn(setup: Setup, *, lines: int = 10) -> None:
    """Run one turn that reads a file of *lines* lines, so /expand has a call."""
    setup.factory.scripts["m1"] = [
        tool_call("file_read", {"path": "notes.txt"}),
        answer("done"),
    ]
    setup.session.agent = await setup.factory.build()
    root = Path(setup.session.agent.workspace_root or "")
    (root / "notes.txt").write_text(
        "\n".join(f"line {number}" for number in range(1, lines + 1)) + "\n",
        encoding="utf-8",
    )
    setup.session.enter("read the notes")
    await setup.idle()


async def test_verbose_and_thinking_toggle(setup: Setup) -> None:
    session, out = setup.session, setup.out
    session.enter("/verbose")
    await _until(lambda: "Verbose rendering on." in out.text())
    assert session.renderer.verbose
    session.enter("/verbose")
    await _until(lambda: "Verbose rendering off." in out.text())
    assert not session.renderer.verbose
    session.enter("/verbose maybe")
    await _until(lambda: "✗ Use /verbose on or /verbose off." in out.text())
    assert not session.renderer.verbose
    session.enter("/thinking off")
    await _until(lambda: "Thinking excerpts off." in out.text())
    assert not session.renderer.thinking
    session.enter("/thinking on")
    await _until(lambda: "Thinking excerpts on." in out.text())
    assert session.renderer.thinking
    session.enter("/thinking yes")
    await _until(lambda: "✗ Use /thinking on or /thinking off." in out.text())
    assert session.renderer.thinking
    session.enter("/stream on")
    await _until(lambda: "Live streaming on" in out.text())
    assert session.stream
    session.enter("/stream off")
    await _until(lambda: "Live streaming off." in out.text())
    assert not session.stream
    session.enter("/stream maybe")
    await _until(lambda: "✗ Use /stream on or /stream off." in out.text())
    assert not session.stream


async def test_expand_prints_a_call_in_full(setup: Setup) -> None:
    await _tool_turn(setup)
    out = setup.out
    assert "line 4" not in out.text()  # the block shows three lines
    setup.session.enter("/expand")
    await _until(lambda: "line 10" in out.text())
    assert "line 4" in out.text()
    setup.session.enter("/expand 1.1")
    await _until(lambda: out.text().count("line 10") >= 2)
    # A call of a resumed turn is not in this process's list.
    setup.session.enter("/expand 1.2")
    await _until(lambda: "✗ Turn 1 has 1 tool call." in out.text())
    setup.session.enter("/expand 2")
    await _until(lambda: out.text().count("✗ Turn 1 has 1 tool call.") == 2)
    setup.session.enter("/expand 9.1")
    await _until(lambda: "✗ No tool calls yet." in out.text())
    setup.session.enter("/expand nope")
    await _until(lambda: "✗ Use /expand [turn.call]." in out.text())


async def test_the_new_commands_follow_the_ascii_symbols(setup: Setup) -> None:
    from localmelo.support.cli.theme import ASCII

    session, out = setup.session, setup.out
    session.renderer.symbols = ASCII
    session.enter("/verbose maybe")
    await _until(lambda: "x Use /verbose on or /verbose off." in out.text())
    session.renderer.progress = {
        "level": 1,
        "state": "running",
        "step": 1,
        "attempt": 1,
        "limits": {},
        "phase": None,
        "cycle": 0,
        "tests": None,
        "items": [{"label": "Edit b", "state": "current", "repeat": 2}],
        "fraction": None,
        "expected_steps": None,
        "past": None,
        "off_reason": None,
        "basis": {
            "routine_id": "r",
            "template_id": "t",
            "version": 1,
            "match": "routine",
            "model": "m",
            "title": None,
        },
    }
    session.enter("/progress")
    await _until(lambda: "Edit b (x2) - current" in out.text())
    assert "basis r - v1 - match routine - model m" in out.text()


async def test_expand_with_no_call_yet_says_so(setup: Setup) -> None:
    setup.session.enter("/expand")
    await _until(lambda: "✗ No tool calls yet." in setup.out.text())


async def test_expand_stops_at_two_hundred_lines(setup: Setup) -> None:
    await _tool_turn(setup, lines=300)
    setup.session.enter("/expand")
    await _until(lambda: "lines not shown" in setup.out.text())
    text = setup.out.text()
    assert "line 200" in text and "line 201" not in text
    assert "… +100 lines not shown (" in text and " chars in all)" in text


async def test_progress_prints_the_latest_facts(setup: Setup) -> None:
    session, out = setup.session, setup.out
    session.enter("/progress")
    await _until(lambda: "No progress facts for this turn yet." in out.text())
    session.enter("hello")
    await setup.idle()
    assert session.renderer.progress is not None  # the turn asked for them
    session.renderer.progress = {
        "level": 1,
        "state": "running",
        "step": 4,
        "attempt": 2,
        "limits": {},
        "phase": "Verify",
        "cycle": 1,
        "tests": {"passed": 7, "failed": 1, "total": 8},
        "items": [
            {"label": "Read a", "state": "done", "repeat": None},
            {"label": "Edit b", "state": "current", "repeat": 2},
        ],
        "fraction": None,
        "expected_steps": None,
        "past": {
            "runs": 2,
            "steps_lo": 6,
            "steps_hi": 8,
            "last_steps": 8,
            "wall_lo_s": 120.0,
            "wall_hi_s": 180.0,
            "last_wall_s": 170.0,
        },
        "off_reason": None,
        "basis": {
            "routine_id": "routine-1",
            "template_id": "routine-1:t",
            "version": 3,
            "match": "routine",
            "model": "stub-model",
            "title": None,
        },
    }
    session.enter("/progress")
    await _until(lambda: "level 1 · step 4 · attempt 2" in out.text())
    text = out.text()
    assert "phase Verify (cycle 1) · tests 7/8" in text
    assert "past 2 runs · 6-8 steps · 2-3 min · last 8 steps, 2m50s" in text
    assert "items\n  Read a · done\n  Edit b (×2) · current" in text
    assert "basis routine-1 · v3 · match routine · model stub-model" in text


async def test_new_forgets_this_process_calls_and_progress(setup: Setup) -> None:
    await _tool_turn(setup)
    session = setup.session
    assert session.renderer.tool_outputs
    assert session.renderer.progress is not None
    session.renderer.verbose = True
    session.renderer.thinking = False
    session.enter("/stream on")
    await _until(lambda: session.stream)
    session.enter("/new")
    await _until(lambda: session.runner is None)
    assert session.renderer.tool_outputs == []
    assert session.renderer.progress is None
    assert not session.renderer.verbose and session.renderer.thinking
    assert not session.stream  # back to [cli] stream, the session's default
    session.enter("/expand")
    await _until(lambda: "✗ No tool calls yet." in setup.out.text())
    session.enter("/progress")
    await _until(lambda: "No progress facts for this turn yet." in setup.out.text())


# ── U2a.3's /approvals ──


async def test_approvals_shows_and_switches_the_mode(setup: Setup) -> None:
    session, out = setup.session, setup.out
    session.enter("/approvals")
    await _until(lambda: "Approvals: ask · grants: none" in out.text())
    session.enter("/approvals auto")
    await _until(lambda: "Approvals: auto" in out.text())
    assert session.approvals.mode == "auto"
    assert session.status_bar().mode == "auto"
    session.enter("/approvals deny")
    await _until(lambda: "Approvals: deny" in out.text())
    assert session.approvals.mode == "deny"
    session.enter("/approvals ask")
    await _until(lambda: out.text().count("Approvals: ask") >= 2)
    session.enter("/approvals nope")
    await _until(
        lambda: "✗ Use /approvals ask, auto, deny, or forget [tool]." in out.text()
    )
    assert session.approvals.mode == "ask"


async def test_approvals_under_yolo_says_how_to_turn_prompts_on(setup: Setup) -> None:
    session, out = setup.session, setup.out
    session.approvals.mode = "yolo"
    session.enter("/approvals")
    await _until(lambda: "Approvals: YOLO (--yolo); no grants." in out.text())
    session.enter("/approvals ask")
    await _until(lambda: "✗ YOLO is on; /approvals ask turns prompts on." in out.text())
    assert session.approvals.mode == "yolo"


# ── U2a.3's /status, /retry and /mcp ──


async def test_status_shows_the_facts(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def list_models(key: str, url: str, **kwargs: object) -> list[str]:
        return ["m1"]

    monkeypatch.setattr("localmelo.support.backends.discovery.list_models", list_models)
    session, out = setup.session, setup.out
    session.enter("hello")
    await setup.idle()
    session.enter("/status")
    await _until(lambda: "version    melo " in out.text())
    text = out.text()
    assert "model      m1 · " in text and " · reachable" in text
    assert "workspace  " in text and "fence      file tools stay here" in text
    assert "session    'hello' · " in text and " · 1 turn" in text
    assert "approvals  ask · allowed in this session: none" in text
    assert "tools      4 built-in (3 ask first)" in text
    assert "memory     2 of 64 messages · " in text
    assert "limits     30 steps a turn, 10 an attempt, 5 attempts" in text
    assert "files      " in text and "sessions.db" in text
    assert "version    melo " in text


@pytest.mark.parametrize(
    ("owned_by", "label"), [("omlx", "oMLX"), ("vllm", "OpenAI-compatible server")]
)
async def test_status_names_the_server_by_what_its_listing_says(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, owned_by: str, label: str
) -> None:
    import httpx

    from localmelo.support.backends import discovery

    real = discovery.list_models

    async def list_models(key: str, url: str, **kwargs: object) -> list[str]:
        reply = {"data": [{"id": "m1", "owned_by": owned_by}]}
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=reply))
        return await real(key, url, transport=transport)

    monkeypatch.setattr(discovery, "list_models", list_models)
    session, out = setup.session, setup.out
    assert setup.factory.backend.label == "OpenAI-compatible server"
    session.enter("/status")
    await _until(lambda: "version    melo " in out.text())
    assert f"model      m1 · {label} · {URL} · reachable" in out.text()
    assert setup.factory.backend.label == label


@pytest.mark.parametrize("cap", [20, 64])
async def test_memory_and_status_show_the_window_cap_the_agent_has(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, cap: int
) -> None:
    async def list_models(key: str, url: str, **kwargs: object) -> list[str]:
        return ["m1"]

    monkeypatch.setattr("localmelo.support.backends.discovery.list_models", list_models)
    session, out = setup.session, setup.out
    options = replace(session.agent.options, window_messages=cap)
    session.agent = Agent(llm=ScriptedLLM([]), options=options)
    await session.agent.start()
    session.enter("/memory")
    await _until(lambda: f"Working memory: 0 of {cap} messages" in out.text())
    session.enter("/status")
    await _until(lambda: f"memory     0 of {cap} messages" in out.text())


async def test_status_hides_the_rows_with_nothing_to_say_until_they_have_data(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def list_models(key: str, url: str, **kwargs: object) -> list[str]:
        return ["m1"]

    monkeypatch.setattr("localmelo.support.backends.discovery.list_models", list_models)
    session, out = setup.session, setup.out
    session.enter("/status")
    await _until(lambda: "version    melo " in out.text())
    text = out.text()
    assert not any(
        row in text for row in ("cache  ", "context  ", "steps  ", "tasks  ")
    )
    assert " yet" not in text
    session.renderer.usage.observe(
        {
            "type": "assistant.message",
            "ts": time.time(),
            "data": {"usage": {"prompt_tokens": 400, "cached_tokens": 300}},
        }
    )
    session.enter("/status")
    await _until(lambda: out.text().count("version    melo ") == 2)
    assert "cache      last request 75% cached" in out.text()


async def test_status_prints_the_usage_rows(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def list_models(key: str, url: str, **kwargs: object) -> list[str]:
        return ["m1"]

    monkeypatch.setattr("localmelo.support.backends.discovery.list_models", list_models)
    session, out = setup.session, setup.out
    session.developer = True  # developer mode shows every row
    session.enter("/status")
    await _until(lambda: "version    melo " in out.text())
    text = out.text()
    assert "cache      no requests yet" in text
    assert "context    no context budget measured yet" in text
    assert "steps      no progress facts yet" in text
    assert "tasks      no task store in this session" in text
    session.renderer.usage.observe(
        {
            "type": "assistant.message",
            "ts": time.time(),
            "data": {"usage": {"prompt_tokens": 400, "cached_tokens": 300}},
        }
    )
    session.enter("/status")
    await _until(lambda: out.text().count("version    melo ") == 2)
    assert "last request 75% cached (300 of 400 prompt tokens)" in out.text()


async def test_retry_runs_the_last_turn_again(setup: Setup) -> None:
    session, out = setup.session, setup.out
    setup.factory.scripts["m1"] = [answer("Paris."), answer("Paris.")]
    session.agent = await setup.factory.build()
    session.enter("What is the capital of France?")
    await setup.idle()
    session.enter("/retry")
    await _until(
        lambda: "↻ Retrying turn 1: What is the capital of France?" in out.text()
    )
    await _until(lambda: session.state == "idle" and out.text().count("● Paris.") == 2)
    turns = await setup.rows()
    assert [(turn.turn, turn.status, turn.undone) for turn in turns] == [
        (1, "completed", True),
        (2, "completed", False),
    ]


async def test_retry_with_nothing_to_retry_says_so(setup: Setup) -> None:
    setup.session.enter("/retry")
    await _until(lambda: "✗ Nothing to retry." in setup.out.text())


async def test_mcp_without_a_config_explains_how_to_add_one(setup: Setup) -> None:
    setup.session.enter("/mcp")
    await _until(lambda: "No MCP servers." in setup.out.text())


async def test_mcp_lists_the_servers_and_a_start_failure(setup: Setup) -> None:
    from localmelo.support.mcp import McpServerConfig

    session, out = setup.session, setup.out
    factory = session.factory
    factory.mcp_config = "~/mcp.json"
    factory.mcp_source = "mcp_config"
    factory.mcp_servers = [
        McpServerConfig(name="notes", command="notes-mcp", args=["--stdio"])
    ]
    session.enter("/mcp")
    await _until(lambda: "notes: running, 0 tools · notes-mcp --stdio" in out.text())
    assert "MCP servers from ~/mcp.json (mcp_config)" in out.text()
    factory.mcp_error = "the server exited"
    session.enter("/mcp")
    await _until(
        lambda: "MCP is off for this session: the server exited." in out.text()
    )
    assert "Fix the config, then restart melo." in " ".join(out.text().split())


async def test_deny_mode_refuses_a_file_write(setup: Setup) -> None:
    from localmelo.tests.sessions.helpers import tool_call

    session, out = setup.session, setup.out
    setup.factory.scripts["m1"] = [
        tool_call("file_write", {"path": "blocked.txt", "content": "x\n"}),
        answer("I did not write it."),
    ]
    session.agent = await setup.factory.build()
    session.approvals.set_mode("deny")
    session.enter("write it")
    await setup.idle()
    assert "denied by policy: approvals are set to de" in out.text()
    root = Path(session.agent.workspace_root or "")
    assert not (root / "blocked.txt").exists()


async def test_auto_mode_writes_and_shows_the_change(setup: Setup) -> None:
    from localmelo.tests.sessions.helpers import tool_call

    session, out = setup.session, setup.out
    setup.factory.scripts["m1"] = [
        tool_call("file_write", {"path": "notes.txt", "content": "new note\n"}),
        answer("Written."),
    ]
    session.agent = await setup.factory.build()
    session.approvals.set_mode("auto")
    session.enter("write the notes")
    await setup.idle()
    assert "auto-allowed" in out.text() or "allowed" in out.text()
    assert "└ +1 −1 lines" in out.text()


async def test_approvals_forget_takes_back_a_grant(setup: Setup) -> None:
    session, out = setup.session, setup.out
    session.approvals._policy._granted.add("file_write")
    session.enter("/approvals forget file_write")
    await _until(lambda: "Approvals: ask · grants: none" in out.text())
    assert session.approvals.allowed_tools == frozenset()


# ── ! commands ──


async def test_a_shell_command_runs_prints_its_block_and_stays_out_of_the_model(
    setup: Setup,
) -> None:
    session = setup.session
    assert session.enter("! echo SHELL_ONLY_MARK; exit 4") is None
    assert session.state == "shell"
    await setup.idle()
    text = setup.out.text()
    assert "! user shell · " in text and "$ echo SHELL_ONLY_MARK; exit 4" in text
    assert "└ exit 4 · " in text
    assert "/attach ! shows this output" in text
    # The output is the child's to draw: this fake terminal has none, and
    # nothing of it is printed by melo, stored, or sent.
    assert session.enter("Summarize it") is None
    await setup.idle()
    model = session.agent._llm
    assert isinstance(model, ScriptedLLM)
    sent = repr(model.requests)
    assert "SHELL_ONLY_MARK" not in sent and "user shell" not in sent
    [turn] = await setup.rows()
    assert "SHELL_ONLY_MARK" not in repr(turn)
    assert session.attachment_line == ""  # nothing is attached silently


async def test_bang_alone_shows_help_and_runs_nothing(setup: Setup) -> None:
    assert setup.session.enter("!") is None
    assert setup.session.state == "idle"
    text = setup.out.text()
    assert "is not a sandbox" in text and "/attach !" in text
    assert "user shell" not in text


async def test_the_escape_sends_a_literal_leading_bang_to_the_model(
    setup: Setup,
) -> None:
    assert setup.session.enter("\\! rm is a command") is None
    await setup.idle()
    [turn] = await setup.rows()
    assert turn.typed_text == "! rm is a command"
    assert "user shell" not in setup.out.text()


async def test_bang_text_is_refused_and_the_draft_kept_while_busy(
    setup: Setup,
) -> None:
    session = setup.session
    model = session.agent._llm
    assert isinstance(model, ScriptedLLM)
    model.replies.clear()  # the turn waits for a reply that never comes
    assert session.enter("work") is None
    await asyncio.wait_for(model.blocked.wait(), 5)
    for text in ("! ls", "!"):
        hint = session.enter(text)
        assert hint is not None and "only when melo is idle" in hint
    assert session.queued is None and "user shell" not in setup.out.text()
    assert session.stop()
    await setup.idle()
    session._state = "switching"
    assert "wait for the switch" in (session.enter("! ls") or "")
    session._state = "idle"
    # While a command runs, another is refused too, and so is a message.
    assert session.enter("! sleep 30") is None
    await _until(
        lambda: session._shell is not None and session._shell._process is not None
    )
    assert session.enter("! ls") == "A ! command is already running."
    assert session.enter("hello") == "A ! command is running."
    assert session.enter("!") == "A ! command is already running."
    await session.shutdown()  # melo exits: the command's group is ended
    assert session.state == "idle"


async def test_attach_bang_attaches_the_last_output_only_when_asked(
    setup: Setup,
) -> None:
    session = setup.session
    assert session.enter("/attach !") is None
    await setup.idle()
    assert "No ! output to attach" in setup.out.text()
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    assert session.enter(f"! echo ok; echo {secret}") is None
    await setup.idle()
    assert session.attachment_line == ""
    setup.app.answers.append("n")
    assert session.enter("/attach !") is None
    await setup.idle()
    assert session.attachment_line == ""  # redacted: shown, and declined
    setup.app.answers.append("y")
    assert session.enter("/attach !") is None
    await setup.idle()
    assert "shell output: echo ok" in session.attachment_line
    # The command line the person typed is echoed as typed; the snapshot is not.
    assert secret not in setup.out.text().split("review the exact snapshot")[1]
    assert session.enter("Why?") is None
    await setup.idle()
    [turn] = await setup.rows()
    assert turn.attachments is not None
    [attached] = turn.attachments
    assert attached.name.startswith("shell output: ") and secret not in attached.name
    assert attached.text.strip() == "ok\n[redacted]"  # exactly what was shown


async def test_a_command_without_output_leaves_nothing_to_attach(
    setup: Setup,
) -> None:
    session = setup.session
    assert session.enter("! echo first") is None
    await setup.idle()
    assert session.enter("! true") is None
    await setup.idle()
    assert "/attach ! attaches" not in setup.out.text().split("$ true")[1]
    assert session.enter("/attach !") is None
    await setup.idle()
    assert "No ! output to attach" in setup.out.text()


async def test_attach_bang_always_shows_the_exact_output_and_asks(
    setup: Setup,
) -> None:
    session = setup.session
    # Nothing is redacted here, and a concealed line was never on screen.
    assert session.enter("! printf 'ok\\n\\033[8mHIDDEN_MARK\\033[0m\\nend\\n'") is None
    await setup.idle()
    setup.app.answers.append("n")
    assert session.enter("/attach !") is None
    await setup.idle()
    assert session.attachment_line == ""  # declined: asking is the default
    shown = setup.out.text().split("Review the exact output that would be sent:")[1]
    assert "ok" in shown and "end" in shown
    assert "HIDDEN_MARK" not in shown.split("$ ")[0]
    setup.app.answers.append("y")
    assert session.enter("/attach !") is None
    await setup.idle()
    assert "shell output: " in session.attachment_line
    assert session.enter("Why?") is None
    await setup.idle()
    [turn] = await setup.rows()
    assert turn.attachments is not None
    [attached] = turn.attachments
    assert attached.text == "ok\n\nend\n" and "HIDDEN_MARK" not in attached.text


def _request_event(prompt: int, cached: int | None) -> StoredEvent:
    usage = {"prompt_tokens": prompt, "completion_tokens": 5}
    if cached is not None:
        usage["cached_tokens"] = cached
    event = {"type": "assistant.message", "ts": time.time(), "data": {"usage": usage}}
    return StoredEvent(None, "s1", 1, event)


async def test_status_bar_redraws_after_an_assistant_message(setup: Setup) -> None:
    session = setup.session
    assert session.status_bar().segments == ()
    before = setup.app.invalidated
    session.on_event(_request_event(1000, 830))
    assert setup.app.invalidated == before + 1
    assert session.status_bar().segments == ("cache 83%",)
    # Without a context_budget event the bar keeps its own ctx, as before.
    assert session.status_bar().context == "ctx 1.0k"


async def test_status_line_off_keeps_the_bar_as_it_was(setup: Setup) -> None:
    session = setup.session
    session.on_event(_request_event(1000, 830))
    auto = session.status_bar()
    session.status_line = "off"
    before = setup.app.invalidated
    session.on_event(_request_event(1000, 830))
    assert setup.app.invalidated == before
    off = session.status_bar()
    assert off.segments == ()
    assert (off.model, off.workspace, off.context, off.title, off.mode) == (
        auto.model,
        auto.workspace,
        "ctx 1.0k",
        auto.title,
        auto.mode,
    )
