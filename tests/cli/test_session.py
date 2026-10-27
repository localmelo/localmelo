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
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import SessionBusyError, SessionStore
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
async def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("HOME", str(tmp_path))  # the store writes its host id there
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
    assert "Model: m2 (OpenAI-compatible). Saved as the default." in setup.out.text()


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
def test_a_config_file_edited_during_the_session_is_not_written_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edited: str, problem: str
) -> None:
    path = tmp_path / "config.toml"
    monkeypatch.setattr("localmelo.support.config.CONFIG_PATH", str(path))
    monkeypatch.setattr("localmelo.support.config.CONFIG_DIR", str(tmp_path))
    factory = _config_factory(tmp_path)
    path.write_text(edited, encoding="utf-8")
    factory.use_model("m2")
    assert factory.save().startswith(problem)
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
