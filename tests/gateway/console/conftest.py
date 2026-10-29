"""Fixtures for the API v2 tests: a real store, a fake Agent, a fake probe."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.events import EVENT_FIELDS
from localmelo.melo.schema import ImagePart, Message
from localmelo.support.config import (
    Config,
    GatewayConfig,
    LocalBackendConfig,
    SchedulerConfig,
)
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.sessions.store import (
    NewSession,
    SessionStore,
    TurnInfo,
)

PORT = 8401
HOLDER = "test-host:1234:abcd"


class FakeProbe:
    """A probe whose answer a test sets; ``None`` reads as a cloud backend."""

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self.state = state
        self.closed = False

    def snapshot(self) -> dict[str, Any] | None:
        return self.state

    async def aclose(self) -> None:
        self.closed = True


def quiet_gate() -> ChatGate:
    """A pass-through gate that records and writes nothing."""
    return ChatGate(policy="off", call_log_dir=None)


class FakeAgent:
    """An Agent stand-in with the conversation methods the registry uses."""

    def __init__(self, workspace: str, *, close_fails: bool = False) -> None:
        self.workspace_root = workspace
        self.options = AgentOptions(record_answers=True, workspace_root=workspace)
        self.messages: list[Message] = []
        self.closed = False
        self.close_fails = close_fails
        self.exited: tuple[str, float] | None = None

    def conversation(self) -> list[Message]:
        return list(self.messages)

    def load_conversation(self, messages: Any) -> int:
        self.messages = list(messages)
        return len(self.messages)

    def begin_exit(self, reason: str, *, within_s: float) -> None:
        self.exited = (reason, within_s)

    async def close(self) -> None:
        self.closed = True
        if self.close_fails:
            raise RuntimeError("could not close the Agent")


class FakeBuilder:
    """Builds a :class:`FakeAgent`; a test may make it fail."""

    def __init__(self, *, fail: bool = False, close_fails: bool = False) -> None:
        self.calls: list[tuple[Config, str, str | None]] = []
        self.fail = fail
        self.close_fails = close_fails
        self.agents: list[FakeAgent] = []

    async def __call__(
        self, settings: Config, workspace: str, mcp_config: str | None, session_id: str
    ) -> Agent:
        self.calls.append((settings, workspace, mcp_config))
        if self.fail:
            raise RuntimeError("no model server")
        agent = FakeAgent(workspace, close_fails=self.close_fails)
        self.agents.append(agent)
        return cast(Agent, agent)


def make_cfg() -> Config:
    return Config(
        chat_backend="mlc",
        embedding_backend="none",
        mlc=LocalBackendConfig(
            chat_url="http://127.0.0.1:9/v1", chat_model="stub-model"
        ),
        gateway=GatewayConfig(port=PORT, host="127.0.0.1"),
        # No call log: a test never writes the user's scheduler directory.
        scheduler=SchedulerConfig(call_log=False),
    )


@pytest.fixture()
def workspace(tmp_path: Path) -> str:
    path = tmp_path / "ws"
    path.mkdir()
    return str(path)


@pytest.fixture()
def builder() -> FakeBuilder:
    return FakeBuilder()


@pytest.fixture()
def probe() -> FakeProbe:
    return FakeProbe()


@pytest.fixture()
async def console(
    tmp_path: Path,
    builder: FakeBuilder,
    probe: FakeProbe,
) -> AsyncIterator[Console]:
    console = Console(
        make_cfg(),
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=builder,
            probe=probe,
            # A gate with no call log: a test never writes the user's
            # scheduler directory.
            gate=quiet_gate(),
        ),
    )
    await console.start()
    try:
        yield console
    finally:
        await console.aclose()


@pytest.fixture()
def app(console: Console) -> FastAPI:
    app = FastAPI()
    console.mount(app)
    return app


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}")


async def open_session(sessions: Any, session_id: str) -> Any:
    """Open a session the way a turn route does: under its own lock."""
    async with sessions.lock(session_id):
        return await sessions.open(session_id)


async def seed_turn(
    store: SessionStore,
    workspace: str,
    *,
    text: str = "hello",
    title: str | None = "Hello",
    key: str | None = None,
    status: str = "completed",
    usage: dict[str, int] | None = None,
    images: list[ImagePart] | None = None,
    run_id: str = "run-1",
) -> TurnInfo:
    """Create a session with one turn, and finish it unless status is running."""
    turn = await store.begin_turn(
        None,
        text,
        holder=HOLDER,
        title=title,
        client_request_id=key,
        new_session=NewSession(model="stub-model", workspace=workspace),
    )
    await store.append_events(
        turn.session_id,
        turn.turn,
        [
            core_event("run.started", run_id=run_id),
            core_event("assistant.message", content="an answer"),
        ],
        holder=HOLDER,
        images=images or [],
    )
    if status == "running":
        turns = await store.list_turns(turn.session_id)
        return turns[-1]
    result = run_result()
    if usage is not None:
        result["usage"] = usage
    return await store.finish_turn(
        turn.session_id,
        turn.turn,
        holder=HOLDER,
        status=status,
        cause="run",
        result=result,
        answer="an answer",
    )


async def all_events(store: SessionStore, session_id: str) -> list[Any]:
    """Every stored event of *session_id*, paged past the store's 1000 cap."""
    events: list[Any] = []
    after = 0
    while True:
        page = await store.list_events(session_id, after_id=after, limit=1000)
        events.extend(page)
        if len(page) < 1000 or page[-1].id is None:
            return events
        after = page[-1].id


async def seed_many_events(
    store: SessionStore, workspace: str, count: int = 1005
) -> TurnInfo:
    """A finished turn with *count* assistant events, in writer-sized batches."""
    turn = await store.begin_turn(
        None,
        "many",
        holder=HOLDER,
        title="Many",
        new_session=NewSession(model="stub-model", workspace=workspace),
    )
    events = [
        core_event("assistant.message", content=f"m{index}") for index in range(count)
    ]
    for start in range(0, count, 200):
        await store.append_events(
            turn.session_id,
            turn.turn,
            events[start : start + 200],
            holder=HOLDER,
        )
    return await store.finish_turn(
        turn.session_id,
        turn.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(),
        answer="done",
    )


# ── The two event helpers the sessions tests use ──

# Copied from tests/sessions/helpers.py: importing that module from here makes
# mypy see tests/ as two packages (the package-dir maps it under localmelo).


def core_event(
    type_: str, *, run_id: str = "run-1", seq: int = 1, ts: float = 0.0, **data: Any
) -> dict[str, Any]:
    """A core event as ``AgentEvent.to_dict()`` gives it, with every pinned key."""
    defaults: dict[str, dict[str, Any]] = {
        "run.started": {
            "query": "q",
            "max_steps": 3,
            "workspace_root": None,
            "approval": False,
        },
        "step.started": {"step": 1, "attempt": 1},
        "assistant.message": {
            "step": 1,
            "content": "",
            "thinking": "",
            "tool_calls": [],
            "usage": {},
            "prefix": None,
            "duration_s": 0.1,
        },
        "tool.started": {
            "step": 1,
            "call_id": "call_1",
            "name": "file_read",
            "arguments": {"path": "a.txt"},
            "preview": "a.txt",
        },
        "tool.finished": {
            "step": 1,
            "call_id": "call_1",
            "name": "file_read",
            "status": "success",
            "error_category": "none",
            "output": "x",
            "output_chars": 1,
            "error": "",
            "duration_s": 0.01,
            "images": [],
            "approval": None,
            "exit_code": None,
        },
        "run.finished": {"result": run_result()},
    }
    payload = dict(defaults[type_])
    payload.update(data)
    assert set(payload) == EVENT_FIELDS[type_]
    return {
        "v": 1,
        "type": type_,
        "run_id": run_id,
        "seq": seq,
        "ts": ts,
        "data": payload,
    }


def run_result(
    status: str = "completed", result: str = "done", error: str = "", **extra: Any
) -> dict[str, Any]:
    """A ``RunResult.to_dict()``."""
    data: dict[str, Any] = {
        "task_id": "run-1",
        "status": status,
        "result": result,
        "error": error,
        "steps": 1,
        "attempts": 1,
        "n_llm": 1,
        "n_main": 1,
        "n_aux": 0,
        "n_tool": 0,
        "llm_errors": 0,
        "usage": {},
        "duration_s": 0.5,
        "config": {},
    }
    data.update(extra)
    return data
