"""Auto mode, part 2 (#339): a chat's mode, the Allow once route, and tasks in Auto.

A real Agent per chat with a scripted model (step reviews read their own
replies) and a ``python_exec`` that records what ran instead of running it.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.melo.schema import Message
from localmelo.support.gateway.auto_blocks import blocked_steps, go_ahead_text
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.sessions.approvals import ChatPolicy, chat_mode
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.agent.test_approvals import _answer, _calls
from localmelo.tests.agent.test_auto_mode import FakeModel, verdict
from localmelo.tests.sessions.helpers import started_agent

from .conftest import PORT, FakeProbe, all_events, make_cfg, quiet_gate

DELETE = "import shutil\nshutil.rmtree('.git', ignore_errors=True)"
SELF = "open('~/.cache/localmelo/config.toml', 'w').write('x')"
TIMEOUT_S = 10.0


def step(code: str, call_id: str = "call_1") -> Message:
    return _calls(("python_exec", {"code": code}, call_id))


class Ran:
    """The code each chat's python_exec was given, in order."""

    def __init__(self) -> None:
        self.codes: list[str] = []

    async def python_exec(self, code: str) -> str:
        self.codes.append(code)
        return "ok"


@contextlib.asynccontextmanager
async def auto_console(
    tmp_path: Path,
    plans: list[Message],
    reviews: list[Message] | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> AsyncIterator[tuple[Console, httpx.AsyncClient, Ran, FakeModel]]:
    """A console whose chats share one scripted model and one recording python_exec.

    With *monkeypatch*, a saved task's Agent is built the same way.
    """
    ran = Ran()
    llm = FakeModel(plans, reviews or [])

    async def build_task_agent(**kwargs: Any) -> Agent:
        options = dataclasses.replace(
            kwargs["options"], workspace_root=kwargs["workspace"], record_answers=True
        )
        agent = Agent(llm=llm, options=options)
        await agent.start()
        agent.executor.register("python_exec", ran.python_exec, requires_approval=True)
        return agent

    if monkeypatch is not None:
        monkeypatch.setattr(
            "localmelo.support.gateway.workers.build_session_agent", build_task_agent
        )

    async def build(
        settings: Any, workspace: str, mcp_config: str | None, session_id: str
    ) -> Any:
        agent = await started_agent(
            Path(workspace),
            llm,
            mode="gateway",
            builtins=frozenset({"python_exec"}),
            tool_exposure="all",
        )
        agent.executor.register("python_exec", ran.python_exec, requires_approval=True)
        return agent

    console = Console(
        make_cfg(),
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=build,
            probe=FakeProbe(),
            gate=quiet_gate(),
        ),
    )
    await console.start()
    app = FastAPI()
    console.mount(app)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=f"http://127.0.0.1:{PORT}"
    )
    try:
        yield console, client, ran, llm
    finally:
        await client.aclose()
        await console.aclose()


async def wait_turn(client: httpx.AsyncClient, session_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + TIMEOUT_S
    while True:
        body = (await client.get(f"/api/sessions/{session_id}/turns")).json()
        if body["turns"] and body["turns"][-1]["status"] != "running":
            return cast(dict[str, Any], body["turns"][-1])
        if time.monotonic() > deadline:
            raise AssertionError("the turn did not finish")
        await asyncio.sleep(0.01)


async def approvals(console: Console, session_id: str) -> list[dict[str, Any]]:
    return [
        e.event["data"]["approval"]
        for e in await all_events(console.store, session_id)
        if e.event["type"] == "tool.finished"
    ]


async def create(
    client: httpx.AsyncClient, tmp_path: Path, text: str, **extra: Any
) -> str:
    folder = tmp_path / "ws"
    folder.mkdir(exist_ok=True)
    response = await client.post(
        "/api/sessions",
        json={"turn": {"text": text}, "workspace": str(folder), **extra},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["session"]["id"])


# ── The chat's mode ──


def test_a_mode_is_read_from_meta_with_the_main_chat_s_default() -> None:
    assert chat_mode({}) == "ask"
    assert chat_mode({"approvals": "auto"}) == "auto"
    assert chat_mode({"approvals": "yolo"}, "auto") == "auto"
    assert chat_mode(None, "auto") == "auto"
    assert chat_mode({"approvals": "ask"}, "auto") == "ask"


@pytest.mark.asyncio
async def test_a_chat_policy_follows_its_mode_at_each_decision() -> None:
    asked: list[str] = []

    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request.tool_name)
        return ApprovalDecision(True, origin="user")

    from localmelo.melo.contracts.approvals import find_auto, review_mode

    policy = ChatPolicy(ask)
    request = ApprovalRequest(
        "r", "run", "c", "python_exec", {"code": "x"}, "x", None, "exec"
    )
    assert review_mode(policy, request) == "ask" and find_auto(policy) is None
    assert (await policy.decide(request)).origin == "user"
    policy.mode = "auto"
    assert review_mode(policy, request) == "auto"
    assert find_auto(policy) is policy.auto
    # Auto asks nobody; a call nothing reviewed is blocked.
    decision = await policy.decide(request)
    assert (decision.allow, decision.origin) == (False, "auto")
    assert asked == ["python_exec"]


@pytest.mark.asyncio
async def test_meta_update_keeps_the_other_keys_and_says_so(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path / "s.db"), writer="gateway")
    await store.open()
    try:
        info = await store.create_session(meta={"other": 1})
        after = await store.update_session(info.id, meta_update={"approvals": "auto"})
        assert after.meta == {"other": 1, "approvals": "auto"}
        [event] = await all_events(store, info.id)
        assert event.event["type"] == "session.updated"
        assert event.event["data"]["meta"] == {"other": 1, "approvals": "auto"}
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_a_new_chat_starts_in_the_settings_mode_and_switches_by_patch(
    tmp_path: Path,
) -> None:
    async with auto_console(tmp_path, [_answer("hi"), _answer("hi")]) as (
        console,
        client,
        _,
        _llm,
    ):
        first = await create(client, tmp_path, "hello")
        await wait_turn(client, first)
        body = (await client.get(f"/api/sessions/{first}")).json()["session"]
        assert (body["approvals"], body["allowances"]) == ("ask", {})
        console.cfg.approvals.mode = "auto"
        # The main chat follows the setting until it is picked.
        main = (await client.get("/api/sessions")).json()["main"]
        assert main["approvals"] == "auto"
        second = await create(client, tmp_path, "hello again")
        await wait_turn(client, second)
        assert console.sessions.get(second).policy.mode == "auto"  # type: ignore[union-attr]
        response = await client.patch(
            f"/api/sessions/{first}", json={"approvals": "auto"}
        )
        assert response.json()["session"]["approvals"] == "auto"
        assert console.sessions.get(first).policy.mode == "auto"  # type: ignore[union-attr]
        refused = await client.patch(
            f"/api/sessions/{first}", json={"approvals": "yolo"}
        )
        assert refused.status_code == 400
        # The chip's choice for a chat not made yet goes with its first turn.
        console.cfg.approvals.mode = "ask"
        third = await create(client, tmp_path, "x", approvals="auto")
        body = (await client.get(f"/api/sessions/{third}")).json()["session"]
        assert body["approvals"] == "auto"
        # Only a chat has a mode; a refused one changes nothing else either.
        worker = await console.store.create_session(
            kind="worker", workspace=str(tmp_path / "ws")
        )
        other = tmp_path / "other"
        other.mkdir()
        refused = await client.patch(
            f"/api/sessions/{worker.id}",
            json={"approvals": "auto", "title": "x", "workspace": str(other)},
        )
        assert (refused.status_code, refused.json()["error"]["code"]) == (
            409,
            "not_a_chat",
        )
        kept = await console.store.get_session(worker.id)
        assert kept is not None and kept.title != "x" and "approvals" not in kept.meta
        assert kept.workspace == str(tmp_path / "ws")


# ── Allow once, end to end ──


@pytest.mark.asyncio
async def test_allow_once_runs_the_blocked_step_once_then_it_is_blocked_again(
    tmp_path: Path,
) -> None:
    plans = [
        step(DELETE),
        _answer("I left .git as it is."),
        step(DELETE, "call_2"),
        _answer("Done."),
        step(DELETE, "call_3"),
        _answer("Blocked again."),
    ]
    # The third turn comes after a block of the same kind: the review asks
    # the model, which says the person did not ask for it this time.
    reviews = [verdict(verdict="ask", category="delete", user_asked="no")]
    async with auto_console(tmp_path, plans, reviews) as (console, client, ran, llm):
        sid = await create(client, tmp_path, "Tidy the notes", approvals="auto")
        await wait_turn(client, sid)
        [block] = await approvals(console, sid)
        assert (block["origin"], block["allowed"]) == ("auto", False)
        assert ran.codes == []

        missing = await client.post(f"/api/sessions/{sid}/blocks/nope/allow")
        assert missing.status_code == 404
        response = await client.post(
            f"/api/sessions/{sid}/blocks/{block['request_id']}/allow"
        )
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["step_key"] == block["step_key"]
        assert body["turn"]["user_text"] == go_ahead_text(["delete"])
        assert body["turn"]["user_text"] == (
            "Go ahead with the step melo blocked: delete files that cannot be "
            "restored."
        )
        await wait_turn(client, sid)
        assert ran.codes == [DELETE]
        allowed = (await approvals(console, sid))[1]
        assert (allowed["origin"], allowed["allowance"]) == ("user", "once")
        session = (await client.get(f"/api/sessions/{sid}")).json()["session"]
        assert set(session["allowances"]) == {block["step_key"]}

        await client.post(f"/api/sessions/{sid}/turns", json={"text": "Do it again"})
        await wait_turn(client, sid)
        assert ran.codes == [DELETE]
        again = (await approvals(console, sid))[2]
        assert (again["origin"], again["allowed"]) == ("auto", False)
        assert len(llm.review_requests) == 1


@pytest.mark.asyncio
async def test_a_step_that_changes_localmelo_itself_cannot_be_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    async with auto_console(tmp_path, [step(SELF), _answer("no")]) as (
        console,
        client,
        ran,
        _,
    ):
        console.cfg.approvals.mode = "auto"
        sid = await create(client, tmp_path, "change the settings file")
        await wait_turn(client, sid)
        [block] = await approvals(console, sid)
        assert block["review"]["categories"] == ["self"]
        response = await client.post(
            f"/api/sessions/{sid}/blocks/{block['request_id']}/allow"
        )
        assert (response.status_code, response.json()["error"]["code"]) == (
            409,
            "not_allowable",
        )
        assert ran.codes == []


# ── Saved tasks keep their chat's mode ──


@pytest.mark.asyncio
async def test_a_task_made_in_auto_runs_in_auto_and_lists_its_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [
        step(DELETE),
        _answer("The task went on without it."),
        step(DELETE, "call_2"),
        _answer("Deleted."),
    ]
    async with auto_console(tmp_path, plans, monkeypatch=monkeypatch) as (
        console,
        client,
        ran,
        llm,
    ):
        folder = tmp_path / "ws"
        folder.mkdir()
        assert console.main_id is not None
        await console.store.update_session(
            console.main_id, meta_update={"approvals": "auto"}
        )
        task, _ = await console.workers.create(
            {"prompt": "tidy the notes", "workspace": str(folder)}
        )
        task_id = task["task_id"]
        assert task["approvals"] == "auto"
        deadline = time.monotonic() + TIMEOUT_S
        while (await console.workers.get(task_id))["state"] not in ("done", "failed"):
            assert time.monotonic() < deadline, "the task did not finish"
            await asyncio.sleep(0.02)
        body = (await client.get(f"/api/tasks/{task_id}")).json()["task"]
        assert body["state"] == "done" and ran.codes == []
        [blocked] = body["blocked"]
        assert blocked["can_allow"] is True and blocked["name"] == "python_exec"
        assert blocked["preview"] and blocked["used_at"] is None
        # The note the task read offers its page, not the chat.
        assert any(
            "from the task's page" in m.content for m in llm.requests[-1]
        ), "the task's note"
        steps = await blocked_steps(console.store, task_id)
        assert steps[0]["review"]["categories"] == ["delete"]
        # A later switch of the chat leaves the task as it was made.
        await console.store.update_session(
            console.main_id, meta_update={"approvals": "ask"}
        )
        assert (await console.workers.get(task_id))["approvals"] == "auto"

        # Allow once from the task's page: the finished task goes ahead with it.
        response = await client.post(
            f"/api/tasks/{task_id}/blocks/{blocked['request_id']}/allow"
        )
        assert response.status_code == 202, response.text
        assert response.json()["message"] == "continued"
        assert set(response.json()["task"]["allowances"]) == {blocked["step_key"]}
        deadline = time.monotonic() + TIMEOUT_S
        while not ran.codes or (await console.workers.get(task_id))["state"] != "done":
            assert time.monotonic() < deadline, "the task did not go ahead"
            await asyncio.sleep(0.02)
        assert ran.codes == [DELETE]
        [blocked] = (await console.workers.get(task_id))["blocked"]
        assert blocked["used_at"] is not None
        missing = await client.post(f"/api/tasks/{task_id}/blocks/nope/allow")
        assert missing.status_code == 404


@pytest.mark.asyncio
async def test_a_task_made_in_ask_keeps_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    async with auto_console(tmp_path, [], monkeypatch=monkeypatch) as (
        console,
        _client,
        _ran,
        _llm,
    ):
        folder = tmp_path / "ws"
        folder.mkdir()
        task, _ = await console.workers.create(
            {"prompt": "tidy the notes", "workspace": str(folder), "level": 5}
        )
        assert task["approvals"] == "ask" and task["blocked"] == []
