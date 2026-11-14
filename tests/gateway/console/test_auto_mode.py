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
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import (
    ALLOW_ONCE_S,
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
)
from localmelo.melo.contracts.tools import BaseToolProvider
from localmelo.melo.schema import Message
from localmelo.support.gateway.auto_blocks import (
    blocked_steps,
    go_ahead_text,
    own_allowances,
)
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.sessions.approvals import ChatPolicy, chat_mode
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore, typed_words
from localmelo.tests.agent.test_approvals import _answer, _calls
from localmelo.tests.agent.test_auto_mode import ActsOnSite, FakeModel, verdict
from localmelo.tests.sessions.helpers import started_agent

from .conftest import PORT, FakeProbe, all_events, make_cfg, quiet_gate, viewer

DELETE = "import shutil\nshutil.rmtree('.git', ignore_errors=True)"
# A step only a click allows: Auto blocks it, with Allow once, even in a chat
# that could ask (#465).
KEYS = "import os\nprint(open(os.path.expanduser('~/.ssh/id_rsa')).read())"
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
    providers: list[BaseToolProvider] = (),  # type: ignore[assignment]
    watched: bool = True,
) -> AsyncIterator[tuple[Console, httpx.AsyncClient, Ran, FakeModel]]:
    """A console whose chats share one scripted model and one recording python_exec.

    With *monkeypatch*, a saved task's Agent is built the same way. A chat's
    Agent also has the tools of *providers*. *watched*: a page shows every
    chat and task, so a step Auto stops asks on the card (#465); without it
    the hub counts the viewers a test opens.
    """
    if watched:
        with patch.object(SessionHub, "viewers", lambda self, session_id: 1):
            async with auto_console(
                tmp_path, plans, reviews, monkeypatch, providers, watched=False
            ) as made:
                yield made
        return
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
        if providers:
            agent = Agent(
                llm=llm,
                options=AgentOptions(
                    workspace_root=workspace,
                    record_answers=True,
                    mode="gateway",
                    builtins=frozenset({"python_exec"}),
                ),
                tool_providers=list(providers),
            )
            await agent.start()
        else:
            agent = await started_agent(
                Path(workspace),
                llm,
                mode="gateway",
                builtins=frozenset({"python_exec"}),
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


# ── A stopped step asks on the chat's card (#465) ──


async def pending(client: httpx.AsyncClient, session_id: str) -> dict[str, Any]:
    """The chat's first waiting approval, once there is one."""
    deadline = time.monotonic() + TIMEOUT_S
    while True:
        body = (await client.get(f"/api/sessions/{session_id}")).json()
        if body["pending_approvals"]:
            return cast(dict[str, Any], body["pending_approvals"][0])
        if time.monotonic() > deadline:
            raise AssertionError("no card waits in the chat")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize(("choice", "runs"), [("once", True), ("deny", False)])
async def test_in_auto_a_stopped_step_asks_on_the_chat_s_card(
    tmp_path: Path, choice: str, runs: bool
) -> None:
    plans = [step(DELETE), _answer("Done.")]
    async with auto_console(tmp_path, plans) as (console, client, ran, llm):
        sid = await create(client, tmp_path, "Tidy the notes", approvals="auto")
        card = await pending(client, sid)
        assert card["review"]["verdict"] == "high"
        assert card["review"]["categories"] == ["delete"]
        body = {"choice": choice, **({"reason": "keep it"} if choice == "deny" else {})}
        answered = await client.post(f"/api/approvals/{card['request_id']}", json=body)
        assert answered.status_code == 200, answered.text
        await wait_turn(client, sid)
        assert (ran.codes == [DELETE]) is runs
        [approval] = await approvals(console, sid)
        assert (approval["origin"], approval["allowed"]) == ("user", runs)
        session = (await client.get(f"/api/sessions/{sid}")).json()["session"]
        assert session["allowances"] == {}  # a card's answer is no allowance
        assert llm.review_requests == []


@pytest.mark.asyncio
async def test_a_site_allowed_for_the_chat_lasts_until_removed(tmp_path: Path) -> None:
    site = ActsOnSite()
    act = _calls(("act_on_site", {"site": "news.example", "access": "act"}, "c1"))
    plans = [act, _answer("a"), act, _answer("b"), act, _answer("c")]
    not_asked = verdict(verdict="ask", category="beyond_request", fits_request="no")
    async with auto_console(tmp_path, plans, [not_asked] * 3, providers=[site]) as (
        console,
        client,
        _ran,
        _llm,
    ):
        sid = await create(client, tmp_path, "check the news", approvals="auto")
        card = await pending(client, sid)
        name = "browser-act:https://news.example"
        assert (card["tool_name"], card["offers_chat"]) == (name, True)
        answered = await client.post(
            f"/api/approvals/{card['request_id']}", json={"choice": "session"}
        )
        assert answered.json()["remember"] is True
        await wait_turn(client, sid)
        assert (await client.get(f"/api/sessions/{sid}")).json()["grants"] == [name]

        # The next message: the same click on the same site runs on the grant.
        await client.post(f"/api/sessions/{sid}/turns", json={"text": "I signed in"})
        await wait_turn(client, sid)
        assert [a.origin for a in site.answers] == ["user", "session"]

        # Remove (the chat's menu): the next one asks again.
        removed = await client.delete(
            f"/api/sessions/{sid}/grants/{name.replace(':', '%3A').replace('/', '%2F')}"
        )
        assert removed.status_code == 200 and removed.json()["everywhere"] is False
        await client.post(f"/api/sessions/{sid}/turns", json={"text": "once more"})
        card = await pending(client, sid)
        await client.post(
            f"/api/approvals/{card['request_id']}", json={"choice": "deny"}
        )
        await wait_turn(client, sid)
        assert [a.allow for a in site.answers] == [True, True, False]


# ── Nobody watching keeps the block; a card nobody answers is declined (#465) ──


async def empty_chat(client: httpx.AsyncClient, tmp_path: Path) -> str:
    """A chat in Auto with no turn yet, so a page can open it before the first."""
    folder = tmp_path / "ws"
    folder.mkdir(exist_ok=True)
    made = await client.post(
        "/api/sessions", json={"workspace": str(folder), "approvals": "auto"}
    )
    assert made.status_code == 201, made.text
    return str(made.json()["session"]["id"])


async def requested(console: Console, session_id: str) -> int:
    return sum(
        e.event["type"] == "approval.requested"
        for e in await all_events(console.store, session_id)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["nobody watches", "auto_asks off"])
async def test_without_a_card_auto_blocks_at_once_as_before(
    tmp_path: Path, how: str
) -> None:
    # A script sends the turn over HTTP and follows no stream as a viewer; or
    # [approvals] auto_asks is off, and even an open page gets no card.
    async with auto_console(
        tmp_path, [step(DELETE), _answer("Done.")], watched=False
    ) as (
        console,
        client,
        ran,
        llm,
    ):
        started = time.monotonic()
        if how == "nobody watches":
            sid = await create(client, tmp_path, "Tidy the notes", approvals="auto")
            await wait_turn(client, sid)
        else:
            console.cfg.approvals.auto_asks = False
            sid = await empty_chat(client, tmp_path)
            async with viewer(console, sid):
                await client.post(f"/api/sessions/{sid}/turns", json={"text": "Tidy"})
                await wait_turn(client, sid)
        assert time.monotonic() - started < TIMEOUT_S / 2  # nothing waited
        assert ran.codes == [] and await requested(console, sid) == 0
        [approval] = await approvals(console, sid)
        assert (approval["origin"], approval["allowed"]) == ("auto", False)
        [note] = llm.tool_messages()
        assert "Auto mode blocked this step, so it did not run." in note


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["closed", "unanswered"])
async def test_a_card_its_viewer_leaves_or_nobody_answers_is_declined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    import localmelo.support.sessions.approvals as approvals_mod

    monkeypatch.setattr(approvals_mod, "CARD_POLL_S", 0.02)
    monkeypatch.setattr(approvals_mod, "CARD_LEFT_S", 0.1)
    if how == "unanswered":
        monkeypatch.setattr(approvals_mod, "AUTO_ASK_WAIT_S", 0.2)
    plans = [step(DELETE), step(DELETE, "call_2"), _answer("Done.")]
    async with auto_console(tmp_path, plans, watched=False) as (
        console,
        client,
        ran,
        llm,
    ):
        sid = await empty_chat(client, tmp_path)
        async with viewer(console, sid):
            sent = await client.post(
                f"/api/sessions/{sid}/turns", json={"text": "Tidy"}
            )
            assert sent.status_code == 202, sent.text
            card = await pending(client, sid)
            assert card["review"]["categories"] == ["delete"]
            if how == "unanswered":
                await wait_turn(client, sid)
        await wait_turn(client, sid)
        assert ran.codes == []
        first, again = await approvals(console, sid)
        assert (first["origin"], first["allowed"], first["mode"]) == (
            "timeout",
            False,
            "auto",
        )
        reason = "the chat was closed" if how == "closed" else "no answer within 1"
        assert reason in first["reason"]
        # The surfaces read the code, never the text (#465).
        code = {"closed": {"unanswered": "left"}, "unanswered": {"unanswered": "time"}}
        assert {k: first[k] for k in code[how]} == code[how]
        assert again["unanswered"] == first["unanswered"]
        # The same step later in the turn is not asked about again.
        assert again["declined"] is True and await requested(console, sid) == 1
        first_note, again_note = llm.tool_messages()
        assert "not answered the card" in first_note and "not a no" in first_note
        assert "Don't allow" not in first_note and "another way" not in first_note
        assert "did not ask again" in again_note


# ── Allow once, end to end ──


@pytest.mark.asyncio
async def test_allow_once_runs_the_blocked_step_once_then_it_is_blocked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[tuple[str, str | None]] = []
    run_task = Agent.run_task

    async def spy(self: Agent, query: str, **kwargs: Any) -> Any:
        runs.append((query, kwargs.get("user_words")))
        return await run_task(self, query, **kwargs)

    monkeypatch.setattr(Agent, "run_task", spy)
    plans = [
        step(KEYS),
        _answer("I left .git as it is."),
        step(KEYS, "call_2"),
        _answer("Done."),
        step(KEYS, "call_3"),
        _answer("Blocked again."),
    ]
    # Only a click allows it: no review asks the model about words.
    async with auto_console(tmp_path, plans) as (console, client, ran, llm):
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
        assert body["turn"]["user_text"] == go_ahead_text(["credentials"])
        assert body["turn"]["user_text"] == (
            "Go ahead with the step melo blocked: use your passwords or keys."
        )
        await wait_turn(client, sid)
        assert ran.codes == [KEYS]
        # The sentence is melo's: the model gets it, but the stored turn is composed,
        # no reader takes it for words the person typed, and the run counts none.
        assert body["turn"]["composed"] is True
        assert runs[-1] == (body["turn"]["user_text"], "")
        first, sent = await console.store.list_turns(sid)
        assert (typed_words(first), typed_words(sent)) == ("Tidy the notes", "")
        allowed = (await approvals(console, sid))[1]
        assert (allowed["origin"], allowed["allowance"]) == ("user", "once")
        session = (await client.get(f"/api/sessions/{sid}")).json()["session"]
        assert set(session["allowances"]) == {block["step_key"]}
        # The step ran on it: the same block cannot be allowed again.
        used = await client.post(
            f"/api/sessions/{sid}/blocks/{block['request_id']}/allow"
        )
        assert (used.status_code, used.json()["error"]["code"]) == (
            409,
            "already_allowed",
        )

        await client.post(f"/api/sessions/{sid}/turns", json={"text": "Do it again"})
        await wait_turn(client, sid)
        assert ran.codes == [KEYS]
        again = (await approvals(console, sid))[2]
        assert (again["origin"], again["allowed"]) == ("auto", False)
        assert llm.review_requests == []


@pytest.mark.asyncio
async def test_try_again_on_the_go_ahead_turn_stays_melos_and_renews_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[tuple[str, str | None]] = []
    run_task = Agent.run_task

    async def spy(self: Agent, query: str, **kwargs: Any) -> Any:
        runs.append((query, kwargs.get("user_words")))
        return await run_task(self, query, **kwargs)

    monkeypatch.setattr(Agent, "run_task", spy)
    plans = [
        step(KEYS),
        _answer("I left .git as it is."),
        step(KEYS, "call_2"),
        _answer("Done."),
        step(KEYS, "call_3"),
        _answer("Blocked again."),
    ]
    async with auto_console(tmp_path, plans) as (console, client, ran, _):
        sid = await create(client, tmp_path, "Tidy the notes", approvals="auto")
        await wait_turn(client, sid)
        [block] = await approvals(console, sid)
        allowed = await client.post(
            f"/api/sessions/{sid}/blocks/{block['request_id']}/allow"
        )
        await wait_turn(client, sid)
        assert ran.codes == [KEYS]
        sentence = allowed.json()["turn"]["user_text"]
        cards = len(await approvals(console, sid))
        again = await client.post(
            f"/api/sessions/{sid}/retry", json={"turn_id": allowed.json()["turn"]["id"]}
        )
        assert again.status_code == 202, again.text
        await wait_turn(client, sid)
        # The sentence is tried again as melo's: no words of the person's, composed.
        tried = (await console.store.list_turns(sid))[-1]
        assert (tried.user_text, tried.composed, typed_words(tried)) == (
            sentence,
            True,
            "",
        )
        assert runs[-1] == (sentence, "")
        # Try again renews nothing: the allowance was used, so the step is blocked
        # again and its card comes back; it did not run a second time.
        assert ran.codes == [KEYS]
        after = await approvals(console, sid)
        assert len(after) == cards + 1  # exactly one new card
        assert (after[-1]["origin"], after[-1]["allowed"]) == ("auto", False)


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


@pytest.mark.asyncio
async def test_an_allowance_its_turn_did_not_use_goes_and_one_waiting_refuses(
    tmp_path: Path,
) -> None:
    plans = [
        step(KEYS),
        _answer("I left .git as it is."),
        _answer("I will not delete it after all."),
        step(KEYS, "call_2"),
        _answer("Done."),
    ]
    async with auto_console(tmp_path, plans) as (console, client, ran, _llm):
        sid = await create(client, tmp_path, "Tidy the notes", approvals="auto")
        await wait_turn(client, sid)
        [block] = await approvals(console, sid)
        url = f"/api/sessions/{sid}/blocks/{block['request_id']}/allow"
        assert (await client.post(url)).status_code == 202
        await wait_turn(client, sid)
        # The model did not try again: the allowance is gone with its turn.
        entry = console.sessions.get(sid)
        assert entry is not None
        deadline = time.monotonic() + TIMEOUT_S
        while entry.policy.auto.allowed_once(block["step_key"]):
            assert time.monotonic() < deadline, "the allowance stayed"
            await asyncio.sleep(0.01)
        session = (await client.get(f"/api/sessions/{sid}")).json()["session"]
        assert session["allowances"] == {} and ran.codes == []
        # One waiting refuses a second click.
        entry.policy.auto.allow_once(block["step_key"], ["credentials"], "python_exec")
        waiting = await client.post(url)
        assert (waiting.status_code, waiting.json()["error"]["code"]) == (
            409,
            "already_allowed",
        )
        entry.policy.auto.withdraw(block["step_key"])
        # Allowed again, it runs.
        assert (await client.post(url)).status_code == 202
        await wait_turn(client, sid)
        assert ran.codes == [KEYS]


# ── Saved tasks keep their chat's mode ──


@pytest.mark.asyncio
async def test_a_task_made_in_auto_runs_in_auto_and_lists_its_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [
        step(KEYS),
        _answer("It needs that step.\nEnding: blocked: may it read the key?"),
        step(KEYS, "call_2"),
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
        # The note the task read promises no allowance after it ended.
        assert any(
            "new work they ask the chat for" in m.content for m in llm.requests[-1]
        ), "the task's note"
        steps = await blocked_steps(console.store, task_id)
        assert steps[0]["review"]["categories"] == ["credentials"]
        # A later switch of the chat leaves a task that ended as it was; the run
        # the Allow continues decides by its chat's mode (#576), so it is put back.
        await console.store.update_session(
            console.main_id, meta_update={"approvals": "ask"}
        )
        assert (await console.workers.get(task_id))["approvals"] == "auto"
        await console.store.update_session(
            console.main_id, meta_update={"approvals": "auto"}
        )

        # Allow once from the task's page: the job it stopped on goes ahead with it.
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
        assert ran.codes == [KEYS]
        [blocked] = (await console.workers.get(task_id))["blocked"]
        assert blocked["used_at"] is not None
        missing = await client.post(f"/api/tasks/{task_id}/blocks/nope/allow")
        assert missing.status_code == 404
        # Finished, the task keeps no policy; the block that ran is not offered again.
        assert task_id not in console.workers._auto
        again = await client.post(
            f"/api/tasks/{task_id}/blocks/{blocked['request_id']}/allow"
        )
        assert (again.status_code, again.json()["error"]["code"]) == (
            409,
            "already_allowed",
        )


@pytest.mark.asyncio
async def test_allow_once_on_a_chats_ended_task_goes_on_in_a_fresh_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker a chat started runs once (#571): the allowed step runs in a fresh task."""
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [
        step(KEYS),
        _answer("It needs that step.\nEnding: blocked: may it read the key?"),
        step(KEYS, "call_2"),
        _answer("Deleted."),
    ]
    async with auto_console(tmp_path, plans, monkeypatch=monkeypatch) as (
        console,
        client,
        ran,
        _llm,
    ):
        folder = tmp_path / "ws"
        folder.mkdir()
        assert console.main_id is not None
        await console.store.update_session(
            console.main_id, meta_update={"approvals": "auto"}
        )
        spec = TaskSpec.from_payload({"objective": "tidy the notes"})
        task, _ = await console.workers.create_spec(
            dataclasses.replace(spec, workspace=str(folder)), chat_started=True
        )
        task_id = task["task_id"]
        deadline = time.monotonic() + TIMEOUT_S
        while (await console.workers.get(task_id))["state"] not in ("done", "failed"):
            assert time.monotonic() < deadline, "the task did not finish"
            await asyncio.sleep(0.02)
        [blocked] = (await console.workers.get(task_id))["blocked"]
        turns = await console.store.list_turns(task_id)
        response = await client.post(
            f"/api/tasks/{task_id}/blocks/{blocked['request_id']}/allow"
        )
        assert response.status_code == 202, response.text
        fresh = response.json()["task"]
        assert response.json()["message"] == "successor"
        assert fresh["task_id"] != task_id
        assert set(fresh["allowances"]) == {blocked["step_key"]}
        deadline = time.monotonic() + TIMEOUT_S
        while not ran.codes or (await console.workers.get(fresh["task_id"]))[
            "state"
        ] not in ("done", "failed"):
            assert time.monotonic() < deadline, "the fresh task did not go ahead"
            await asyncio.sleep(0.02)
        assert ran.codes == [KEYS], "the allowed step ran once, in the fresh task"
        # The ended task's block reads that it ran, from the fresh task's record:
        # its page and its chat's card say so instead of offering it again.
        [after] = (await console.workers.get(task_id))["blocked"]
        assert after["can_allow"] is False and after["used_at"] is not None
        # The ended task was never reopened, and keeps no policy.
        assert await console.store.list_turns(task_id) == turns
        assert task_id not in console.workers._auto
        # A pending allowance in the fresh task (seeded, so no run can use it
        # first) is the ended task's block's: its key, never another step's.
        policy = AutoPolicy(task=True)
        policy.allow_once(blocked["step_key"], ["credentials"])
        policy.allow_once("another-step", ["delete"])
        console.workers._auto[fresh["task_id"]] = policy
        body = await console.workers.get(task_id)
        assert set(body["allowances"]) == {blocked["step_key"]}
        assert body["blocked"][0]["allowed_until"] == policy.given[blocked["step_key"]]


@pytest.mark.asyncio
async def test_a_job_that_finished_without_a_blocked_step_is_not_reopened_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = [step(KEYS), _answer("The task went on without it.")]
    async with auto_console(tmp_path, plans, monkeypatch=monkeypatch) as (
        console,
        client,
        ran,
        _,
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
        deadline = time.monotonic() + TIMEOUT_S
        while (await console.workers.get(task_id))["state"] != "done":
            assert time.monotonic() < deadline, "the task did not finish"
            await asyncio.sleep(0.02)
        [blocked] = (await console.workers.get(task_id))["blocked"]
        assert blocked["can_allow"] is False, "a finished job offers no Allow once"
        assert blocked["finished"] is True
        turns = await console.store.list_turns(task_id)
        response = await client.post(
            f"/api/tasks/{task_id}/blocks/{blocked['request_id']}/allow"
        )
        assert (response.status_code, response.json()["error"]["code"]) == (
            409,
            "invalid_transition",
        )
        assert (await console.workers.get(task_id))["state"] == "done"
        assert await console.store.list_turns(task_id) == turns
        assert ran.codes == []


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


# ── The gateway is closed to Auto before its first chat (round 3) ──


@pytest.mark.asyncio
async def test_the_gateway_is_registered_before_the_console_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.melo.contracts import own_gateway
    from localmelo.support.gateway import GatewayServer

    monkeypatch.setenv("HOME", str(tmp_path))
    server = GatewayServer(
        make_cfg(),
        console_options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            probe=FakeProbe(),
            gate=quiet_gate(),
        ),
    )
    console = server.console
    assert console is not None
    seen: list[own_gateway.OwnGateway] = []
    start = console.start

    async def starting() -> None:
        seen.append(own_gateway.current())
        await start()

    monkeypatch.setattr(console, "start", starting)
    async with server.app.router.lifespan_context(server.app):
        pass
    [guard] = seen
    assert server.port in guard.ports
    assert str((tmp_path / "sessions.db").resolve()) in guard.files
    assert server.port not in own_gateway.current().ports


def test_a_later_block_of_a_step_does_not_take_an_earlier_blocks_allowance() -> None:
    """An allowance is kept by step; the block it was given for is one made before it."""
    given = {"k1": 1000.0 + ALLOW_ONCE_S}
    blocks: list[dict[str, Any]] = [
        {"step_key": "k1", "ts": 900.0},
        {"step_key": "k1", "ts": 1100.0},
        {"step_key": "k2", "ts": 900.0},
    ]
    own_allowances(blocks, given)
    assert [b["allowed_until"] for b in blocks] == [given["k1"], None, None]
