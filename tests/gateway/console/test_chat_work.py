"""Work that belongs to a chat: a task browses, its card shows in the chat that asked, and the chat stays free.

Each test drives the console's routes with real workers on scripted models, and a scripted browser that streams. The
fail-closed case for a task no person's chat owns is in ``tests/gateway/test_task_folders.py``; admission is in
``test_front.py``, renewal in ``tests/agent/test_extend_steps.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.contracts.approvals import ApprovalRequest
from localmelo.melo.contracts.sign_in import handles_in, hide_passwords
from localmelo.melo.schema import EXTEND_STEPS, MAX_AGENT_STEPS
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.gateway import workers as workers_module
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.browser_cards import BROWSER_EVENT
from localmelo.support.gateway.finishing import is_out_of_steps
from localmelo.tests.browser.test_live import HOME, site
from localmelo.tests.sessions.helpers import answer, tool_call

from .test_ask_owner import Reply, ask_console, ended, held, state, until
from .test_turn_writes import _post_turn, _wait_turn

OPEN = tool_call("browser_open", {"url": HOME}, "b-1")
ASK = f"/task read {HOME} for me"
# Working, or waiting a moment for the model while the chat has it.
LIVE = ("running", "held")


def browser() -> BrowserToolProvider:
    return BrowserToolProvider(site())  # type: ignore[arg-type]


async def chat_of(client: httpx.AsyncClient, console: Console, kind: str) -> str:
    """The main chat, or a new side chat that has said hello."""
    if kind == "main":
        return str(console.main_id)
    folder = Path.home() / "side"
    folder.mkdir(exist_ok=True)
    made = await client.post(
        "/api/sessions", json={"turn": {"text": "hi"}, "workspace": str(folder)}
    )
    chat = str(made.json()["session"]["id"])
    await _wait_turn(client, chat)
    return chat


async def begin(client: httpx.AsyncClient, chat: str, words: str = ASK) -> str:
    response = await _post_turn(client, chat, words)
    assert response.status_code == 200, response.text
    return str(response.json()["echo"]["task_id"])


async def allow_site(client: httpx.AsyncClient, console: Console, task: str) -> None:
    """The person's Allow of the site the task's browser asks about, as in a chat."""
    await until(lambda: console.broker.pending(task))
    [request] = console.broker.pending(task)
    answered = await client.post(
        f"/api/approvals/{request.request_id}", json={"choice": "once"}
    )
    assert answered.status_code == 200, answered.text


async def card(console: Console, chat: str, wanted: str = "working") -> dict[str, Any]:
    """The newest card of *chat*, once it reads *wanted*."""

    async def check() -> bool:
        rows = await console.store.events_of_type(chat, BROWSER_EVENT, limit=50)
        return bool(rows) and rows[-1].event["data"]["state"] == wanted

    await until(check)
    rows = await console.store.events_of_type(chat, BROWSER_EVENT, limit=50)
    data: dict[str, Any] = rows[-1].event["data"]
    return data


async def stop(
    client: httpx.AsyncClient, task: str, named: dict[str, str]
) -> httpx.Response:
    """Stop on a task's card: the task, and the source the card names."""
    return await client.patch(
        f"/api/tasks/{task}", json={"state": "stop", "source": named}
    )


def source(found: dict[str, Any], **change: str) -> dict[str, str]:
    return {
        "gen": found["browser_session_id"],
        "run": found["run_id"],
        "owner": found["owner_chat_id"],
        **change,
    }


@pytest.mark.parametrize("kind", ["main", "side"])
async def test_a_tasks_card_shows_in_the_chat_that_asked_while_that_chat_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    go = asyncio.Event()
    work: list[Reply] = [OPEN, held(go, answer("Read it."))]
    says: list[Reply] = [answer("hi"), answer("Yes, it is working on that.")]
    async with ask_console(
        tmp_path,
        monkeypatch,
        main=says[1:],
        sides=[says],
        workers=[work],
        browser=browser,
    ) as (console, client, _):
        chat = await chat_of(client, console, kind)
        task = await begin(client, chat)
        await allow_site(client, console, task)
        found = await card(console, chat)
        assert found["owner_chat_id"] == chat
        assert (found["source_session_id"], found["source_kind"]) == (task, "task")
        assert found["goal"] == f"read {HOME} for me"
        assert found["card_key"] == f"{task}:{found['run_id']}"
        # Nothing of the task's reaches the chat as a turn: the chat is free.
        entry = console.sessions.get(chat)
        assert entry is None or not entry.runner.busy
        await _post_turn(client, chat, "is that going ok?")
        turn = await _wait_turn(client, chat)
        assert turn["answer"] == "Yes, it is working on that."
        assert await state(console, task) in LIVE
        go.set()
        await ended(console, task)
        assert (await card(console, chat, "done"))["goal"] == found["goal"]
        task_rows = await console.store.events_of_type(task, BROWSER_EVENT)
        assert task_rows == [], "the card is the chat's, not the task's"


async def test_a_task_given_with_no_chat_has_no_browser_and_one_naming_its_chat_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stored under the main chat either way: only a chat named when it was made counts."""
    async with ask_console(
        tmp_path,
        monkeypatch,
        workers=[[answer("Read it.")], [answer("Read it.")]],
        browser=browser,
    ) as (console, client, models):
        main = str(console.main_id)
        given = await client.post("/api/tasks", json={"objective": "read the news"})
        assert given.status_code == 202, given.text
        assert given.json()["task"]["front_id"] == main
        await ended(console, given.json()["task"]["task_id"])
        assert models["browsers"] == []
        named = await client.post(
            "/api/tasks", json={"objective": "read more news", "front_id": main}
        )
        assert named.status_code == 202, named.text
        await ended(console, named.json()["task"]["task_id"])
        assert len(models["browsers"]) == 1


async def test_a_tasks_browser_is_reached_only_by_its_exact_source_and_moves_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    go = asyncio.Event()
    work: list[Reply] = [OPEN, held(go, answer("Read it."))]
    async with ask_console(
        tmp_path, monkeypatch, sides=[[answer("hi")]], workers=[work], browser=browser
    ) as (console, client, _):
        chat = await chat_of(client, console, "side")
        task = await begin(client, chat)
        await allow_site(client, console, task)
        found = await card(console, chat)
        url = f"/api/sessions/{task}/browser"
        exact = source(found)

        async def page_open() -> bool:
            state_now = await client.get(url, params=exact)
            return state_now.json()["open"] is True

        await until(page_open)
        for wrong in (
            {},
            {"gen": exact["gen"], "run": exact["run"]},
            source(found, gen="00000000"),
            source(found, run="another-run"),
            source(found, owner=str(console.main_id)),
        ):
            for method, path in (
                ("GET", url),
                ("POST", f"{url}/done"),
                ("POST", f"{url}/window"),
            ):
                answered = await client.request(method, path, params=wrong)
                assert answered.status_code == 409, (method, wrong)
            if wrong.keys() == exact.keys():
                refused = await stop(client, task, wrong)
                assert refused.status_code == 409, refused.text
        assert await state(console, task) in LIVE

        # The chat is deleted: the card and its controls are the main chat's now.
        deleted = await client.delete(f"/api/sessions/{chat}")
        assert deleted.status_code in (200, 204), deleted.text
        main = str(console.main_id)
        moved = await card(console, main)
        assert moved["card_key"] == found["card_key"]
        old = await client.post(f"{url}/done", params=exact)
        assert old.status_code == 409
        assert (await stop(client, task, exact)).status_code == 409
        new = await client.post(f"{url}/done", params=source(moved))
        assert new.status_code == 200
        entry = console.sessions.get(main)
        assert entry is None or not entry.runner.busy, "it never takes the chat's turn"
        go.set()
        await ended(console, task)


@pytest.mark.parametrize("how", ["done", "stop"])
async def test_a_page_the_person_takes_holds_that_run_until_done_or_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    go = asyncio.Event()
    work: list[Reply] = [
        OPEN,
        held(go, tool_call("browser_read", {}, "b-2")),
        answer("Read it."),
    ]
    async with ask_console(tmp_path, monkeypatch, workers=[work], browser=browser) as (
        console,
        client,
        models,
    ):
        chat = str(console.main_id)
        task = await begin(client, chat)
        await allow_site(client, console, task)
        found = await card(console, chat)
        [provider] = models["browsers"]
        provider.owners.take("t1")
        yours = await card(console, chat, "you")
        assert (yours["card_key"], yours["run_id"]) == (
            found["card_key"],
            found["run_id"],
        )
        model = models["workers"][0]
        # The worker's held request is asked before it is released: a request that
        # came after the count would read as a call made during the hold.
        await until(lambda: len(model.requests) == 2)
        go.set()
        asked = len(model.requests)
        await asyncio.sleep(0.05)
        assert len(model.requests) == asked, "no model call while the person has it"
        record = await console.workers.tasks.get_task(task)
        assert not console.workers._can_yield(record), "never parked from under them"
        # Health says whose page it is, with the exact source the app's Done sends.
        health = (await client.get("/api/health")).json()["status"]["control"]
        assert health == [{"kind": "task", "id": task, **source(yours)}]
        if how == "done":
            held_page = {k: health[0][k] for k in ("gen", "run", "owner")}
            answered = await client.post(
                f"/api/sessions/{task}/browser/done", params=held_page
            )
            assert answered.json() == {"ok": True}
            await ended(console, task)
            assert (await _turns(console, task)) == 1, "the same run went on"
            assert (await card(console, chat, "done"))["run_id"] == found["run_id"]
        else:
            response = await client.patch(f"/api/tasks/{task}", json={"state": "stop"})
            assert response.status_code == 200
            await ended(console, task, "cancelled")
            stopped = await card(console, chat, "stopped")
            assert (stopped["by"], stopped["run_id"]) == ("you", found["run_id"])
            assert not provider.owners.held


async def _turns(console: Console, task: str) -> int:
    return len(await console.store.list_turns(task))


async def test_stop_on_a_tasks_card_reaches_its_run_and_never_a_later_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, later = asyncio.Event(), asyncio.Event()
    runs: list[list[Reply]] = [
        [OPEN, held(first, answer("Read it."))],
        [OPEN, held(later, answer("Read it again."))],
    ]
    async with ask_console(tmp_path, monkeypatch, workers=runs, browser=browser) as (
        console,
        client,
        _,
    ):
        chat = str(console.main_id)
        # A task given directly, which goes on in its own session (a chat's own
        # worker runs once, #571).
        made, _ = await console.workers.create(
            {"prompt": f"read {HOME} for me", "front_id": chat}
        )
        task = str(made["task_id"])
        await allow_site(client, console, task)
        old = await card(console, chat)
        first.set()
        await ended(console, task)
        await console.workers.message(task, f"read {HOME} once more")
        await allow_site(client, console, task)

        async def next_card() -> bool:
            rows = await console.store.events_of_type(chat, BROWSER_EVENT, limit=50)
            data = rows[-1].event["data"]
            return bool(data["run_id"] != old["run_id"] and data["state"] == "working")

        await until(next_card)
        new = await card(console, chat)
        refused = await stop(client, task, source(old))
        assert refused.status_code == 409, refused.text
        assert await state(console, task) in LIVE
        stopped = await stop(client, task, source(new))
        assert stopped.status_code == 200, stopped.text
        later.set()
        await ended(console, task, "cancelled")


async def test_a_password_is_good_for_the_persons_run_and_not_a_chats_own_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[list[Reply]] = [[answer("Done.")] for _ in range(3)]
    named = "example.com"
    async with ask_console(tmp_path, monkeypatch, workers=runs) as (
        console,
        client,
        models,
    ):
        task = await begin(
            client,
            str(console.main_id),
            f"/task sign in to {named}, my password is hunter2-xy",
        )
        await ended(console, task)
        record = await console.workers.tasks.get_task(task)
        [given] = handles_in(str(record.spec["objective"]))
        assert models["agents"][0]._reviewer.sign_in_value(given, named)

        # The chat's own continue goes on in a fresh task (#571): no handle moves.
        fresh, _ = await console.workers.successor(
            record, "look at it again", by="chat"
        )
        await ended(console, fresh["task_id"])
        assert models["agents"][1]._reviewer.sign_in_value(given, named) is None
        assert not models["agents"][1]._reviewer.signs_in(named)

        # The person's own message gives one again, and only that one.
        later, _ = await console.workers.successor(
            await console.workers.tasks.get_task(fresh["task_id"]),
            hide_passwords(f"sign in to {named}, my password is other-pass-9"),
            by="person",
        )
        await ended(console, later["task_id"])
        again = models["agents"][2]._reviewer
        assert again.sign_in_value(given, named) is None
        assert again.signs_in(named)


async def test_a_job_past_the_fifth_step_goes_on_while_the_chat_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    go = asyncio.Event()
    looks: list[Reply] = [
        tool_call("search_files", {"pattern": f"word{n}"}, f"s-{n}") for n in range(6)
    ]
    work: list[Reply] = [*looks, held(go, answer("Found it."))]
    async with ask_console(
        tmp_path,
        monkeypatch,
        main=[answer("It is still looking.")],
        workers=[work],
    ) as (console, client, models):
        chat = str(console.main_id)
        task = await begin(client, chat)
        await until(
            lambda: models["workers"] and len(models["workers"][0].requests) == 7
        )
        await _post_turn(client, chat, "how is it going?")
        assert (await _wait_turn(client, chat))["answer"] == "It is still looking."
        assert await state(console, task) in LIVE
        go.set()
        await ended(console, task)
        # One run from the first step to the last: nothing cancelled, nothing sent again.
        assert await _turns(console, task) == 1
        assert len(models["workers"]) == 1
        assert len(await console.workers.tasks.list_tasks()) == 1


async def test_work_that_keeps_finding_new_calls_ends_at_its_bound_and_can_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound is the ordinary step limit, further on: the same stop, the same Continue."""
    monkeypatch.setattr(workers_module, "TASK_EXTENSIONS", 1)
    last = MAX_AGENT_STEPS + EXTEND_STEPS
    work: list[Reply] = [
        tool_call("search_files", {"pattern": f"word{n}"}, f"s-{n}")
        for n in range(last + 5)
    ]
    async with ask_console(
        tmp_path, monkeypatch, workers=[work, [answer("Finished.")]]
    ) as (console, client, _):
        task = await begin(client, str(console.main_id))
        await ended(console, task, "failed")
        record = await console.workers.tasks.get_task(task)
        assert is_out_of_steps(record.state, record.wait_reason)
        [turn] = await console.store.list_turns(task)
        assert turn.result is not None and turn.result["steps"] == last
        again = await client.post(f"/api/tasks/{task}/continue")
        assert again.status_code == 202, again.text


async def test_a_step_a_chat_task_must_ask_about_waits_for_the_person_not_its_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backgrounding gives no broader grant: what asks in a chat asks here, and only the person's answer counts."""
    go = asyncio.Event()
    other = "https://example.org/"
    work: list[Reply] = [
        tool_call("browser_open", {"url": other}, "b-1"),
        held(go, answer("Done.")),
    ]
    async with ask_console(tmp_path, monkeypatch, workers=[work], browser=browser) as (
        console,
        client,
        _,
    ):
        chat = str(console.main_id)
        task = await begin(client, chat)

        def waiting() -> list[ApprovalRequest]:
            return list(console.broker.pending(task))

        await until(waiting)
        [request] = waiting()
        assert request.tool_name == f"browser:{other.rstrip('/')}"
        # Its call can hold a page's words: no model writes beside the card.
        assert console.explanations.get(request.request_id)["state"] == "none"
        # What the chat's model says to the task approves nothing: the card still waits.
        said = await console.asks.answer(console.front, task, "yes, go ahead")
        assert said.startswith("Not ")
        assert waiting() == [request]
        answered = await client.post(
            f"/api/approvals/{request.request_id}", json={"choice": "deny"}
        )
        assert answered.status_code == 200, answered.text
        go.set()
        await ended(console, task)
