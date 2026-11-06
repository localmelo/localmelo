"""A chat does what the person asks (#567).

The person says what they want; the chat's model reads it and acts. Each test
runs the console's routes on scripted models: nothing reaches a model server,
a browser or the network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.chores import CHECKIN_S
from localmelo.support.gateway.console.front import (
    CHAT_NOTE,
    FRONT_TEXT,
    NO_ANSWER_TEXT,
    NO_ANSWER_TEXT_ZH,
    TASK_TOOLS,
)
from localmelo.support.gateway.front_rules import create_rule
from localmelo.support.gateway.workers import NO_BROWSER_NOTE
from localmelo.tests.sessions.helpers import answer, tool_call

from .test_ask_owner import Reply, rows, tool_results, until
from .test_chat_owns_tasks import (
    creates,
    owner_console,
    say,
    started_task,
    state,
    stop_run,
)
from .test_chores import Summary, chore_console
from .test_front_followup import _finished
from .test_side_equals import chats_console
from .test_turn_writes import _post_turn, _wait_turn

OK = answer("It is on its way.")

# ── 1. No word list vetoes work ──


@pytest.mark.parametrize(
    "typed",
    [
        "去签到",
        "帮我订明天的机票",
        "帮我在淘宝上买一个充电器",
        "帮我登录 example.com",
        "帮我投简历",
        "sign in to example.com",
        "order me a pizza",
        "book the table for tonight and just tell me the time",
        # A polite opening with no listed verb, long or short.
        "please buy the blue one from the list on example.com and pay with the card I "
        "keep there, then send me nothing more than the order number when it is done",
        "能帮我订一张机票吗？",
        # Words that also steer a task do not stop work on something else.
        "取消我在 example.com 的订阅",
        "cancel my subscription on example.com",
        # A question the model reads as work.
        "what is the cheapest flight to Oslo on example.com?",
    ],
)
async def test_the_models_call_on_a_persons_turn_starts_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, typed: str
) -> None:
    async with owner_console(
        tmp_path, monkeypatch, main=[creates("The work"), OK], run=False
    ) as (console, client, models, _):
        await say(client, console, typed)
        assert tool_results(models["main"])[-1].startswith("Started task ")
        record = await console.workers.tasks.get_task(await started_task(console))
        # The person's words are the task's request, never the model's reading.
        assert record.spec["objective"] == typed
        assert not [
            r for r in await rows(console, console.main_id) if r["echo"] == "ask"
        ]


@pytest.mark.parametrize(
    "typed",
    [
        "",
        "don't change anything, only tell me what is wrong",
        "不用开任务，看一下我的简历然后告诉我适合什么工作",
        "don't do anything yet",
        "先别动手，只解释一下这个错误",
        # A word that looks like a verb in what it is about changes nothing.
        "the fix report: don't start a task, just explain it",
    ],
)
def test_words_that_plainly_say_not_to_act_are_offered(typed: str) -> None:
    assert create_rule(typed) == "offer"


@pytest.mark.parametrize(
    "typed",
    [
        # A limit on the work is no word against the work.
        "fix the login bug without changing anything else",
        "修复登录的 bug，别改其他文件",
        "帮我订机票，别改其他东西",
        "book the hotel, don't touch anything else",
        "don't do it before noon, book it at one",
        # A limit on what comes back is no word against the work.
        "do it and just tell me the result",
        "book the table and only tell me the time",
        "订好机票，只告诉我结果就行",
        # A question and work in one message: the model read them both.
        "到底好了没，好了就去订票",
        "is it done, and book the table for tonight",
        "briefly, what is in my inbox and can you clear it?",
        # Words after a colon are not always text to explain.
        "总结一下：我这周的邮件",
        "translate my site: all pages to French",
        # No list stands in the way of the model's own reading.
        "is it done?",
        "is the fix report done?",
        "hi",
    ],
)
def test_no_other_words_stand_between_the_model_and_the_work(typed: str) -> None:
    assert create_rule(typed) == "start"


def test_nothing_sends_the_person_to_rephrase() -> None:
    told = " ".join(
        (FRONT_TEXT, CHAT_NOTE, NO_ANSWER_TEXT, NO_ANSWER_TEXT_ZH)
    ).casefold()
    assert "as a task" not in told and "作为任务" not in told
    assert "with more detail" not in told and "说得更具体" not in told


# ── 2. A worker is one-time ──

REPORT = "Wrote vendors.md with three vendors.\nEnding: ready"


async def task_of(client: httpx.AsyncClient, console: Console, work: str) -> str:
    """A task the person started in code, without a model call."""
    response = await _post_turn(client, str(console.main_id), f"/task {work}")
    assert response.status_code == 200, response.text
    return str(response.json()["echo"]["task_id"])


async def others(console: Console, task_id: str) -> list[Any]:
    records = await console.workers.tasks.list_tasks()
    return [record for record in records if record.task_id != task_id]


def messages(ids: dict[str, str]) -> Reply:
    """The model sends the person's message to the task of ``ids["task"]``."""
    return lambda _: tool_call("task_message", {"task_id": ids["task"]}, "m-1")


async def test_new_work_for_a_finished_worker_starts_a_new_one_with_its_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids: dict[str, str] = {}
    typed = "add the prices"
    async with owner_console(
        tmp_path, monkeypatch, main=[messages(ids), OK], follow_up=False, run=False
    ) as (console, client, models, _):
        old = ids["task"] = await task_of(client, console, "write the vendors list")
        await _finished(console, old, REPORT)
        turns = await console.store.list_turns(old)
        await say(client, console, typed)
        assert tool_results(models["main"])[-1].startswith("Started task ")
        [new] = await others(console, old)
        # The new worker has the person's words as its request, and the old one's
        # record as data; the old worker's session is as it ended.
        assert new.spec["objective"] == typed and new.spec["user_words"] == typed
        assert "Wrote vendors.md" in new.spec["chat_picked"]
        assert "write the vendors list" in new.spec["chat_picked"]
        assert await state(console, old) == "done"
        assert await console.store.list_turns(old) == turns
        assert [r["echo"] for r in await rows(console, console.main_id)][
            -1
        ] == "started"


async def test_the_record_reaches_the_new_worker_with_memory_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids: dict[str, str] = {}
    async with owner_console(
        tmp_path, monkeypatch, main=[messages(ids), OK], follow_up=False, run=False
    ) as (console, client, _, _):
        console.workers._memory_on = lambda: False  # type: ignore[method-assign]
        old = ids["task"] = await task_of(client, console, "write the vendors list")
        await _finished(console, old, REPORT)
        await say(client, console, "add the prices")
        [new] = await others(console, old)
        assert "Wrote vendors.md" in new.spec["chat_picked"]


async def test_a_follow_word_for_a_finished_worker_starts_a_new_one_with_no_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(tmp_path, monkeypatch, follow_up=False, run=False) as (
        console,
        client,
        models,
        _,
    ):
        old = await task_of(client, console, "write the vendors list")
        await _finished(console, old, REPORT)
        response = await _post_turn(
            client, str(console.main_id), "commit the vendors list"
        )
        assert response.json()["echo"]["echo"] == "started"
        assert models["main"].requests == []
        [new] = await others(console, old)
        assert "Wrote vendors.md" in new.spec["chat_picked"]
        assert await state(console, old) == "done"


async def test_a_suggestion_tapped_after_the_worker_finished_starts_a_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(tmp_path, monkeypatch, follow_up=False, run=False) as (
        console,
        client,
        _,
        _,
    ):
        old = await task_of(client, console, "write the vendors list")
        record = await console.workers.tasks.get_task(old)
        card = await console.front.suggest(record, "send", text="add the prices")
        await _finished(console, old, REPORT)
        tapped = await client.post(
            f"/api/front/echoes/{card['id']}", json={"do": "apply"}
        )
        assert tapped.status_code == 200, tapped.text
        assert tapped.json()["echo"]["echo"] == "started"
        assert await state(console, old) == "done"
        [new] = await others(console, old)
        assert new.spec["objective"] == "add the prices"


async def test_one_message_still_starts_one_task_through_a_finished_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids: dict[str, str] = {}
    main: list[Reply] = [creates("Prices"), messages(ids), OK]
    async with owner_console(
        tmp_path, monkeypatch, main=main, follow_up=False, run=False
    ) as (console, client, models, _):
        old = ids["task"] = await task_of(client, console, "write the vendors list")
        await _finished(console, old, REPORT)
        await say(client, console, "add the prices")
        assert tool_results(models["main"])[-1].startswith("Not done: one task")
        assert len(await others(console, old)) == 1


async def test_words_that_say_not_to_act_start_no_worker_through_a_finished_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids: dict[str, str] = {}
    async with owner_console(
        tmp_path, monkeypatch, main=[messages(ids), OK], follow_up=False, run=False
    ) as (console, client, _, _):
        old = ids["task"] = await task_of(client, console, "write the vendors list")
        await _finished(console, old, REPORT)
        typed = "don't start a task, only tell me what the vendors list says"
        await say(client, console, typed)
        assert await others(console, old) == []
        assert await state(console, old) == "done"


async def test_a_worker_that_stopped_before_finishing_continues_its_own_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids: dict[str, str] = {}
    async with owner_console(
        tmp_path, monkeypatch, main=[messages(ids), OK], follow_up=False, run=False
    ) as (console, client, _, _):
        old = ids["task"] = await task_of(client, console, "write the vendors list")
        await stop_run(console, old)
        await say(client, console, "the vendors need their prices too")
        assert await state(console, old) == "queued"
        assert await others(console, old) == []


async def test_a_draft_is_a_job_not_finished_and_continue_reaches_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(tmp_path, monkeypatch, follow_up=False, run=False) as (
        console,
        client,
        _,
        _,
    ):
        old = await task_of(client, console, "write the vendors list")
        draft = {"state": "draft", "reasons": ["tests_missing"]}
        await console.workers.tasks.change(
            old, state="done", actor="scheduler", ending=draft
        )
        console.gate.finish_task(old, "done")
        response = await _post_turn(
            client, str(console.main_id), "continue the vendors list task"
        )
        assert response.json()["echo"]["echo"] == "sent"
        assert await state(console, old) == "queued"
        assert await others(console, old) == []


async def test_continue_on_a_finished_worker_goes_to_the_model_with_all_its_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    typed = "continue the vendors list task and add the prices"
    async with owner_console(
        tmp_path, monkeypatch, main=[creates("Prices"), OK], follow_up=False, run=False
    ) as (console, client, _, _):
        old = await task_of(client, console, "write the vendors list")
        await _finished(console, old, REPORT)
        await say(client, console, typed)
        [new] = await others(console, old)
        assert new.spec["objective"] == typed
        assert await state(console, old) == "done"


async def test_resume_on_a_finished_worker_never_says_it_is_still_working(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids: dict[str, str] = {}

    def resumes(_: Any) -> Any:
        return tool_call(
            "task_control", {"task_id": ids["task"], "action": "resume"}, "r-1"
        )

    async with owner_console(
        tmp_path, monkeypatch, main=[resumes, OK], follow_up=False, run=False
    ) as (console, client, models, _):
        ids["task"] = await task_of(client, console, "write the vendors list")
        await _finished(console, ids["task"], REPORT)
        await say(client, console, "resume the vendors list please, once more")
        assert tool_results(models["main"])[-1].startswith("Not done: that task")
        assert await state(console, ids["task"]) == "done"


# ── 3. "Do it, and every day at 9" does both ──

BOTH = "download my invoice from example.com now and every day at 9"
ONLY = "every day at 9 brief me on the news"


def schedules(run_now: bool, reading: str = "") -> Reply:
    args = {"title": "The work", "run_now": run_now, "reading": reading}
    return tool_call("task_schedule", args, "s-1")


async def chores(console: Console) -> list[Any]:
    return list(await console.chore_runner.store.list_chores())


async def test_now_and_every_day_saves_the_whole_request_and_runs_it_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(
        tmp_path, monkeypatch, main=[schedules(True), OK], run=False
    ) as (console, client, models, _):
        await say(client, console, BOTH)
        [chore] = await chores(console)
        assert chore.request == BOTH
        assert (chore.schedule.kind, chore.schedule.minute) == ("daily", 9 * 60)
        # It runs now as the person's work in this chat: their words, this chat's.
        [task] = await console.workers.tasks.list_tasks()
        assert task.origin == "user" and task.spec["user_words"] == BOTH
        assert task.spec["objective"].startswith(BOTH)
        assert task.front_id == console.main_id
        kinds = [r["echo"] for r in await rows(console, console.main_id)]
        assert kinds == ["chore_saved", "started"]
        assert "started" in tool_results(models["main"])[-1].lower()
        # The next run's slot stands, and the check-in does not tell this run again.
        assert (await chores(console))[0].last_task_id == task.task_id
        assert (await chores(console))[0].next_run == chore.next_run


async def test_the_run_started_now_works_in_the_chores_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One folder across the chore's runs, the one started with the schedule too (#570)."""
    async with owner_console(
        tmp_path, monkeypatch, main=[schedules(True), OK], run=False
    ) as (console, client, models, _):
        await say(client, console, BOTH)
        [chore] = await chores(console)
        [task] = await console.workers.tasks.list_tasks()
        info = await console.store.get_session(task.task_id)
        assert info is not None and info.workspace is not None
        assert Path(info.workspace).name == f"chore-{chore.chore_id}"


@pytest.mark.parametrize(
    "main",
    [[schedules(True), OK], [creates("Invoice"), schedules(False), OK]],
    ids=["schedule first", "task first"],
)
async def test_a_chores_first_run_is_linked_while_the_chore_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, main: list[Reply]
) -> None:
    """The tick cannot start a second run between the chore's save and its run's link (#570)."""
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        runner = console.chore_runner
        held: list[bool] = []
        link = runner.ran_in_chat

        async def spy(chore_id: str, task_id: str) -> None:
            held.append(runner._locks[chore_id].locked())
            await link(chore_id, task_id)

        monkeypatch.setattr(runner, "ran_in_chat", spy)
        await say(client, console, BOTH)
        assert held == [True]


async def test_a_message_that_only_sets_a_schedule_just_saves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(
        tmp_path, monkeypatch, main=[schedules(False), OK], run=False
    ) as (console, client, _, _):
        await say(client, console, ONLY)
        [chore] = await chores(console)
        assert chore.request == ONLY
        assert await console.workers.tasks.list_tasks() == []
        saved = await rows(console, console.main_id)
        assert [r["echo"] for r in saved] == ["chore_saved"]
        # Undo takes it back, as it does after the card.
        undone = await client.post(
            f"/api/front/echoes/{saved[0]['id']}", json={"do": "undo"}
        )
        assert undone.status_code == 200, undone.text
        assert await chores(console) == []


async def test_a_fragment_keeps_all_its_words_and_says_what_it_points_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    typed = "please also do it every day at 9"
    reading = "Download the invoice from example.com."
    async with owner_console(
        tmp_path, monkeypatch, main=[schedules(False, reading), OK], run=False
    ) as (console, client, _, _):
        await say(client, console, typed)
        [chore] = await chores(console)
        assert chore.request.startswith(typed) and reading in chore.request
        # Its run reads the person's words as theirs, and the reading as data only.
        task = await console.chore_runner.run_now(chore)
        record = await console.workers.tasks.get_task(task["task_id"])
        assert record.spec["user_words"] == typed


async def test_a_chores_run_is_told_the_schedule_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(
        tmp_path, monkeypatch, main=[schedules(False), OK], run=False
    ) as (console, client, _, _):
        await say(client, console, ONLY)
        [chore] = await chores(console)
        task = await console.chore_runner.run_now(chore)
        record = await console.workers.tasks.get_task(task["task_id"])
        assert record.spec["objective"].startswith(ONLY)
        assert "one run" in record.spec["objective"]


async def test_the_card_stays_for_a_turn_that_saved_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = answer("Sure, that is a good habit.")
    async with owner_console(tmp_path, monkeypatch, main=[reply], run=False) as (
        console,
        client,
        _,
        _,
    ):
        await say(client, console, ONLY)
        [card] = await rows(console, console.main_id)
        assert card["echo"] == "chore" and card["text"] == ONLY
        assert await chores(console) == []
        saved = await client.post(
            f"/api/front/echoes/{card['id']}", json={"do": "save"}
        )
        assert saved.status_code == 200, saved.text
        assert (await chores(console))[0].request == ONLY


async def test_a_task_the_model_started_is_told_the_message_names_a_schedule_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [creates("Invoice"), schedules(False), schedules(True), OK]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, BOTH)
        assert "task_schedule" in tool_results(models["main"], 1)[-1]
        # One task and one schedule for the message, however the model got there.
        assert len(await console.workers.tasks.list_tasks()) == 1
        assert len(await chores(console)) == 1
        assert tool_results(models["main"])[-1].startswith("Not done: one schedule")


async def test_a_schedule_melo_cannot_keep_saves_nothing_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(
        tmp_path, monkeypatch, main=[schedules(True), OK], run=False
    ) as (console, client, models, _):
        await say(client, console, "every Monday at 9 brief me on the news")
        assert tool_results(models["main"])[-1].startswith("Not done:")
        assert await chores(console) == []
        assert await console.workers.tasks.list_tasks() == []


async def test_words_that_say_not_to_act_get_the_card_not_a_saved_chore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    typed = "every day at 9 brief me on the news, but don't start anything yet"
    async with owner_console(
        tmp_path, monkeypatch, main=[schedules(True), OK], run=False
    ) as (console, client, _, _):
        await say(client, console, typed)
        assert await chores(console) == []
        assert await console.workers.tasks.list_tasks() == []
        assert [r["echo"] for r in await rows(console, console.main_id)] == ["chore"]


async def test_only_a_persons_own_words_save_a_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn melo began has no typed words: its text never saves a chore."""
    async with owner_console(tmp_path, monkeypatch, run=False) as (console, _, _, _):
        tools = console.front.tools
        assert (await tools.schedule(title="X", run_now=True)).startswith("Not done:")
        assert await chores(console) == []


def test_the_chat_has_the_tool_and_is_told_when_to_use_it() -> None:
    assert "task_schedule" in {tool.name for tool in TASK_TOOLS}
    assert "task_schedule" in FRONT_TEXT


# ── 4. Every chat the same ──


async def test_a_new_chats_first_message_reads_codes_rules_like_any_other(
    tmp_path: Path,
) -> None:
    folder = Path.home() / "side-folder"
    folder.mkdir(exist_ok=True)
    async with chats_console(tmp_path) as (console, client, models):

        async def first(text: str, key: str | None = None) -> dict[str, Any]:
            turn = {"text": text, **({"client_request_id": key} if key else {})}
            made = await client.post(
                "/api/sessions", json={"turn": turn, "workspace": str(folder)}
            )
            assert made.status_code == 201, made.text
            body: dict[str, Any] = made.json()
            return body

        # /task and a change to named files start a task the new chat owns.
        body = await first("/task tidy the notes", "first-1")
        assert body["echo"]["echo"] == "started" and "turn" not in body
        chat = body["session"]["id"]
        record = await console.workers.tasks.get_task(body["echo"]["task_id"])
        assert record.front_id == chat
        assert (await first("fix notes.md"))["echo"]["echo"] == "started"
        # A reminder is proposed there, and a control phrase reaches its task.
        reminder = await first("remind me to call Sam on Friday at 9")
        assert reminder["echo"]["echo"] == "chore"
        assert (await first("pause tidy the notes"))["echo"]["echo"] == "control"
        # A repeat of the first message's key is the same chat, not a second one.
        again = await client.post(
            "/api/sessions",
            json={
                "turn": {
                    "text": "/task tidy the notes",
                    "client_request_id": "first-1",
                },
                "workspace": str(folder),
            },
        )
        assert again.status_code == 200 and again.json()["session"]["id"] == chat
        # None of these called a model, and a plain message is a turn as ever.
        assert not any(llm.requests for llm in models.values())
        plain = await first("hello there")
        assert plain["turn"]["status"] == "running"


REPEATS = {
    "a reminder": "remind me to call Sam on Friday at 9",
    "a change to a task": "pause tidy the notes",
    "a choice between tasks": "pause tidy",
    "new work on a finished task": "redo the vendors list",
}


@pytest.mark.parametrize("what", list(REPEATS))
async def test_a_repeated_first_message_is_one_chat_and_one_effect(
    tmp_path: Path, what: str
) -> None:
    folder = Path.home() / "side-folder"
    folder.mkdir(exist_ok=True)
    async with chats_console(tmp_path) as (console, client, _):
        made: list[str] = []
        for work in ("tidy the notes", "tidy the desk", "write the vendors list"):
            made.append(await task_of(client, console, work))
        await _finished(console, made[2], REPORT)
        turn = {"text": REPEATS[what], "client_request_id": "repeat-1"}
        body = {"turn": turn, "workspace": str(folder)}
        first = await client.post("/api/sessions", json=body)
        assert first.status_code == 201, first.text
        chat = first.json()["session"]["id"]
        lines = await rows(console, chat)
        tasks = await console.workers.tasks.list_tasks()
        # The answer was lost, and the words may read otherwise by now.
        await console.workers.update(made[0], state="stop")
        again = await client.post("/api/sessions", json=body)
        assert again.status_code == 200, again.text
        assert again.json()["session"]["id"] == chat
        assert again.json()["echo"]["id"] == first.json()["echo"]["id"]
        assert again.json()["repeated"] is True
        assert await rows(console, chat) == lines
        assert len(await console.workers.tasks.list_tasks()) == len(tasks)
        # The key is this message's: another one under it is a conflict.
        turn["text"] = "pause tidy the desk"
        other = await client.post("/api/sessions", json=body)
        assert other.status_code == 409, other.text


# ── 6. A waiting task shows in the check-in ──


def held_ask() -> Any:
    """An open question to the person about a task, as ``OwnerAsks`` keeps it."""

    def ask(task_id: str) -> Any:
        return SimpleNamespace(task_id=task_id, title="Slides")

    return ask


async def test_a_task_held_on_an_approval_shows_in_the_check_in(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path, summary=Summary("Slides waits for you.")) as (
        console, client, clock, summary, _
    ):  # fmt: skip
        task, _ = await console.workers.create(
            {"objective": "Make slides", "title": "Slides"}
        )
        async with console.store.component_transaction("test_state") as conn:
            await conn.execute(
                "UPDATE tasks SET state = 'held', wait_reason = 'approval', "
                "updated_at = ? WHERE task_id = ?",
                (clock.now, task["task_id"]),
            )
        await console.chore_runner.checkin()
        assert summary.facts == [], "the approval card told the person just now"
        clock.now += CHECKIN_S + 1
        card = await console.chore_runner.checkin()
        assert card is not None
        assert summary.facts == [["Slides: waiting for the user"]]
        clock.now += CHECKIN_S
        assert await console.chore_runner.checkin() is None, "mentioned once"


async def test_a_task_held_for_a_model_slot_is_not_waiting_for_the_person(
    tmp_path: Path,
) -> None:
    async with chore_console(tmp_path) as (console, client, clock, summary, _):
        task, _ = await console.workers.create(
            {"objective": "Make slides", "title": "Slides"}
        )
        async with console.store.component_transaction("test_state") as conn:
            await conn.execute(
                "UPDATE tasks SET state = 'held', wait_reason = 'Resources', "
                "updated_at = ? WHERE task_id = ?",
                (clock.now, task["task_id"]),
            )
        clock.now += CHECKIN_S + 1
        assert await console.chore_runner.checkin() is None


async def test_a_task_whose_question_waits_for_the_person_shows_in_the_check_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task on a question to the person stays ``running``: the ask says it waits."""
    question = held_ask()
    async with chore_console(tmp_path) as (console, client, clock, summary, _):
        task, _ = await console.workers.create(
            {"objective": "Make slides", "title": "Slides"}
        )
        chat = str(console.main_id)
        async with console.store.component_transaction("test_state") as conn:
            await conn.execute(
                "UPDATE tasks SET state = 'running', updated_at = ? WHERE task_id = ?",
                (clock.now, task["task_id"]),
            )
        monkeypatch.setattr(
            console.asks,
            "waiting_on_person",
            lambda owner: [question(task["task_id"])] if owner == chat else [],
        )
        await console.chore_runner.checkin()
        assert summary.facts == [], "just asked: the reply box told the person"
        clock.now += CHECKIN_S + 1
        await console.chore_runner.checkin()
        assert summary.facts == [["Slides: waiting for the user"]]


# ── One message, one receipt ──

PROMISE = answer("I'll take care of that now.")
SAVED = answer("It is saved for every day at 9.")


async def retry(client: httpx.AsyncClient, console: Console) -> None:
    """The person's Retry of the chat's newest turn."""
    chat = str(console.main_id)
    turn = await _wait_turn(client, chat)
    done = await client.post(
        f"/api/sessions/{chat}/retry", json={"turn_id": turn["id"]}
    )
    assert done.status_code == 202, done.text
    await _wait_turn(client, chat)


async def card_of(console: Console) -> dict[str, Any]:
    [card] = [r for r in await rows(console, console.main_id) if r["echo"] == "chore"]
    return card


async def test_just_once_after_the_message_started_its_work_starts_nothing_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(
        tmp_path, monkeypatch, main=[creates("Invoice"), OK], run=False
    ) as (console, client, _, _):
        await say(client, console, BOTH)
        [task] = await console.workers.tasks.list_tasks()
        card = await card_of(console)
        tapped = await client.post(
            f"/api/front/echoes/{card['id']}", json={"do": "once"}
        )
        assert len(await console.workers.tasks.list_tasks()) == 1
        assert tapped.status_code == 400 and card["ran"] == task.task_id
        # What is left of the message saves, and that task is its first run.
        saved = await client.post(
            f"/api/front/echoes/{card['id']}", json={"do": "save"}
        )
        assert saved.status_code == 200, saved.text
        [chore] = await chores(console)
        assert chore.last_task_id == task.task_id


async def test_just_once_and_a_retry_start_the_same_work_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [answer("That is a good habit."), PROMISE]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _,
    ):
        await say(client, console, BOTH)
        card = await card_of(console)
        tapped = await client.post(
            f"/api/front/echoes/{card['id']}", json={"do": "once"}
        )
        assert tapped.status_code == 200, tapped.text
        await retry(client, console)
        assert len(await console.workers.tasks.list_tasks()) == 1


@pytest.mark.parametrize(
    "again", [[PROMISE], [schedules(True), PROMISE]], ids=["promise", "schedule"]
)
async def test_a_retry_of_a_message_that_only_saved_a_schedule_runs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, again: list[Reply]
) -> None:
    main: list[Reply] = [schedules(False), SAVED, *again, SAVED]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _,
    ):
        await say(client, console, ONLY)
        await retry(client, console)
        assert len(await chores(console)) == 1
        assert await console.workers.tasks.list_tasks() == []


@pytest.mark.parametrize(
    "calls",
    [
        [creates("Invoice"), schedules(True)],
        [creates("Invoice"), schedules(False)],
        [schedules(True)],
        [schedules(False), creates("Invoice")],
    ],
    ids=["create, now", "create, later", "schedule, now", "schedule, then create"],
)
async def test_the_run_a_message_started_is_its_chores_run_in_either_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[Reply]
) -> None:
    async with owner_console(tmp_path, monkeypatch, main=[*calls, OK], run=False) as (
        console,
        client,
        _,
        _,
    ):
        await say(client, console, BOTH)
        [task] = await console.workers.tasks.list_tasks()
        [chore] = await chores(console)
        assert chore.last_task_id == task.task_id
        assert chore.reported_task_id == task.task_id
        # The next slot comes while that run is held: none starts beside it.
        runner = console.chore_runner
        await runner.store.advance(chore, next_run=runner.clock() - 1)
        assert await runner.tick() == []
        assert len(await console.workers.tasks.list_tasks()) == 1


# ── The worker's lock decides ──


async def test_a_worker_that_finishes_before_the_message_arrives_is_not_reopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(tmp_path, monkeypatch, follow_up=False, run=False) as (
        console,
        client,
        _,
        _,
    ):
        old = await task_of(client, console, "write the vendors list")
        read = await console.workers.tasks.get_task(old)
        await _finished(console, old, REPORT)
        turns = await console.store.list_turns(old)
        typed = "add the prices"
        echo = await console.front.follow(read, typed, typed=typed, show=typed)
        assert echo["echo"] == "started" and echo["task_id"] != old
        assert await state(console, old) == "done"
        assert await console.store.list_turns(old) == turns
        [new] = await others(console, old)
        assert "Wrote vendors.md" in new.spec["chat_picked"]


async def test_a_turn_melo_began_saves_no_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [creates("Long report"), OK, schedules(True), SAVED]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, "write the long report from my notes")
        await stop_run(console, await started_task(console))
        await until(lambda: len(models["main"].requests) >= 4)
        refused = tool_results(models["main"], 3)[-1]
        assert refused.startswith("Not done: only the user's own message")
        assert await chores(console) == []


# ── 7. A website request with browsing off ──


def web_task(title: str = "Sign in") -> Reply:
    args = {"title": title, "objective": title, "uses_website": True}
    return tool_call("task_create", args, "create-web")


async def off_lines(console: Console) -> list[dict[str, Any]]:
    found = await rows(console, console.main_id)
    return [
        r for r in found if r.get("said", {}).get("key") == "front.echo.browsingOff"
    ]


async def test_browsing_off_is_said_once_with_how_to_turn_it_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [web_task(), OK, web_task("Buy it"), OK]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, "去签到")
        await say(client, console, "buy it on example.com")
        [line] = await off_lines(console)
        assert "/browser chrome" in line["line"] and "Settings" in line["line"]
        # The task started as it should: this is a word about it, not a failure.
        assert line["echo"] == "notice"
        assert [r["echo"] for r in await rows(console, console.main_id)][:2] == [
            "started",
            "notice",
        ]
        # The model hears it: it is told the state, and promises no browser.
        assert (
            "browsing is off" in str(models["main"].requests[0][-1].content).casefold()
        )
        assert "browsing is off" in tool_results(models["main"], 1)[-1].casefold()


async def test_a_saved_chore_that_needs_a_website_says_browsing_is_off_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same words as for a task: a chore's runs have no browser either (#570)."""
    args = {"title": "Invoice", "run_now": False, "uses_website": True}
    saves = tool_call("task_schedule", args, "s-web")
    main: list[Reply] = [saves, OK, web_task(), OK]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(
            client, console, "download my invoice from example.com every day at 9"
        )
        assert len(await chores(console)) == 1
        [line] = await off_lines(console)
        assert "/browser chrome" in line["line"] and "Settings" in line["line"]
        assert "browsing is off" in tool_results(models["main"], 1)[-1].casefold()
        # Once for the chat: a task that needs a website after it adds no second line.
        await say(client, console, "buy it on example.com")
        assert len(await off_lines(console)) == 1


async def test_with_browsing_on_nothing_is_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(
        tmp_path, monkeypatch, main=[web_task(), OK], run=False
    ) as (console, client, models, _):
        console.cfg.browser.driver = "chrome"
        await say(client, console, "去签到")
        assert await off_lines(console) == []
        assert "browsing is off" not in str(models["main"].requests[0]).casefold()


async def test_work_with_no_website_says_nothing_of_browsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(
        tmp_path, monkeypatch, main=[creates("Notes"), OK], run=False
    ) as (console, client, _, _):
        await say(client, console, "tidy my notes folder")
        assert await off_lines(console) == []


async def test_a_worker_with_no_browser_is_told_so_and_one_with_it_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(tmp_path, monkeypatch, follow_up=False) as (
        console,
        client,
        models,
        _,
    ):
        chat = str(console.main_id)
        await _post_turn(client, chat, "/task tidy my notes folder")
        await until(
            lambda: len(models["workers"]) == 1 and models["workers"][0].requests
        )
        assert NO_BROWSER_NOTE in str(models["workers"][0].requests[0][0].content)
        console.cfg.browser.driver = "chrome"
        await _post_turn(client, chat, "/task tidy my photos folder")
        await until(
            lambda: len(models["workers"]) == 2 and models["workers"][1].requests
        )
        assert NO_BROWSER_NOTE not in str(models["workers"][1].requests[0][0].content)


def test_no_words_promise_a_browser_unconditionally() -> None:
    create = next(tool for tool in TASK_TOOLS if tool.name == "task_create")
    for words in (FRONT_TEXT, create.description):
        assert "A task has the browser" not in words
        assert "its browser shows live here" not in words
