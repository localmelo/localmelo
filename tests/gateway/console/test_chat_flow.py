"""A task from the chat runs smoothly, end to end (#440).

- A request for work starts at once, in Chinese as in English.
- A call a policy check refused does not use up the chat's steps, and a turn
  that still runs out of steps answers with what it has: never "Start this as
  a task?" for a question the chat answers, nor for a task's status.
- When a task the chat started ends Ready, the chat tells the answer unasked,
  and "is it done?" is answered from the task's state and result.

The person's messages from a real test stand in as generic examples. Scripted
models only: nothing reaches a model server or the network.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.front import TASK_TOOLS
from localmelo.support.gateway.front_rules import ASK_TEXT, ASK_TEXT_ZH
from localmelo.tests.sessions.helpers import answer, tool_call

from .conftest import HOLDER, core_event, run_result
from .test_ask_owner import Reply, rows, tool_results
from .test_chat_owns_tasks import (
    internal_turns,
    owner_console,
    say,
    settled,
    started_task,
)

# The person's messages in the real test (generic examples).
CAREER = "你去我的网站看一下我的简历 然后告诉我什么工作适合我 特别是大公司的什么岗位"
IS_IT_DONE = "跑完了吗"
DIDNT_IT = "之前worker不是已经跑完了吗"
JUST_TELL_ME = "直接和我说就行了"
NOT_SAID = (
    "Not remembered: not something the user said. 简历: not in what the user "
    "typed in this chat or in a finished task's result shown here. Keep only "
    "what the user told you, in their words."
)


class RefusingMemory(BaseToolProvider):
    """A ``remember`` that refuses every note, in the memory's own words."""

    def __init__(self) -> None:
        self.calls = 0

    async def start(self) -> list[ProvidedTool]:
        definition = ToolDef(
            name="remember",
            description="Keep a note about the user.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            annotations={"readOnlyHint": True},
        )
        return [ProvidedTool(definition=definition, function=self.remember)]

    async def remember(self, text: str = "", **_: Any) -> str:
        self.calls += 1
        return NOT_SAID

    async def close(self) -> None:
        return None


def remembers(n: int) -> Message:
    return tool_call("remember", {"text": f"The user has a CV ({n})."}, f"r-{n}")


def lists(n: int) -> Message:
    return tool_call("task_list", {}, f"l-{n}")


def gets(n: int) -> Message:
    return tool_call("task_get", {"task_id": f"nope{n}"}, f"g-{n}")


async def turn_of(console: Console) -> Any:
    turns = await console.store.list_turns(str(console.main_id))
    return [turn for turn in turns if not turn.origin][-1]


async def echo_kinds(console: Console) -> list[str]:
    return [str(row["echo"]) for row in await rows(console, console.main_id)]


# ── 1. A request for work starts at once ──


async def test_the_chinese_request_starts_its_task_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        tool_call(
            "task_create",
            {"title": "适合我的工作", "objective": "Read the site and the CV."},
            "create-1",
        ),
        answer("已经作为任务开始了，结果会回到这里。"),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, CAREER)
        task_id = await started_task(console)
        assert tool_results(models["main"], 1)[-1].startswith("Started task ")
        assert await echo_kinds(console) == ["started"], "started, not offered"
        record = await console.workers.tasks.get_task(task_id)
        assert record.spec["user_words"] == CAREER
        turn = await turn_of(console)
        assert turn.answer == "已经作为任务开始了，结果会回到这里。"


# ── 2. Refused calls don't use up the steps ──


async def test_refused_calls_do_not_use_up_the_chats_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = RefusingMemory()
    main: list[Reply] = [
        remembers(1),
        remembers(2),
        lists(1),
        gets(2),
        gets(3),
        answer("还没有任务在跑。"),
    ]
    async with owner_console(
        tmp_path, monkeypatch, main=main, run=False, front_tools=[memory]
    ) as (console, client, models, _):
        await say(client, console, IS_IT_DONE)
        turn = await turn_of(console)
        assert (turn.status, turn.answer) == ("completed", "还没有任务在跑。")
        assert memory.calls == 2
        assert len(models["main"].requests) == 6, "two refused steps were free"
        assert await echo_kinds(console) == []


async def test_a_refused_task_create_is_free_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        tool_call("task_create", {"title": "Check", "objective": "Check"}, "c-1"),
        lists(1),
        gets(2),
        gets(3),
        answer("No task is running."),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, "is it done?")
        refused = tool_results(models["main"], 1)[-1]
        assert refused.startswith("Not done: this is a question or small talk")
        assert "task_get" in refused, "a status question is answered from the task"
        turn = await turn_of(console)
        assert (turn.status, turn.answer) == ("completed", "No task is running.")
        assert await console.workers.tasks.list_tasks() == []


def _out_of_steps() -> list[Reply]:
    return [lists(1), gets(2), lists(3), gets(4)]


@pytest.mark.parametrize("typed", [IS_IT_DONE, DIDNT_IT, JUST_TELL_ME, "is it done?"])
async def test_a_turn_out_of_steps_answers_with_what_it_has(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, typed: str
) -> None:
    best = "任务已经完成：它的回答在这里。"
    main: list[Reply] = [*_out_of_steps(), answer(best)]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, typed)
        turn = await turn_of(console)
        assert (turn.status, turn.answer) == ("completed", best)
        # The last request asked for the answer, with no tools.
        assert models["main"].offered[-1] == []
        assert await echo_kinds(console) == [], "no Start this as a task?"
        assert await console.workers.tasks.list_tasks() == []


async def test_a_preamble_to_a_status_question_gets_an_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [answer("我先看一下任务列表。"), answer("任务已经完成了。")]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        await say(client, console, IS_IT_DONE)
        turn = await turn_of(console)
        assert turn.answer == "任务已经完成了。"
        assert turn.answer not in (ASK_TEXT, ASK_TEXT_ZH)
        assert await echo_kinds(console) == []


async def test_with_no_answer_at_all_the_turn_says_so_and_offers_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [*_out_of_steps(), answer("")]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        await say(client, console, IS_IT_DONE)
        turn = await turn_of(console)
        assert turn.status == "completed" and turn.answer
        assert turn.answer not in (ASK_TEXT, ASK_TEXT_ZH)
        assert await echo_kinds(console) == []


async def test_a_question_out_of_steps_answers_and_still_offers_a_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    best = "From the files I read, the fixture is missing; a run would tell for sure."
    main: list[Reply] = [*_out_of_steps(), answer(best)]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        await say(client, console, "Why does pytest fail in tests/cli?")
        turn = await turn_of(console)
        assert turn.answer == best
        assert await echo_kinds(console) == ["ask"]


# ── 3. The answer comes back to the chat ──

# A long answer: its key points first, then many lines of detail.
LONG = "\n".join(
    [
        "## Jobs that fit",
        "1. Machine learning systems engineer at a large company.",
        "2. Research engineer on model serving.",
        *[f"- Detail line {n}: what the CV shows for this role." for n in range(100)],
        "Wrote career-advice.md with the full list.",
    ]
)


async def finish_task(
    console: Console, task_id: str, answer_text: str, writes: tuple[str, ...] = ()
) -> None:
    """*task_id* ran one turn that wrote *writes* with file_write and answered *answer_text*."""
    store = console.store
    record = await console.workers.tasks.get_task(task_id)
    info = await store.get_session(task_id)
    assert info is not None and info.workspace
    turn = await store.begin_turn(task_id, record.spec["objective"], holder=HOLDER)
    events = [core_event("run.started", seq=1)]
    for n, name in enumerate(writes):
        (Path(info.workspace) / name).write_text("# Advice\n")
        call: dict[str, Any] = {"call_id": f"w{n}", "name": "file_write"}
        events.append(
            core_event(
                "tool.started",
                seq=2 + 2 * n,
                arguments={"path": name, "content": "# Advice"},
                preview=name,
                **call,
            )
        )
        events.append(core_event("tool.finished", seq=3 + 2 * n, **call))
    await store.append_events(task_id, turn.turn, events, holder=HOLDER)
    await store.finish_turn(
        task_id,
        turn.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(result=answer_text),
        answer=f"{answer_text}\nEnding: ready",
    )
    await console.workers.tasks.change(task_id, state="done", actor="scheduler")


def reads_task(holder: dict[str, str]) -> Reply:
    def reply(messages: list[Message]) -> Message:
        return tool_call("task_get", {"task_id": holder["task_id"]}, "get-1")

    return reply


@pytest.mark.parametrize("typed", [IS_IT_DONE, DIDNT_IT, JUST_TELL_ME])
async def test_is_it_done_is_answered_from_the_tasks_whole_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, typed: str
) -> None:
    holder: dict[str, str] = {}
    told = (
        "跑完了：最适合你的是大公司的机器学习系统工程师；完整建议在 career-advice.md。"
    )
    main: list[Reply] = [
        tool_call("task_create", {"title": "适合我的工作", "objective": "x"}, "c-1"),
        answer("已经作为任务开始了。"),
        reads_task(holder),
        answer(told),
    ]
    async with owner_console(
        tmp_path, monkeypatch, main=main, run=False, follow_up=False
    ) as (console, client, models, _):
        await say(client, console, CAREER)
        holder["task_id"] = task_id = await started_task(console)
        await finish_task(console, task_id, LONG, writes=("career-advice.md",))
        await say(client, console, typed)
        result = tool_results(models["main"], 3)[-1]
        # The whole answer, not its first lines: the last line too, and the file.
        assert "Machine learning systems engineer" in result
        assert "Detail line 99" in result
        assert "Wrote career-advice.md with the full list." in result
        assert "Ending: ready" not in result
        assert "files it wrote: career-advice.md" in result
        turn = await turn_of(console)
        assert turn.answer == told
        assert await echo_kinds(console) == ["started"], "no new offer"
        assert [t.task_id for t in await console.workers.tasks.list_tasks()] == [
            task_id
        ]


async def test_a_very_long_answer_is_cut_with_a_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder: dict[str, str] = {}
    huge = "\n".join(f"Line {n}: " + "word " * 20 for n in range(400))
    main: list[Reply] = [
        tool_call("task_create", {"title": "Report", "objective": "x"}, "c-1"),
        answer("Started."),
        reads_task(holder),
        answer("Done; the report is long, so here are its key points."),
    ]
    async with owner_console(
        tmp_path, monkeypatch, main=main, run=False, follow_up=False
    ) as (console, client, models, _):
        await say(client, console, "research three laptops and write a long report")
        holder["task_id"] = task_id = await started_task(console)
        await finish_task(console, task_id, huge)
        await say(client, console, "is it done?")
        result = tool_results(models["main"], 3)[-1]
        assert "Line 0:" in result and "Line 399:" not in result
        assert "more characters" in result and "Open task" in result
        assert len(result) < 9000


TOLD = "跑完了：最适合你的是大公司的机器学习系统工程师和模型服务方向的研究工程师。完整建议写在 career-advice.md 里。"


async def test_a_ready_tasks_answer_comes_back_to_the_chat_unasked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        tool_call("task_create", {"title": "适合我的工作", "objective": "x"}, "c-1"),
        answer("已经作为任务开始了。"),
        answer(TOLD),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, CAREER)
        task_id = await started_task(console)
        await finish_task(console, task_id, LONG, writes=("career-advice.md",))
        await settled(console)
        [turn] = await internal_turns(console)
        assert (turn.typed_text, turn.status, turn.answer) == ("", "completed", TOLD)
        request = turn.user_text
        assert request.startswith(f"[localmelo · follow-up on task {task_id} · ")
        # It reads the whole answer and the file, and is told how to tell it.
        assert "Jobs that fit" in request and "Detail line 99" in request
        assert "Ending: ready" not in request
        assert "career-advice.md" in request
        assert "in Chinese" in request and "Open task" in request
        assert "task_continue(" not in request
        # Only the task tools; it never continues or starts a task.
        assert set(models["main"].offered[2]) <= {d.name for d in TASK_TOOLS}
        found = await rows(console, console.main_id)
        assert [r["echo"] for r in found] == ["started", "followup", "followed"]
        assert found[1]["why"] == "ready" and found[1]["may_continue"] is False
        assert found[1]["said"]["key"] == "followup.row"
        assert found[2]["did"] == "told"
        activity = (await client.get("/api/activity")).json()
        results = " / ".join(str(row.get("result")) for row in activity["rows"])
        assert "Told you the answer: 适合我的工作" in results
        # The next turn's news does not tell the ending again.
        assert await console.front.news() == ""


async def test_a_ready_follow_up_may_not_continue_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder: dict[str, str] = {}

    def continues(messages: list[Message]) -> Message:
        args = {"task_id": holder["task_id"], "message": "Do more."}
        return tool_call("task_continue", args, "k-1")

    main: list[Reply] = [
        tool_call("task_create", {"title": "Report", "objective": "x"}, "c-1"),
        answer("Started."),
        continues,
        answer("It finished: three laptops compared; the light one wins."),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, "research three laptops and write a report")
        holder["task_id"] = task_id = await started_task(console)
        await finish_task(console, task_id, "Compared three laptops.")
        await settled(console)
        refused = tool_results(models["main"], 3)[-1]
        assert refused.startswith("Not done: the task finished")
        assert (await console.workers.tasks.get_task(task_id)).state == "done"


async def test_a_ready_report_is_dropped_when_the_person_wrote_after_the_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        tool_call("task_create", {"title": "Report", "objective": "x"}, "c-1"),
        answer("Started."),
        answer("It is done: the light laptop wins."),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, "research three laptops and write a report")
        task_id = await started_task(console)
        entered, release = asyncio.Event(), asyncio.Event()
        followups = console.followups
        original = followups.run_next

        async def pending(chat_id: str) -> None:
            entered.set()
            await release.wait()
            await original(chat_id)

        monkeypatch.setattr(followups, "run_next", pending)
        await finish_task(console, task_id, "Compared three laptops.")
        await asyncio.wait_for(entered.wait(), 5.0)
        await say(client, console, "is it done?")
        assert "Compared three laptops" in str(models["main"].requests[-1][-1].content)
        release.set()
        await settled(console)
        assert await internal_turns(console) == [], "the person's turn read it"


async def test_a_task_the_chat_did_not_start_gets_no_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(tmp_path, monkeypatch, run=False) as (
        console,
        _client,
        models,
        _,
    ):
        page, _ = await console.workers.create({"objective": "tidy the notes"})
        await finish_task(console, str(page["task_id"]), "Tidied.")
        await settled(console)
        assert await internal_turns(console) == []
        assert models["main"].requests == []


async def test_with_follow_ups_off_a_ready_task_gets_no_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        tool_call("task_create", {"title": "Report", "objective": "x"}, "c-1"),
        answer("Started."),
    ]
    async with owner_console(
        tmp_path, monkeypatch, main=main, run=False, follow_up=False
    ) as (console, client, models, _):
        await say(client, console, "research three laptops and write a report")
        task_id = await started_task(console)
        await finish_task(console, task_id, "Compared three laptops.")
        await settled(console)
        assert await internal_turns(console) == []
        assert len(models["main"].requests) == 2
