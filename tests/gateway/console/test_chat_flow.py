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

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.front_rules import ASK_TEXT, ASK_TEXT_ZH
from localmelo.tests.sessions.helpers import answer, tool_call

from .test_ask_owner import Reply, rows, tool_results
from .test_chat_owns_tasks import owner_console, say, started_task

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
