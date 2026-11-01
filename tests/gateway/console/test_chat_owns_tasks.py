"""The main chat starts a task itself (#410) and follows up when it stops (#411).

Each test runs the console on scripted models: the main chat's FrontAgent, and
each task run's Agent with a ``python_exec`` that records the code it was given
instead of running it. Nothing reaches a model server or the network.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.schema import Message
from localmelo.support.gateway.chat_gate import GatedLLM
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.console.followups import (
    FOLLOWUPS_PER_MESSAGE,
    asked_question,
)
from localmelo.support.gateway.console.front import (
    FRONT_TEXT,
    FrontAgent,
    front_options,
)
from localmelo.support.gateway.finishing import OUT_OF_STEPS
from localmelo.support.gateway.front_rules import hands_off, start_rule
from localmelo.support.sessions.agents import session_options
from localmelo.tests.sessions.helpers import answer, tool_call

from .conftest import (
    HOLDER,
    PORT,
    FakeProbe,
    all_events,
    make_cfg,
    quiet_gate,
    run_result,
)
from .test_ask_owner import Reply, Script, held, rows, tool_results, until
from .test_turn_writes import _post_turn, _wait_turn

INSTALL = (
    'import subprocess\nsubprocess.run(["pip", "install", "requests"], check=True)'
)
_FOLLOWUP = re.compile(r"\[localmelo · follow-up on task ([0-9a-f]{12}) · ")


class Ran:
    """The code each task run's python_exec was given, in order."""

    def __init__(self) -> None:
        self.codes: list[str] = []

    async def python_exec(self, code: str) -> str:
        self.codes.append(code)
        return "ok"


async def no_explanation(messages: list[Message]) -> str:
    return "It runs a step of the task."


async def first_words(request: str) -> str:
    return ""


@contextlib.asynccontextmanager
async def owner_console(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    main: list[Reply] | None = None,
    workers: list[list[Reply]] | None = None,
    follow_up: bool = True,
    approvals: str = "ask",
    run: bool = True,
) -> AsyncIterator[tuple[Console, httpx.AsyncClient, dict[str, Any], Ran]]:
    """A console whose main chat and task runs work on scripted models.

    *follow_up* is ``[tasks] follow_up``, *approvals* the mode new chats (and
    the main chat) start in; without *run* no task starts.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    models: dict[str, Any] = {"main": Script(main or []), "workers": []}
    scripts = [list(script) for script in workers or []]
    ran = Ran()
    holder: list[Console] = []

    async def front(settings: Any, workspace: str, mcp: Any, session_id: str) -> Any:
        options = session_options(front_options(settings), workspace)
        llm = GatedLLM(models["main"], holder[0].gate.client(session_id))
        agent = FrontAgent(
            llm=llm, options=options, tool_providers=[holder[0].front.tools]
        )
        agent.front = holder[0].front
        await agent.start()
        return agent

    async def side(settings: Any, workspace: str, mcp: Any, session_id: str) -> Any:
        raise AssertionError("a side chat is not opened here")

    async def worker(**kwargs: Any) -> Agent:
        llm = Script(scripts.pop(0) if scripts else [answer("Done.\nEnding: ready")])
        models["workers"].append(llm)
        agent = Agent(
            llm=kwargs["wrap_llm"](llm),
            options=session_options(kwargs["options"], kwargs["workspace"]),
            tool_providers=kwargs["tools"],
        )
        await agent.start()
        agent.executor.register("python_exec", ran.python_exec, requires_approval=True)
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", worker)
    folder = tmp_path / "work"
    folder.mkdir(exist_ok=True)
    cfg = make_cfg(follow_up=follow_up)
    cfg.gateway.workspace = str(folder)
    cfg.approvals.mode = approvals
    console = Console(
        cfg,
        port=PORT,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=side,
            front_builder=front,
            probe=FakeProbe(),
            gate=quiet_gate(),
            titler=first_words,
            explainer=no_explanation,
        ),
    )
    holder.append(console)
    if not run:

        async def no_schedule() -> None:
            pass

        console.workers._schedule = no_schedule  # type: ignore[method-assign]
    console.asks.idle_s = 0.01
    await console.start()
    app = FastAPI()
    console.mount(app)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=f"http://127.0.0.1:{PORT}"
    )
    try:
        yield console, client, models, ran
    finally:
        await client.aclose()
        await console.aclose()


def followed_task(messages: list[Message]) -> str:
    """The task a follow-up's request is about."""
    for message in reversed(messages):
        found = _FOLLOWUP.findall(str(message.content))
        if message.role == "user" and found:
            return str(found[-1])
    raise AssertionError("no follow-up in the request")


def continues(text: str) -> Callable[[list[Message]], Message]:
    def reply(messages: list[Message]) -> Message:
        args = {"task_id": followed_task(messages), "message": text}
        return tool_call("task_continue", args, "continue-1")

    return reply


def creates(title: str) -> Message:
    return tool_call("task_create", {"title": title, "objective": title}, "create-1")


async def say(client: httpx.AsyncClient, console: Console, text: str) -> None:
    chat = str(console.main_id)
    response = await _post_turn(client, chat, text)
    assert response.status_code == 202, response.text
    await _wait_turn(client, chat)


async def started_task(console: Console) -> str:
    found = [r for r in await rows(console, console.main_id) if r["echo"] == "started"]
    assert found, "no task started"
    return str(found[-1]["task_id"])


async def state(console: Console, task_id: str) -> str:
    return (await console.workers.tasks.get_task(task_id)).state


async def stop_run(
    console: Console, task_id: str, report: str = "Wrote half of it."
) -> None:
    """*task_id* ran one turn that used all its steps: it is stopped."""
    store = console.store
    record = await console.workers.tasks.get_task(task_id)
    turn = await store.begin_turn(task_id, record.spec["objective"], holder=HOLDER)
    await store.finish_turn(
        task_id,
        turn.turn,
        holder=HOLDER,
        status="budget_exhausted",
        cause="run",
        result=run_result(status="budget_exhausted"),
        answer=report,
    )
    await console.workers.tasks.change(
        task_id, state="failed", actor="scheduler", failure_reason=OUT_OF_STEPS
    )
    console.gate.finish_task(task_id, "failed")


async def internal_turns(console: Console) -> list[Any]:
    turns = await console.store.list_turns(str(console.main_id))
    return [turn for turn in turns if turn.origin == "followup"]


async def settled(console: Console) -> None:
    """Every ending heard was looked at, and no follow-up waits or runs."""
    await console.followups.settle()
    chat = str(console.main_id)

    def idle() -> bool:
        entry = console.sessions.get(chat)
        busy = entry is not None and entry.runner.busy
        followups = console.followups
        return not followups.pending(chat) and not followups.running(chat) and not busy

    await until(idle)
    await console.followups.settle()


# ── #410: the chat starts the task itself ──


def test_the_start_rule_starts_requests_and_answers_questions() -> None:
    for typed in (
        "open https://example.com/news and tell me the top story",
        "run the test suite in my project and tell me what fails",
        "can you make slides from my notes today?",
        "帮我装一下 numpy 然后跑一下",
        "fix tests/cli/test_e2e.py",
    ):
        assert start_rule(typed) == "start", typed
    for typed in ("hi", "what is a monad?", "简单说一下什么是 monad？"):
        assert start_rule(typed) == "answer", typed
    for typed in (
        "",
        "Why does pytest fail in tests/cli?",
        "just explain how pip works",
        "don't change anything, only tell me what is wrong",
        "只解释一下这个错误",
        "every day at 9 brief me on the news",
    ):
        assert start_rule(typed) == "offer", typed
    assert hands_off("please don't start a task for this")
    assert not hands_off("start the build and tell me when it is done")
    assert "task_create right away" in FRONT_TEXT and "one short question" in FRONT_TEXT


async def test_a_web_or_command_request_starts_a_task_with_one_line_and_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        creates("Read the news page"),
        answer("Started it as a task; its result comes back here."),
        creates("Run the tests"),
        answer("Started it as a task."),
    ]
    web = "open https://example.com/news and tell me the top story"
    command = "run the test suite in my project and tell me what fails"
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, web)
        await say(client, console, command)
        found = await rows(console, console.main_id)
        started = [r for r in found if r["echo"] == "started"]
        assert [r["title"] for r in started] == ["Read the news page", "Run the tests"]
        assert not [r for r in found if r["echo"] == "ask"], "started, not offered"
        records = [await console.workers.tasks.get_task(r["task_id"]) for r in started]
        # The person's words are each task's request; the model's only its reading.
        assert [r.spec["user_words"] for r in records] == [web, command]
        assert records[0].spec["objective"].startswith(web)
        assert tool_results(models["main"], 1)[-1].startswith("Started task ")
        # Stop, from the one line.
        stopped = await client.post(
            f"/api/front/echoes/{started[0]['id']}", json={"do": "stop"}
        )
        assert stopped.status_code == 200, stopped.text
        echo = stopped.json()["echo"]
        assert (echo["echo"], echo["action"], echo["of"]) == (
            "control",
            "stop",
            started[0]["id"],
        )
        assert await state(console, started[0]["task_id"]) == "cancelled"
        assert await state(console, started[1]["task_id"]) == "queued"


async def test_an_answerable_question_is_answered_and_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        creates("Monads"),
        answer("A monad chains steps that carry some context along."),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, "what is a monad?")
        refused = tool_results(models["main"], 1)[-1]
        assert refused.startswith("Not done: this is a question or small talk")
        assert await rows(console, console.main_id) == []
        assert await console.workers.tasks.list_tasks() == []


async def test_an_unclear_request_gets_one_question_and_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = "Which report do you mean: the weekly one or the sales one?"
    async with owner_console(
        tmp_path, monkeypatch, main=[answer(question)], run=False
    ) as (console, client, _, _ran):
        chat = str(console.main_id)
        assert (
            await _post_turn(client, chat, "do the report thing")
        ).status_code == 202
        turn = await _wait_turn(client, chat)
        assert (turn["status"], turn["answer"]) == ("completed", question)
        assert await rows(console, console.main_id) == []
        assert await console.workers.tasks.list_tasks() == []


async def test_ask_mode_still_asks_inside_a_task_the_chat_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = "print('sunny')"
    main: list[Reply] = [creates("Weather script"), answer("Started it as a task.")]
    workers: list[list[Reply]] = [
        [
            tool_call("python_exec", {"code": code}, "w-1"),
            answer("Done.\nEnding: ready"),
        ]
    ]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=workers, approvals="ask"
    ) as (console, client, _, ran):
        await say(client, console, "write a script that prints the weather and run it")
        task_id = await started_task(console)
        await until(lambda: console.broker.pending(task_id))
        [request] = console.broker.pending(task_id)
        # Starting was not a risky step; the step inside the task asks the person.
        assert request.tool_name == "python_exec" and ran.codes == []
        assert (await console.workers.get(task_id))["approvals"] == "ask"
        console.broker.answer(request.request_id, ApprovalDecision(True))
        await until(lambda: state(console, task_id))
        await until(lambda: _is(console, task_id, "done"))
        assert ran.codes == [code]
        await settled(console)
        assert await internal_turns(console) == [], "a ready task needs no follow-up"


async def _is(console: Console, task_id: str, wanted: str) -> bool:
    return await state(console, task_id) == wanted


async def test_a_turn_melo_began_cannot_start_a_task_from_its_own_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        creates("Long report"),
        answer("Started it as a task."),
        # The follow-up tries to start a task of its own.
        creates("Another report"),
        answer("It ran out of steps. Should I split it into two tasks?"),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, "write the long report from my notes")
        task_id = await started_task(console)
        await stop_run(console, task_id)
        await settled(console)
        [turn] = await internal_turns(console)
        assert (turn.typed_text, turn.status) == ("", "completed")
        refused = tool_results(models["main"], 3)[-1]
        assert refused.startswith("Not done: only the user's own message starts a task")
        assert [t.task_id for t in await console.workers.tasks.list_tasks()] == [
            task_id
        ]


# ── #411: the chat follows up on a task it started ──


async def test_a_blocked_install_is_worked_around_and_never_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other_way = "Write it with urllib from the standard library; install nothing."
    main: list[Reply] = [
        creates("Weather script"),
        answer("Started it as a task."),
        continues(other_way),
        answer(
            "The package install was blocked, so the task goes on with urllib. "
            "Nothing was installed."
        ),
    ]
    workers: list[list[Reply]] = [
        [
            tool_call("python_exec", {"code": INSTALL}, "w-1"),
            answer("The script needs a package I could not install.\nEnding: failed"),
        ],
        [
            # The same install again: refused before any review.
            tool_call("python_exec", {"code": INSTALL}, "w-2"),
            answer("Wrote it with urllib instead.\nEnding: ready"),
        ],
    ]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=workers, approvals="auto"
    ) as (console, client, models, ran):
        await say(client, console, "write a script that fetches today's weather")
        task_id = await started_task(console)
        await until(lambda: len(models["workers"]) == 2)
        await until(lambda: _is(console, task_id, "done"))
        await settled(console)
        assert ran.codes == [], "the install never ran"
        # The follow-up read the ending and the blocked step, as data.
        [turn] = await internal_turns(console)
        assert turn.typed_text == "" and turn.status == "completed"
        assert turn.user_text.startswith(f"[localmelo · follow-up on task {task_id} · ")
        assert "Steps blocked in it" in turn.user_text
        assert "pip install requests" in turn.user_text
        assert "blocked by Auto mode" in turn.user_text
        assert "task_continue(" in turn.user_text
        found = await rows(console, console.main_id)
        kinds = [r["echo"] for r in found]
        assert kinds == ["started", "followup", "sent", "followed"]
        followup, sent, followed = found[1], found[2], found[3]
        assert (followup["why"], followup["lead"]) == (
            "failed",
            turn.user_text.splitlines()[0],
        )
        assert followup["said"]["key"] == "followup.row"
        assert (sent["how"], sent["task_id"]) == ("continued", task_id)
        assert (followed["did"], followed["task_id"]) == ("continued", task_id)
        # The continued run read the other way and the blocked step.
        first = str(models["workers"][1].requests[0][-1].content)
        assert other_way in first and "pip install requests" in first
        # The same call was refused before any review, saying who blocked it.
        [refused] = tool_results(models["workers"][1], 1)
        assert "Auto mode blocked this exact step earlier in this task" in refused
        approvals = [
            e.event["data"]["approval"]
            for e in await all_events(console.store, task_id)
            if e.event["type"] == "tool.finished"
        ]
        assert [a["origin"] for a in approvals] == ["auto", "policy"]
        assert approvals[1]["earlier"] == "Auto mode" and "review" not in approvals[1]
        assert approvals[0]["call_key"] == approvals[1]["call_key"]
        info = await console.store.get_session(task_id)
        assert info is not None and info.meta["followup_guard"] is True
        assert info.meta["followup"]["continued"] is True
        # Activity says what the chat did, plainly.
        activity = (await client.get("/api/activity")).json()
        results = " / ".join(str(row.get("result")) for row in activity["rows"])
        assert "Continued the task: Weather script" in results


async def test_out_of_steps_continues_once_then_asks_then_stops_following_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    go_on = "Go on from where you stopped and finish the remaining sections."
    question = "It stopped again after two runs. Should I split it into smaller tasks?"
    main: list[Reply] = [
        creates("Long report"),
        answer("Started it as a task."),
        # The first follow-up continues it once.
        continues(go_on),
        answer("It ran out of steps, so I continued it once."),
        # The second may not continue: it reports and asks one question.
        continues("Try once more."),
        answer(question),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, "write the long report from my notes")
        task_id = await started_task(console)
        await stop_run(console, task_id)
        await settled(console)
        record = await console.workers.tasks.get_task(task_id)
        assert record.state == "queued" and record.card.startswith(go_on)
        await stop_run(console, task_id, "Wrote another part.")
        await settled(console)
        first, second = await internal_turns(console)
        assert "continued the task once already" in second.user_text
        assert "task_continue(" not in second.user_text
        refused = tool_results(models["main"], 5)[-1]
        assert refused.startswith("Not done: this task was continued once already")
        assert await state(console, task_id) == "failed", "not continued again"
        assert second.answer == question
        followed = [
            r for r in await rows(console, console.main_id) if r["echo"] == "followed"
        ]
        assert [(r["did"], r["question"]) for r in followed] == [
            ("continued", ""),
            ("asked", "Should I split it into smaller tasks?"),
        ]
        # A third stop (the person's Continue on the card) gets no follow-up: no loop.
        await console.workers.continue_stopped(task_id)
        await stop_run(console, task_id, "Wrote a third part.")
        await settled(console)
        assert len(await internal_turns(console)) == FOLLOWUPS_PER_MESSAGE
        assert [t.task_id for t in await console.workers.tasks.list_tasks()] == [
            task_id
        ]


async def test_the_persons_message_wins_over_a_pending_follow_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        creates("Long report"),
        answer("Started it as a task."),
        answer("The report task stopped: it ran out of steps. Shall I go on?"),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        await say(client, console, "write the long report from my notes")
        task_id = await started_task(console)
        entered, release = asyncio.Event(), asyncio.Event()
        followups = console.followups
        original = followups.run_next

        async def pending(chat_id: str) -> None:
            entered.set()
            await release.wait()  # the follow-up waits here, not yet begun
            await original(chat_id)

        monkeypatch.setattr(followups, "run_next", pending)
        await stop_run(console, task_id)
        await asyncio.wait_for(entered.wait(), 5.0)
        # The person writes meanwhile: their turn runs, and reads the ending as news.
        await say(client, console, "how is it going?")
        news = str(models["main"].requests[-1][-1].content)
        assert "Long report" in news and "ran out of steps" in news
        release.set()
        await settled(console)
        assert await internal_turns(console) == [], "the person's turn folded it in"
        kinds = [r["echo"] for r in await rows(console, console.main_id)]
        assert "followup" not in kinds


async def test_a_follow_up_waits_for_the_persons_running_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [
        creates("Long report"),
        answer("Started it as a task."),
        held(release, answer("Here is the outline you asked for.")),
        answer("The report task ran out of steps. Should I continue it?"),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        chat = str(console.main_id)
        await say(client, console, "write the long report from my notes")
        task_id = await started_task(console)
        assert (await _post_turn(client, chat, "draft an outline")).status_code == 202
        await until(lambda: console.sessions.get(chat).runner.busy)  # type: ignore[union-attr]
        await stop_run(console, task_id)
        await console.followups.settle()
        await asyncio.sleep(0.05)
        assert await internal_turns(console) == [], "it never interrupts the person"
        release.set()
        await settled(console)
        turns = await console.store.list_turns(chat)
        assert [t.origin for t in turns] == ["", "", "followup"]
        assert turns[2].started_at >= (turns[1].finished_at or 0)


async def test_tasks_the_chat_did_not_start_keep_the_card_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(tmp_path, monkeypatch, run=False) as (
        console,
        _client,
        models,
        _,
    ):
        page, _ = await console.workers.create({"objective": "tidy the notes"})
        await stop_run(console, str(page["task_id"]))
        await settled(console)
        assert await internal_turns(console) == []
        assert models["main"].requests == []
        info = await console.store.get_session(str(page["task_id"]))
        assert info is not None and "started_by" not in info.meta


async def test_the_switch_off_means_no_follow_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    async with owner_console(
        tmp_path, monkeypatch, main=main, run=False, follow_up=False
    ) as (console, client, models, _):
        await say(client, console, "write the long report from my notes")
        task_id = await started_task(console)
        await stop_run(console, task_id)
        await settled(console)
        assert await internal_turns(console) == []
        assert len(models["main"].requests) == 2
        assert await state(console, task_id) == "failed"


def test_a_follow_ups_question_is_its_last_question() -> None:
    assert (
        asked_question("It stopped twice. Should I split it?") == "Should I split it?"
    )
    assert asked_question("要我把它拆成两个任务吗？") == "要我把它拆成两个任务吗？"
    assert asked_question("It is done.") == ""
