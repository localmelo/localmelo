"""The main chat starts a task itself (#410) and follows up when it stops (#411).

Each test runs the console on scripted models: the main chat's FrontAgent, and
each task run's Agent with a ``python_exec`` that records the code it was given
instead of running it. Nothing reaches a model server or the network.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from localmelo.melo.agent import Agent
from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.contracts.approvals import (
    UNKNOWN_RESULT,
    AllowAllPolicy,
    ApprovalDecision,
    AutoPolicy,
    call_key,
    find_earlier,
)
from localmelo.melo.contracts.providers import StreamFragment
from localmelo.melo.contracts.review import step_key
from localmelo.melo.contracts.tools import BaseToolProvider
from localmelo.melo.schema import INTERRUPT_LABEL, Message, ToolDef
from localmelo.support.gateway.chat_gate import GatedLLM
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.console.followups import (
    FOLLOWUPS_PER_MESSAGE,
    asked_question,
)
from localmelo.support.gateway.console.front import (
    TASK_TOOLS,
    FrontAgent,
)
from localmelo.support.gateway.console.registry import chat_options
from localmelo.support.gateway.finishing import OUT_OF_STEPS
from localmelo.support.gateway.handoff import (
    HANDOFF_HEADER,
    UNKNOWN_EFFECT,
    Handoff,
    handoff_block,
)
from localmelo.support.gateway.task_policy import TaskSpec, chore_workspace
from localmelo.support.gateway.task_store import TaskRecord, TaskStoreError
from localmelo.support.gateway.workers import _turn_text
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.live import LIVE_EVENT
from localmelo.support.sessions.store import SessionBusyError
from localmelo.tests.checker.test_redaction import GITHUB
from localmelo.tests.sessions.helpers import answer, tool_call

from .conftest import (
    HOLDER,
    PORT,
    FakeProbe,
    all_events,
    core_event,
    make_cfg,
    quiet_gate,
    run_result,
)
from .test_ask_owner import Reply, Script, held, rows, tool_results, until
from .test_turn_writes import _post_turn, _wait_turn

INSTALL = (
    'import subprocess\nsubprocess.run(["pip", "install", "requests"], check=True)'
)
# The same install that also reads a key file: a step only a click allows, so
# Auto blocks it even in a task that asks its chat (#465).
INSTALL_KEYS = (
    "import os, subprocess\n"
    'subprocess.run(["pip", "install", "requests"], check=True)\n'
    'print(open(os.path.expanduser("~/.netrc")).read())'
)
_FOLLOWUP = re.compile(r"\[localmelo · follow-up on task ([0-9a-f]{12}) · ")


class Streamed(Script):
    """A :class:`Script` whose replies stream: reasoning first, then a line a piece."""

    async def stream_chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Any:
        reply = await self.chat(messages, tools)
        yield StreamFragment(kind="reasoning", text="Thinking it over.")
        for line in reply.content.splitlines(keepends=True):
            yield StreamFragment(kind="content", text=line)
        yield StreamFragment(kind="reply", reply=reply)


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
    front_tools: list[BaseToolProvider] | None = None,
    streams: bool = False,
) -> AsyncIterator[tuple[Console, httpx.AsyncClient, dict[str, Any], Ran]]:
    """A console whose main chat and task runs work on scripted models.

    *follow_up* is ``[tasks] follow_up``, *approvals* the mode new chats (and
    the main chat) start in; without *run* no task starts. *front_tools* are
    the main chat's tools beside its task tools; with *streams* every model's
    replies stream (:class:`Streamed`).
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    model = Streamed if streams else Script
    models: dict[str, Any] = {"main": model(main or []), "workers": []}
    scripts = [list(script) for script in workers or []]
    ran = Ran()
    holder: list[Console] = []

    async def front(settings: Any, workspace: str, mcp: Any, session_id: str) -> Any:
        options = session_options(chat_options(settings, workspace), workspace)
        llm = GatedLLM(models["main"], holder[0].gate.client(session_id))
        agent = FrontAgent(
            llm=llm,
            options=options,
            tool_providers=[holder[0].front.tools, *(front_tools or [])],
        )
        agent.front = holder[0].front
        await agent.start()
        return agent

    async def side(settings: Any, workspace: str, mcp: Any, session_id: str) -> Any:
        raise AssertionError("a side chat is not opened here")

    async def worker(**kwargs: Any) -> Agent:
        llm = model(scripts.pop(0) if scripts else [answer("Done.\nEnding: ready")])
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


async def successor_of(console: Console, task_id: str) -> str:
    """The task that holds *task_id*'s job now (#571); fails when none went on."""
    record = await console.workers.tasks.get_task(task_id)
    head = await console.workers.newest(record)
    assert head.task_id != task_id, "no fresh task went on"
    return head.task_id


async def calls_of(console: Console, *task_ids: str) -> list[dict[str, Any]]:
    """The approvals of every finished call of *task_ids*' runs, in order."""
    return [
        e.event["data"]["approval"]
        for task_id in task_ids
        for e in await all_events(console.store, task_id)
        if e.event["type"] == "tool.finished"
    ]


async def state(console: Console, task_id: str) -> str:
    return (await console.workers.tasks.get_task(task_id)).state


async def stop_run(
    console: Console,
    task_id: str,
    report: str = "Wrote half of it.",
    events: Sequence[dict[str, Any]] = (),
) -> None:
    """*task_id* ran one turn that used all its steps, with *events*: it is stopped."""
    store = console.store
    record = await console.workers.tasks.get_task(task_id)
    turn = await store.begin_turn(task_id, record.spec["objective"], holder=HOLDER)
    if events:
        await store.append_events(
            task_id,
            turn.turn,
            [{**event, "run_id": turn.run_id} for event in events],
            holder=HOLDER,
        )
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


async def test_the_models_task_create_starts_a_task_with_one_line_and_stop(
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


async def test_a_task_the_chat_starts_never_streams_into_the_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = "Started it as a task.\nIts result comes back here.\n"
    main: list[Reply] = [
        creates("Plan the trip"),
        answer(reply),
        answer("It is ready.\n"),
    ]
    workers: list[list[Reply]] = [[answer("The task's own words.\nEnding: ready")]]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=workers, follow_up=False, streams=True
    ) as (console, client, _, _ran):
        seen: list[Any] = []

        async def follow() -> None:
            async for event in console.hub.subscribe(str(console.main_id), viewer=True):
                seen.append(event)

        following = asyncio.create_task(follow())
        await say(client, console, "plan my trip as a task")
        task_id = await started_task(console)
        await until(lambda: _is(console, task_id, "done"))
        await settled(console)
        following.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await following
    live = [e.event["data"] for e in seen if e.event["type"] == LIVE_EVENT]
    assert live, "the chat's own reply streams"
    assert not any("own words" in data["text"] for data in live)
    first = [d["text"] for d in live if d["turn"] == live[0]["turn"]]
    assert "".join(first) == reply, "the chat's live text is its own reply's"


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
    told = "It is done: the script prints the weather."
    main: list[Reply] = [
        creates("Weather script"),
        answer("Started it as a task."),
        answer(told),
    ]
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
        # A ready task's answer comes back to the chat in one turn of its own (#440).
        [turn] = await internal_turns(console)
        assert turn.answer == told and "it finished" in turn.user_text


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
        # The continued task ends ready: its answer comes back (#440).
        answer("It is done: the script fetches the weather with urllib."),
    ]
    workers: list[list[Reply]] = [
        [
            tool_call("python_exec", {"code": INSTALL_KEYS}, "w-1"),
            answer("The script needs a package I could not install.\nEnding: failed"),
        ],
        [
            # The same install again: refused before any review.
            tool_call("python_exec", {"code": INSTALL_KEYS}, "w-2"),
            answer("Wrote it with urllib instead.\nEnding: ready"),
        ],
    ]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=workers, approvals="auto"
    ) as (console, client, models, ran):
        await say(client, console, "write a script that fetches today's weather")
        task_id = await started_task(console)
        await until(lambda: len(models["workers"]) == 2)
        # The chat's continue went to a fresh task; the ended one stays as it was.
        fresh = await successor_of(console, task_id)
        await until(lambda: _is(console, fresh, "done"))
        await settled(console)
        # Never reopened: its one run is all its session holds.
        assert len(await console.store.list_turns(task_id)) == 1
        assert ran.codes == [], "the install never ran"
        # The follow-up read the ending and the blocked step, as data; the
        # second turn told the ready answer.
        turn, told = await internal_turns(console)
        assert "it finished" in told.user_text and "urllib" in told.answer
        assert turn.typed_text == "" and turn.status == "completed"
        assert turn.user_text.startswith(f"[localmelo · follow-up on task {task_id} · ")
        assert "Steps blocked in it" in turn.user_text
        assert "pip install requests" in turn.user_text
        assert "blocked by Auto mode" in turn.user_text
        assert "task_continue(" in turn.user_text
        # What the task wrote is fenced with a nonce, as a page is.
        fence = re.search(r"<<answer nonce=([0-9a-f]{8})>>", turn.user_text)
        assert fence is not None
        assert f"<<end blocked nonce={fence.group(1)}>>" in turn.user_text
        # The follow-up is offered the task tools only.
        task_tools = {definition.name for definition in TASK_TOOLS}
        assert (
            models["main"].offered[2] and set(models["main"].offered[2]) <= task_tools
        )
        assert "file_read" in models["main"].offered[0], "the person's turn has it"
        found = await rows(console, console.main_id)
        kinds = [r["echo"] for r in found]
        assert kinds == [
            "started",
            "followup",
            "started",
            "followed",
            "followup",
            "followed",
        ]
        followup, sent, followed = found[1], found[2], found[3]
        assert (followup["why"], followup["lead"]) == (
            "failed",
            turn.user_text.splitlines()[0],
        )
        assert followup["said"]["key"] == "followup.row"
        assert followup["turn_id"] == turn.id, "written once its turn began"
        assert (sent["follows"], sent["task_id"]) == (task_id, fresh)
        assert sent["said"]["key"] == "front.echo.continued"
        assert (followed["did"], followed["task_id"]) == ("continued", task_id)
        # The fresh task read the other way and the blocked step in its packet,
        # as the chat's note after the person's request.
        first = "\n".join(
            str(message.content) for message in models["workers"][1].requests[0]
        )
        packet = first[first.index(HANDOFF_HEADER) :]
        assert first.index("fetches today's weather") < first.index(HANDOFF_HEADER)
        assert other_way in packet and "pip install requests" in packet
        # The same call was refused before any review, saying who blocked it.
        [refused] = tool_results(models["workers"][1], 1)
        assert "Auto mode blocked this exact step earlier in this task" in refused
        approvals = await calls_of(console, task_id, fresh)
        assert [a["origin"] for a in approvals] == ["auto", "policy"]
        assert approvals[1]["earlier"] == "Auto mode" and "review" not in approvals[1]
        assert "install:pip:requests" in approvals[0]["loose_keys"]
        assert "call_key" not in approvals[1], "a refused repeat is no new block"
        info = await console.store.get_session(fresh)
        assert info is not None and info.meta["followup_guard"] is True
        assert info.meta["followup"]["continued"] is True
        # Activity says what the chat did, plainly.
        activity = (await client.get("/api/activity")).json()
        results = " / ".join(str(row.get("result")) for row in activity["rows"])
        assert "Told you the answer: Weather script" in results


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
        # The ended task stays as it was; a fresh one goes on with the chat's note.
        assert await state(console, task_id) == "failed"
        fresh = await successor_of(console, task_id)
        record = await console.workers.tasks.get_task(fresh)
        assert record.state == "queued" and not record.card
        assert record.spec["handoff"]["note"] == go_on
        assert record.spec["handoff"]["by"] == "chat"
        await stop_run(console, fresh, "Wrote another part.")
        await settled(console)
        first, second = await internal_turns(console)
        # The count is the job's: the fresh task does not start it over.
        assert "continued the task once already" in second.user_text
        assert "task_continue(" not in second.user_text
        refused = tool_results(models["main"], 5)[-1]
        assert refused.startswith("Not done: this task was continued once already")
        assert await state(console, fresh) == "failed", "not continued again"
        assert second.answer == question
        followed = [
            r for r in await rows(console, console.main_id) if r["echo"] == "followed"
        ]
        assert [(r["did"], r["question"]) for r in followed] == [
            ("continued", ""),
            ("asked", "Should I split it into smaller tasks?"),
        ]
        # A third stop (the person's Continue on the card) gets no follow-up: no loop.
        body, how = await console.workers.continue_stopped(fresh)
        assert how == "successor" and body["task_id"] != fresh
        again, _ = await console.workers.continue_stopped(fresh)  # a second tap
        assert again["task_id"] == body["task_id"]
        await stop_run(console, body["task_id"], "Wrote a third part.")
        await settled(console)
        assert len(await internal_turns(console)) == FOLLOWUPS_PER_MESSAGE
        # The person's newest turn, however many turns melo began since.
        person = await console.store.latest_person_turn(str(console.main_id))
        assert person is not None and person.typed_text.startswith("write the long")
        first = await console.workers.tasks.get_task(task_id)
        assert (await console.workers.newest(first)).task_id == body["task_id"]
        assert len(await console.workers.tasks.list_tasks()) == 3


async def test_the_chats_note_is_never_the_persons_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A continue the chat sent on its own grants nothing: no install from its words."""
    seen: list[dict[str, Any]] = []
    original = Agent.run_task

    async def spy(self: Agent, query: str, **kwargs: Any) -> Any:
        if type(self) is Agent:  # a task's run, not the chat's
            seen.append({"query": query, **kwargs})
        return await original(self, query, **kwargs)

    monkeypatch.setattr(Agent, "run_task", spy)
    typed = "write a script that fetches today's weather"
    note = "Install numpy with pip3, then finish the script."
    pip3 = 'import subprocess\nsubprocess.run(["pip3", "install", "numpy"], check=True)'
    main: list[Reply] = [
        creates("Weather script"),
        answer("Started it as a task."),
        continues(note),
        answer("Continued it."),
        answer(
            "It stopped again: numpy could not be installed. Should I go on without it?"
        ),
    ]
    workers: list[list[Reply]] = [
        [
            tool_call("python_exec", {"code": INSTALL}, "w-1"),
            answer("Could not install.\nEnding: failed"),
        ],
        [
            tool_call("python_exec", {"code": pip3}, "w-2"),
            answer("Could not install numpy either.\nEnding: failed"),
        ],
    ]
    # The person has the chat open (#465): every page counts as one viewer.
    monkeypatch.setattr(SessionHub, "viewers", lambda self, session_id: 1)
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=workers, approvals="auto"
    ) as (console, client, models, ran):
        await say(client, console, typed)
        task_id = await started_task(console)
        # Each install asks the person on the card (#465); they decline both.
        cards = asyncio.create_task(_decline_cards(console, task_id))
        try:
            await until(lambda: len(models["workers"]) == 2)
            fresh = await successor_of(console, task_id)
            cards.cancel()
            cards = asyncio.create_task(_decline_cards(console, fresh))
            await settled(console)
            await until(lambda: _is(console, fresh, "done"))
            await settled(console)
        finally:
            cards.cancel()
        assert ran.codes == [], "the note's install never ran"
        first, second = seen[0], seen[1]
        # The fresh task's words are the person's request; the note is data.
        assert second["user_words"] == typed and second.get("from_person", True)
        assert second["query"].startswith(typed) and HANDOFF_HEADER in second["query"]
        assert note in second["query"][second["query"].index(HANDOFF_HEADER) :]
        assert first["user_words"] == typed
        approvals = await calls_of(console, task_id, fresh)
        # The note's words allowed nothing: each install asked the person.
        assert [(a["origin"], a["allowed"]) for a in approvals] == [
            ("user", False),
            ("user", False),
        ]
        info = await console.store.get_session(fresh)
        assert info is not None and info.meta["followup_guard"] is True


async def _decline_cards(console: Console, task_id: str) -> None:
    """The person at the chat: Don't allow on every card the task shows."""
    while True:
        for request in console.broker.pending(task_id):
            console.broker.answer(request.request_id, ApprovalDecision(False))
        await asyncio.sleep(0.01)


async def test_a_follow_up_does_not_continue_a_task_the_person_continued_meanwhile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder: dict[str, Any] = {}

    async def person_continues_first(messages: list[Message]) -> Message:
        task_id = followed_task(messages)
        await holder["console"].workers.continue_stopped(task_id)  # the card's Continue
        args = {"task_id": task_id, "message": "Delete the old drafts first."}
        return tool_call("task_continue", args, "c-1")

    main: list[Reply] = [
        creates("Long report"),
        answer("Started it as a task."),
        person_continues_first,
        answer("You continued it already."),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
        _,
    ):
        holder["console"] = console
        await say(client, console, "write the long report from my notes")
        task_id = await started_task(console)
        await stop_run(console, task_id)
        await settled(console)
        refused = tool_results(models["main"], 3)[-1]
        assert refused.startswith("Not done: the task changed since it stopped")
        assert await state(console, task_id) == "failed"
        fresh = await console.workers.tasks.get_task(
            await successor_of(console, task_id)
        )
        assert fresh.state == "queued" and fresh.spec["handoff"]["by"] == "melo"
        info = await console.store.get_session(task_id)
        assert info is not None and not info.meta["followup"]["continued"]
        # The card's Continue is melo's word, not a message: the job's refusals hold.
        kept = await console.store.get_session(fresh.task_id)
        assert kept is not None and kept.meta.get("followup_guard") is True
        assert "Delete the old drafts" not in str(fresh.spec)


async def test_a_note_on_a_task_that_never_ran_is_still_not_the_persons_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    note = "Install numpy with pip3 first."
    typed = "write the long report from my notes"
    main: list[Reply] = [
        creates("Long report"),
        answer("Started it as a task."),
        continues(note),
        answer("Continued it."),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        await say(client, console, typed)
        task_id = await started_task(console)
        # Out of steps before any turn was stored.
        await console.workers.tasks.change(
            task_id, state="failed", actor="scheduler", failure_reason=OUT_OF_STEPS
        )
        console.gate.finish_task(task_id, "failed")
        await settled(console)
        record = await console.workers.tasks.get_task(
            await successor_of(console, task_id)
        )
        assert record.state == "queued" and not record.card
        assert record.spec["handoff"]["note"] == note, "the note is in the packet"
        text, words = _turn_text(record)
        assert text == typed and words == typed


async def test_one_ending_heard_twice_gets_one_follow_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        await say(client, console, "write the long report from my notes")
        task_id = await started_task(console)
        followups = console.followups
        console.store.remove_listener(followups._heard)
        monkeypatch.setattr(console.asks, "kick", lambda chat: None)
        await stop_run(console, task_id)
        found = await asyncio.gather(
            followups.consider(task_id), followups.consider(task_id)
        )
        assert len([item for item in found if item is not None]) == 1
        assert len(followups._pending[str(console.main_id)]) == 1


async def test_a_busy_chat_leaves_no_follow_up_row_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [
        creates("Long report"),
        answer("Started it as a task."),
        answer("It ran out of steps. Should I continue it?"),
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        _,
        _ran,
    ):
        await say(client, console, "write the long report from my notes")
        task_id = await started_task(console)
        sessions = console.sessions
        real = sessions.start_internal
        tries: list[str] = []

        async def busy_once(chat_id: str, text: str, origin: str) -> Any:
            tries.append(origin)
            if len(tries) == 1:
                raise SessionBusyError("a turn began meanwhile")
            return await real(chat_id, text, origin)

        monkeypatch.setattr(sessions, "start_internal", busy_once)
        await stop_run(console, task_id)
        await settled(console)
        assert tries == ["followup", "followup"]
        [turn] = await internal_turns(console)
        found = [
            r for r in await rows(console, console.main_id) if r["echo"] == "followup"
        ]
        assert [r["turn_id"] for r in found] == [
            turn.id
        ], "one row, for the turn that ran"


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


# ── #571: one job, one successor, and nothing starts over ──

REPORT = "write the long report from my notes"


async def test_one_ended_job_gets_one_successor_whatever_continues_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    async with owner_console(
        tmp_path, monkeypatch, main=main, run=False, follow_up=False
    ) as (console, client, _, _):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        await stop_run(console, task_id)
        record = await console.workers.tasks.get_task(task_id)
        echo = await console.front.follow(record, "go on", typed="go on")
        assert echo["echo"] == "started"
        first = str(echo["task_id"])
        # The ended task's Continue, and a second message, reach the one successor.
        body, how = await console.workers.continue_stopped(task_id)
        assert (body["task_id"], how) == (first, "running")
        more = await console.front.follow(
            record, "add a summary", typed="add a summary"
        )
        assert (more["echo"], more["task_id"]) == ("sent", first)
        tasks = await console.workers.tasks.list_tasks()
        assert sorted(t.task_id for t in tasks) == sorted([task_id, first])
        # Its card names where the job went and offers nothing that forks it.
        card = (await client.get(f"/api/tasks/{task_id}")).json()["task"]
        assert card["successor"]["task_id"] == first


async def test_a_successor_of_a_repeating_chore_run_still_asks_before_each_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with owner_console(tmp_path, monkeypatch, run=False, follow_up=False) as (
        console,
        _,
        _,
        _,
    ):
        spec = TaskSpec(
            title="Weekly summary",
            objective="summarize the week",
            user_words="summarize the week",
            repeats=True,
            workspace=chore_workspace(console.workers.settings(), "c0ffee"),
        )
        body, _ = await console.workers.create_spec(
            spec, front_id=console.main_id, origin="user_rule", chat_started=True
        )
        await stop_run(console, str(body["task_id"]))
        fresh, how = await console.workers.continue_stopped(str(body["task_id"]))
        assert how == "successor"
        record = await console.workers.tasks.get_task(str(fresh["task_id"]))
        assert record.spec.get("repeats") is True


async def test_a_long_chain_keeps_one_head_its_request_and_its_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    async with owner_console(
        tmp_path, monkeypatch, main=main, run=False, follow_up=False
    ) as (console, client, _, _):
        await say(client, console, REPORT)
        first = await started_task(console)
        info = await console.store.get_session(first)
        assert info is not None
        refused = {"request_id": "r0", "allowed": False, "reason": "no",
                   "remember": False, "origin": "user", "loose_keys": [],
                   "call_key": call_key("python_exec", {"code": INSTALL}, info.workspace)}  # fmt: skip
        await stop_run(
            console, first, events=[core_event("tool.finished", approval=refused)]
        )
        ended = await console.workers.tasks.get_task(first)
        for hop in range(10):
            # Each message names the first task; each reaches the job's newest.
            note = (f"note {hop}: " + "more detail " * 1000).strip()
            echo = await console.front.follow(ended, note, typed=note)
            assert echo["echo"] == "started", echo
            await stop_run(console, str(echo["task_id"]))
        head = await console.workers.newest(ended)
        assert head.task_id == echo["task_id"]
        assert len(await console.workers.tasks.list_tasks()) == 11
        assert head.spec["objective"].startswith(REPORT)
        assert head.spec["objective"].endswith(note)
        assert head.spec["user_words"].count(note) == 1
        # melo's Continue: the step the person refused in the first task is refused.
        body, how = await console.workers.continue_stopped(first)
        assert how == "successor"
        fresh = await console.workers.tasks.get_task(str(body["task_id"]))
        meta = await console.store.get_session(fresh.task_id)
        assert meta is not None
        policy = await console.workers._guard(
            fresh, meta.meta, AllowAllPolicy(), lambda _: None
        )
        earlier = find_earlier(policy)
        assert earlier is not None
        assert earlier.match("python_exec", {"code": INSTALL}, info.workspace)


async def test_a_call_whose_effect_is_unknown_asks_the_person_again_down_the_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each call is fenced by the live key it started with, whatever the store kept
    of it (redacted, no finish kept); run again, it is the person's card, and their
    Allow runs it once."""
    cut = f"import os\nos.environ['TOKEN'] = '{GITHUB}'\nos.system('deploy')"
    worker: list[Reply] = [
        tool_call("python_exec", {"code": cut}, "p-1"),
        answer("Done.\nEnding: ready"),
    ]
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=[worker], run=False, follow_up=False
    ) as (console, client, _, ran):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        info = await console.store.get_session(task_id)
        assert info is not None
        crash = {"command": f"deploy --token {GITHUB}"}
        events = [  # what the store keeps of them: their arguments redacted
            core_event("tool.started", seq=1, call_id="p-0", name="python_exec",
                       arguments={"code": redact_secrets(cut)[0]}, preview="python",
                       run_key=call_key("python_exec", {"code": cut}, info.workspace)),
            core_event("tool.finished", seq=2, call_id="p-0", name="python_exec",
                       status="timeout", error="Execution timed out"),
            core_event("tool.started", seq=3, call_id="s-0", name="shell_exec",
                       arguments={"command": redact_secrets(crash["command"])[0]},
                       run_key=call_key("shell_exec", crash, info.workspace)),
        ]  # fmt: skip
        for n in range(25):  # more calls after them than the packet lists
            events += [
                core_event("tool.started", seq=4 + 2 * n, step=2 + n, call_id=f"r-{n}"),
                core_event(
                    "tool.finished", seq=5 + 2 * n, step=2 + n, call_id=f"r-{n}"
                ),
            ]
        # The task's last work checkpoint: what failed and what works go on.
        notes = {"failed": ["deploy timed out"], "works": ["read a.txt first"]}
        events.append(
            core_event("notice", seq=60, kind="checkpoint", message="Noted.", step=3,
                       attempt=1, detail={"checkpoint": notes})
        )  # fmt: skip
        await stop_run(console, task_id, INTERRUPT_LABEL, events=events)
        del console.workers._schedule  # tasks run from here on
        body, how = await console.workers.continue_stopped(task_id)
        fresh = str(body["task_id"])
        record = await console.workers.tasks.get_task(fresh)
        assert record.spec["handoff"]["checkpoint"]["state"] == "", "no placeholder"
        said = handoff_block(record.spec["handoff"])
        assert "Did not work: deploy timed out" in said
        assert "Works: read a.txt first" in said
        assert "(effect unknown)" in said and f"Goal: {REPORT}" in said
        fenced = [b["tool"] for b in record.spec["handoff"]["blocks"]]
        assert fenced == ["python_exec", "shell_exec"]
        await until(lambda: console.broker.pending(fresh), timeout=15.0)
        [card] = console.broker.pending(fresh)
        assert card.review is not None and UNKNOWN_RESULT in card.review.reasons
        console.broker.answer(card.request_id, ApprovalDecision(True, origin="user"))
        await until(lambda: ran.codes, timeout=15.0)
        assert ran.codes == [cut]


async def test_a_call_the_person_allowed_that_timed_out_asks_again_in_the_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ask: the person allows a call (kept for the chat's later tasks), it times
    out and the task fails; the chat's own continue runs the same call in a fresh
    task: its effect is unknown, so it asks the person again, and the Allow kept
    for the chat does not run it."""
    export = "import time\nprint('exporting')\ntime.sleep(120)"
    again = "Run the same export again and tell me what it printed."
    main: list[Reply] = [
        creates("Nightly export"),
        answer("Started it as a task."),
        continues(again),
        answer("It goes on in a fresh task."),
    ]
    workers: list[list[Reply]] = [
        [
            tool_call("python_exec", {"code": export}, "e-1"),
            answer("It did not finish in time.\nEnding: failed"),
        ],
        [tool_call("python_exec", {"code": export}, "e-2"), answer("Done.")],
    ]
    async with owner_console(tmp_path, monkeypatch, main=main, workers=workers) as (
        console,
        client,
        models,
        ran,
    ):

        async def times_out(code: str) -> str:
            ran.codes.append(code)
            raise TimeoutError("python_exec timed out")

        ran.python_exec = times_out  # type: ignore[method-assign]
        await say(client, console, "run the nightly export")
        task_id = await started_task(console)
        await until(lambda: console.broker.pending(task_id), timeout=15.0)
        [card] = console.broker.pending(task_id)
        console.broker.answer(card.request_id, ApprovalDecision(True, origin="user"))
        await until(lambda: len(models["workers"]) == 2, timeout=15.0)
        fresh = await successor_of(console, task_id)

        async def asked_or_ran() -> bool:
            return bool(console.broker.pending(fresh)) or len(ran.codes) > 1

        await until(asked_or_ran, timeout=15.0)
        assert ran.codes == [export], "it ran again unasked"
        [again_card] = console.broker.pending(fresh)
        assert again_card.review is not None
        assert UNKNOWN_RESULT in again_card.review.reasons


async def test_a_message_that_reaches_a_worker_as_it_ends_goes_on_in_a_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    worker: list[Reply] = [held(release, answer("Done.\nEnding: ready"))]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=[worker], follow_up=False
    ) as (console, client, models, _):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        await until(lambda: models["workers"] and models["workers"][0].requests)
        _, how = await console.workers.message(task_id, "also add a short summary")
        assert how == "queued"
        release.set()  # its last step ends with the message unread
        fresh = await _went_on(console, task_id)
        assert len(await console.store.list_turns(task_id)) == 1, "never reopened"
        assert "also add a short summary" in fresh.spec["objective"]


async def test_an_allow_once_still_unread_at_the_end_goes_on_as_the_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allow once on a running worker queues melo's go-ahead (``allow_once``); when
    the run ends without reading it, the successor runs the allowed step on that
    allowance, with no card."""
    release = asyncio.Event()
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    first: list[Reply] = [held(release, answer("Done.\nEnding: ready"))]
    then: list[Reply] = [
        tool_call("python_exec", {"code": INSTALL}, "i-1"),
        answer("Done.\nEnding: ready"),
    ]
    async with owner_console(
        tmp_path,
        monkeypatch,
        main=main,
        workers=[first, then],
        approvals="auto",
        follow_up=False,
    ) as (console, client, models, ran):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        await until(lambda: models["workers"] and models["workers"][0].requests)
        workers = console.workers
        info = await console.store.get_session(task_id)
        assert info is not None
        allowed = workers._auto.setdefault(task_id, AutoPolicy(task=True))
        allowed.allow_once(
            step_key("python_exec", {"code": INSTALL}, info.workspace),
            ["install"],
            "python_exec",
        )
        _, how = await workers.message(task_id, "Go ahead.", same_job=True)
        assert how == "queued"
        release.set()
        fresh = await _went_on(console, task_id)
        await until(lambda: _is(console, fresh.task_id, "done"), timeout=15.0)
        assert ran.codes == [INSTALL]


async def test_a_message_whose_successor_cannot_start_is_said_in_the_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    worker: list[Reply] = [held(release, answer("Done.\nEnding: ready"))]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=[worker], follow_up=False
    ) as (console, client, models, _):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        await until(lambda: models["workers"] and models["workers"][0].requests)
        _, how = await console.workers.message(task_id, "also add a short summary")
        assert how == "queued"

        async def full(*_: Any, **__: Any) -> Any:
            raise TaskStoreError("the task store is full")

        monkeypatch.setattr(console.workers, "successor", full)
        release.set()

        async def told() -> bool:
            return any(
                r["echo"] == "failed" and "the task store is full" in r["line"]
                for r in await rows(console, console.main_id)
            )

        await until(told, timeout=10.0)


async def test_a_second_message_for_an_ended_job_never_reopens_its_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    worker: list[Reply] = [held(release, answer("Done.\nEnding: ready"))]
    async with owner_console(
        tmp_path, monkeypatch, main=main, workers=[worker], run=False, follow_up=False
    ) as (console, client, models, _):
        await say(client, console, REPORT)
        first = await started_task(console)
        await stop_run(console, first)
        del console.workers._schedule  # tasks run from here on
        body, _ = await console.workers.continue_stopped(first)
        fresh = str(body["task_id"])
        await until(lambda: models["workers"] and models["workers"][0].requests)
        ended = await console.workers.tasks.get_task(first)
        note = "also add a short summary"
        echo = await console.front.follow(ended, note, typed=note)
        assert (echo["echo"], echo["task_id"]) == ("sent", fresh)
        release.set()
        newer = await _went_on(console, fresh)
        assert len(await console.store.list_turns(fresh)) == 1, "never reopened"
        assert note in newer.spec["objective"]


async def _went_on(console: Console, task_id: str) -> TaskRecord:
    """The task that went on with *task_id*'s job once it ended."""
    workers = console.workers
    heads: list[TaskRecord] = []

    async def moved() -> bool:
        heads.append(await workers.newest(await workers.tasks.get_task(task_id)))
        return heads[-1].task_id != task_id

    await until(moved, timeout=10.0)
    return heads[-1]


async def test_every_turn_of_an_ended_task_keeps_its_unknown_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-time task holds more than one turn after a resume: a call cut off in
    its first turn stays fenced in its successor."""
    main: list[Reply] = [creates("Long report"), answer("Started it as a task.")]
    async with owner_console(
        tmp_path, monkeypatch, main=main, run=False, follow_up=False
    ) as (console, client, _, _):
        await say(client, console, REPORT)
        task_id = await started_task(console)
        record = await console.workers.tasks.get_task(task_id)
        info = await console.store.get_session(task_id)
        assert info is not None
        written = {"path": "out.txt", "content": "part one"}
        cut = core_event("tool.started", call_id="w-0", name="file_write",
                         arguments=written, preview="out.txt",
                         run_key=call_key("file_write", written, info.workspace))  # fmt: skip
        turn = await console.store.begin_turn(
            task_id, record.spec["objective"], holder=HOLDER
        )
        await console.store.append_events(
            task_id, turn.turn, [{**cut, "run_id": turn.run_id}], holder=HOLDER
        )
        await console.store.finish_turn(
            task_id, turn.turn, holder=HOLDER, status="failed", cause="run",
            result=run_result(status="failed"), answer="It stopped.",
        )  # fmt: skip
        await stop_run(console, task_id)  # the resumed turn, which ran out of steps
        body, _ = await console.workers.continue_stopped(task_id)
        fresh = await console.workers.tasks.get_task(str(body["task_id"]))
        blocks = fresh.spec["handoff"]["blocks"]
        assert [b["tool"] for b in blocks if b["who"] == UNKNOWN_EFFECT] == [
            "file_write"
        ]


def _tool(name: str, read: bool, tags: tuple[str, ...] = ()) -> ToolDef:
    return ToolDef(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        annotations={"readOnlyHint": read},
        semantic_tags=list(tags),
    )


async def test_only_a_read_cut_off_may_run_again_whoever_moved_the_job_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool's own tier decides: a cut-off read may run again, a cut-off page
    action or write asks the person first, in a successor the person's message
    started too; a step the person refused gets a fresh decision there, and the
    packet says so."""
    tools = {
        "web_fetch": _tool("web_fetch", read=True),
        "browser_click": _tool("browser_click", read=False, tags=("browser",)),
    }

    def block(name: str, who: str) -> dict[str, Any]:
        return {"call_key": call_key(name, {"n": 1}, None), "loose_keys": [],
                "who": who, "step_key": "", "what": name, "tool": name}  # fmt: skip

    blocks = (
        block("web_fetch", UNKNOWN_EFFECT),
        block("browser_click", UNKNOWN_EFFECT),
        block("python_exec", "the user"),
    )
    async with owner_console(tmp_path, monkeypatch, run=False, follow_up=False) as (
        console,
        _,
        _,
        _,
    ):
        for by in ("person", "melo"):
            packet = Handoff(origin="task", source="t1", by=by, blocks=blocks)
            spec = packet.onto(TaskSpec(title="Report", objective=REPORT))
            body, _ = await console.workers.create_spec(
                spec, front_id=console.main_id, chat_started=True, handoff=packet
            )
            record = await console.workers.tasks.get_task(str(body["task_id"]))
            info = await console.store.get_session(record.task_id)
            assert info is not None
            policy = await console.workers._guard(
                record, info.meta, AllowAllPolicy(), tools.get
            )
            earlier = find_earlier(policy)
            assert earlier is not None

            def key(name: str) -> str:
                return call_key(name, {"n": 1}, None)

            assert earlier.fresh == {key("browser_click")}
            refused = earlier.match("python_exec", {"n": 1}, None) is not None
            assert refused is (by != "person")
            said = handoff_block(record.spec["handoff"])
            assert "browser_click" in said and "web_fetch" in said
            assert "melo asks the person" in said
