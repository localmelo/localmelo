"""The main chat does what it is asked, and reaches its tasks by what the person says.

``search_files`` answers "which file defines X?", a follow-up reaches the task
it names (a running task at its next step, a finished one as a new turn), and
the main chat reads its tasks' answers. Each test drives the console's routes
with a real Agent on a scripted model; no worker starts.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console import front as front_module
from localmelo.support.gateway.console.front import (
    FRONT_TEXT,
    answer_head,
    clean_title,
)
from localmelo.support.gateway.finishing import OUT_OF_STEPS
from localmelo.support.sessions.display import short_title
from localmelo.tests.sessions.helpers import answer, tool_call

from ..conftest import REAL_ASK_TITLE
from .conftest import HOLDER, run_result
from .test_front import _echoes, _say, _start, _task, front_console
from .test_turn_writes import _wait_turn

FIX = (
    "Please fix this by today: in tests/cli/test_e2e_pty.py the list_files test "
    "only checks one word. Make it check a file name and commit."
)
REPORT = "Fixed the check in test_e2e_pty.py.\n24 passed.\nCommitted b60d0892."


async def _finished(
    console: Console,
    task_id: str,
    reply: str = REPORT,
    ending: dict[str, Any] | None = None,
) -> None:
    """*task_id* ran one turn that answered *reply* and is done, as *ending* says."""
    store = console.store
    record = await _task(console, task_id)
    turn = await store.begin_turn(task_id, record.spec["objective"], holder=HOLDER)
    await store.finish_turn(
        task_id,
        turn.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(),
        answer=reply,
    )
    await console.workers.tasks.change(
        task_id, state="done", actor="scheduler", ending=ending
    )
    console.gate.finish_task(task_id, "done")


async def _delegated(console: Console, client: Any, text: str) -> str:
    echo = (await _say(client, console, f"/task {text}")).json()["echo"]
    return str(echo["task_id"])


def _tool_result(llm: Any, request: int = 1) -> str:
    return str(llm.requests[request][-1].content)


async def _fresh(console: Console, task_id: str) -> Any:
    """The fresh task that went on with *task_id*'s job (#571): it ran no more."""
    assert (await _task(console, task_id)).state in ("done", "failed", "cancelled")
    head = await console.workers.newest(await _task(console, task_id))
    assert head.task_id != task_id, "no fresh task went on"
    return head


# ── A reply is what the model said ──


async def test_a_real_answer_is_kept(tmp_path: Path) -> None:
    reply = answer("Let me check. It is defined in support/gateway/task_policy.py.")
    async with front_console(tmp_path, reply) as (console, client, _):
        await _say(client, console, "where is the default folder set")
        turn = await _wait_turn(client, str(console.main_id))
        assert turn["answer"].endswith("task_policy.py.")
        assert await _echoes(console) == []


async def test_a_slash_task_with_a_change_to_named_files_is_a_task_without_a_model_call(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        response = await _say(client, console, f"/task {FIX}")
        assert response.status_code == 200, response.text
        echo = response.json()["echo"]
        assert echo["line"] == f"Started: {short_title(FIX)} · P1"
        record = await _task(console, echo["task_id"])
        assert (record.spec["objective"], record.level) == (FIX, 1)
        assert llm.requests == []


async def test_every_request_tells_the_chat_to_act(tmp_path: Path) -> None:
    async with front_console(tmp_path, answer("Done.")) as (console, client, llm):
        await _say(client, console, "tidy my notes")
        await _wait_turn(client, str(console.main_id))
        assert FRONT_TEXT in llm.requests[0][0].content
    # One instruction for handing the rest on (#571): no step count, no meter.
    [rule] = [line for line in FRONT_TEXT.split(". ") if "continue_current" in line]
    assert "finish it here" in FRONT_TEXT and "large tool or page material" in rule
    assert not any(ch.isdigit() for ch in rule)


# ── search_files: "which file defines X?" in one or two steps ──


async def test_the_main_chat_answers_which_file_defines_x_with_its_search(
    tmp_path: Path,
) -> None:
    replies = (
        tool_call(
            "search_files", {"pattern": "DEFAULT_WORKSPACE =", "path_glob": "*.py"}
        ),
        answer("support/gateway/task_policy.py, line 3."),
    )
    async with front_console(tmp_path, *replies) as (console, client, llm):
        main = str(console.main_id)
        async with console.sessions.lock(main):
            await console.sessions.open(main)
        folder = Path(str(console.front.folder()))
        (folder / "support" / "gateway").mkdir(parents=True, exist_ok=True)
        (folder / "support" / "gateway" / "task_policy.py").write_text(
            'import os\n\nDEFAULT_WORKSPACE = "~/localmelo-workspace"\n'
        )
        await _say(
            client,
            console,
            "Quick question: in this repository, which file defines "
            "DEFAULT_WORKSPACE? Answer in one line.",
        )
        turn = await _wait_turn(client, main)
        assert (turn["status"], turn["answer"]) == (
            "completed",
            "support/gateway/task_policy.py, line 3.",
        )
        assert len(llm.requests) == 2
        assert "support/gateway/task_policy.py:3: DEFAULT_WORKSPACE =" in _tool_result(
            llm
        )
        assert await _echoes(console) == []


# ── Follow-ups reach the task they name ──


async def test_finish_it_continues_the_newest_task_that_did_not_finish(
    tmp_path: Path,
) -> None:
    draft = {"state": "draft", "reasons": ["commit_missing"]}
    async with front_console(tmp_path) as (console, client, llm):
        task_id = await _delegated(console, client, "Fix the e2e list_files check")
        await _finished(console, task_id, "Done except the commit.", draft)
        llm.replies.extend(
            [
                tool_call("task_message", {"task_id": task_id}),
                answer("Sent it to the e2e task."),
            ]
        )
        await _say(client, console, "finish it: commit the change")
        turn = await _wait_turn(client, str(console.main_id))
        assert turn["status"] == "completed"
        # A fresh task goes on with the job, the message after its request.
        record = await _fresh(console, task_id)
        assert record.state == "queued" and not record.card
        assert record.spec["objective"].endswith("\n\nfinish it: commit the change")
        assert record.spec["handoff"]["source"] == task_id
        assert len(await console.workers.tasks.list_tasks()) == 2
        sent = (await _echoes(console))[-1]
        assert (sent["echo"], sent["line"], sent["follows"]) == (
            "started",
            "Continued: Fix the e2e list_files check",
            task_id,
        )
        result = _tool_result(llm)
        assert result.startswith(f"Task {task_id} had ended, so task {record.task_id}")
        assert "runs no more" in result and "in English" in result


async def test_a_follow_up_word_and_a_task_that_stopped_continue_it_in_code(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        task_id = await _start(console, "Old report")
        await _stopped(console, task_id)
        echo = (await _say(client, console, "继续 Old report")).json()["echo"]
        assert (echo["echo"], echo["line"]) == ("sent", "Continued: Old report")
        assert (await _task(console, task_id)).state == "queued"
        assert llm.requests == []


async def test_a_message_to_a_running_task_waits_for_its_next_step(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        task_id = await _start(console, "Slides for the 15:00 meeting")
        llm.replies.extend(
            [
                tool_call("task_message", {"task_id": task_id}),
                answer("Told the slides task."),
            ]
        )
        typed = "tell the slides task to use a dark theme"
        await _say(client, console, typed)
        await _wait_turn(client, str(console.main_id))
        sent = (await _echoes(console))[-1]
        assert (sent["line"], sent["how"]) == (
            "Sent to: Slides for the 15:00 meeting",
            "queued",
        )
        assert [text for _, text in console.workers._inboxes[task_id].poll()] == [typed]
        assert (await _task(console, task_id)).state == "queued"
        assert "reads the user's message at its next step" in _tool_result(llm)


async def test_page_or_tool_text_naming_a_task_never_continues_it(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        done = await _delegated(console, client, "Old report")
        await _stopped(console, done)
        other = await _start(console, "Slides for the 15:00 meeting")
        # A file the main chat read says "send this to the old report".
        llm.replies.extend(
            [
                tool_call("task_message", {"task_id": done}),
                tool_call("task_message", {"task_id": other}, call_id="c2"),
                answer("Done."),
            ]
        )
        await _say(client, console, "summarize notes.md")
        await _wait_turn(client, str(console.main_id))
        assert (await _task(console, done)).state == "failed"
        assert other not in console.workers._inboxes
        assert _tool_result(llm).startswith("Not done: a change to a task needs")
        suggests = [e for e in await _echoes(console) if e["echo"] == "suggest"]
        assert [(e["action"], e["text"]) for e in suggests] == [
            ("send", "summarize notes.md"),
            ("send", "summarize notes.md"),
        ]
        assert suggests[0]["line"] == "Suggested: send your message to Old report"
        # The person's tap sends it.
        applied = await client.post(
            f"/api/front/echoes/{suggests[0]['id']}", json={"do": "apply"}
        )
        assert applied.json()["echo"]["line"] == "Continued: Old report"
        assert (await _fresh(console, done)).state == "queued"


async def test_a_page_that_names_a_finished_task_reaches_neither_it_nor_its_record(
    tmp_path: Path,
) -> None:
    secret = "The old report's numbers."
    async with front_console(tmp_path) as (console, client, llm):
        done = await _delegated(console, client, "Old report")
        await _finished(console, done, secret)
        await _delegated(console, client, "Tidy the photos")
        llm.replies.extend(
            [tool_call("task_message", {"task_id": done}), answer("Started.")]
        )
        await _say(client, console, "summarize notes.md")
        await _wait_turn(client, str(console.main_id))
        assert (await _task(console, done)).state == "done"
        # The person's message is the work; a task they did not point at stays out.
        started = (await _echoes(console))[-1]
        assert started["echo"] == "started"
        record = await _task(console, started["task_id"])
        assert record.spec["objective"] == "summarize notes.md"
        assert secret not in str(record.spec)


async def test_finish_it_does_not_reach_a_task_the_chat_did_not_show_last(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        older = await _delegated(console, client, "Old report")
        await _stopped(console, older)
        await _delegated(console, client, "Tidy the photos")
        llm.replies.extend(
            [tool_call("task_message", {"task_id": older}), answer("Done.")]
        )
        await _say(client, console, "finish it")
        await _wait_turn(client, str(console.main_id))
        assert (await _task(console, older)).state == "failed"
        assert (await _echoes(console))[-1]["echo"] == "suggest"


# ── The main chat reads its tasks' state and answers ──


async def test_what_got_done_names_the_commits_tasks_reported(tmp_path: Path) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        task_id = await _delegated(console, client, "Fix the e2e list_files check")
        await _finished(console, task_id)
        llm.replies.extend(
            [
                tool_call("task_list", {}),
                tool_call("task_get", {"task_id": task_id}, call_id="c2"),
                answer("The e2e check is fixed and committed as b60d0892."),
            ]
        )
        await _say(client, console, "What did your tasks get done? Two lines.")
        await _wait_turn(client, str(console.main_id))
        # The turn reads the tasks that ended since the last one…
        query = llm.requests[0][-1].content
        assert "tasks that ended since the last message" in query
        assert "Committed b60d0892" in query
        # …task_list carries each task's state and the first lines of its answer…
        listed = _tool_result(llm, 1)
        assert f"{task_id} · Fix the e2e list_files check · done · P2 · " in listed
        assert "24 passed. / Committed b60d0892." in listed
        # …and task_get its whole answer, line by line (#440).
        got = _tool_result(llm, 2)
        assert "its answer:" in got
        assert (
            "Fixed the check in test_e2e_pty.py.\n24 passed.\nCommitted b60d0892."
            in got
        )
        # Read once: the next turn does not hear it again.
        llm.replies.append(answer("Nothing new."))
        await _say(client, console, "anything else?")
        await _wait_turn(client, str(console.main_id))
        assert "tasks that ended" not in llm.requests[-1][-1].content


def test_answer_heads_keep_the_first_lines() -> None:
    assert answer_head("<think>x</think>## Done\n- tests pass\n\nsha abc123") == (
        "Done / tests pass / sha abc123"
    )
    assert answer_head("word " * 100, 20) == "word word word word…"
    assert answer_head("key sk-" + "a" * 30) == "key [redacted]"


# ── "Continue the … task" (#341) ──


async def _stopped(console: Console, task_id: str, reply: str = REPORT) -> None:
    """*task_id* ran one turn that used all its steps, so it is stopped."""
    store = console.store
    record = await _task(console, task_id)
    turn = await store.begin_turn(task_id, record.spec["objective"], holder=HOLDER)
    await store.finish_turn(
        task_id,
        turn.turn,
        holder=HOLDER,
        status="budget_exhausted",
        cause="run",
        result=run_result(),
        answer=reply,
    )
    await console.workers.tasks.change(
        task_id, state="failed", actor="scheduler", failure_reason=OUT_OF_STEPS
    )
    console.gate.finish_task(task_id, "failed")


CONTINUE = (
    "Please continue the recovery card task where it stopped: run the two changed "
    "test files, and commit if they pass."
)


async def test_continue_the_task_continues_the_one_stopped_task_at_once(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        stopped = await _delegated(console, client, "Hide the recovery card section")
        other = await _delegated(console, client, "Improve the Activity row text")
        await _stopped(console, stopped)
        await _finished(console, other)
        response = await _say(client, console, CONTINUE)
        assert response.status_code == 200, response.text
        echo = response.json()["echo"]
        record = await _fresh(console, stopped)
        assert (echo["echo"], echo["follows"], echo["task_id"]) == (
            "started",
            stopped,
            record.task_id,
        )
        assert echo["line"] == "Continued: Hide the recovery card section"
        assert echo["typed"] == CONTINUE
        # No model call and no suggestion: a fresh task goes on with the message.
        assert record.state == "queued" and record.spec["objective"].endswith(CONTINUE)
        assert llm.requests == []
        assert [e["echo"] for e in await _echoes(console)][-1:] == ["started"]
        assert "suggest" not in [e["echo"] for e in await _echoes(console)]
        assert (await _task(console, other)).state == "done"


async def test_a_stopped_task_stays_stopped_and_more_work_is_one_fresh_task(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        stopped = await _delegated(console, client, "Hide the recovery card section")
        await console.workers.update(stopped, state="stop")
        message = tool_call("task_message", {"task_id": stopped})
        llm.replies.extend([message, answer("Sent."), message, answer("Sent again.")])
        await _say(client, console, "use a toggle for that section instead")
        chat = str(console.main_id)
        turn = await _wait_turn(client, chat)
        fresh = await _fresh(console, stopped)
        # Stop stands: the stopped task never runs again, nor reloads its session.
        assert (await _task(console, stopped)).state == "cancelled"
        assert await console.store.list_turns(stopped) == []
        assert fresh.spec["handoff"]["ended"].startswith("Stopped")
        console.front.made.clear()  # what a restart forgets
        retried = await client.post(
            f"/api/sessions/{chat}/retry", json={"turn_id": turn["id"]}
        )
        assert retried.status_code == 202, retried.text
        await _wait_turn(client, chat)
        # The retry reaches the same fresh task; it admits no other.
        assert len(await console.workers.tasks.list_tasks()) == 2


async def test_continue_the_task_in_chinese(tmp_path: Path) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        stopped = await _delegated(console, client, "修复恢复卡片的测试")
        await _stopped(console, stopped)
        echo = (await _say(client, console, "继续恢复卡片任务，然后提交")).json()[
            "echo"
        ]
        assert (echo["echo"], echo["follows"]) == ("started", stopped)
        assert (await _fresh(console, stopped)).state == "queued"
        assert llm.requests == []


async def test_several_stopped_tasks_are_each_offered_to_pick_from(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        first = await _delegated(console, client, "Fix the recovery card test")
        second = await _delegated(console, client, "Hide the recovery card section")
        await _stopped(console, first)
        await _stopped(console, second)
        message = "continue the recovery card task"
        response = await _say(client, console, message)
        assert response.status_code == 200, response.text
        offers = [e for e in await _echoes(console) if e["echo"] == "suggest"]
        assert {e["task_id"] for e in offers} == {first, second}
        assert {e["action"] for e in offers} == {"send"}
        # The first carries the person's message; applying one continues that task.
        assert [e.get("typed") for e in offers].count(message) == 1
        assert {(await _task(console, t)).state for t in (first, second)} == {"failed"}
        applied = await client.post(
            f"/api/front/echoes/{offers[0]['id']}", json={"do": "apply"}
        )
        assert applied.status_code == 200, applied.text
        assert (await _fresh(console, offers[0]["task_id"])).state == "queued"
        assert llm.requests == []


async def test_continue_the_task_says_when_it_is_still_working(tmp_path: Path) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        busy = await _delegated(console, client, "Hide the recovery card section")
        said = (await _say(client, console, "continue the recovery card task")).json()
        assert (
            said["echo"]["line"] == "Hide the recovery card section is still working."
        )
        assert (await _task(console, busy)).state == "queued"
        assert llm.requests == []


async def test_continue_the_task_resumes_a_paused_task_whatever_the_length(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        paused = await _delegated(console, client, "Hide the recovery card section")
        await console.workers.update(paused, state="pause")
        assert (await _task(console, paused)).state == "parked"
        echo = (await _say(client, console, CONTINUE)).json()["echo"]
        assert (echo["echo"], echo["action"]) == ("control", "resume")
        assert (await _task(console, paused)).state == "queued"
        assert llm.requests == []


# ── A question that asks for a short answer has the ordinary steps ──


async def test_a_question_that_asks_for_a_short_answer_keeps_the_ordinary_steps(
    tmp_path: Path,
) -> None:
    searches = [
        tool_call("search_files", {"pattern": pattern}, call_id=f"s{n}")
        for n, pattern in enumerate(("a =", "b =", "c ="), start=1)
    ]
    replies = [*searches, answer("It is set in support/gateway/task_policy.py.")]
    async with front_console(tmp_path, *replies) as (console, client, llm):
        await _say(client, console, "briefly, which file defines DEFAULT_WORKSPACE?")
        turn = await _wait_turn(client, str(console.main_id))
        assert (turn["status"], turn["answer"]) == (
            "completed",
            "It is set in support/gateway/task_policy.py.",
        )
        assert len(llm.requests) == 4  # no small cap of a chat's own
        assert await _echoes(console) == []


# ── A task code starts is named in a few words ──


class Titles:
    """A stand-in for the model's reply to the title request."""

    def __init__(self, reply: str | Exception = "") -> None:
        self.reply = reply
        self.asked: list[str] = []

    async def __call__(self, request: str) -> str:
        self.asked.append(request)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


async def _session_title(console: Console, task_id: str) -> str:
    info = await console.store.get_session(task_id)
    assert info is not None
    return info.title


async def _title_events(console: Console, task_id: str) -> list[str]:
    events = await console.store.list_events(task_id)
    return [
        e.event["data"]["title"] for e in events if e.event["type"] == "session.title"
    ]


async def test_a_task_starts_at_once_and_is_named_by_the_model_when_it_answers(
    tmp_path: Path,
) -> None:
    release = asyncio.Event()
    asked: list[str] = []

    async def slow(request: str) -> str:
        asked.append(request)
        await release.wait()
        return '"Hide the recovery card in default mode."\nbecause it is empty'

    async with front_console(tmp_path, titler=slow) as (console, client, llm):
        echo = (await _say(client, console, f"/task {FIX}")).json()["echo"]
        # The answer did not wait for the title: the task runs under the first words.
        first = short_title(FIX)
        assert echo["line"] == f"Started: {first} · P1"
        task_id = echo["task_id"]
        assert (await _task(console, task_id)).title == first
        assert await _session_title(console, task_id) == first
        assert asked == [FIX]
        release.set()
        await console.front.drain()
        # The task, its session and the event the page reads all say the new title.
        title = "Hide the recovery card in default mode"
        record = await _task(console, task_id)
        assert (record.title, record.spec["objective"]) == (title, FIX)
        assert await _session_title(console, task_id) == title
        assert await _title_events(console, task_id) == [title]
        assert [
            r["title"] for r in (await client.get("/api/tasks")).json()["tasks"]
        ] == [title]
        assert llm.requests == []
        # The same for /task once the request is longer than a title.
        again = (await _say(client, console, "/task " + FIX + " " + FIX)).json()["echo"]
        await console.front.drain()
        assert (await _task(console, again["task_id"])).title == title
        assert asked[-1] == FIX + " " + FIX


async def test_a_short_request_is_its_own_title_and_asks_no_model(
    tmp_path: Path,
) -> None:
    titles = Titles("Never used")
    async with front_console(tmp_path, titler=titles) as (console, client, _):
        echo = (await _say(client, console, "/task make slides from notes.md")).json()[
            "echo"
        ]
        await console.front.drain()
        assert echo["line"] == "Started: make slides from notes.md · P2"
        assert titles.asked == []


@pytest.mark.parametrize(
    "reply",
    [
        RuntimeError("no model server"),
        "",
        "Fix",
        "The title of this work is a very long sentence that goes on and on and on",
        short_title(FIX),  # the model's title is the first words: nothing changes
    ],
)
async def test_a_title_the_model_cannot_give_leaves_the_first_words(
    tmp_path: Path, reply: str | Exception
) -> None:
    async with front_console(tmp_path, titler=Titles(reply)) as (console, client, _):
        echo = (await _say(client, console, f"/task {FIX}")).json()["echo"]
        await console.front.drain()
        record = await _task(console, echo["task_id"])
        assert record.title == await _session_title(console, record.task_id)
        assert record.title == short_title(FIX)
        assert await _title_events(console, record.task_id) == []


async def test_a_title_that_takes_too_long_leaves_the_first_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(front_module, "TITLE_TIMEOUT_S", 0.05)

    async def slow(request: str) -> str:
        await asyncio.sleep(5)
        return "Too late for this task"

    async with front_console(tmp_path, titler=slow) as (console, client, _):
        echo = (await _say(client, console, f"/task {FIX}")).json()["echo"]
        await console.front.drain()
        assert (await _task(console, echo["task_id"])).title == short_title(FIX)
        assert await _title_events(console, echo["task_id"]) == []


async def test_a_task_the_person_renamed_meanwhile_keeps_their_name(
    tmp_path: Path,
) -> None:
    release = asyncio.Event()

    async def late(request: str) -> str:
        await release.wait()
        return "Hide the recovery card in default mode"

    async with front_console(tmp_path, titler=late) as (console, client, _):
        echo = (await _say(client, console, f"/task {FIX}")).json()["echo"]
        task_id = echo["task_id"]
        assert await console.store.rename_session(task_id, "My own name")
        release.set()
        await console.front.drain()
        assert await _session_title(console, task_id) == "My own name"
        assert await _title_events(console, task_id) == ["My own name"]
        # The task row was not named either: its first words stay for the list.
        assert (await _task(console, task_id)).title == short_title(FIX)


async def test_the_title_request_asks_at_the_lowest_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, int]] = []

    class Provider(BaseLLMProvider):
        async def chat(self, messages: list[Message], tools: Any = None) -> Message:
            gate = holder[0].gate
            seen.extend(
                (name, gate.level(name))
                for name in list(gate._levels)  # noqa: SLF001
                if name.startswith("title:")
            )
            return Message(role="assistant", content="Hide the recovery card now")

        async def close(self) -> None:
            pass

    class Backend:
        def build_chat_provider(self, cfg: Any) -> Provider:
            return Provider()

    holder: list[Console] = []
    async with front_console(tmp_path) as (console, client, _):
        holder.append(console)
        monkeypatch.setattr(
            "localmelo.support.backends.registry.get_backend", lambda name: Backend()
        )
        # The suite's stand-in refuses the request; call the real one.
        title = await REAL_ASK_TITLE(console.front, FIX)
        assert title == "Hide the recovery card now"
        assert [level for _, level in seen] == [4]
        # Its bookkeeping goes with the request.
        assert not [
            n for n in console.gate._levels if n.startswith("title:")
        ]  # noqa: SLF001


async def test_the_model_names_its_own_tasks_without_a_second_request(
    tmp_path: Path,
) -> None:
    titles = Titles("Never used")
    create = tool_call(
        "task_create", {"title": "Improve Activity row Done text", "objective": FIX}
    )
    async with front_console(tmp_path, create, answer("Started."), titler=titles) as (
        console,
        client,
        _,
    ):
        # The model's task_create is the person's, with a title of its own.
        await _start(console, "Activity row text")
        typed = (
            "Fix the Activity row text in support/gateway/console/activity.py "
            "and finish it"
        )
        await _say(client, console, typed)
        await _wait_turn(client, str(console.main_id))
        [echo] = [e for e in await _echoes(console) if e["echo"] == "started"]
        assert echo["title"] == "Improve Activity row Done text"
        assert titles.asked == []
        # The person's message is the objective; the model's reading is not sent.
        record = await _task(console, echo["task_id"])
        assert record.spec["objective"] == typed


@pytest.mark.parametrize(
    ("raw", "title"),
    [
        ("Fix the list_files test", "Fix the list_files test"),
        ("**Title:** Fix the list_files test.", "Fix the list_files test"),
        ("标题：修复文件列表测试", "修复文件列表测试"),
        ('<think>hm</think>\n"Fix the list_files test"', "Fix the list_files test"),
        ("Fix", ""),
        ("", ""),
        ("one two three four five six seven eight nine ten eleven", ""),
    ],
)
def test_a_models_reply_is_cleaned_into_a_title(raw: str, title: str) -> None:
    assert clean_title(raw) == title
