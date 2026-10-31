"""The main chat does what it is asked, or hands it off with its context.

A preamble with no work is handed off, an imperative change to named files
becomes a task without a model call, ``search_files`` answers "which file
defines X?", a question at the cap asks, the hand-off sentence follows the
message's language, a follow-up reaches the task it names (a running task at
its next step, a finished one as a new turn), and the main chat reads its
tasks' answers. Each test drives the console's routes with a real Agent on a
scripted model; no worker starts.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console import front as front_module
from localmelo.support.gateway.console.front import (
    ASK_TEXT,
    BRIEF_NOTE,
    FRONT_TEXT,
    STARTED_TEXT,
    STARTED_TEXT_ZH,
    answer_head,
    clean_title,
)
from localmelo.support.gateway.finishing import OUT_OF_STEPS
from localmelo.support.gateway.front_rules import short_title
from localmelo.tests.sessions.helpers import answer, tool_call

from .conftest import HOLDER, run_result
from .test_front import _echoes, _four_steps, _say, _start, _task, front_console
from .test_turn_writes import _wait_turn

FIX = (
    "Please fix this by today: in tests/cli/test_e2e_pty.py the list_files test "
    "only checks one word. Make it check a file name and commit."
)
REPORT = "Fixed the check in test_e2e_pty.py.\n24 passed.\nCommitted b60d0892."


async def _finished(console: Console, task_id: str, reply: str = REPORT) -> None:
    """*task_id* ran one turn that answered *reply* and is done."""
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
    await console.workers.tasks.change(task_id, state="done", actor="scheduler")
    console.gate.finish_task(task_id, "done")


async def _delegated(console: Console, client: Any, text: str) -> str:
    echo = (await _say(client, console, f"/task {text}")).json()["echo"]
    return str(echo["task_id"])


def _tool_result(llm: Any, request: int = 1) -> str:
    return str(llm.requests[request][-1].content)


# ── A reply that only announces work is handed off ──


async def test_a_preamble_with_no_tool_call_hands_the_work_off(tmp_path: Path) -> None:
    preamble = answer(
        "I'll start by looking at the test file and finding the relevant code."
    )
    async with front_console(tmp_path, preamble) as (console, client, llm):
        typed = "the e2e check for list_files is too weak, make it stronger and commit"
        await _say(client, console, typed)
        turn = await _wait_turn(client, str(console.main_id))
        assert (turn["status"], turn["answer"]) == ("completed", STARTED_TEXT)
        assert len(llm.requests) == 1
        [echo] = await _echoes(console)
        assert echo["echo"] == "started"
        record = await _task(console, echo["task_id"])
        assert (record.title, record.spec["objective"]) == (short_title(typed), typed)
        # The next turn reads the hand-off, not the preamble.
        entry = console.sessions.get(str(console.main_id))
        assert entry is not None and entry.agent is not None
        assert entry.agent.conversation()[-1].content == STARTED_TEXT
        # The main chat is told to act, not announce, in the message's language.
        assert FRONT_TEXT in llm.requests[0][0].content


async def test_a_chinese_preamble_hands_off_in_chinese(tmp_path: Path) -> None:
    async with front_console(tmp_path, answer("我先看一下测试文件。")) as (
        console,
        client,
        _,
    ):
        await _say(client, console, "把端到端测试改得更严格一些，然后提交")
        turn = await _wait_turn(client, str(console.main_id))
        assert turn["answer"] == STARTED_TEXT_ZH
        assert [e["echo"] for e in await _echoes(console)] == ["started"]


async def test_a_question_answered_with_a_preamble_asks(tmp_path: Path) -> None:
    async with front_console(tmp_path, answer("Let me look into it.")) as (
        console,
        client,
        _,
    ):
        await _say(client, console, "why is the e2e check so weak?")
        turn = await _wait_turn(client, str(console.main_id))
        assert turn["answer"] == ASK_TEXT
        [ask] = await _echoes(console)
        assert ask["echo"] == "ask"
        assert await console.workers.tasks.list_tasks() == []


async def test_a_real_answer_is_kept(tmp_path: Path) -> None:
    reply = answer("Let me check. It is defined in support/gateway/task_policy.py.")
    async with front_console(tmp_path, reply) as (console, client, _):
        await _say(client, console, "where is the default folder set")
        turn = await _wait_turn(client, str(console.main_id))
        assert turn["answer"].endswith("task_policy.py.")
        assert await _echoes(console) == []


# ── Rules first: work on named files is a task at once ──


async def test_an_imperative_change_to_named_files_is_a_task_without_a_model_call(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        response = await _say(client, console, FIX)
        assert response.status_code == 200, response.text
        echo = response.json()["echo"]
        assert echo["line"] == "Started: fix this by today: in test_e2e_pty.py · P1"
        assert echo["typed"] == FIX
        record = await _task(console, echo["task_id"])
        assert (record.spec["objective"], record.level) == (FIX, 1)
        assert llm.requests == []


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


async def test_a_quick_question_at_the_cap_asks(tmp_path: Path) -> None:
    async with front_console(tmp_path, *_four_steps()) as (console, client, _):
        await _say(
            client,
            console,
            "Quick question: which file defines DEFAULT_WORKSPACE?",
        )
        turn = await _wait_turn(client, str(console.main_id))
        assert turn["answer"] == ASK_TEXT
        assert [e["echo"] for e in await _echoes(console)] == ["ask"]
        assert await console.workers.tasks.list_tasks() == []


async def test_the_hand_off_sentence_follows_the_messages_script(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path, *_four_steps()) as (console, client, _):
        await _say(client, console, "帮我调研三款轻薄笔记本电脑")
        turn = await _wait_turn(client, str(console.main_id))
        assert turn["answer"] == STARTED_TEXT_ZH


# ── Follow-ups reach the task they name ──


async def test_finish_it_continues_the_newest_done_task(tmp_path: Path) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        task_id = await _delegated(console, client, "Fix the e2e list_files check")
        await _finished(console, task_id, "Done except the commit.")
        llm.replies.extend(
            [
                tool_call("task_message", {"task_id": task_id}),
                answer("Sent it to the e2e task."),
            ]
        )
        await _say(client, console, "finish it: commit the change")
        turn = await _wait_turn(client, str(console.main_id))
        assert turn["status"] == "completed"
        record = await _task(console, task_id)
        # The same task, queued again with the message as its next turn.
        assert (record.state, record.card) == ("queued", "finish it: commit the change")
        assert len(await console.workers.tasks.list_tasks()) == 1
        sent = (await _echoes(console))[-1]
        assert (sent["echo"], sent["line"], sent["how"]) == (
            "sent",
            "Continued: Fix the e2e list_files check",
            "continued",
        )
        result = _tool_result(llm)
        assert result.startswith(f"Sent to task {task_id}: Fix the e2e list_files")
        assert "continues in its own session" in result and "in English" in result


async def test_a_follow_up_word_and_a_named_done_task_continue_it_in_code(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        task_id = await _start(console, "Old report")
        await _finished(console, task_id)
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
        await _finished(console, done)
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
        assert (await _task(console, done)).state == "done"
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
        assert (await _task(console, done)).state == "queued"


async def test_finish_it_does_not_reach_a_task_the_chat_did_not_show_last(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        older = await _delegated(console, client, "Old report")
        await _finished(console, older)
        await _delegated(console, client, "Tidy the photos")
        llm.replies.extend(
            [tool_call("task_message", {"task_id": older}), answer("Done.")]
        )
        await _say(client, console, "finish it")
        await _wait_turn(client, str(console.main_id))
        assert (await _task(console, older)).state == "done"
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
        # …and the task tools carry each task's state and the first lines of its answer.
        listed = _tool_result(llm, 1)
        assert f"{task_id} · Fix the e2e list_files check · done · P2 · " in listed
        assert "24 passed. / Committed b60d0892." in listed
        assert "answer: Fixed the check" in _tool_result(llm, 2)
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
        assert (echo["echo"], echo["how"], echo["task_id"]) == (
            "sent",
            "continued",
            stopped,
        )
        assert echo["line"] == "Continued: Hide the recovery card section"
        assert echo["typed"] == CONTINUE
        # No model call and no suggestion: the task runs again with the message.
        record = await _task(console, stopped)
        assert (record.state, record.card) == ("queued", CONTINUE)
        assert llm.requests == []
        assert [e["echo"] for e in await _echoes(console)][-1:] == ["sent"]
        assert "suggest" not in [e["echo"] for e in await _echoes(console)]
        assert (await _task(console, other)).state == "done"


async def test_continue_the_task_in_chinese(tmp_path: Path) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        stopped = await _delegated(console, client, "修复恢复卡片的测试")
        await _stopped(console, stopped)
        echo = (await _say(client, console, "继续恢复卡片任务，然后提交")).json()[
            "echo"
        ]
        assert (echo["echo"], echo["how"]) == ("sent", "continued")
        assert (await _task(console, stopped)).state == "queued"
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
        assert (await _task(console, offers[0]["task_id"])).state == "queued"
        assert llm.requests == []


async def test_continue_the_task_says_when_it_is_done_or_still_working(
    tmp_path: Path,
) -> None:
    async with front_console(tmp_path) as (console, client, llm):
        done = await _delegated(console, client, "Improve the Activity row text")
        busy = await _delegated(console, client, "Hide the recovery card section")
        await _finished(console, done)
        said = (await _say(client, console, "continue the activity row task")).json()
        assert said["echo"]["echo"] == "failed"
        assert (
            said["echo"]["line"]
            == "Improve the Activity row text is done. There is nothing to continue."
        )
        said = (await _say(client, console, "continue the recovery card task")).json()
        assert (
            said["echo"]["line"] == "Hide the recovery card section is still working."
        )
        assert (await _task(console, done)).state == "done"
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


# ── A question that asks for a short answer is answered here ──


async def test_a_brief_question_at_the_cap_gets_its_stop_answer(tmp_path: Path) -> None:
    replies = [*_four_steps(), answer("It reads Done only when the answer is empty.")]
    async with front_console(tmp_path, *replies) as (console, client, llm):
        await _say(
            client,
            console,
            "主对话里卡片上的「Done.」现在是怎么处理的？简单说一下就行。",
        )
        turn = await _wait_turn(client, str(console.main_id))
        assert (turn["status"], turn["answer"]) == (
            "completed",
            "It reads Done only when the answer is empty.",
        )
        # No "Start this as a task?", no task; the model was asked to be brief.
        assert await _echoes(console) == []
        assert await console.workers.tasks.list_tasks() == []
        assert BRIEF_NOTE in llm.requests[0][-1].content
        entry = console.sessions.get(str(console.main_id))
        assert entry is not None and entry.agent is not None
        assert entry.agent.options.answer_on_stop is False  # only this turn asked
        assert entry.agent.conversation()[-1].content == turn["answer"]


async def test_a_brief_question_with_no_stop_answer_still_asks(tmp_path: Path) -> None:
    # The stop answer is asked for, and the model keeps calling tools instead.
    steps = [*_four_steps(), tool_call("task_list", {"x": 2}, call_id="c5")]
    async with front_console(tmp_path, *steps) as (console, client, _):
        await _say(client, console, "briefly, how is the Done card made?")
        turn = await _wait_turn(client, str(console.main_id))
        assert turn["answer"] == ASK_TEXT


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


async def test_a_task_code_starts_is_named_by_the_model_in_a_few_words(
    tmp_path: Path,
) -> None:
    titles = Titles('"Hide the recovery card in default mode."\nbecause it is empty')
    async with front_console(tmp_path, titler=titles) as (console, client, llm):
        echo = (await _say(client, console, FIX)).json()["echo"]
        assert echo["line"] == "Started: Hide the recovery card in default mode · P1"
        record = await _task(console, echo["task_id"])
        assert record.title == "Hide the recovery card in default mode"
        assert record.spec["objective"] == FIX
        assert titles.asked == [FIX]
        assert llm.requests == []
        # The same for /task, once the request is longer than a title.
        longer = "/task " + FIX + " " + FIX
        again = (await _say(client, console, longer)).json()["echo"]
        assert again["title"] == "Hide the recovery card in default mode"
        assert titles.asked[-1] == FIX + " " + FIX


async def test_a_short_request_is_its_own_title_and_asks_no_model(
    tmp_path: Path,
) -> None:
    titles = Titles("Never used")
    async with front_console(tmp_path, titler=titles) as (console, client, _):
        echo = (await _say(client, console, "/task make slides from notes.md")).json()[
            "echo"
        ]
        assert echo["line"] == "Started: make slides from notes.md · P2"
        assert titles.asked == []


@pytest.mark.parametrize(
    "reply",
    [
        RuntimeError("no model server"),
        "",
        "Fix",
        "The title of this work is a very long sentence that goes on and on and on",
    ],
)
async def test_a_title_the_model_cannot_give_falls_back_to_the_first_words(
    tmp_path: Path, reply: str | Exception
) -> None:
    async with front_console(tmp_path, titler=Titles(reply)) as (console, client, _):
        echo = (await _say(client, console, FIX)).json()["echo"]
        assert echo["title"] == short_title(FIX)


async def test_a_title_that_takes_too_long_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(front_module, "TITLE_TIMEOUT_S", 0.05)

    async def slow(request: str) -> str:
        await asyncio.sleep(5)
        return "Too late"

    async with front_console(tmp_path, titler=slow) as (console, client, _):
        echo = (await _say(client, console, FIX)).json()["echo"]
        assert echo["title"] == short_title(FIX)


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
        await _say(client, console, "tidy up the Activity row text, no rush")
        await _wait_turn(client, str(console.main_id))
        [echo] = [e for e in await _echoes(console) if e["echo"] == "started"]
        assert echo["title"] == "Improve Activity row Done text"
        assert titles.asked == []


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
