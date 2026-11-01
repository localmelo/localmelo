"""What runs now and what ran today: ``status.items[*].activity``, ``status.focus`` and ``GET /api/activity``."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localmelo.melo.checker.redaction import REDACTED
from localmelo.melo.contracts.approvals import ApprovalRequest
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console import activity as activity_module
from localmelo.support.gateway.console.activity import (
    MAX_ROWS,
    MAX_SPAN_S,
    answer_lines,
    approvals_of,
    left_line,
    local_midnight,
    result_line,
)
from localmelo.support.gateway.console.worker_reader import WorkerReader
from localmelo.support.gateway.finishing import OUT_OF_STEPS
from localmelo.support.sessions.store import (
    NewSession,
    SessionStore,
    SessionStoreError,
    StoredEvent,
)

from .conftest import HOLDER, core_event, open_session, run_result, seed_turn

# How long ago "today" starts in these tests: their events sit minutes before
# now, which must not fall on yesterday when a run passes midnight.
DAY_SO_FAR_S = 6 * 3600.0


@pytest.fixture(autouse=True)
def _day_started_hours_ago(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.node.name.startswith("test_since_is_today_by_default"):
        return  # that test checks the real local midnight
    start = time.time() - DAY_SO_FAR_S  # one fixed start, as midnight is
    monkeypatch.setattr(activity_module, "local_midnight", lambda now: start)


@pytest.fixture()
def cli_path(tmp_path: Path, console: Console) -> Path:
    path = tmp_path / "cli" / "sessions.db"
    console.worker_history = WorkerReader(str(path))
    return path


async def cli_turn(
    path: Path, kind: str, title: str, *, answer: str = "", events: Any = ()
) -> str:
    """A run (``worker``) or terminal chat (``chat``); finished when it has an *answer*."""
    store = SessionStore(path, writer="cli")
    try:
        turn = await store.begin_turn(
            None,
            "do it",
            holder=HOLDER,
            title=title,
            new_session=NewSession(kind=kind, model="test"),
        )
        if events:
            await store.append_events(
                turn.session_id, turn.turn, list(events), holder=HOLDER
            )
        if answer:
            await store.finish_turn(
                turn.session_id,
                turn.turn,
                holder=HOLDER,
                status="completed",
                cause="run",
                result=run_result(result=answer),
                answer=answer,
            )
        return turn.session_id
    finally:
        await store.aclose()


def set_turn(path: str | Path, session_id: str, **columns: Any) -> None:
    sets = ", ".join(f"{name} = ?" for name in columns)
    with sqlite3.connect(path) as conn:
        conn.execute(
            f"UPDATE turns SET {sets} WHERE session_id = ?",
            (*columns.values(), session_id),
        )


class Task:
    """A task row as ``tasks_since`` gives it, with the fields the list reads."""

    def __init__(self, task_id: str, state: str, title: str, **more: Any) -> None:
        now = time.time()
        self.task_id, self.state, self.title = task_id, state, title
        self.front_id = more.get("front_id")
        self.level_eff = more.get("level", 2)
        self.wait_reason = more.get("wait_reason", "")
        self.finished_at = more.get("finished_at")
        self.updated_at = more.get("updated_at", now)
        self.ending = more.get("ending")


def tasks(monkeypatch: pytest.MonkeyPatch, console: Console, *rows: Task) -> None:
    async def since(_: float) -> list[Task]:
        return list(rows)

    monkeypatch.setattr(console.workers.tasks, "tasks_since", since)


def activity(client: TestClient, **params: Any) -> dict[str, Any]:
    response = client.get("/api/activity", params=params)
    assert response.status_code == 200, response.text
    return response.json()  # type: ignore[no-any-return]


# ── status.items and status.focus ──


def ask(console: Console, owner: str) -> Any:
    request = ApprovalRequest(
        request_id=f"req-{owner}",
        run_id="",
        call_id="call_1",
        tool_name="file_write",
        arguments={},
        preview="a.txt",
        workspace_root=None,
    )
    return console.broker.begin(owner, request)


async def test_each_running_item_says_what_it_does_and_focus_prefers_a_wait(
    client: TestClient,
    console: Console,
    workspace: str,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = await seed_turn(console.store, workspace, title="Slides", status="running")
    await console.store.append_events(
        chat.session_id,
        chat.turn,
        [
            core_event(
                "tool.started",
                ts=time.time() - 60,
                name="file_write",
                arguments={"path": "/w/slides.md", "content": "x"},
            )
        ],
        holder=HOLDER,
    )
    entry = await open_session(console.sessions, chat.session_id)
    monkeypatch.setattr(entry.runner, "_active", object())
    command = core_event(
        "tool.started",
        ts=time.time(),
        name="shell_exec",
        arguments={"command": "uv run pytest -q"},
    )
    run = await cli_turn(cli_path, "worker", "Check the build", events=[command])
    found = client.get("/api/health").json()["status"]
    words = {item["id"]: (item["activity"], item["icon"]) for item in found["items"]}
    assert words == {
        chat.session_id: ("Writing slides.md", "write"),
        run: ("Running the tests", "run"),
    }
    # The most recently changed item is the one a glance shows ...
    assert found["focus"]["id"] == run
    assert found["focus"]["activity"] == "Running the tests"
    # ... until one waits for the person.
    waiting = ask(console, chat.session_id)
    found = client.get("/api/health").json()["status"]
    assert found["focus"]["id"] == chat.session_id
    assert found["focus"]["activity"] == "Waiting for your approval"
    console.broker.withdraw_all(reason="done")
    await waiting


async def test_an_unreadable_event_store_reads_thinking(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = await seed_turn(console.store, workspace, status="running")
    entry = await open_session(console.sessions, chat.session_id)
    monkeypatch.setattr(entry.runner, "_active", object())

    async def broken(*_: Any, **__: Any) -> Any:
        raise SessionStoreError("cannot read")

    monkeypatch.setattr(console.store, "latest_events", broken)
    item = client.get("/api/health").json()["status"]["items"][0]
    assert (item["activity"], item["icon"]) == ("Thinking", "think")


# ── GET /api/activity ──


async def test_the_day_merges_chats_tasks_runs_and_terminal_chats(
    client: TestClient,
    console: Console,
    workspace: str,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    since = now - 3600
    old = await seed_turn(console.store, workspace, title="Yesterday's chat")
    set_turn(console.store.path, old.session_id, finished_at=since - 60)
    chat = await seed_turn(console.store, workspace, title="Plan the trip")
    set_turn(console.store.path, chat.session_id, finished_at=now - 300)
    task_session = await console.store.begin_turn(
        None,
        "find jobs",
        holder=HOLDER,
        title="",
        new_session=NewSession(kind="task", model="m", workspace=workspace),
    )
    await console.store.finish_turn(
        task_session.session_id,
        task_session.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(),
        answer="Two new roles. Both match.",
    )
    set_turn(console.store.path, task_session.session_id, finished_at=now - 200)
    await cli_turn(
        cli_path, "worker", "Check the build", answer="All 412 tests passed."
    )
    terminal = await cli_turn(
        cli_path, "chat", "Read the notes", answer="Version 0.7.1 adds batching. More."
    )
    set_turn(cli_path, terminal, finished_at=now - 100)
    tasks(
        monkeypatch,
        console,
        Task(task_session.session_id, "done", "Find jobs", finished_at=now - 200),
        Task("t-queued", "queued", "Later", level=3),
        Task("t-parked", "parked", "Paused one"),
        Task(
            "t-failed", "failed", "Broke", finished_at=now - 50, wait_reason="no disk"
        ),
    )
    body = activity(client, since=since)
    assert body["since"] == since and body["unavailable"] == [] and not body["more"]
    rows = {row["title"]: row for row in body["rows"]}
    assert "Yesterday's chat" not in rows
    assert [row["kind"] for row in body["rows"]].count("chat") == 1
    assert {row["kind"] for row in body["rows"]} == {"chat", "task", "run", "terminal"}
    # What waits comes first, then the rest, newest first.
    states = [row["state"] for row in body["rows"]]
    assert states[:2] == ["queued", "paused"] or states[:2] == ["paused", "queued"]
    finished = [row["at"] for row in body["rows"][2:]]
    assert finished == sorted(finished, reverse=True)
    assert (
        rows["Later"]["activity"] == "Waiting to start" and rows["Later"]["level"] == 3
    )
    assert rows["Paused one"]["activity"] == "Paused"
    assert (rows["Broke"]["state"], rows["Broke"]["result"]) == ("failed", "no disk")
    assert rows["Find jobs"]["result"] == "Two new roles."
    assert rows["Check the build"]["kind"] == "run"
    assert rows["Check the build"]["result"] == "All 412 tests passed."
    assert rows["Read the notes"]["result"] == "Version 0.7.1 adds batching."
    assert rows["Plan the trip"] | {"at": 0} == {
        "kind": "chat",
        "id": chat.session_id,
        "title": "Plan the trip",
        "state": "done",
        "activity": "",
        "icon": "done",
        "result": "an answer",
        "at": 0,
        "approvals": [],
        "turns": 1,
    }


async def test_a_running_chat_says_what_it_does_and_a_finished_one_its_last_kind(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = await seed_turn(console.store, workspace, title="Live", status="running")
    await console.store.append_events(
        running.session_id,
        running.turn,
        [
            core_event(
                "tool.started",
                name="browser_open",
                arguments={"url": "https://example.com"},
            )
        ],
        holder=HOLDER,
    )
    tasks(monkeypatch, console)
    row = activity(client)["rows"][0]
    assert (row["state"], row["activity"], row["icon"]) == (
        "running",
        "Reading example.com",
        "web",
    )
    await console.store.finish_turn(
        running.session_id,
        running.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(),
        answer="Done.",
    )
    row = activity(client)["rows"][0]
    assert (row["state"], row["icon"], row["activity"]) == ("done", "web", "")


async def test_the_list_is_capped_and_says_so(
    client: TestClient,
    console: Console,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks(monkeypatch, console)
    for n in range(MAX_ROWS + 3):
        await cli_turn(cli_path, "worker", f"run {n}", answer="ok")
    body = activity(client)
    assert len(body["rows"]) == MAX_ROWS and body["more"] is True


def test_a_held_task_says_why_it_waits(
    client: TestClient, console: Console, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks(
        monkeypatch,
        console,
        Task("t-priority", "held", "Behind a higher one"),
        Task("t-slot", "held", "No slot"),
        Task("t-plain", "held", "Unknown reason"),
    )
    reasons = {"t-priority": "Priority", "t-slot": "Resources"}
    monkeypatch.setattr(console.gate, "held", lambda task_id: reasons.get(task_id, ""))
    rows = {row["title"]: row for row in activity(client)["rows"]}
    assert rows["Behind a higher one"]["activity"] == (
        "Waiting for a higher-priority task"
    )
    assert rows["No slot"]["activity"] == "Waiting for a free model slot"
    assert rows["Unknown reason"]["activity"] == "Waiting for the model"
    assert {row["state"] for row in rows.values()} == {"held"}
    assert {row["icon"] for row in rows.values()} == {"held"}


def test_a_task_held_on_an_approval_says_so_and_a_slot_wait_keeps_its_words(
    client: TestClient, console: Console, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks(
        monkeypatch,
        console,
        Task("t-ask", "held", "Needs me", wait_reason="approval"),
        Task("t-input", "held", "Wants an answer", wait_reason="input"),
        Task("t-slot", "held", "No slot"),
        Task("t-late", "held", "Behind another", wait_reason="dependency"),
    )
    reasons = {"t-slot": "Resources", "t-late": "Priority"}
    monkeypatch.setattr(console.gate, "held", lambda task_id: reasons.get(task_id, ""))
    rows = {row["title"]: row for row in activity(client)["rows"]}
    assert rows["Needs me"]["activity"] == "Needs your approval"
    assert rows["Wants an answer"]["activity"] == "Needs your approval"
    assert rows["Needs me"]["icon"] == "wait"
    # The scheduler's holds keep the model's words.
    assert rows["No slot"]["activity"] == "Waiting for a free model slot"
    assert rows["Behind another"]["activity"] == "Waiting for a higher-priority task"
    assert {row["state"] for row in rows.values()} == {"held"}
    chinese = client.get("/api/activity", headers={"X-LocalMelo-Language": "zh-Hans"})
    words = {row["title"]: row["activity"] for row in chinese.json()["rows"]}
    assert words["Needs me"] == "需要你批准"
    assert words["No slot"] == "在等空闲的模型位置"


async def test_a_task_that_ran_out_of_steps_reads_stopped_and_can_continue(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    session = await console.store.begin_turn(
        None,
        "fix it",
        holder=HOLDER,
        title="",
        new_session=NewSession(kind="task", model="m", workspace=workspace),
    )
    await console.store.finish_turn(
        session.session_id,
        session.turn,
        holder=HOLDER,
        status="budget_exhausted",
        cause="run",
        result=run_result(),
        answer="Fixed the check. Remaining: run the tests and commit.",
    )
    tasks(
        monkeypatch,
        console,
        Task(
            session.session_id,
            "failed",
            "Fix the check",
            wait_reason=OUT_OF_STEPS,
            finished_at=now,
        ),
        Task("t-broke", "failed", "Broke", wait_reason="no disk", finished_at=now - 5),
        Task(
            "t-bare", "failed", "No turn", wait_reason=OUT_OF_STEPS, finished_at=now - 9
        ),
    )
    rows = {row["title"]: row for row in activity(client)["rows"]}
    stopped = rows["Fix the check"]
    assert (stopped["state"], stopped["icon"]) == ("stopped", "stopped")
    assert stopped["result"] == "Remaining: run the tests and commit."
    assert stopped["can_continue"] is True
    # A task with no turn left reads the plain words, never the stored reason.
    assert rows["No turn"]["state"] == "stopped"
    assert rows["No turn"]["result"] == "Ran out of steps"
    # A real failure stays failed, with its reason and no Continue.
    assert (rows["Broke"]["state"], rows["Broke"]["result"]) == ("failed", "no disk")
    assert "can_continue" not in rows["Broke"]


def test_a_task_code_found_not_finished_says_so_and_can_finish(
    monkeypatch: pytest.MonkeyPatch, console: Console, client: TestClient
) -> None:
    now = time.time()
    draft = {"state": "draft", "reasons": ["tests_missing"], "at": now}
    tasks(
        monkeypatch,
        console,
        Task("t-draft", "done", "Fix", finished_at=now, ending=draft),
        Task(
            "t-ready",
            "done",
            "Notes",
            finished_at=now - 1,
            ending={"state": "ready", "at": now - 1},
        ),
        Task(
            "t-old", "done", "Old", finished_at=now - 2, ending={**draft, "at": now - 9}
        ),
    )
    rows = {row["title"]: row for row in activity(client)["rows"]}
    assert rows["Fix"]["ending"] == draft and rows["Fix"]["can_continue"] is True
    assert rows["Notes"]["ending"]["state"] == "ready"
    assert "can_continue" not in rows["Notes"]
    # An ending decided at another transition is not this one's.
    assert rows["Old"]["ending"] is None


@pytest.mark.parametrize(
    ("answer", "left"),
    [
        (
            "Fixed the check. Done except for the final commit: `git add a && git commit`.",
            "Done except for the final commit: git add a && git commit.",
        ),
        (
            "All changed.\n- Still need to run the tests.\n- Pushed.",
            "Still need to run the tests.",
        ),
        ("代码已改好。还需要运行测试并提交。", "还需要运行测试并提交。"),
        ("Fixed the check and committed abc1234.", ""),
        ("```\nremaining = 3\n```", ""),
        ("", ""),
    ],
)
def test_a_stop_answer_says_what_is_left(answer: str, left: str) -> None:
    assert left_line(answer) == left


@pytest.mark.parametrize(
    ("answer", "bare"),
    [
        ("Fixed it. Remaining: run the tests.", "run the tests."),
        ("Fixed it. REMAINING：run the tests.", "run the tests."),
        ("Fixed it. What's left: run the tests.", "run the tests."),
        ("Fixed it. What is left: run the tests.", "run the tests."),
        ("Fixed it. Left: run the tests.", "run the tests."),
        ("代码已改好。剩下：运行测试。", "运行测试。"),
        ("代码已改好。还剩：运行测试。", "运行测试。"),
        ("代码已改好。还剩下: 运行测试。", "运行测试。"),
        # No label: the sentence stays as it is.
        ("Fixed it. Still need to run the tests.", "Still need to run the tests."),
        (
            "Fixed it. Done except for the final commit.",
            "Done except for the final commit.",
        ),
        # A label alone has nothing after it: the next sentence that says it is used.
        ("Remaining:\nStill need to run the tests.", "Still need to run the tests."),
    ],
)
def test_the_task_page_shows_what_is_left_without_the_answers_own_label(
    answer: str, bare: str
) -> None:
    assert left_line(answer, bare=True) == bare


def test_a_card_and_a_row_keep_the_sentence_as_the_answer_wrote_it() -> None:
    assert (
        left_line("Fixed it. Remaining: run the tests.") == "Remaining: run the tests."
    )
    assert left_line("代码已改好。剩下：运行测试。") == "剩下：运行测试。"


def test_a_long_sentence_of_what_is_left_is_cut_at_a_word() -> None:
    got = left_line("Remaining: " + "word " * 60)
    assert got.startswith("Remaining: word") and got.endswith("…")
    assert len(got) <= 121


@pytest.mark.parametrize("broken", ["day_turns", "tasks_since", "worker_day_turns"])
async def test_an_unreadable_store_counts_as_empty(
    client: TestClient,
    console: Console,
    workspace: str,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broken: str,
) -> None:
    await seed_turn(console.store, workspace, title="A chat")
    await cli_turn(cli_path, "worker", "A run", answer="ok")
    tasks(monkeypatch, console, Task("t1", "queued", "A task"))

    async def fail(*_: Any, **__: Any) -> Any:
        raise SessionStoreError("cannot read")

    target, name, gone = {
        "day_turns": (console.store, "day_turns", ("chats", "A chat")),
        "tasks_since": (console.workers.tasks, "tasks_since", ("tasks", "A task")),
        "worker_day_turns": (console.worker_history, "day_turns", ("runs", "A run")),
    }[broken]
    monkeypatch.setattr(target, name, fail)
    body = activity(client)
    assert body["unavailable"] == [gone[0]]
    titles = {row["title"] for row in body["rows"]}
    assert titles == {"A chat", "A task", "A run"} - {gone[1]}


async def test_a_missing_cli_file_is_an_empty_source(
    client: TestClient,
    console: Console,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks(monkeypatch, console)
    assert activity(client) == {
        "since": activity(client)["since"],
        "scope": "all",
        "rows": [],
        "more": False,
        "unavailable": [],
    }
    assert not cli_path.exists()


async def test_a_terminal_chat_is_redacted(
    client: TestClient,
    console: Console,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks(monkeypatch, console)
    key = "sk-" + "a1b2c3d4e5" * 3
    await cli_turn(cli_path, "chat", f"use {key}", answer=f"Set {key} as the key.")
    row = activity(client)["rows"][0]
    assert key not in row["title"] and REDACTED in row["title"]
    assert key not in row["result"]


def test_since_is_today_by_default_a_past_time_and_at_most_a_week(
    client: TestClient, console: Console, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks(monkeypatch, console)
    now = time.time()
    assert activity(client)["since"] == pytest.approx(local_midnight(now), abs=2)
    assert activity(client, since=0)["since"] == pytest.approx(now - MAX_SPAN_S, abs=5)
    future = client.get("/api/activity", params={"since": now + 3600})
    assert future.status_code == 400
    assert future.json()["error"]["code"] == "invalid_request"
    assert client.get("/api/activity", params={"since": "nan"}).status_code == 400


@pytest.mark.parametrize(
    "note, answer, line",
    [
        ("", "Two new roles. Both match.", "Two new roles."),
        ("Saved jobs.md", "Two new roles. Both match.", "Saved jobs.md"),
        (
            "",
            "<think>why</think>## Results\n- Latency fell 38%.",
            "Results - Latency fell 38%.",
        ),
        ("", "找到了两个职位。都在湾区。", "找到了两个职位。"),
        ("", "", ""),
    ],
)
def test_the_result_is_one_line(note: str, answer: str, line: str) -> None:
    assert result_line(note, answer) == line


@pytest.mark.parametrize(
    "answer, line",
    [
        # A "Done." that only says it is done gives way to the line after it.
        ("Done.\nFixed the parser.\nTests pass.", "Fixed the parser."),
        ("Done. Fixed the parser. Tests pass.", "Fixed the parser."),
        ("已完成。修改了规则。测试通过。", "修改了规则。"),
        # Nothing else follows, so the row says what the answer says.
        ("Done.", "Done."),
        ("All done!\n\n", "All done!"),
        # Only an opening ack goes; the word inside a sentence stays.
        ("Done by Friday for the report.", "Done by Friday for the report."),
    ],
)
def test_a_result_line_reads_past_a_leading_done(answer: str, line: str) -> None:
    assert result_line("", answer) == line


def test_a_long_result_is_cut_at_a_word() -> None:
    found = result_line("", "word " * 60)
    assert len(found) <= 140 and found.endswith("word…")


@pytest.mark.parametrize(
    "answer, line",
    [
        ("[interrupt: no assistant output for this turn]", "Stopped before answering"),
        ("[no answer: the run ended with status failed]", "Ended without an answer"),
        ("[answer not kept in working memory: too long]", ""),
        ("图已生成：/Users/me/out/dog_1.png", "Made a picture"),
        ("image generated: /tmp/work/chart.png", "Made a picture"),
        ("/Users/me/out/dog.webp", "Made a picture"),
        ("/Users/me/out/report.pdf", "report.pdf"),
        ("~/Documents/notes.md", "notes.md"),
        (
            "Saved the report to /Users/me/out/report.pdf. Check it.",
            "Saved the report to report.pdf.",
        ),
        ("```python\nimport os\nprint(os.getcwd())\n```", "Wrote some code"),
        ("import os\nimport sys\n\ndef main():\n    print(1)\n", "Wrote some code"),
        (
            "Here is the script. It prints the folder.\n```python\nimport os\n```",
            "Here is the script.",
        ),
        (
            "Use /memory to see notes, and/or on 10/30.",
            "Use /memory to see notes, and/or on 10/30.",
        ),
        (
            "[docs](https://x.org/a/b) has the details.",
            "[docs](https://x.org/a/b) has the details.",
        ),
    ],
)
def test_a_result_line_never_shows_a_marker_a_path_or_code(
    answer: str, line: str
) -> None:
    assert result_line("", answer) == line


def test_a_digest_that_is_a_path_reads_as_its_name() -> None:
    assert result_line("/Users/me/out/slides.pdf", "ignored") == "slides.pdf"


@pytest.mark.parametrize(
    ("answer", "lines"),
    [
        # What changed, the tests and the commit: the first two lines, markup gone.
        (
            "Done.\n\nChanged `front_rules.py` and its tests.\nTests: 12 passed.\n"
            "Committed abc1234.",
            "Changed front_rules.py and its tests.\nTests: 12 passed.",
        ),
        (
            "- **Changed** the deadline rule\n- Tests: 12 passed\n- Commit abc1234",
            "Changed the deadline rule\nTests: 12 passed",
        ),
        # One paragraph gives its first two sentences, a lead "Done." left out.
        (
            "Done. Changed the rule. All 12 tests pass. Committed abc1234.",
            "Changed the rule.\nAll 12 tests pass.",
        ),
        (
            "已完成。修改了 front_rules.py。测试通过。",
            "修改了 front_rules.py。\n测试通过。",
        ),
        # A single line stays one line.
        ("Committed abc1234 on dogfood/1030.", "Committed abc1234 on dogfood/1030."),
        # An answer that only says it is done says so.
        ("Done.", "Done."),
        ("All done!\n", "All done!"),
        # Code is not shown; a marker reads as what happened.
        ("```py\nprint(1)\n```", "Wrote some code"),
        ("[interrupt: stopped by the user]", "Stopped before answering"),
        ("", ""),
        # A path reads as its file's name.
        (
            "Edited /Users/me/work/test_e2e_pty.py\nTests: 3 passed",
            "Edited test_e2e_pty.py\nTests: 3 passed",
        ),
    ],
)
def test_a_done_card_carries_the_first_lines_of_the_answer(
    answer: str, lines: str
) -> None:
    assert answer_lines(answer) == lines


def test_a_long_line_of_an_answer_is_cut_at_a_word() -> None:
    first, second = answer_lines("word " * 60 + "\nsecond line").split("\n")
    assert first.endswith("word…") and len(first) <= 120
    assert second == "second line"


# ── GET /api/activity: the open chat, and approvals ──


async def front_chat(console: Console, workspace: str, **new: Any) -> str:
    """A finished chat of the given session kind or meta; kind ``front`` is the console's main chat."""
    if new.get("kind") == "front":
        # The console made its one main chat when it opened the store.
        assert console.main_id is not None
        turn = await console.store.begin_turn(console.main_id, "hello", holder=HOLDER)
    else:
        turn = await console.store.begin_turn(
            None,
            "hello",
            holder=HOLDER,
            title="Main chat",
            new_session=NewSession(model="m", workspace=workspace, **new),
        )
    await console.store.finish_turn(
        turn.session_id,
        turn.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=run_result(),
        answer="Hi.",
    )
    return turn.session_id


async def test_a_side_chat_lists_its_own_row_and_its_tasks_and_the_main_chat_the_day(
    client: TestClient,
    console: Console,
    workspace: str,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mine = await seed_turn(console.store, workspace, title="Slides chat")
    other = await seed_turn(console.store, workspace, title="Taxes chat")
    main = await front_chat(console, workspace, kind="front")
    tasks(
        monkeypatch,
        console,
        Task("t-mine", "queued", "Slides worker", front_id=mine.session_id),
        Task("t-other", "queued", "Taxes worker", front_id=other.session_id),
        Task("t-free", "queued", "Loose task"),
    )
    await cli_turn(cli_path, "worker", "Direct run", answer="ok")
    await cli_turn(cli_path, "chat", "Terminal talk", answer="ok")
    day = {"Slides chat", "Taxes chat", "Main chat", "Slides worker", "Taxes worker"}
    day |= {"Loose task", "Direct run", "Terminal talk"}

    side = activity(client, session_id=mine.session_id)
    assert side["scope"] == "chat"
    assert {row["title"] for row in side["rows"]} == {"Slides chat", "Slides worker"}
    for scope in (
        activity(client, session_id=main),
        activity(client),
        activity(client, session_id=""),
    ):
        assert scope["scope"] == "all"
        assert {row["title"] for row in scope["rows"]} == day
    assert [row["kind"] for row in activity(client, session_id=main)["rows"]].count(
        "chat"
    ) == 3, "the main chat's own turns are rows too"
    # An unknown id is a side chat with nothing of its own; so is a task's session.
    assert activity(client, session_id="gone")["rows"] == []
    assert [r["title"] for r in activity(client, session_id="t-other")["rows"]] == [
        "Taxes worker"
    ]


async def test_a_session_marked_main_lists_the_day_too(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await seed_turn(console.store, workspace, title="Side")
    main = await front_chat(console, workspace, meta={"main": True})
    tasks(monkeypatch, console)
    body = activity(client, session_id=main)
    assert body["scope"] == "all" and len(body["rows"]) == 2


def approval(
    type_: str, request_id: str, ts: float, tool: str = "file_write", **data: Any
) -> dict[str, Any]:
    """An ``approval.*`` session event as the runner stores it."""
    if type_ == "approval.requested":
        body: dict[str, Any] = {
            "request_id": request_id,
            "call_id": "call_1",
            "tool_name": tool,
            "arguments": {"path": "/w/notes/slides.md", "content": "secret text"},
            "preview": "slides.md",
            "workspace_root": "/w",
        }
    else:
        body = {
            "request_id": request_id,
            "call_id": "call_1",
            "tool_name": tool,
            "outcome": "allowed",
            "origin": "user",
            "reason": "",
            "remember": False,
        }
    body.update(data)
    return {"v": 1, "type": type_, "run_id": "", "seq": None, "ts": ts, "data": body}


async def asked(console: Console, session_id: str, turn: int, *events: Any) -> None:
    await console.store.append_events(session_id, turn, list(events), holder=HOLDER)


async def test_a_row_lists_its_approvals_in_plain_words_with_their_answers(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    quiet = await seed_turn(console.store, workspace, title="Quiet")
    busy = await seed_turn(console.store, workspace, title="Busy", status="running")
    asks: list[tuple[str, str, dict[str, Any]]] = [
        ("a", "file_write", {"outcome": "allowed", "origin": "user"}),
        ("b", "shell_exec", {"outcome": "allowed", "origin": "user", "remember": True}),
        ("c", "python_exec", {"outcome": "denied", "origin": "user", "reason": "no"}),
        ("d", "file_write", {"outcome": "allowed", "origin": "session"}),
        ("e", "shell_exec", {"outcome": "denied", "origin": "timeout"}),
        ("f", "file_write", {"outcome": "withdrawn", "origin": "withdrawn"}),
    ]
    for n, (request, tool, answer) in enumerate(asks):
        await asked(
            console,
            busy.session_id,
            busy.turn,
            approval("approval.requested", request, now - 100 + n, tool),
            approval("approval.resolved", request, now - 90 + n, tool, **answer),
        )
    await asked(
        console,
        busy.session_id,
        busy.turn,
        approval(
            "approval.requested",
            "w",
            now - 5,
            "shell_exec",
            arguments={"command": "rm -rf ~/x --token sk-aaaa"},
        ),
        # An answer whose request is not in the day adds nothing.
        approval("approval.resolved", "orphan", now - 4),
    )
    tasks(monkeypatch, console)
    rows = {row["title"]: row for row in activity(client)["rows"]}
    assert rows["Quiet"]["approvals"] == []
    listed = [(item["what"], item["answer"]) for item in rows["Busy"]["approvals"]]
    assert listed == [
        ("Write slides.md", "allowed once"),
        ("Run a command", "allowed for this chat"),
        ("Run Python", "denied"),
        ("Write slides.md", "allowed for this chat"),
        ("Run a command", "no answer"),
        ("Write slides.md", "withdrawn"),
        ("Run a command", "waiting"),
    ]
    assert [item["id"] for item in rows["Busy"]["approvals"]][:2] == ["a", "b"]
    assert rows["Busy"]["approvals"][0]["at"] == pytest.approx(now - 100, abs=1)
    text = str(rows["Busy"]["approvals"])
    assert "secret text" not in text and "rm -rf" not in text and "sk-aaaa" not in text
    assert quiet.session_id != busy.session_id


async def test_only_the_approvals_of_the_day_are_listed(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    turn = await seed_turn(console.store, workspace, title="Chat", status="running")
    await asked(
        console,
        turn.session_id,
        turn.turn,
        approval("approval.requested", "old", now - 7200),
        approval("approval.resolved", "old", now - 7190),
        approval("approval.requested", "new", now - 60),
    )
    tasks(monkeypatch, console)
    row = activity(client, since=now - 3600)["rows"][0]
    assert [item["id"] for item in row["approvals"]] == ["new"]


async def test_a_task_a_worker_run_and_a_terminal_chat_list_their_approvals_too(
    client: TestClient,
    console: Console,
    workspace: str,
    cli_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    task = await console.store.begin_turn(
        None,
        "slides",
        holder=HOLDER,
        title="Slides",
        new_session=NewSession(kind="task", model="m", workspace=workspace),
    )
    await asked(
        console,
        task.session_id,
        task.turn,
        approval("approval.requested", "t1", now - 30),
    )
    key = "sk-" + "a1b2c3d4e5" * 3
    await cli_turn(
        cli_path,
        "worker",
        "Check the build",
        answer="ok",
        events=[
            approval("approval.requested", "r1", now - 20, "shell_exec"),
            approval("approval.resolved", "r1", now - 19, "shell_exec"),
        ],
    )
    await cli_turn(
        cli_path,
        "chat",
        "Terminal talk",
        answer="ok",
        events=[
            approval(
                "approval.requested",
                "c1",
                now - 10,
                arguments={"path": f"/w/{key}.txt", "content": "x"},
            )
        ],
    )
    tasks(monkeypatch, console, Task(task.session_id, "running", "Slides"))
    body = activity(client)
    rows = {row["title"]: row for row in body["rows"]}
    assert body["unavailable"] == []
    assert [(i["id"], i["answer"]) for i in rows["Slides"]["approvals"]] == [
        ("t1", "waiting")
    ]
    assert [(i["what"], i["answer"]) for i in rows["Check the build"]["approvals"]] == [
        ("Run a command", "allowed once")
    ]
    terminal = rows["Terminal talk"]["approvals"]
    # A turn that ended with the request open withdrew it.
    assert [i["answer"] for i in terminal] == ["withdrawn"]
    assert key not in str(terminal)


async def test_an_unreadable_approval_source_is_named_and_the_rows_stay(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await seed_turn(console.store, workspace, title="A chat")
    tasks(monkeypatch, console)

    async def fail(*_: Any, **__: Any) -> Any:
        raise SessionStoreError("cannot read")

    monkeypatch.setattr(console.store, "approval_events", fail)
    body = activity(client)
    assert body["unavailable"] == ["approvals"]
    assert [(row["title"], row["approvals"]) for row in body["rows"]] == [
        ("A chat", [])
    ]


def requested(session: str, request: str, ts: float, **data: Any) -> StoredEvent:
    event = approval("approval.requested", request, ts, **data)
    return StoredEvent(1, session, 1, event)


def test_approvals_of_reads_the_events_oldest_first_and_keeps_the_newest_twenty() -> (
    None
):
    events = [requested("s", f"r{n}", float(n)) for n in range(25)]
    found = approvals_of(events)
    assert [item["id"] for item in found["s"]] == [f"r{n}" for n in range(5, 25)]
    assert approvals_of([]) == {}
    odd = approval("approval.requested", "x", 0, arguments="not a mapping")
    odd["ts"] = "late"
    found = approvals_of([StoredEvent(2, "t", 1, odd)])
    assert found["t"][0]["at"] == 0.0 and found["t"][0]["what"] == "Write a file"
