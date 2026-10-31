"""What a running session is doing, in the plain words every surface shares."""

from __future__ import annotations

import os
from typing import Any

import pytest

from localmelo.support.sessions.display import (
    ActivityText,
    activity_text,
    event_activity,
    request_words,
)


@pytest.mark.parametrize(
    "tool, arguments, words, icon",
    [
        ("file_read", {"path": "/w/talk/notes.md"}, "Reading notes.md", "read"),
        ("file_read", {}, "Reading a file", "read"),
        ("list_files", {}, "Looking through the folder", "read"),
        (
            "file_write",
            {"path": "slides.md", "content": "x"},
            "Writing slides.md",
            "write",
        ),
        ("shell_exec", {"command": "uv run pytest -q"}, "Running the tests", "run"),
        ("shell_exec", {"command": "npm test"}, "Running the tests", "run"),
        (
            "shell_exec",
            {"command": "cd app && cargo test --all"},
            "Running the tests",
            "run",
        ),
        ("shell_exec", {"command": "python -m unittest"}, "Running the tests", "run"),
        ("shell_exec", {"command": "cat pytest.ini"}, "Running a command", "run"),
        ("shell_exec", {"command": "rm -rf build"}, "Running a command", "run"),
        ("python_exec", {"code": "print(1)"}, "Running Python", "run"),
        (
            "browser_open",
            {"url": "https://www.example.com/a?b"},
            "Reading example.com",
            "web",
        ),
        ("browser_open", {"url": "not a url"}, "Opening a page", "web"),
        ("browser_read", {}, "Reading the page", "web"),
        ("browser_follow", {"ref": "e12"}, "Following a link", "web"),
        ("browser_screenshot", {}, "Looking at the page", "web"),
        ("recall", {"query": "x"}, "Remembering", "memory"),
        ("search_memory", {"query": "x"}, "Remembering", "memory"),
        (
            "remember",
            {"text": "You prefer tea.", "topic": "x"},
            "Remembering",
            "memory",
        ),
        ("mcp__files__file_read", {"path": "a/b.txt"}, "Reading b.txt", "read"),
        ("mcp__calendar__list_events", {}, "Using a tool", "tool"),
        ("", None, "Thinking", "think"),
    ],
)
def test_each_tool_reads_by_what_it_does(
    tool: str, arguments: Any, words: str, icon: str
) -> None:
    assert activity_text(tool, arguments) == ActivityText(words, icon)


def test_a_folder_inside_the_workspace_reads_relative_to_it() -> None:
    home = os.path.expanduser("~")
    found = activity_text("list_files", {"path": f"{home}/talk"})
    assert found.text == "Looking through ~/talk"
    found = activity_text("list_files", {"path": "/w/talk/src"}, root="/w/talk")
    assert found.text == "Looking through src"


@pytest.mark.parametrize(
    "state, words, icon",
    [
        ("waiting", "Waiting for your approval", "wait"),
        ("held", "Waiting for the model", "held"),
        ("queued", "Waiting to start", "held"),
        ("paused", "Paused", "paused"),
        ("reflecting", "Checking its work", "think"),
        ("remembering", "Remembering", "memory"),
        ("thinking", "Thinking", "think"),
    ],
)
def test_each_state_has_its_words_and_wins_over_a_tool(
    state: str, words: str, icon: str
) -> None:
    assert activity_text("shell_exec", {"command": "ls"}, state=state) == (
        ActivityText(words, icon)
    )


def test_no_command_line_code_or_text_is_ever_shown() -> None:
    secret = "sk-" + "a1b2c3" * 4
    for tool, arguments in [
        ("shell_exec", {"command": f"curl -H 'Authorization: {secret}' x"}),
        ("python_exec", {"code": f"key = '{secret}'"}),
        ("file_write", {"path": "a.txt", "content": secret}),
        ("browser_open", {"url": f"https://example.com/?token={secret}"}),
    ]:
        assert secret not in activity_text(tool, arguments).text


def event(kind: str, **data: Any) -> dict[str, Any]:
    return {"type": kind, "data": data}


def words(events: list[dict[str, Any]]) -> str | None:
    found = event_activity(events)
    return None if found is None else found.text


def test_the_newest_events_say_what_a_turn_does() -> None:
    started = event(
        "tool.started", call_id="c1", name="file_read", arguments={"path": "a.md"}
    )
    finished = event("tool.finished", call_id="c1", name="file_read")
    step = event("step.started", step=2)
    assert event_activity([started, step]) == ActivityText("Reading a.md", "read")
    # A call that ended is no longer what the turn does: the step after it is.
    assert words([finished, started, step]) == "Thinking"
    asked = event("approval.requested", request_id="r1")
    assert words([asked, started]) == "Waiting for your approval"
    answered = event("approval.resolved", request_id="r1")
    assert words([answered, asked, started]) == "Reading a.md"
    assert words([event("reflection.started"), finished]) == "Checking its work"
    assert words([event("recall")]) == "Remembering"
    assert words([event("turn.finished"), started]) is None
    assert event_activity([]) == ActivityText("Thinking", "think")


@pytest.mark.parametrize(
    "tool, arguments, words",
    [
        (
            "file_write",
            {"path": "/w/talk/slides.md", "content": "secret"},
            "Write slides.md",
        ),
        ("file_read", {}, "Read a file"),
        ("list_files", {"path": "/w"}, "Look through the folder"),
        ("shell_exec", {"command": "rm -rf ~/x --token abc"}, "Run a command"),
        ("shell_exec", {"command": "uv run pytest"}, "Run the tests"),
        ("python_exec", {"code": "print(1)"}, "Run Python"),
        ("browser_open", {"url": "https://www.example.com/a"}, "Read example.com"),
        ("browser_screenshot", {}, "Look at the page"),
        ("mcp__calendar__list_events", {}, "Use a tool"),
        ("", None, "Thinking"),
    ],
)
def test_an_approval_puts_the_call_as_a_request_without_its_command_or_text(
    tool: str, arguments: Any, words: str
) -> None:
    assert request_words(tool, arguments, root="/w") == words


@pytest.mark.parametrize(
    "reason, words",
    [
        ("Priority", "Waiting for a higher-priority task"),
        ("Resources", "Waiting for a free model slot"),
        ("Reservation", "Waiting while another job uses the model"),
        ("Memory", "Waiting for free memory"),
        ("Battery", "Waiting for the battery"),
        ("Thermal", "Waiting for the Mac to cool down"),
        ("IdleOnly", "Waiting until the Mac is idle"),
        ("Paused", "Waiting for the model"),
        ("", "Waiting for the model"),
    ],
)
def test_a_held_call_says_why_in_words(reason: str, words: str) -> None:
    assert activity_text(state="held", reason=reason) == ActivityText(words, "held")


def test_a_reason_changes_only_a_held_state() -> None:
    assert activity_text(state="waiting", reason="Priority") == ActivityText(
        "Waiting for your approval", "wait"
    )
