"""Evidence and safety properties of deterministic resumption cards."""

from typing import Any

from localmelo.support.gateway.park_card import MAX_CARD_CHARS, build_park_card
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.sessions.store import StoredEvent


def event(
    kind: str, data: Any, *, id: int | None = 1, run_id: str = "run"
) -> StoredEvent:
    return StoredEvent(id, "session", 1, {"type": kind, "run_id": run_id, "data": data})


def test_card_uses_committed_evidence_without_inventing_completion() -> None:
    spec = TaskSpec(title="Example", objective="Update the file")
    events = [
        event(
            "tool.started",
            {
                "name": "file_write",
                "call_id": "1",
                "arguments": {"path": "a.txt", "content": "do not copy"},
            },
        ),
        event(
            "tool.finished",
            {
                "name": "file_write",
                "call_id": "1",
                "status": "success",
                "exit_code": 0,
                "output": "do not copy",
            },
        ),
        event("reflection", {"summary": "Review the output"}),
        event("assistant.message", {"content": "A draft exists"}),
        event("tool.started", {"name": "other", "call_id": "2"}),
        event("assistant.message", {"content": "Unstored"}, id=None),
    ]
    card = build_park_card(spec, events)
    assert card == build_park_card(spec, events)
    for text in (
        "Update the file",
        "a.txt",
        "success",
        "exit code 0",
        "Review the output",
        "A draft exists",
        "without a recorded finish",
        "completion unverified",
        "side effects",
    ):
        assert text in card
    assert "do not copy" not in card and "Unstored" not in card


def test_latest_nonempty_reply_survives_empty_tool_only_messages() -> None:
    events = [
        event("assistant.message", {"content": "Earlier reply"}),
        event("assistant.message", {"content": "Latest reply"}),
        event(
            "assistant.message", {"content": "", "tool_calls": [{"name": "file_read"}]}
        ),
        event("assistant.message", {"content": "  \n "}),
        event("assistant.message", {"content": None}),
    ]
    card = build_park_card({}, events)
    assert "Last reply (reported): Latest reply" in card
    assert "Earlier reply" not in card


def test_a_tool_call_written_as_text_is_not_the_last_reply() -> None:
    events = [
        event("assistant.message", {"content": "Saved the note."}),
        event("assistant.message", {"content": "<tool_call>\n<function=remember>"}),
        event("assistant.message", {"content": "<|python_tag|>{}"}),
    ]
    card = build_park_card({}, events)
    assert "Last reply (reported): Saved the note." in card
    assert "tool_call" not in card and "python_tag" not in card


def test_malformed_inputs_and_missing_evidence_are_unknown() -> None:
    card = build_park_card(
        {"objective": None, "done_when": ["First", 3, "Second"]},
        [
            None,  # type: ignore[list-item]
            event("tool.finished", {"name": [], "status": {}, "exit_code": True}),
            event("reflection", None),
        ],
    )
    assert "Unknown" in card and "unknown tool: unknown" in card
    assert "First; Second (completion unverified)" in card
    assert "exit code" not in card
    assert "If resuming" in card and "before deciding" in card
    assert "No recorded tool activity" in build_park_card(None, None)  # type: ignore[arg-type]


def test_length_and_cautions_survive_large_inputs() -> None:
    huge = "x" * 10000
    events = [
        event("tool.started", {"name": huge, "call_id": str(i), "arguments": {}})
        for i in range(10)
    ]
    events += [
        event("reflection", {"summary": huge}),
        event("assistant.message", {"content": huge}),
    ]
    card = build_park_card({"objective": huge, "done_when": [huge] * 20}, events)
    assert len(card) <= MAX_CARD_CHARS
    assert "Cautions:" in card and "side effects" in card
    assert "Next action:" in card


def test_secrets_are_redacted_and_calls_are_scoped_to_run() -> None:
    events = [
        event(
            "tool.started",
            {
                "name": "file_write",
                "call_id": "1",
                "arguments": {"path": "API_KEY=supersecret"},
            },
            run_id="first",
        ),
        event(
            "tool.finished",
            {"name": "file_write", "call_id": "1", "status": "success"},
            run_id="second",
        ),
    ]
    card = build_park_card({"objective": "API_KEY=supersecret"}, events)
    assert "supersecret" not in card
    assert "without a recorded finish" in card
