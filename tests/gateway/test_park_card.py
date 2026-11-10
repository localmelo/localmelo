"""The card a parked or cut-off task goes on with: where its work stands, from stored events."""

from typing import Any

from localmelo.support.gateway.park_card import (
    CARD_START,
    MAX_CARD_CHARS,
    build_park_card,
)
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.sessions.store import StoredEvent


def event(
    kind: str, data: Any, *, id: int | None = 1, run_id: str = "run"
) -> StoredEvent:
    return StoredEvent(id, "session", 1, {"type": kind, "run_id": run_id, "data": data})


def test_the_card_is_where_the_work_stands_and_claims_no_more_than_the_record() -> None:
    spec = TaskSpec(title="Example", objective="Update the file")
    notes = {
        "state": "parts 1-40 read",
        "failed": ["a.md is locked"],
        "works": ["write b.md"],
    }
    events = [
        event("notice", {"kind": "checkpoint", "detail": {"checkpoint": notes}}),
        event("tool.started", {"name": "file_write", "call_id": "1",
                               "arguments": {"path": "b.md", "content": "do not copy"}}),
        event("tool.finished", {"name": "file_write", "call_id": "1",
                                "status": "error", "output": "do not copy"}),
        event("assistant.message", {"content": "A draft exists"}),
        event("tool.started", {"name": "file_read", "call_id": "2",
                               "arguments": {"path": "c.md"}}),
        event("assistant.message", {"content": "Unstored"}, id=None),
    ]  # fmt: skip
    card = build_park_card(spec, events)
    assert card == build_park_card(spec, events)
    assert card.startswith(CARD_START) and "side effects" in card
    for text in (
        "Goal: Update the file",
        # The model's newest words after the last checkpoint take its state.
        "State: A draft exists",
        "Did not work: a.md is locked",
        "Works: write b.md",
        "- file_write b.md (failed)",
        "- file_read c.md (effect unknown)",
    ):
        assert text in card
    assert "do not copy" not in card and "Unstored" not in card
    assert "parts 1-40 read" not in card


def test_the_card_fits_the_store_and_reads_no_malformed_evidence() -> None:
    huge = "x" * 5000
    events = [
        event("tool.started", {"name": "file_read", "call_id": str(n),
                               "arguments": {"path": f"{n}-{huge}"}})
        for n in range(80)
    ] + [event("reflection", {"summary": huge}), event("tool.started", "not a dict")]  # fmt: skip
    card = build_park_card({"objective": huge}, events)
    assert len(card) <= MAX_CARD_CHARS and "side effects" in card
    assert build_park_card(None, None).startswith(CARD_START)  # type: ignore[arg-type]


def test_secrets_are_redacted_and_calls_are_scoped_to_run() -> None:
    events = [
        event("tool.started", {"name": "file_write", "call_id": "1",
                               "arguments": {"path": "API_KEY=supersecret"}},
              run_id="first"),
        event("tool.finished", {"name": "file_write", "call_id": "1",
                                "status": "success"}, run_id="second"),
    ]  # fmt: skip
    card = build_park_card({"objective": "API_KEY=supersecret"}, events)
    assert "supersecret" not in card and "(effect unknown)" in card
