"""``SessionStore.turn_tool_results``: how a turn's tool calls ended, by turn."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo

from .helpers import core_event

HOLDER = "testhost:1:results"


def _finished(seq: int, **data: Any) -> dict[str, Any]:
    return core_event("tool.finished", seq=seq, call_id=f"call_{seq}", **data)


async def _run(
    store: SessionStore, session_id: str | None, events: list[dict[str, Any]]
) -> TurnInfo:
    turn = await store.begin_turn(
        session_id,
        "start",
        holder=HOLDER,
        new_session=NewSession() if session_id is None else None,
    )
    await store.append_events(turn.session_id, turn.turn, events, holder=HOLDER)
    await store.finish_turn(
        turn.session_id,
        turn.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer="done",
    )
    return turn


async def test_results_come_in_the_order_the_calls_finished(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    first = await _run(
        store,
        None,
        [
            core_event("step.started", seq=1),
            _finished(2, output="fine"),
            _finished(
                3,
                status="blocked",
                error_category="approval_denied",
                output="",
                error="The user denied this call.",
            ),
        ],
    )
    second = await _run(store, first.session_id, [_finished(1, output="other turn")])
    assert await store.turn_tool_results(first.session_id, first.turn) == [
        ("success", "none", "fine"),
        ("blocked", "approval_denied", "The user denied this call."),
    ]
    assert await store.turn_tool_results(second.session_id, second.turn) == [
        ("success", "none", "other turn")
    ]
    assert await store.turn_tool_results(first.session_id, 99) == []
    assert await store.turn_tool_results("no-such-session", 1) == []


async def test_only_the_last_results_can_be_asked_for(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    turn = await _run(
        store, None, [_finished(n, output=f"result {n}") for n in range(1, 5)]
    )
    assert await store.turn_tool_results(turn.session_id, turn.turn, last=1) == [
        ("success", "none", "result 4")
    ]
    assert [
        words
        for *_, words in await store.turn_tool_results(
            turn.session_id, turn.turn, last=2
        )
    ] == ["result 3", "result 4"]
    assert len(await store.turn_tool_results(turn.session_id, turn.turn)) == 4


async def test_the_text_is_the_error_else_the_output_and_only_its_head(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    turn = await _run(
        store,
        None,
        [
            _finished(
                1, status="error", error_category="tool_error", output="", error="boom"
            ),
            _finished(2, output="x" * 5_000),
            _finished(3, output="", error=""),
            core_event("run.started", seq=4),
        ],
    )
    results = await store.turn_tool_results(turn.session_id, turn.turn, head=50)
    assert results == [
        ("error", "tool_error", "boom"),
        ("success", "none", "x" * 50),
        ("success", "none", ""),
    ]
