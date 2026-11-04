"""``SessionStore.read_notes``: the words a run read from the person, by turn."""

from __future__ import annotations

from pathlib import Path

from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo

from .helpers import core_event

HOLDER = "testhost:1:notes"


async def _run(
    store: SessionStore, session_id: str | None, notes: list[str]
) -> TurnInfo:
    """A finished turn whose run read *notes*, one ``inbox.accepted`` each."""
    turn = await store.begin_turn(
        session_id,
        "start",
        holder=HOLDER,
        new_session=NewSession() if session_id is None else None,
    )
    events = [
        core_event("step.started", seq=1),
        *[
            core_event(
                "inbox.accepted",
                seq=number + 1,
                offset=number,
                text=text,
                step=number + 1,
            )
            for number, text in enumerate(notes, 1)
        ],
        core_event("step.started", seq=len(notes) + 2, step=len(notes) + 2),
    ]
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


async def test_read_notes_are_the_accepted_texts_of_that_turn_in_order(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    first = await _run(store, None, ["one", "two", "three"])
    second = await _run(store, first.session_id, ["only"])
    other = await _run(store, None, ["elsewhere"])
    assert await store.read_notes(first.session_id, first.turn) == [
        "one",
        "two",
        "three",
    ]
    assert await store.read_notes(second.session_id, second.turn) == ["only"]
    assert await store.read_notes(other.session_id, other.turn) == ["elsewhere"]
    assert await store.read_notes(first.session_id, 99) == []
    assert await store.read_notes("no-such-session", 1) == []


async def test_a_turn_without_notes_and_empty_texts_give_nothing(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    plain = await _run(store, None, [])
    assert await store.read_notes(plain.session_id, plain.turn) == []
    blank = await _run(store, plain.session_id, ["", "  \n", "kept"])
    assert await store.read_notes(blank.session_id, blank.turn) == ["kept"]


async def test_only_inbox_accepted_events_count_as_notes(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    turn = await store.begin_turn(
        None, "start", holder=HOLDER, new_session=NewSession()
    )
    queued = {
        "v": 1,
        "type": "turn.note",
        "run_id": "",
        "seq": None,
        "ts": 1.0,
        "data": {
            "turn": turn.turn,
            "turn_id": turn.id,
            "note": 1,
            "text": "queued only",
        },
    }
    await store.append_events(
        turn.session_id,
        turn.turn,
        [queued, core_event("inbox.accepted", seq=1, text="read")],
        holder=HOLDER,
    )
    assert await store.read_notes(turn.session_id, turn.turn) == ["read"]
