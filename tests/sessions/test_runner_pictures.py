"""SessionRunner(pictures=True): the ``turn.pictures`` event a console turn stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.support.sessions import pictures
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore, StoredEvent

from .helpers import ScriptedLLM, answer, started_agent, tool_call

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels"
SVG = "<svg xmlns='http://www.w3.org/2000/svg'></svg>"


async def run_turn(
    tmp_path: Path,
    replies: list[Any],
    *,
    pictures_on: bool = True,
    before: dict[str, bytes] | None = None,
) -> tuple[list[StoredEvent], Path]:
    """One turn on a scripted model in a folder holding *before*; its stored events."""
    root = tmp_path / "ws"
    agent = await started_agent(root, ScriptedLLM(replies))
    for name, data in (before or {}).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    runner = SessionRunner(
        agent, store, None, new_session=NewSession(), pictures=pictures_on
    )
    try:
        result = await runner.run_turn("make a chart")
        assert result.status == "completed"
        assert runner.session_id is not None
        return await store.list_events(runner.session_id), root
    finally:
        await runner.close()
        await agent.close()
        await store.aclose()


def types_of(events: list[StoredEvent]) -> list[str]:
    return [event.event["type"] for event in events]


def listed(events: list[StoredEvent]) -> list[dict[str, Any]]:
    found = [e.event["data"] for e in events if e.event["type"] == "turn.pictures"]
    assert len(found) <= 1
    return found[0]["pictures"] if found else []


async def test_a_picture_the_turn_wrote_is_stored_after_the_run(tmp_path: Path) -> None:
    events, _ = await run_turn(
        tmp_path,
        [
            tool_call("file_write", {"path": "chart.svg", "content": SVG}),
            answer("Done."),
        ],
    )
    names = types_of(events)
    assert names.index("run.finished") < names.index("turn.pictures")
    assert names.index("turn.pictures") < names.index("turn.finished")
    found = listed(events)
    assert [(item["path"], item["state"]) for item in found] == [("chart.svg", "new")]
    assert found[0]["size"] == len(SVG) and found[0]["mtime"] > 0
    data = next(e.event["data"] for e in events if e.event["type"] == "turn.pictures")
    assert data["turn"] == 1 and data["turn_id"]


async def test_created_changed_and_untouched_pictures(tmp_path: Path) -> None:
    events, _ = await run_turn(
        tmp_path,
        [
            tool_call("file_write", {"path": "old.svg", "content": SVG + "<!-- 2 -->"}),
            answer("Updated."),
        ],
        before={"old.svg": SVG.encode(), "kept.png": PNG},
    )
    assert [(i["path"], i["state"]) for i in listed(events)] == [
        ("old.svg", "modified")
    ]


async def test_a_picture_path_in_the_answer_is_listed_once(tmp_path: Path) -> None:
    events, _ = await run_turn(
        tmp_path,
        [answer(f"The chart is `{tmp_path}/ws/shots/a.png` (also shots/a.png).")],
        before={"shots/a.png": PNG},
    )
    assert [(i["path"], i["state"]) for i in listed(events)] == [
        ("shots/a.png", "mentioned")
    ]


async def test_no_picture_means_no_event(tmp_path: Path) -> None:
    events, _ = await run_turn(tmp_path, [answer("Nothing drawn.")])
    assert "turn.pictures" not in types_of(events)


async def test_a_runner_without_pictures_stores_none(tmp_path: Path) -> None:
    events, _ = await run_turn(
        tmp_path,
        [tool_call("file_write", {"path": "c.svg", "content": SVG}), answer("Done.")],
        pictures_on=False,
    )
    assert "turn.pictures" not in types_of(events)


async def test_a_folder_that_cannot_be_scanned_never_fails_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(root: str) -> pictures.Snapshot:
        raise OSError("no")

    monkeypatch.setattr(pictures, "scan", broken)
    events, _ = await run_turn(tmp_path, [answer("Fine.")])
    assert "turn.pictures" not in types_of(events)
    assert types_of(events)[-1] == "turn.finished"
