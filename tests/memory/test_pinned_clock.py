"""``LOCALMELO_NOW``: a developer variable that pins "now" for two reads.

The model's ``Date:`` line and recall's age decay read it; unset, empty or
invalid, they read the wall clock (an invalid value logs one warning).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import pytest

from localmelo.melo import clock
from localmelo.melo.agent._chat import local_date_line
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.index import Document, document_text

MAY_20 = datetime(2023, 5, 20, 13, 40).timestamp()


@pytest.fixture(autouse=True)
def fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(clock.ENV, raising=False)
    monkeypatch.setattr(clock, "_warned", set())


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("2023-05-20T13:40:00", MAY_20),  # no zone: local time
        ("2023-05-20T13:40:00Z", 1684590000.0),
        ("2023-05-20T15:40:00+02:00", 1684590000.0),
        ("2023-05-20", datetime(2023, 5, 20).timestamp()),
        ("1684590000", 1684590000.0),
        (" 1684590000.5 ", 1684590000.5),
        (1684590000, 1684590000.0),
    ],
)
def test_a_time_is_iso_8601_or_epoch_seconds(value: object, seconds: float) -> None:
    assert clock.parse(value) == seconds


@pytest.mark.parametrize(
    "value",
    ["", "  ", "yesterday", "2023-13-45", "nan", "inf", True, None, [], "9" * 400],
)
def test_anything_else_is_not_a_time(value: object) -> None:
    assert clock.parse(value) is None


def test_the_date_line_reads_the_pinned_day(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(clock.ENV, "2023-05-20T13:40:00")
    assert local_date_line().startswith("Date: 2023-05-20 (UTC")
    monkeypatch.setenv(clock.ENV, str(datetime(2024, 2, 29, 12).timestamp()))
    assert local_date_line().startswith("Date: 2024-02-29 (UTC")
    assert abs(clock.now() - datetime(2024, 2, 29, 12).timestamp()) < 1e-6


def test_unset_and_empty_read_the_wall_clock_without_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    assert abs(clock.now() - time.time()) < 5
    monkeypatch.setenv(clock.ENV, "")
    assert clock.pinned() is None
    assert local_date_line().startswith(f"Date: {datetime.now():%Y-%m-%d} (UTC")
    assert caplog.records == []


def test_an_invalid_value_falls_back_with_one_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv(clock.ENV, "next tuesday")
    for _ in range(3):
        assert abs(clock.now() - time.time()) < 5
        assert local_date_line().startswith(f"Date: {datetime.now():%Y-%m-%d} (UTC")
    warnings = [r for r in caplog.records if r.name == clock.__name__]
    assert len(warnings) == 1
    assert "LOCALMELO_NOW" in warnings[0].getMessage()
    assert "'next tuesday'" in warnings[0].getMessage()


async def _ranked_first(tmp_path: Path) -> str:
    """The first of two turns that match a question: one from the pinned day, one new.

    The stronger match is from 2023-05-20 12:00; the weaker was written a moment
    ago. Read on 2023-05-20 13:40 the old one has hardly aged; read now it is
    three years old and the new one comes first.
    """
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        _, index = memory.stores()
        made = {
            "old": (
                "Notes about the quarterly budget review",
                datetime(2023, 5, 20, 12).timestamp(),
            ),
            "new": ("Budget review", time.time() - 60),
        }
        for turn, (typed, at) in enumerate(made.values(), 1):
            await index.add(
                Document(
                    session_id=f"s{turn}",
                    turn=1,
                    surface="gateway",
                    kind="chat",
                    title="",
                    text=document_text(typed, "Noted."),
                    created_at=at,
                )
            )
        found = await memory.search("quarterly budget review", records_k=0, docs_k=2)
        assert len(found) == 2
        return "old" if "quarterly" in found[0].text else "new"
    finally:
        await memory.aclose()


async def test_recall_ages_turns_from_the_pinned_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert await _ranked_first(tmp_path / "wall") == "new"
    monkeypatch.setenv(clock.ENV, "2023-05-20T13:40:00")
    assert await _ranked_first(tmp_path / "pinned") == "old"
