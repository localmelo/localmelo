"""The developer routes of sleep: ``POST /api/memory/sleep`` and its undo.

They answer 404 outside developer mode (``[ui] mode = "developer"``) unless the
console was given ``dev_routes``; in developer mode the first runs the sleep
the setting or the body names and answers its report, the second puts that run
back. Over the console's app with a scripted model that never keeps a note, as
``test_memory_routes`` does.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.support.gateway.console import Console
from localmelo.support.memory.sleep import SleepReport
from localmelo.tests.sessions.helpers import answer

from .test_memory_routes import ScriptedLLM, say, setup  # noqa: F401  (fixture)

Setup = tuple[Console, httpx.AsyncClient, list[ScriptedLLM]]


async def test_budget_override_is_strict_bounded_and_developer_only(
    setup: Setup, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    console, client, _ = setup
    console.cfg.ui.mode = "developer"
    seen: list[int | None] = []

    async def run(variant: str | None = None, **kwargs: Any) -> SleepReport:
        seen.append(kwargs.get("budget"))
        return SleepReport("test", variant or "off", 0, 0)

    monkeypatch.setattr(console.memory, "sleep", run)
    for budget in (1, 100):
        assert (
            await client.post("/api/memory/sleep", json={"budget": budget})
        ).status_code == 200
        assert seen[-1] == budget
    for invalid in (0, 101, True, 1.5, "3"):
        assert (
            await client.post("/api/memory/sleep", json={"budget": invalid})
        ).status_code == 400
    assert len(seen) == 2
    console.cfg.ui.mode = "user"
    console.options = dataclasses.replace(console.options, dev_routes=True)
    assert (
        await client.post("/api/memory/sleep", json={"budget": 3})
    ).status_code == 200
    assert seen[-1] is None


async def test_the_routes_are_not_there_outside_developer_mode(
    setup: Setup,  # noqa: F811
) -> None:
    console, client, _ = setup
    for call in (
        client.post("/api/memory/sleep"),
        client.delete("/api/memory/sleep/x"),
    ):
        refused = await call
        assert refused.status_code == 404
        assert refused.json()["error"]["code"] == "not_found"
    # A test may open them without developer mode.
    console.options = dataclasses.replace(console.options, dev_routes=True)
    assert (await client.post("/api/memory/sleep")).status_code == 200


async def test_developer_mode_runs_a_sleep_and_undoes_it(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    console.cfg.ui.mode = "developer"
    await say(client, tmp_path, "I live in Utrecht.")
    await console.memory.drain()
    # The setting is off: nothing happens, and a body can name the variant.
    off = (await client.post("/api/memory/sleep")).json()
    assert (off["variant"], off["changes"], off["aborted"]) == (
        "off",
        [],
        "sleep is off",
    )
    ran = (await client.post("/api/memory/sleep", json={"variant": "rules"})).json()
    assert (ran["variant"], ran["model_calls"], ran["aborted"]) == ("rules", 0, "")
    (change,) = ran["changes"]
    assert (change["op"], change["text"]) == ("O3", "You live in Utrecht.")
    notes = (await client.get("/api/memory/records")).json()["records"]
    assert [n["text"] for n in notes] == ["You live in Utrecht."]

    undone = await client.delete(f"/api/memory/sleep/{ran['run_id']}")
    assert undone.status_code == 200
    assert (undone.json()["removed"], undone.json()["kept"]) == (1, 0)
    assert (await client.get("/api/memory/records")).json()["records"] == []
    again = await client.delete(f"/api/memory/sleep/{ran['run_id']}")
    assert (again.status_code, again.json()["error"]["code"]) == (404, "not_found")


async def test_model_variant_runs_checked_ops_and_undo_through_the_api(
    setup: Setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    console, client, _ = setup
    console.cfg.ui.mode = "developer"
    quote = "I'm allergic to peanuts."
    await say(client, tmp_path, quote)
    provider = ScriptedLLM(
        [
            answer(
                json.dumps(
                    [
                        {
                            "op": "ADD",
                            "note_id": None,
                            "text": "You are allergic to peanuts.",
                            "quote": quote,
                        },
                    ]
                )
            )
        ]
    )
    monkeypatch.setattr(console.memory, "chat_provider", lambda: provider)
    response = await client.post("/api/memory/sleep", json={"variant": "model+tags"})
    assert response.status_code == 200
    report = response.json()
    assert (report["variant"], report["model_calls"], report["refused"]) == (
        "model+tags",
        1,
        0,
    )
    assert report["ms"] >= 0 and report["changes"][0]["op"] == "ADD"
    assert report["selected"] == report["processed"]
    assert len(report["proposals"]) == 1
    label = report["proposals"][0]
    assert label["verdict"] == "applied" and label["quote"] == quote
    assert report["changes"][0]["proposal"] == label
    assert report["changes"][0]["verdict"] == "applied"
    assert (await client.get("/api/memory/records")).json()["records"][0][
        "text"
    ] == "You are allergic to peanuts."
    undone = await client.delete(f"/api/memory/sleep/{report['run_id']}")
    assert undone.status_code == 200 and undone.json()["removed"] == 1
    assert (await client.get("/api/memory/records")).json()["records"] == []


async def test_the_setting_runs_when_no_variant_is_named_and_bad_ones_are_refused(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    console.cfg.ui.mode = "developer"
    console.cfg.memory.sleep = "rules"
    assert (await client.post("/api/memory/sleep")).json()["variant"] == "rules"
    built = await client.post("/api/memory/sleep", json={"variant": "model"})
    assert built.status_code == 200
    assert (built.json()["variant"], built.json()["model_calls"]) == ("model", 0)
    for body in ({"variant": "deep"}, {"variant": 3}, {"x": 1}):
        refused = await client.post("/api/memory/sleep", json=body)
        assert (refused.status_code, refused.json()["error"]["code"]) == (
            400,
            "invalid_request",
        )
