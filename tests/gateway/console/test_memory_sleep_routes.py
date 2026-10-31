"""The developer routes of sleep: ``POST /api/memory/sleep`` and its undo.

They answer 404 outside developer mode (``[ui] mode = "developer"``) unless the
console was given ``dev_routes``; in developer mode the first runs the sleep
the setting or the body names and answers its report, the second puts that run
back. Over the console's app with a scripted model that never keeps a note, as
``test_memory_routes`` does.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import httpx

from localmelo.support.gateway.console import Console

from .test_memory_routes import ScriptedLLM, say, setup  # noqa: F401  (fixture)

Setup = tuple[Console, httpx.AsyncClient, list[ScriptedLLM]]


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


async def test_the_setting_runs_when_no_variant_is_named_and_bad_ones_are_refused(
    setup: Setup, tmp_path: Path  # noqa: F811
) -> None:
    console, client, _ = setup
    console.cfg.ui.mode = "developer"
    console.cfg.memory.sleep = "rules"
    assert (await client.post("/api/memory/sleep")).json()["variant"] == "rules"
    built = await client.post("/api/memory/sleep", json={"variant": "model"})
    assert (built.status_code, built.json()["error"]["code"]) == (501, "not_built")
    assert "not built yet" in built.json()["error"]["message"]
    for body in ({"variant": "deep"}, {"variant": 3}, {"x": 1}):
        refused = await client.post("/api/memory/sleep", json=body)
        assert (refused.status_code, refused.json()["error"]["code"]) == (
            400,
            "invalid_request",
        )
