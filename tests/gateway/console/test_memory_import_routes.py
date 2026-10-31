"""Bringing in other assistants' notes over HTTP: scan, write the chosen sources, undo.

The HOME is the synthetic one of the memory tests, and ``Path.home`` points at
it, so the routes (which scan the user's home) never read the real one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.support.gateway.console import Console
from localmelo.tests.memory.import_home import build

from .test_memory_routes import ScriptedLLM, setup  # noqa: F401  (fixture)

Setup = tuple[Console, httpx.AsyncClient, list[ScriptedLLM]]


@pytest.fixture(autouse=True)
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = build(tmp_path.resolve())
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    return home


async def test_scan_apply_list_and_undo(setup: Setup) -> None:  # noqa: F811
    _, client, _ = setup
    scan = (await client.get("/api/memory/import/scan")).json()
    assert set(scan) == {"batch_id", "sources", "partial"} and scan["partial"] is False
    assert {
        "id",
        "harness",
        "path",
        "scope",
        "workspace",
        "items_found",
        "items_kept",
        "filtered",
        "sample",
    } == set(scan["sources"][0])
    assert (await client.get("/api/memory/records")).json()[
        "records"
    ] == []  # a scan writes nothing

    gemini = next(s["id"] for s in scan["sources"] if s["harness"] == "gemini")
    done = await client.post(
        "/api/memory/import",
        json={"batch_id": scan["batch_id"], "source_ids": [gemini]},
    )
    body: dict[str, Any] = done.json()
    assert (
        done.status_code == 200
        and len(body["written_ids"]) == 3
        and body["skipped"] == 0
    )
    assert body["event"] == {
        "type": "memory.import",
        "data": {
            "batch_id": scan["batch_id"],
            "sources": [{"id": gemini, "harness": "gemini", "written": 3}],
        },
    }
    rows = (await client.get("/api/memory/records")).json()["records"]
    assert {r["imported"] for r in rows} == {True} and {
        r["batch_id"] for r in rows
    } == {scan["batch_id"]}
    assert rows[0]["source"].startswith("imported from the file GEMINI.md, ")

    gone = await client.delete(f"/api/memory/import/{scan['batch_id']}")
    assert gone.json() == {"batch_id": scan["batch_id"], "removed": 3}
    assert (await client.get("/api/memory/records")).json()["records"] == []
    assert (
        await client.delete(f"/api/memory/import/{scan['batch_id']}")
    ).status_code == 404


async def test_an_unknown_scan_or_source_is_refused(setup: Setup) -> None:  # noqa: F811
    _, client, _ = setup
    unknown = await client.post(
        "/api/memory/import", json={"batch_id": "nope", "source_ids": []}
    )
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "not_found"
    scan = (await client.get("/api/memory/import/scan")).json()
    bad = await client.post(
        "/api/memory/import",
        json={"batch_id": scan["batch_id"], "source_ids": ["claude-nope"]},
    )
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_request"
    extra = await client.post(
        "/api/memory/import",
        json={"batch_id": scan["batch_id"], "source_ids": [], "x": 1},
    )
    assert extra.status_code == 400
