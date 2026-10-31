"""Bringing in other assistants' notes over HTTP: scan, write the chosen sources, undo.

The HOME is the synthetic one of the memory tests, and ``Path.home`` points at
it, so the routes (which scan the user's home) never read the real one.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest

from localmelo.support import config
from localmelo.support.gateway.console import Console
from localmelo.tests.memory.import_home import build

from .test_memory_routes import ScriptedLLM, setup  # noqa: F401  (fixture)

Setup = tuple[Console, httpx.AsyncClient, list[ScriptedLLM]]


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path) -> Iterator[Path]:
    """The config file the memory switch writes is the test's own."""
    path = tmp_path / "cfg" / "config.toml"
    with (
        mock.patch.object(config, "CONFIG_DIR", str(path.parent)),
        mock.patch.object(config, "CONFIG_PATH", str(path)),
    ):
        yield path


@pytest.fixture(autouse=True)
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = build(tmp_path.resolve())
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    return home


async def test_scan_apply_list_and_undo(
    setup: Setup,  # noqa: F811
    fake_home: Path,
) -> None:
    _, client, _ = setup
    scan = (await client.get("/api/memory/import/scan")).json()
    assert set(scan) == {"batch_id", "sources", "partial", "home"}
    assert scan["partial"] is False and scan["home"] == str(fake_home)
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


async def test_the_offer_is_open_until_it_is_answered_or_the_notes_are_brought_in(
    setup: Setup,  # noqa: F811
) -> None:
    console, client, _ = setup
    assert (await client.get("/api/memory/import/offer")).json() == {"pending": True}
    # A scan only looks: the offer stays open.
    scan = (await client.get("/api/memory/import/scan")).json()
    assert (await client.get("/api/memory/import/offer")).json() == {"pending": True}
    # "Not now" is the one answer, and it is not the undo of a batch called "offer".
    declined = await client.delete("/api/memory/import/offer")
    assert (declined.status_code, declined.json()) == (200, {"pending": False})
    assert (await client.get("/api/memory/import/offer")).json() == {"pending": False}
    assert (await client.get("/api/memory/records")).json()["records"] == []
    # Writing notes later still works and the offer stays answered.
    gemini = next(s["id"] for s in scan["sources"] if s["harness"] == "gemini")
    body = {"batch_id": scan["batch_id"], "source_ids": [gemini]}
    assert (await client.post("/api/memory/import", json=body)).status_code == 200
    assert (await client.get("/api/memory/import/offer")).json() == {"pending": False}
    assert console.memory.import_pending() is False


async def test_bringing_notes_in_by_hand_counts_as_the_offer_made(
    setup: Setup,  # noqa: F811
) -> None:
    _, client, _ = setup
    scan = (await client.get("/api/memory/import/scan")).json()
    gemini = next(s["id"] for s in scan["sources"] if s["harness"] == "gemini")
    body = {"batch_id": scan["batch_id"], "source_ids": [gemini]}
    assert (await client.post("/api/memory/import", json=body)).status_code == 200
    assert (await client.get("/api/memory/import/offer")).json() == {"pending": False}


async def test_no_offer_while_memory_is_off(setup: Setup) -> None:  # noqa: F811
    _, client, _ = setup
    assert (await client.put("/api/memory", json={"on": False})).status_code == 200
    assert (await client.get("/api/memory/import/offer")).json() == {"pending": False}
    assert (await client.put("/api/memory", json={"on": True})).status_code == 200
    assert (await client.get("/api/memory/import/offer")).json() == {"pending": True}


async def test_an_imported_note_says_which_file_it_came_from_in_the_readers_language(
    setup: Setup,  # noqa: F811
) -> None:
    _, client, _ = setup
    scan = (await client.get("/api/memory/import/scan")).json()
    gemini = next(s["id"] for s in scan["sources"] if s["harness"] == "gemini")
    body = {"batch_id": scan["batch_id"], "source_ids": [gemini]}
    await client.post("/api/memory/import", json=body)
    row = (await client.get("/api/memory/records")).json()["records"][0]
    assert row["from"].startswith("from the file GEMINI.md, ")
    assert row["source"].startswith("imported from the file GEMINI.md, ")
    zh = await client.get(
        "/api/memory/records", headers={"X-LocalMelo-Language": "zh-Hans"}
    )
    assert zh.json()["records"][0]["from"].startswith("来自文件 GEMINI.md，")


async def test_not_now_says_so_only_when_the_record_was_kept(
    setup: Setup,  # noqa: F811
) -> None:
    """A folder that cannot be written keeps the offer open, and the page is told."""
    if os.geteuid() == 0:
        pytest.skip("root writes anywhere")
    console, client, _ = setup
    folder = Path(console.memory.directory)
    folder.mkdir(parents=True, exist_ok=True)
    folder.chmod(0o500)
    try:
        refused = await client.delete("/api/memory/import/offer")
        assert (refused.status_code, refused.json()) == (200, {"pending": True})
        assert (await client.get("/api/memory/import/offer")).json() == {
            "pending": True
        }
    finally:
        folder.chmod(0o700)
    kept = await client.delete("/api/memory/import/offer")
    assert kept.json() == {"pending": False}


async def test_the_scan_names_the_home_as_the_paths_spell_it_even_when_it_is_a_link(
    setup: Setup,  # noqa: F811
    fake_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, client, _ = setup
    link = tmp_path / "link"
    link.symlink_to(fake_home)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: link))
    monkeypatch.setenv("HOME", str(link))
    scan = (await client.get("/api/memory/import/scan")).json()
    assert scan["home"] == str(fake_home) and scan["sources"]
    assert all(
        source["path"].startswith(scan["home"] + "/") for source in scan["sources"]
    )


async def test_undo_leaves_notes_the_person_reworded_and_says_how_many(
    setup: Setup,  # noqa: F811
) -> None:
    """`kept` is in the answer only when above 0; a batch with nothing left to take back is still 404."""
    _, client, _ = setup
    scan = (await client.get("/api/memory/import/scan")).json()
    gemini = next(s["id"] for s in scan["sources"] if s["harness"] == "gemini")
    body = {"batch_id": scan["batch_id"], "source_ids": [gemini]}
    assert (await client.post("/api/memory/import", json=body)).status_code == 200
    first = (await client.get("/api/memory/records")).json()["records"][0]
    edited = await client.patch(
        f"/api/memory/records/{first['id']}", json={"text": "You prefer short answers."}
    )
    assert edited.status_code == 200
    gone = await client.delete(f"/api/memory/import/{scan['batch_id']}")
    assert gone.json() == {"batch_id": scan["batch_id"], "removed": 2, "kept": 1}
    rows = (await client.get("/api/memory/records")).json()["records"]
    assert [r["text"] for r in rows] == ["You prefer short answers."]
    again = await client.delete(f"/api/memory/import/{scan['batch_id']}")
    assert again.json() == {"batch_id": scan["batch_id"], "removed": 0, "kept": 1}
    assert (await client.delete("/api/memory/import/no-such-batch")).status_code == 404
