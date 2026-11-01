"""The console and a folder outside the workspace (#405).

A chat's read outside its folder waits for the person with a card that
names the folder; "this chat" is listed with the chat's grants and can be
taken back; "always" writes ``[workspace] extra_folders``, which Settings
lists and removes. HOME is a temporary folder: nothing here touches the
real settings.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localmelo.support import config
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.tests.sessions.helpers import answer, tool_call

from .test_turn_writes import _create, _wait_pending, _wait_turn, console_with


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setenv("HOME", str(folder))
    yield Path(os.path.realpath(folder))


@pytest.fixture
def letters(home: Path) -> Path:
    path = home / "Documents" / "letters"
    path.mkdir(parents=True)
    (path / "cover-letter.md").write_text("Engineer\n")
    return path


async def _ask(
    client: Any, tmp_path: Path, letters: Path
) -> tuple[str, dict[str, Any]]:
    body = await _create(client, tmp_path, "Look at my letters folder")
    session_id = body["session"]["id"]
    [pending] = await _wait_pending(client, session_id)
    return session_id, pending


async def test_the_card_names_the_folder_and_this_chat_is_a_grant_to_take_back(
    tmp_path: Path, letters: Path
) -> None:
    async with console_with(
        tmp_path,
        tool_call("file_read", {"path": str(letters / "cover-letter.md")}),
        answer("It says engineer."),
    ) as (_console, client, _):
        session_id, pending = await _ask(client, tmp_path, letters)

        assert pending["tool_name"] == "file_read"
        assert pending["outside"] == {
            "path": str(letters / "cover-letter.md"),
            "folder": str(letters),
            "access": "read",
            "grantable": True,
            "path_label": "~/Documents/letters/cover-letter.md",
            "folder_label": "~/Documents/letters",
        }
        answered = await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "session"}
        )
        assert answered.json()["remember"] is True
        assert answered.json()["always"] is False
        turn = await _wait_turn(client, session_id)
        assert turn["status"] == "completed"

        listed = (await client.get(f"/api/sessions/{session_id}/tools")).json()
        assert listed["folders"] == [
            {"path": str(letters), "label": "~/Documents/letters"}
        ]
        # A folder grant is no grant of the tool.
        detail = (await client.get(f"/api/sessions/{session_id}")).json()
        assert detail["grants"] == [] and detail["granted_at"] == {}

        gone = await client.delete(
            f"/api/sessions/{session_id}/folders", params={"path": str(letters)}
        )
        again = await client.delete(
            f"/api/sessions/{session_id}/folders", params={"path": str(letters)}
        )
        assert (gone.status_code, again.status_code) == (204, 404)
        listed = (await client.get(f"/api/sessions/{session_id}/tools")).json()
        assert listed["folders"] == []


async def test_always_keeps_the_folder_in_the_settings(
    tmp_path: Path, letters: Path
) -> None:
    async with console_with(
        tmp_path,
        tool_call("file_read", {"path": str(letters / "cover-letter.md")}),
        answer("done"),
    ) as (_console, client, _):
        session_id, pending = await _ask(client, tmp_path, letters)

        answered = await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "always"}
        )
        await _wait_turn(client, session_id)

        assert answered.json()["always"] is True
        assert config.load().workspace.extra_folders == ["~/Documents/letters"]


async def test_always_is_refused_for_an_ordinary_call(tmp_path: Path) -> None:
    target = tmp_path / "ws" / "todo.md"
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(target), "content": "# TODO"}),
        answer("done"),
    ) as (_console, client, _):
        body = await _create(client, tmp_path, "Write the todo file")
        session_id = body["session"]["id"]
        [pending] = await _wait_pending(client, session_id)

        refused = await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "always"}
        )
        assert refused.status_code == 400
        assert pending["outside"] is None
        await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "deny"}
        )
        await _wait_turn(client, session_id)


async def test_a_write_outside_is_allowed_once_whatever_the_answer(
    tmp_path: Path, letters: Path
) -> None:
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(letters / "new.md"), "content": "x"}),
        answer("done"),
    ) as (_console, client, _):
        session_id, pending = await _ask(client, tmp_path, letters)
        assert pending["outside"]["access"] == "write"
        assert pending["outside"]["grantable"] is False

        answered = await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "always"}
        )
        await _wait_turn(client, session_id)

        assert answered.json()["remember"] is False
        assert answered.json()["always"] is False
        assert (letters / "new.md").read_text() == "x"
        assert config.load().workspace.extra_folders == []


# ── Settings ──


def _saved(**folders: Any) -> None:
    cfg = Config(
        chat_backend="mlc",
        embedding_backend="none",
        mlc=LocalBackendConfig(chat_url="http://127.0.0.1:9/v1", chat_model="m"),
    )
    cfg.workspace.extra_folders = list(folders.get("folders", []))
    config.save(cfg)


def _put(revision: str, **extra: Any) -> dict[str, Any]:
    return {
        "chat_backend": "mlc",
        "embedding_backend": "none",
        "workspace": "~/ws",
        "sections": {},
        "revision": revision,
        **extra,
    }


def test_settings_list_the_extra_folders_and_remove_one(client: TestClient) -> None:
    _saved(folders=["~/Documents/letters", "~/.ssh"])

    body = client.get("/api/settings").json()
    assert body["extra_folders"] == [
        {"path": "~/Documents/letters", "name": "letters", "valid": True},
        {"path": "~/.ssh", "name": ".ssh", "valid": False},
    ]

    bad = client.put(
        "/api/settings", json=_put(body["revision"], extra_folders=["~/.ssh"])
    )
    assert bad.status_code == 400
    assert "never opens" in bad.json()["error"]["message"]

    kept = client.put(
        "/api/settings",
        json=_put(body["revision"], extra_folders=["~/Documents/letters"]),
    )
    assert kept.status_code == 200, kept.text
    assert [f["path"] for f in kept.json()["extra_folders"]] == ["~/Documents/letters"]
    # Absent leaves it; an empty list removes the table.
    same = client.put("/api/settings", json=_put(kept.json()["revision"]))
    assert len(same.json()["extra_folders"]) == 1
    empty = client.put(
        "/api/settings", json=_put(same.json()["revision"], extra_folders=[])
    )
    assert empty.json()["extra_folders"] == []
    assert "[workspace]" not in Path(config.config_path()).read_text()
