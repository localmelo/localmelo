"""``/api/grants``: one list of every grant, Revoke, and the main chat's counting everywhere.

A side chat's "allow for this chat" holds in that chat; what the person
allows in the main chat holds in every chat and in the tasks' chats. The
list comes in the order the page numbers it, a revoke is felt at once, a
deleted chat takes its grants along, and a restart keeps them. The routes
sit behind the gateway's token and origin guard like every other route.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.support.sessions.store import SessionStore, SessionStoreError
from localmelo.tests.sessions.helpers import answer, tool_call

from ..test_guard import FOREIGN_ORIGIN, ORIGIN_ERROR, _build_gateway
from .test_turn_writes import _create, _wait_pending, _wait_turn, console_with

pytestmark = pytest.mark.filterwarnings(
    r"ignore:\s*on_event is deprecated:DeprecationWarning"
)


async def _side_chat_with_a_grant(client: object, tmp_path: Path) -> str:
    """A side chat whose person allowed ``file_write`` for the chat."""
    body = await _create(client, tmp_path, "Write the todo file")  # type: ignore[arg-type]
    session_id = str(body["session"]["id"])
    [pending] = await _wait_pending(client, session_id)  # type: ignore[arg-type]
    await client.post(  # type: ignore[attr-defined]
        f"/api/approvals/{pending['request_id']}", json={"choice": "session"}
    )
    await _wait_turn(client, session_id)  # type: ignore[arg-type]
    return session_id


async def test_the_list_shows_everywhere_first_then_each_chat(tmp_path: Path) -> None:
    target = tmp_path / "ws" / "todo.md"
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(target), "content": "# TODO"}),
        answer("done"),
    ) as (console, client, _):
        assert (await client.get("/api/grants")).json() == {"grants": []}
        side = await _side_chat_with_a_grant(client, tmp_path)
        main = console.main_id
        assert main is not None
        await console.sessions.book.view(main, "main").grant("shell_exec")

        rows = (await client.get("/api/grants")).json()["grants"]

        assert [(r["key"], r["scope"], r["session_kind"]) for r in rows] == [
            ("shell_exec", "global", "main"),
            ("file_write", "session", "side"),
        ]
        first = rows[0]
        assert first["chat_title"] == "Main chat" and first["by"] == "person"
        assert first["label"] == "Run shell commands" and first["action"] == "tool"
        assert rows[1]["session_id"] == side and rows[1]["chat_title"]
        assert set(first) == {
            "id",
            "scope",
            "session_id",
            "session_kind",
            "chat_title",
            "kind",
            "key",
            "label",
            "action",
            "target",
            "by",
            "granted_at",
        }


async def test_a_main_chat_grant_holds_in_a_side_chat_and_its_tasks(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("hi")) as (console, client, _):
        body = await _create(client, tmp_path, "Hello")
        side = body["session"]["id"]
        await _wait_turn(client, side)
        main = console.main_id
        assert main is not None
        await console.sessions.book.view(main, "main").grant("python_exec")

        detail = (await client.get(f"/api/sessions/{side}")).json()
        assert detail["grants"] == ["python_exec"]
        assert detail["granted_at"]["python_exec"] > 0
        # A task the side chat started uses its chat's grants (#369).
        assert "python_exec" in console.workers.ledger.grants_of(side)
        # A side chat's own grant stays its own.
        await console.sessions.book.view(side, "side").grant("file_write")
        main_detail = (await client.get(f"/api/sessions/{main}")).json()
        assert main_detail["grants"] == ["python_exec"]


async def test_a_revoke_is_felt_at_once_and_a_second_one_is_404(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ws" / "todo.md"
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(target), "content": "# TODO"}),
        answer("done"),
    ) as (console, client, _):
        side = await _side_chat_with_a_grant(client, tmp_path)
        [row] = (await client.get("/api/grants")).json()["grants"]

        revoked = await client.delete(f"/api/grants/{row['id']}")

        assert revoked.status_code == 204 and revoked.content == b""
        entry = console.sessions.get(side)
        assert entry is not None and "file_write" not in entry.policy.allowed_tools
        assert (await client.get(f"/api/sessions/{side}")).json()["grants"] == []
        again = await client.delete(f"/api/grants/{row['id']}")
        assert again.status_code == 404
        assert again.json()["error"]["code"] == "not_found"
        assert (await client.delete("/api/grants/not-an-id")).status_code == 404


async def test_deleting_a_chat_deletes_its_grants(tmp_path: Path) -> None:
    target = tmp_path / "ws" / "todo.md"
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(target), "content": "# TODO"}),
        answer("done"),
    ) as (console, client, _):
        side = await _side_chat_with_a_grant(client, tmp_path)
        assert (await client.delete(f"/api/sessions/{side}")).status_code == 204
        assert (await client.get("/api/grants")).json() == {"grants": []}
        assert console.sessions.book.live() == []
        async with console.store.component_transaction("test") as conn:
            rows = list(await conn.execute_fetchall("SELECT COUNT(*) FROM grants"))
        assert rows == [(0,)]


async def test_grants_are_listed_again_after_a_restart(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (console, _client, _):
        main = console.main_id
        assert main is not None
        await console.sessions.book.view(main, "main").grant("shell_exec")
    async with console_with(tmp_path) as (console, client, _):
        rows = (await client.get("/api/grants")).json()["grants"]
        assert [(r["key"], r["scope"]) for r in rows] == [("shell_exec", "global")]
        assert console.main_id is not None
        assert "shell_exec" in console.sessions.grants(console.main_id)


async def test_a_change_of_folder_takes_back_the_chat_s_grants_but_not_the_main_chat_s(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("hi")) as (console, client, _):
        body = await _create(client, tmp_path, "Hello")
        side = body["session"]["id"]
        await _wait_turn(client, side)
        assert console.main_id is not None
        await console.sessions.book.view(console.main_id, "main").grant("shell_exec")
        await console.sessions.book.view(side, "side").grant("file_write")
        other = tmp_path / "other"
        other.mkdir()
        moved = await client.patch(
            f"/api/sessions/{side}", json={"workspace": str(other)}
        )
        assert moved.status_code == 200, moved.text
        assert (await client.get(f"/api/sessions/{side}")).json()["grants"] == [
            "shell_exec"
        ]


async def test_store_path_is_the_console_s(tmp_path: Path) -> None:
    """The table lives in the gateway's sessions file, not a file of its own."""
    async with console_with(tmp_path) as (console, _client, _):
        assert isinstance(console.grants.store, SessionStore)
        assert console.grants.store.path == str(tmp_path / "sessions.db")


async def test_a_side_chat_shows_and_takes_back_the_main_chat_s_grants_everywhere(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("hi")) as (console, client, _):
        body = await _create(client, tmp_path, "Hello")
        side = body["session"]["id"]
        await _wait_turn(client, side)
        main = console.main_id
        assert main is not None
        book = console.sessions.book
        await book.view(main, "main").grant("python_exec")
        await book.view(main, "main").grant("shell_exec")
        await book.view(side, "side").grant("file_write")

        detail = (await client.get(f"/api/sessions/{side}")).json()
        assert detail["grants_everywhere"] == ["python_exec", "shell_exec"]
        main_detail = (await client.get(f"/api/sessions/{main}")).json()
        assert main_detail["grants_everywhere"] == []

        # Clear: only the side chat's own go; the main chat's still hold here.
        cleared = await client.delete(f"/api/sessions/{side}/grants")
        assert cleared.json() == {"revoked": 1, "still_everywhere": 2}
        # By name: a main-chat grant goes everywhere, and the answer says so.
        gone = await client.delete(f"/api/sessions/{side}/grants/python_exec")
        assert gone.status_code == 200
        assert gone.json() == {
            "revoked": [{"key": "python_exec", "scope": "global"}],
            "everywhere": True,
        }
        assert console.sessions.grants(main) == {"shell_exec"}


async def test_a_main_chat_folder_taken_back_in_a_side_chat_goes_everywhere(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "letters"
    folder.mkdir()
    async with console_with(tmp_path, answer("hi")) as (console, client, _):
        body = await _create(client, tmp_path, "Hello")
        side = body["session"]["id"]
        await _wait_turn(client, side)
        main = console.main_id
        assert main is not None
        await console.sessions.book.view(main, "main").grant(str(folder), kind="folder")
        assert console.sessions.folder_grants(side) == (str(folder),)

        gone = await client.delete(
            f"/api/sessions/{side}/folders", params={"path": str(folder)}
        )

        assert gone.status_code == 200
        assert gone.json() == {
            "revoked": [{"key": str(folder), "scope": "global"}],
            "everywhere": True,
        }
        assert console.sessions.folder_grants(main) == ()


async def test_a_revoke_the_store_cannot_save_is_503_and_holds_here(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("hi")) as (console, client, _):
        body = await _create(client, tmp_path, "Hello")
        side = body["session"]["id"]
        await _wait_turn(client, side)
        await console.sessions.book.view(side, "side").grant("file_write")

        async def fail(*args: object, **kwargs: object) -> None:
            raise SessionStoreError("disk full")

        console.sessions.book._mark_revoked = fail  # type: ignore[method-assign]
        refused = await client.delete(f"/api/sessions/{side}/grants/file_write")
        assert refused.status_code == 503
        # The check stopped using it all the same.
        assert "file_write" not in console.sessions.grants(side)


# ── The guard ──


def test_the_grants_routes_need_the_token() -> None:
    gateway = _build_gateway()
    try:
        bare = gateway.client.get(
            "/api/grants", headers={"Authorization": "Bearer wrong"}
        )
        assert bare.status_code == 401
        refused = gateway.client.delete(
            "/api/grants/0123456789ab", headers={"Authorization": ""}
        )
        assert refused.status_code == 401
    finally:
        gateway.server.auth.close()


def test_a_page_on_another_origin_cannot_revoke() -> None:
    gateway = _build_gateway()
    try:
        refused = gateway.client.delete(
            "/api/grants/0123456789ab", headers={"Origin": FOREIGN_ORIGIN}
        )
        assert refused.status_code == 403
        assert refused.json()["error"]["code"] == ORIGIN_ERROR["code"]
    finally:
        gateway.server.auth.close()
