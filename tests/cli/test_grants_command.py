"""``/grants`` in the terminal: what you allowed, where it holds, and Revoke.

In the terminal's own chat the list is this chat's grants, kept in its
sessions file, and the main chat's, read from the gateway's file; only the
main chat takes back one of those. In the main chat and its side chats the
list and Revoke are the gateway's.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from localmelo.support.cli import grants
from localmelo.support.sessions.grants import GrantBook
from localmelo.support.sessions.store import GATEWAY_SESSIONS_DB, SessionStore

from .test_session import Setup, _until
from .test_session import setup as setup


async def _first_turn(setup: Setup) -> None:
    setup.session.enter("one")
    await setup.idle()


async def test_grants_lists_and_revokes_the_chat_s_own(setup: Setup) -> None:
    session, out = setup.session, setup.out
    session.enter("/grants")
    await _until(lambda: "No grants." in out.text())
    await _first_turn(setup)
    session.approvals._policy.grants.grant_now("shell_exec")
    await session.grant_book.flush()

    session.enter("/grants")
    await _until(lambda: "1. Run shell commands ·" in out.text())
    assert "· you ·" in out.text() and "· everywhere ·" not in out.text()
    session.enter("/grants revoke 1")
    await _until(lambda: "Revoked: Run shell commands." in out.text())
    assert session.approvals.allowed_tools == frozenset()
    # The row is marked in the terminal's file: a new book reads none.
    again = GrantBook(setup.store)
    await again.open()
    assert again.live() == []


async def test_grants_takes_a_number_or_says_how(setup: Setup) -> None:
    session, out = setup.session, setup.out
    session.enter("/grants nope")
    await _until(lambda: grants.USAGE in out.text())
    session.enter("/grants revoke 3")
    await _until(lambda: "There is no grant 3" in out.text())


async def test_a_grant_stays_with_its_chat_across_new(setup: Setup) -> None:
    session = setup.session
    await _first_turn(setup)
    first = session.runner.session_id if session.runner else None
    session.approvals._policy.grants.grant_now("file_write")
    await session.grant_book.flush()
    session.enter("/new")
    await setup.idle()
    assert session.approvals.allowed_tools == frozenset()
    kept = [g for g in session.grant_book.live() if g.session_id == first]
    assert [g.key for g in kept] == ["file_write"]


async def test_the_main_chat_s_grants_hold_in_the_terminal_s_own_chat(
    setup: Setup, tmp_path: Path
) -> None:
    session, out = setup.session, setup.out
    gateway = SessionStore(Path(GATEWAY_SESSIONS_DB).expanduser(), writer="gateway")
    await gateway.open()
    try:
        book = GrantBook(gateway)
        await book.open()
        main = (await gateway.main_chat()).id
        await book.view(main, "main").grant("python_exec")
    finally:
        await gateway.aclose()

    await _first_turn(setup)
    assert "python_exec" in session.approvals.allowed_tools
    session.enter("/grants")
    await _until(lambda: "1. Run Python code · everywhere · you" in out.text())
    session.enter("/grants revoke 1")
    await _until(lambda: "allowed in the main chat" in out.text())
    assert "python_exec" in session.approvals.allowed_tools


async def test_approvals_forget_says_a_main_chat_grant_is_taken_back_there(
    setup: Setup,
) -> None:
    session, out = setup.session, setup.out
    gateway = SessionStore(Path(GATEWAY_SESSIONS_DB).expanduser(), writer="gateway")
    await gateway.open()
    try:
        book = GrantBook(gateway)
        await book.open()
        main = (await gateway.main_chat()).id
        await book.view(main, "main").grant("python_exec")
    finally:
        await gateway.aclose()
    await _first_turn(setup)
    session.approvals._policy.grants.grant_now("file_write")

    session.enter("/approvals forget python_exec")
    await _until(lambda: "was allowed in the main chat" in out.text())
    assert "python_exec" in session.approvals.allowed_tools
    session.enter("/approvals forget")
    await _until(lambda: "these still hold here" in out.text())
    assert session.approvals.allowed_tools == {"python_exec"}


async def test_approvals_forget_names_a_main_chat_folder_too(
    setup: Setup, tmp_path: Path
) -> None:
    session, out = setup.session, setup.out
    letters = Path(os.path.realpath(tmp_path)) / "Documents" / "letters"
    letters.mkdir(parents=True)
    gateway = SessionStore(Path(GATEWAY_SESSIONS_DB).expanduser(), writer="gateway")
    await gateway.open()
    try:
        book = GrantBook(gateway)
        await book.open()
        main = (await gateway.main_chat()).id
        await book.view(main, "main").grant(str(letters), kind="folder")
    finally:
        await gateway.aclose()
    await _first_turn(setup)
    assert session.approvals.allowed_folders == (str(letters),)

    def said() -> str:
        return " ".join(out.text().split())

    session.enter(f"/approvals forget {letters}")
    await _until(lambda: "was allowed in the main chat" in said())
    session.enter("/approvals forget")
    await _until(lambda: "~/Documents/letters" in said().split("these still")[-1])
    assert session.approvals.allowed_folders == (str(letters),)


class FakeGateway:
    """The gateway's grants routes, as the main chat's client calls them."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, str]] = []

    async def call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, path))
        if method == "DELETE":
            self.rows = [r for r in self.rows if path != f"/api/grants/{r['id']}"]
            return {}
        return {"grants": list(self.rows)}


async def test_in_the_main_chat_the_list_and_revoke_are_the_gateway_s(
    setup: Setup,
) -> None:
    session, out = setup.session, setup.out
    gateway = FakeGateway(
        [
            {
                "id": "0123456789ab",
                "scope": "global",
                "label": "Open example.com",
                "by": "person",
                "chat_title": "Main chat",
                "granted_at": 1.0,
            },
            {
                "id": "ba9876543210",
                "scope": "session",
                "label": "Write files",
                "by": "person",
                "chat_title": "Trip",
                "granted_at": 2.0,
            },
        ]
    )
    session.remote = SimpleNamespace(client=gateway)  # type: ignore[assignment]
    try:
        await grants.command(session, "")
        assert "1. Open example.com · everywhere · you" in out.text()
        assert "2. Write files · Trip · you" in out.text()
        await grants.command(session, "revoke 2")
        assert gateway.calls[-1] == ("DELETE", "/api/grants/ba9876543210")
        assert "Revoked: Write files." in out.text()
    finally:
        session.remote = None
