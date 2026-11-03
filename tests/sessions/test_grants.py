"""The grants table: scope, persistence, revoke, refusals and cleanup.

What the person allows in the main chat counts in every session; what they
allow in a side chat, the terminal's own chat or a task holds there only;
what Auto keeps is the chat's own, the main chat's too. Nothing that must
ask every time is written, and a row that breaks a rule is never used.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.melo.contracts.approvals import (
    ALLOWED_ONCE,
    ApprovalDecision,
    ApprovalRequest,
    holds_grant,
)
from localmelo.melo.contracts.review import ActionReview
from localmelo.support.sessions.approvals import (
    ApprovalLedger,
    ChatPolicy,
    OwnedTaskPolicy,
    RememberingPolicy,
)
from localmelo.support.sessions.grants import (
    COMPONENT,
    GRANTS_VERSION,
    Grant,
    GrantBook,
    GrantRefusedError,
    check_grant,
    grant_kind,
    listed,
    read_global_grants,
    trusted,
)
from localmelo.support.sessions.store import SessionStore, SessionStoreError

SITE = "browser:https://example.com"


def _request(
    tool: str = "shell_exec",
    *,
    review: ActionReview | None = None,
    arguments: dict[str, object] | None = None,
    request_id: str = "r1",
) -> ApprovalRequest:
    return ApprovalRequest(
        request_id=request_id,
        run_id="",
        call_id="c1",
        tool_name=tool,
        arguments=arguments if arguments is not None else {"command": "ls"},
        preview="ls",
        workspace_root="/work",
        review=review,
    )


class Asker:
    """Answers each request with the next decision, and keeps what it was asked."""

    def __init__(self, *decisions: ApprovalDecision) -> None:
        self.decisions = list(decisions)
        self.requests: list[ApprovalRequest] = []

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self.decisions.pop(0)


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    opened = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await opened.open()
    yield opened
    await opened.aclose()


async def _chats(store: SessionStore) -> tuple[str, str, str]:
    main = (await store.main_chat()).id
    side = (await store.create_session(workspace="/work")).id
    other = (await store.create_session(workspace="/work")).id
    return main, side, other


# ── Scope ──


async def test_the_person_s_word_in_the_main_chat_counts_in_every_session(
    store: SessionStore,
) -> None:
    book = GrantBook(store)
    await book.open()
    main, side, other = await _chats(store)
    granted = await book.view(main, "main").grant("shell_exec", request=_request())
    assert granted is not None and granted.scope == "global"

    # A side chat runs it without asking, as on its own grant.
    asker = Asker()
    policy = ChatPolicy(asker, grants=book.view(side, "side"))
    decision = await policy.decide(_request(review=ActionReview("routine")))
    assert decision.allow and decision.origin == "session" and asker.requests == []

    # A task of a side chat, through its chat (#369), after the review window.
    ledger = ApprovalLedger()
    ledger.grants_of = book.allowed_for
    task_policy = OwnedTaskPolicy(
        RememberingPolicy(Asker(), grants=book.view("t1", "task", inherit=False)),
        ledger,
        "t1",
        owner=lambda: _chat(side),
        started=_start,
    )
    assert await task_policy.review_window() == 1.0
    ran = await task_policy.decide(_request(review=ActionReview("routine")))
    assert ran.allow and ran.origin == "session"
    # A worker's tool that checks a grant by name sees it too.
    assert holds_grant(task_policy, "shell_exec")
    assert "shell_exec" in book.allowed_for(other)
    await book.flush()


async def _chat(chat: str) -> str | None:
    return chat


async def _start(_chat_id: str) -> float | None:
    return 1.0


async def test_a_side_chat_s_grant_holds_in_that_chat_only(
    store: SessionStore,
) -> None:
    book = GrantBook(store)
    await book.open()
    main, side, other = await _chats(store)
    granted = await book.view(side, "side").grant("file_write")
    assert granted is not None and granted.scope == "session"
    assert "file_write" in book.view(side, "side").names()
    assert "file_write" not in book.view(other, "side").names()
    assert "file_write" not in book.view(main, "main").names()
    assert book.allowed_for(main) == frozenset()


async def test_auto_s_site_in_the_main_chat_stays_the_main_chat_s(
    store: SessionStore,
) -> None:
    book = GrantBook(store)
    await book.open()
    main, side, _ = await _chats(store)
    policy = RememberingPolicy(Asker(), grants=book.view(main, "main"))
    policy.passed_in_auto(SITE)
    await book.flush()
    [kept] = book.live()
    assert (kept.scope, kept.decided_by, kept.kind) == ("session", "auto", "site")
    assert SITE in policy.allowed_tools
    assert SITE not in book.view(side, "side").passed()
    # Auto keeps a site read only: a click on it is not kept.
    policy.passed_in_auto("browser-act:https://example.com")
    assert len(book.live()) == 1


async def test_a_folder_the_person_allows_in_the_main_chat_reads_everywhere(
    store: SessionStore, tmp_path: Path
) -> None:
    book = GrantBook(store)
    await book.open()
    main, side, _ = await _chats(store)
    folder = tmp_path / "home" / "letters"
    folder.mkdir(parents=True)
    granted = await book.view(main, "main").grant(str(folder), kind="folder")
    assert granted is not None and granted.scope == "global"
    assert book.view(side, "side").folders() == (str(folder),)


# ── Persistence ──


async def test_grants_outlive_a_restart_and_are_used(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    first = SessionStore(path, writer="gateway")
    await first.open()
    main, side, _ = await _chats(first)
    book = GrantBook(first)
    await book.open()
    await book.view(main, "main").grant("python_exec")
    await book.view(side, "side").grant(SITE)
    await first.aclose()

    again = SessionStore(path, writer="gateway")
    await again.open()
    try:
        reopened = GrantBook(again)
        await reopened.open()
        assert sorted(g.key for g in reopened.live()) == [SITE, "python_exec"]
        asker = Asker()
        policy = ChatPolicy(asker, grants=reopened.view(side, "side"))
        decision = await policy.decide(_request("python_exec"))
        assert decision.allow and decision.origin == "session"
        assert asker.requests == []
        assert SITE in policy.allowed_tools
    finally:
        await again.aclose()


async def test_the_component_is_versioned_and_a_newer_one_is_refused(
    store: SessionStore,
) -> None:
    await GrantBook(store).open()
    async with store.component_transaction("test") as conn:
        rows = list(
            await conn.execute_fetchall(
                "SELECT version FROM schema_components WHERE name = ?", (COMPONENT,)
            )
        )
        assert rows[0][0] == GRANTS_VERSION
        await conn.execute(
            "UPDATE schema_components SET version = ? WHERE name = ?",
            (GRANTS_VERSION + 1, COMPONENT),
        )
    with pytest.raises(SessionStoreError, match="grants schema version"):
        await GrantBook(store).open()


async def test_a_grant_given_before_the_book_opens_is_written_when_it_does(
    store: SessionStore,
) -> None:
    main, _, _ = await _chats(store)
    book = GrantBook(store)
    book.view(main, "main").grant_now("shell_exec")
    await book.open()
    await book.flush()
    other = GrantBook(store)
    await other.open()
    assert [g.key for g in other.live()] == ["shell_exec"]


# ── Revoke ──


async def test_a_revoke_takes_effect_at_once_in_a_running_session(
    store: SessionStore,
) -> None:
    book = GrantBook(store)
    await book.open()
    main, side, _ = await _chats(store)
    granted = await book.view(main, "main").grant("shell_exec")
    assert granted is not None
    asker = Asker(ApprovalDecision(False, "no"))
    policy = ChatPolicy(asker, grants=book.view(side, "side"))
    assert "shell_exec" in policy.allowed_tools

    write = asyncio.ensure_future(book.revoke(granted.id))
    await asyncio.sleep(0)
    # The check stops using it before the row is written.
    assert "shell_exec" not in policy.allowed_tools
    assert (await write) is not None
    decision = await policy.decide(_request(review=ActionReview("routine")))
    assert not decision.allow and len(asker.requests) == 1
    assert await book.revoke(granted.id) is None

    reopened = GrantBook(store)
    await reopened.open()
    assert reopened.live() == []


async def test_taking_back_a_main_chat_grant_by_name_in_a_side_chat_takes_it_everywhere(
    store: SessionStore,
) -> None:
    book = GrantBook(store)
    await book.open()
    main, side, _ = await _chats(store)
    await book.view(main, "main").grant("shell_exec")
    await book.view(side, "side").grant("file_write")
    side_policy = RememberingPolicy(Asker(), grants=book.view(side, "side"))
    # Every grant of a side chat: its own go, the main chat's stay.
    side_policy.forget()
    assert side_policy.allowed_tools == {"shell_exec"}
    side_policy.forget("shell_exec")
    assert book.allowed_for(main) == frozenset()
    await book.flush()


# ── Refusals ──


@pytest.mark.parametrize(
    "key",
    [
        "browser-submit:https://example.com",
        "browser-type:https://example.com",
        "browser-point:https://example.com",
        "desktop-act:Notes",
        "browser:",
        "",
        "shell_exec\n",
    ],
)
def test_a_question_that_asks_every_time_is_never_a_grant(key: str) -> None:
    assert grant_kind(key) is None
    with pytest.raises(GrantRefusedError):
        check_grant("tool", key, "person")


def test_money_account_and_allow_once_are_never_kept() -> None:
    money = _request(
        "browser-act:https://shop.example",
        review=ActionReview("routine", ("money",)),
        arguments={"site": "shop.example"},
    )
    with pytest.raises(GrantRefusedError, match="money"):
        check_grant("site", money.tool_name, "person", request=money)
    account = _request(review=ActionReview("undecided", ("account",)))
    with pytest.raises(GrantRefusedError):
        check_grant("tool", "shell_exec", "person", request=account)
    for once in (
        ApprovalDecision(True),
        ApprovalDecision(True, ALLOWED_ONCE, remember=True),
    ):
        with pytest.raises(GrantRefusedError, match="Allow once"):
            check_grant("tool", "shell_exec", "person", decision=once)
    high = _request(review=ActionReview("high", ("install",)))
    with pytest.raises(GrantRefusedError, match="high-impact"):
        check_grant("tool", "shell_exec", "person", request=high)
    with pytest.raises(GrantRefusedError, match="Auto keeps"):
        check_grant("tool", "shell_exec", "auto")
    with pytest.raises(GrantRefusedError):
        check_grant("folder", "/", "person")


async def test_a_refused_grant_writes_no_row(store: SessionStore) -> None:
    book = GrantBook(store)
    await book.open()
    main, _, _ = await _chats(store)
    view = book.view(main, "main")
    money = _request(review=ActionReview("routine", ("money",)))
    assert await view.grant("shell_exec", request=money) is None
    assert await view.grant("browser-submit:https://example.com") is None
    assert await view.grant("shell_exec", decision=ApprovalDecision(True)) is None
    await book.flush()
    async with store.component_transaction("test") as conn:
        assert list(await conn.execute_fetchall("SELECT COUNT(*) FROM grants")) == [
            (0,)
        ]


async def test_a_card_only_step_answered_for_the_chat_stays_unasked_never(
    store: SessionStore,
) -> None:
    """The person's "for this chat" on a paying step allows that step, and keeps nothing."""
    book = GrantBook(store)
    await book.open()
    main, _, _ = await _chats(store)
    asker = Asker(ApprovalDecision(True, remember=True), ApprovalDecision(False, "no"))
    policy = RememberingPolicy(asker, grants=book.view(main, "main"))
    request = _request(review=ActionReview("undecided", ("money",)))
    assert (await policy.decide(request)).allow
    assert policy.allowed_tools == frozenset()
    assert not (await policy.decide(request)).allow


async def test_a_forged_row_is_never_used(store: SessionStore, tmp_path: Path) -> None:
    main, side, _ = await _chats(store)
    await GrantBook(store).open()
    rows = [
        # global from a side chat, global from Auto, Auto beyond a site read
        ("a1", side, "side", "global", "tool", "shell_exec", "person"),
        ("a2", main, "main", "global", "site", SITE, "auto"),
        ("a3", side, "side", "session", "tool", "file_write", "auto"),
        # questions that ask every time, a kind that does not match, a card-only kind
        (
            "a4",
            main,
            "main",
            "global",
            "tool",
            "browser-submit:https://x.example",
            "person",
        ),
        ("a5", main, "main", "global", "tool", "desktop-act:Notes", "person"),
        ("a6", main, "main", "global", "site", "shell_exec", "person"),
        ("a7", main, "main", "global", "money", "pay", "person"),
        # a folder that is never offered: the home folder and the system's
        ("a8", main, "main", "global", "folder", str(Path.home()), "person"),
        ("a9", main, "main", "global", "folder", "/etc", "person"),
        ("a10", main, "terminal", "global", "tool", "python_exec", "person"),
    ]
    async with store.component_transaction("forge") as conn:
        for gid, sid, skind, scope, kind, key, by in rows:
            await conn.execute(
                "INSERT INTO grants (id, session_id, session_kind, scope, kind, key, "
                "decided_by, granted_at) VALUES (?, ?, ?, ?, ?, ?, ?, 1.0)",
                (gid, sid, skind, scope, kind, key, by),
            )
    book = GrantBook(store)
    await book.open()
    assert book.live() == []
    assert book.allowed_for(side) == frozenset()
    assert book.folders_for(side) == ()
    assert await read_global_grants(store.path) == []


def test_trusted_keeps_the_rule_of_the_write() -> None:
    good = Grant(
        "g1", "m", "main", "global", "tool", "shell_exec", "", "", "person", 1.0
    )
    assert trusted(good)
    assert not trusted(
        Grant("g2", "s", "side", "global", "tool", "shell_exec", "", "", "person", 1.0)
    )
    assert not trusted(
        Grant("g3", "m", "main", "global", "site", SITE, "", "", "auto", 1.0)
    )
    assert not trusted(
        Grant(
            "g4",
            "m",
            "main",
            "global",
            "tool",
            "shell_exec",
            "",
            "",
            "person",
            1.0,
            revoked_at=2.0,
        )
    )


# ── Cleanup ──


async def test_deleting_a_chat_deletes_its_grants(store: SessionStore) -> None:
    book = GrantBook(store)
    await book.open()
    main, side, _ = await _chats(store)
    await book.view(side, "side").grant("file_write")
    await book.view(main, "main").grant("shell_exec")
    assert await store.delete_session(side)
    book.forget_session(side)
    assert [g.key for g in book.live()] == ["shell_exec"]
    async with store.component_transaction("test") as conn:
        keys = list(await conn.execute_fetchall("SELECT key FROM grants"))
    assert keys == [("shell_exec",)]


async def test_retention_drops_old_revoked_rows_only(store: SessionStore) -> None:
    book = GrantBook(store, clock=lambda: 1.0)
    await book.open()
    main, _, _ = await _chats(store)
    kept = await book.view(main, "main").grant("shell_exec")
    gone = await book.view(main, "main").grant("python_exec")
    assert kept is not None and gone is not None
    await book.revoke(gone.id)
    assert await book.apply_retention() == 1
    async with store.component_transaction("test") as conn:
        ids = list(await conn.execute_fetchall("SELECT id FROM grants"))
    assert ids == [(kept.id,)]


# ── Lists and the terminal ──


def test_a_list_shows_everywhere_first_then_each_chat_newest_first() -> None:
    def grant(gid: str, sid: str, scope: str, at: float) -> Grant:
        kind = "main" if scope == "global" else "side"
        return Grant(gid, sid, kind, scope, "tool", gid, "", "", "person", at)  # type: ignore[arg-type]

    rows = [
        grant("g1", "m", "global", 1.0),
        grant("s1", "a", "session", 2.0),
        grant("s2", "b", "session", 5.0),
        grant("g2", "m", "global", 3.0),
        grant("s3", "a", "session", 4.0),
    ]
    assert [g.id for g in listed(rows)] == ["g2", "g1", "s2", "s3", "s1"]


async def test_the_terminal_reads_the_main_chat_s_grants_without_writing(
    store: SessionStore, tmp_path: Path
) -> None:
    book = GrantBook(store)
    await book.open()
    main, side, _ = await _chats(store)
    await book.view(main, "main").grant("shell_exec")
    await book.view(side, "side").grant("file_write")

    terminal_store = SessionStore(tmp_path / "cli.db", writer="cli")
    await terminal_store.open()
    try:
        terminal = GrantBook(
            terminal_store, outside=lambda: read_global_grants(store.path)
        )
        await terminal.open()
        await terminal.refresh()
        view = terminal.view("t-chat", "terminal")
        assert view.names() == {"shell_exec"}
        assert terminal.live() == []  # the main chat's rows are not this book's
        assert await read_global_grants(tmp_path / "missing.db") == []
    finally:
        await terminal_store.aclose()


def test_holds_grant_never_answers_for_a_question_that_asks_every_time() -> None:
    policy = RememberingPolicy(Asker())
    policy.grants.grant_now("window-read:com.example.notes")
    assert holds_grant(policy, "window-read:com.example.notes")
    assert holds_grant(
        ChatPolicy(Asker(), grants=policy.grants), "window-read:com.example.notes"
    )
    assert not holds_grant(policy, "desktop-act:Notes")
    assert not holds_grant(policy, "window-read:com.example.mail")
