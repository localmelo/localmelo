"""A task's approvals go through its chat: the chat's grants, and nothing else kept."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import (
    Allowance,
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
    find_auto,
    review_mode,
)
from localmelo.melo.contracts.review import ActionReview, step_key
from localmelo.support.sessions.approvals import OwnedTaskPolicy, RememberingPolicy
from localmelo.support.sessions.grants import GrantBook

ROUTINE = ActionReview("routine")
UNDECIDED = ActionReview("undecided", (), ("melo could not tell.",))
HIGH = ActionReview("high", ("install",), ("Installs requests with pip.",))


def request(
    folder: str,
    *,
    code: str = "print(1)",
    tool: str = "python_exec",
    review: ActionReview | None = ROUTINE,
    arguments: Mapping[str, Any] | None = None,
) -> ApprovalRequest:
    return ApprovalRequest(
        request_id=uuid.uuid4().hex,
        run_id="task-1",
        call_id="call-1",
        tool_name=tool,
        arguments=dict(arguments) if arguments is not None else {"code": code},
        preview=code,
        workspace_root=folder,
        tier="exec",
        review=review,
    )


def key_of(asked: ApprovalRequest) -> str:
    return step_key(asked.tool_name, asked.arguments, asked.workspace_root)


class Asker:
    """The person: answers each ask with *decision*, counting the asks."""

    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.asked: list[ApprovalRequest] = []

    async def __call__(self, asked: ApprovalRequest) -> ApprovalDecision:
        self.asked.append(asked)
        return self.decision


@pytest.fixture()
def folder(tmp_path: Path) -> str:
    path = tmp_path / "ws"
    path.mkdir()
    return str(path)


# When each chat began: its review window starts there.
STARTED = {"chat-1": 1000.0, "chat-2": 1000.0, "main": 500.0}


async def started(chat: str) -> float | None:
    return STARTED.get(chat)


async def owned(
    book: GrantBook, inner: Any, chat: str | None = "chat-1"
) -> OwnedTaskPolicy:
    """A task's policy whose turn read its owner chat's start, as the runner does."""

    async def owner() -> str | None:
        return chat

    policy = OwnedTaskPolicy(inner, book, owner, started=started)
    await policy.review_window()
    return policy


async def test_allow_once_is_one_call_in_the_task_and_in_the_chats_next_task(
    folder: str,
) -> None:
    book = GrantBook()
    person = Asker(ApprovalDecision(True))
    first = await owned(book, RememberingPolicy(person))
    assert (await first.decide(request(folder))).origin == "user"
    assert (await first.decide(request(folder))).origin == "user"
    second = await owned(book, RememberingPolicy(person))
    assert (await second.decide(request(folder))).origin == "user"
    assert len(person.asked) == 3
    assert book.live() == []


async def test_in_auto_the_chats_grant_is_not_used_and_auto_decides(
    folder: str,
) -> None:
    book = GrantBook()
    book.add_now("chat-1", "side", "tool", "python_exec")
    auto = AutoPolicy(task=True)
    policy = await owned(book, auto)
    # A routine step the grant would cover in Ask: Auto decides it, not the grant.
    assert (await policy.decide(request(folder))).origin == "auto"
    blocked = request(folder, review=UNDECIDED)
    auto.allow_once(key_of(blocked), [], "python_exec")
    # Auto's allowance passes through the task's policy as its type.
    assert isinstance(await policy.decide(blocked), Allowance)
    # The grant makes nothing silent in Auto: the rules review every step.
    assert find_auto(policy) is auto and review_mode(policy, blocked) == "auto"


async def test_a_task_gets_no_more_from_autos_site_read_than_its_chat(
    folder: str,
) -> None:
    site = "browser:https://news.example"
    book = GrantBook()
    book.view("chat-1", "side").grant_now(site, by="auto")
    person = Asker(ApprovalDecision(False, "no"))
    policy = await owned(book, RememberingPolicy(person))

    def read(url: str) -> ApprovalRequest:
        return request(
            folder,
            tool=site,
            review=None,
            arguments={"site": "news.example", "url": url, "via": "web_fetch"},
        )

    # Its plain read runs, as in the chat; an address that can carry data asks.
    assert (await policy.decide(read("https://news.example/today"))).allow
    assert (await policy.decide(read("https://news.example/a?q=1"))).origin == "user"
    assert len(person.asked) == 1


async def test_the_owner_chats_grant_covers_its_tasks_routine_calls(
    folder: str,
) -> None:
    book = GrantBook()
    book.add_now("chat-1", "side", "tool", "python_exec")
    person = Asker(ApprovalDecision(True))
    policy = await owned(book, RememberingPolicy(person))
    routine = await policy.decide(request(folder, code="print(3)"))
    assert routine.origin == "session" and person.asked == []
    # The gate lets the user's model review an undecided call first, as under a grant.
    assert review_mode(policy, request(folder, review=UNDECIDED)) == "silent"
    assert review_mode(policy, request(folder, tool="file_write")) == "ask"
    # A high-impact call asks every time.
    assert (await policy.decide(request(folder, review=HIGH))).origin == "user"
    assert len(person.asked) == 1
    # Another chat's grant covers nothing here; in Auto the chat's grants are not used.
    other = await owned(book, RememberingPolicy(person), chat="chat-2")
    assert (await other.decide(request(folder, code="print(4)"))).origin == "user"
    auto = await owned(book, AutoPolicy(task=True))
    assert (await auto.decide(request(folder, review=UNDECIDED))).origin == "auto"


async def test_the_review_window_starts_at_the_owner_chats_start(folder: str) -> None:
    book = GrantBook()
    policy = await owned(book, RememberingPolicy(Asker(ApprovalDecision(True))))
    assert await policy.review_window() == 1000.0
    other = await owned(
        book, RememberingPolicy(Asker(ApprovalDecision(True))), chat="main"
    )
    assert await other.review_window() == 500.0
    # Auto uses no grant of the chat: its steps keep the task's own window.
    auto = await owned(book, AutoPolicy(task=True), chat="main")
    assert await auto.review_window() is None


@pytest.mark.parametrize("how", ["never read", "unreadable", "raises"])
async def test_without_the_owner_chats_start_no_grant_is_used(
    folder: str, how: str
) -> None:
    book = GrantBook()
    book.add_now("chat-1", "side", "tool", "python_exec")

    async def owner() -> str | None:
        return "chat-1"

    async def unknown(chat: str) -> float | None:
        if how == "raises":
            raise RuntimeError("store closed")
        return None

    person = Asker(ApprovalDecision(True))
    policy = OwnedTaskPolicy(RememberingPolicy(person), book, owner, started=unknown)
    if how != "never read":
        assert await policy.review_window() is None
    # Fail closed: no grant of the chat; the person is asked.
    assert review_mode(policy, request(folder)) == "ask"
    assert (await policy.decide(request(folder))).origin == "user"
    assert len(person.asked) == 1


async def test_a_window_read_for_another_chat_uses_no_grant(folder: str) -> None:
    book = GrantBook()
    book.add_now("main", "main", "tool", "python_exec")
    # The turn began in chat-1; the chat went, and the step's owner is the main chat.
    owners = iter(["chat-1", "main"])

    async def owner() -> str | None:
        return next(owners, "main")

    person = Asker(ApprovalDecision(True))
    policy = OwnedTaskPolicy(RememberingPolicy(person), book, owner, started=started)
    assert await policy.review_window() == 1000.0
    assert (await policy.decide(request(folder))).origin == "user"
    assert len(person.asked) == 1
