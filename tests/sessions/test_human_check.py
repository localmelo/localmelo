"""A human check asks once per chat (#583): what "Always in this chat" keeps, and where it holds.

The kept answer lives in the one grant store, is listed and revocable like the
others, and is read by the one rule (``kept_answer``) in the chat, in Auto with
nobody watching, in the chat's tasks and in its chores. It covers a known
provider's check on every site and nothing else; the main chat's covers every
chat, as every answer kept there does.
"""

from __future__ import annotations

import pytest

from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
    check_question,
)
from localmelo.melo.contracts.review import ActionReview
from localmelo.support.sessions.approvals import (
    ChatPolicy,
    ModePolicy,
    OwnedTaskPolicy,
    RememberingPolicy,
)
from localmelo.support.sessions.grants import Grant, GrantBook, describe, trusted

PAGE, OTHER = "https://shop.example", "https://news.example"
BOX = "https://captcha.boxes.example"
# How Auto hands the question to the person: only a card settles it.
OPAQUE = ActionReview("high", ("opaque",), ("A human check on this page.",))


def check(
    page: str = PAGE, review: ActionReview | None = None, name: str = ""
) -> ApprovalRequest:
    return ApprovalRequest(
        request_id="r1",
        run_id="task-1",
        call_id="c1",
        tool_name=name or check_question(page),
        arguments={"site": page.split("://")[1], "access": "check", "box": "x"},
        preview="press the human check",
        workspace_root="/ws",
        review=review,
    )


def other(name: str, review: ActionReview | None = None) -> ApprovalRequest:
    return ApprovalRequest(
        request_id="r2",
        run_id="task-1",
        call_id="c2",
        tool_name=name,
        arguments={"site": "shop.example", "access": "point", "on": "a box"},
        preview="another question",
        workspace_root="/ws",
        review=review,
    )


APP = "window-read:com.example.notes"


def window(review: ActionReview | None = None) -> ApprovalRequest:
    """Whether melo may see another app's window: a question only the person answers too."""
    return ApprovalRequest(
        request_id="r3",
        run_id="task-1",
        call_id="c3",
        tool_name=APP,
        arguments={"app": "Notes"},
        preview="see the Notes window",
        workspace_root="/ws",
        review=review,
    )


class Person:
    """Answers every card with *answer* and counts the cards."""

    def __init__(self, answer: ApprovalDecision) -> None:
        self.answer = answer
        self.asked: list[str] = []

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.asked.append(request.tool_name)
        return self.answer


ALWAYS = ApprovalDecision(True, remember=True)


async def test_always_in_this_chat_keeps_checks_on_every_site_and_nothing_else() -> (
    None
):
    person = Person(ALWAYS)
    chat = ChatPolicy(person)
    assert (await chat.decide(check())).allow and person.asked == [check_question(PAGE)]
    [grant] = chat.grants.book.live()
    assert (grant.kind, grant.action, grant.label) == (
        "check",
        "check",
        "Press human checks on every site",
    )
    # Another site's check runs unasked, in Ask and in Auto as Auto shows it.
    assert (await chat.decide(check(OTHER))).origin == "session"
    chat.mode = "auto"
    assert (await chat.decide(check(OTHER, review=OPAQUE))).origin == "session"
    assert person.asked == [check_question(PAGE)]
    # Never a name of a box of its own, a point press, a click, or a card-only step.
    chat.mode = "ask"
    person.answer = ApprovalDecision(False, "no")
    for asked in (
        check(name=f"browser-check:{PAGE}#{BOX}"),
        other(f"browser-point:{PAGE}"),
        other(f"browser-act:{PAGE}"),
        check(OTHER, review=ActionReview("high", ("money",), ("Pays.",))),
    ):
        assert not (await chat.decide(asked)).allow, asked.tool_name
    assert len(person.asked) == 5


async def test_a_chat_s_answer_covers_that_chat_and_the_main_chat_s_every_chat() -> (
    None
):
    """A side chat keeps its answer for itself; the main chat's counts everywhere, as its card says."""
    book = GrantBook()
    side, later = book.view("side-1", "side"), book.view("side-2", "side")
    assert side.grant_now(check_question(PAGE), request=check(), decision=ALWAYS)
    assert side.covers(check_question(OTHER))
    assert not later.covers(check_question(PAGE))
    main = book.view("main", "main")
    assert main.grant_now(check_question(PAGE), request=check(), decision=ALWAYS)
    assert later.covers(check_question(OTHER))
    assert book.view("task-9", "task").covers(check_question(OTHER))


async def test_allow_once_keeps_nothing_and_a_revoke_asks_again() -> None:
    person = Person(ApprovalDecision(True))
    chat = ChatPolicy(person)
    await chat.decide(check())
    await chat.decide(check())
    assert len(person.asked) == 2 and chat.grants.book.live() == []
    person.answer = ALWAYS
    await chat.decide(check())
    [grant] = chat.grants.book.live()
    assert grant.key in chat.allowed_tools  # listed like any other
    assert await chat.grants.book.revoke(grant.id) is not None
    await chat.decide(check(OTHER))
    assert len(person.asked) == 4


def test_a_check_row_is_trusted_only_as_written_and_never_as_auto_s() -> None:
    def row(key: str, by: str = "person") -> Grant:
        label, target = describe("check", key, "person")
        return Grant("g1", "chat", "side", "session", "check", key, label, target, by, 1.0)  # type: ignore[arg-type]

    assert trusted(row(check_question(PAGE)))
    # A forged row that would read wider, or as Auto's, is ignored: a box of its
    # own and another spelling of the page are no check question.
    for key in (
        "browser-check:*",
        f"browser-check:{PAGE}/x",
        f"browser-check:{PAGE}#{BOX}",
        "browser-check:https://Shop.example",
        f"browser-act:{PAGE}",
    ):
        assert not trusted(row(key)), key
    assert not trusted(row(check_question(PAGE), by="auto"))


async def owner() -> str:
    return "chat"


async def started(chat: str) -> float:
    return 1.0


@pytest.mark.parametrize("mode", ["ask", "auto"])
async def test_a_task_follows_its_chat_s_answer_in_either_mode_from_its_first_step(
    mode: str,
) -> None:
    """No review window is read, and Auto uses it too: only the check is covered so."""
    book = GrantBook()
    chat = book.view("chat", "side")
    assert chat.grant_now(check_question(PAGE), request=check(), decision=ALWAYS)
    assert chat.grant_now("python_exec") is not None
    assert chat.grant_now(APP, request=window(), decision=ALWAYS) is not None
    person = Person(ApprovalDecision(False, "no"))
    asking = RememberingPolicy(
        person, grants=book.view("task-1", "task", inherit=False)
    )
    inner = ModePolicy(asking, AutoPolicy(task=True, asking=asking), mode)  # type: ignore[arg-type]
    task = OwnedTaskPolicy(inner, book, owner, started=started)
    review = OPAQUE if mode == "auto" else None
    assert (await task.decide(check(OTHER, review=review))).origin == "session"
    assert person.asked == []
    # The chat's tool grant still needs the review window, and never holds in Auto;
    # nor does its answer for another app's window (only the check is passed on).
    code = other("python_exec", ActionReview("routine"))
    assert (await task.decide(code)).origin != "session"
    assert (await task.decide(window(review))).origin != "session"


@pytest.mark.parametrize("auto", [False, True])
async def test_a_chore_follows_its_chat_for_a_check_and_for_nothing_else(
    auto: bool,
) -> None:
    book = GrantBook()
    chat = book.view("chat", "side")
    assert chat.grant_now(f"browser-act:{PAGE}") is not None
    assert chat.grant_now(APP, request=window(), decision=ALWAYS) is not None
    person = Person(ApprovalDecision(False, "no"))
    own: object = (
        AutoPolicy(task=True)
        if auto
        else RememberingPolicy(person, grants=book.view("chore", "task", inherit=False))
    )
    chore = OwnedTaskPolicy(own, book, owner)  # type: ignore[arg-type]
    review = OPAQUE if auto else None
    # Without the chat's answer: Ask's card waits for the person, Auto blocks.
    first = await chore.decide(check(review=review))
    assert not first.allow and (first.origin == "auto") is auto
    assert chat.grant_now(check_question(PAGE), request=check(), decision=ALWAYS)
    assert (await chore.decide(check(OTHER, review=review))).origin == "session"
    # The chat's click grant and its answer for an app's window are not the chore's,
    # and a refusal there leaves the chat's.
    act = other(f"browser-act:{PAGE}", ActionReview("routine"))
    assert (await chore.decide(act)).origin != "session"
    assert (await chore.decide(window(review))).origin != "session"
    assert await chore.take_back_site("ask:site:shop.example", "person")
    assert f"browser-act:{PAGE}" in book.allowed_for("chat")
