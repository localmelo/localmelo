"""In Auto, the steps a task needs run without asking; the risky ones still stop (#465).

A regression set from a person's test: a daily check-in on a forum that needs
signing in first, over eight messages, in a chat in Auto mode. The steps are
the questions melo's browser asks (opening a site, a click that names what it
is on, typing, a form sent), on generic sites. The user's model is a fake whose
review replies are scripted: where the real run's reply is known it is used
(the sign-in click was filed under ``credentials``), elsewhere the reply the
review prompt asks for.

- The six steps the check-in needs run with no card: opening the forum, the
  sign-in clicks done the way the person described, picking the mood, writing
  the line, sending it, reading the result.
- Steps off the goal or of a high-impact kind still stop or ask: changing the
  account's email, posting in another thread, typing a password, paying.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._step_review import StepReviewer, holds_secret
from localmelo.melo.checker.review.model import (
    CHAT_REQUESTS,
    EARLIER_REQUESTS,
    ReviewInputs,
    user_message,
)
from localmelo.melo.contracts.approvals import (
    TOOL_ASK,
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
)
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.memory.block import BLOCK_HEADER
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.sessions.approvals import (
    PASSED_IN_AUTO,
    ChatPolicy,
    RememberingPolicy,
)
from localmelo.tests.agent.test_approvals import Recorder, _answer, _calls, _no_wait
from localmelo.tests.agent.test_auto_choice import Card
from localmelo.tests.agent.test_auto_mode import FakeModel, verdict
from localmelo.tests.browser.fakes import FakeDriver, button, page

FORUM = "www.forum.example"
SIGN_IN = "auth.forum.example"
CHAT_APP = "login.chat.example"

# The questions melo's browser asks, by step: (name, arguments, preview).
Q = tuple[str, dict[str, Any], str]


def _read(site: str) -> Q:
    return (
        f"browser:https://{site}",
        {"site": site, "origin": f"https://{site}", "access": "read"},
        f"open {site} in Safari and read it",
    )


def _click(site: str, on: str) -> Q:
    return (
        f"browser-act:https://{site}",
        {"site": site, "access": "act", "how": "click", "on": on},
        f"click {on} on {site} in Safari",
    )


def _type(site: str, on: str, text: str) -> Q:
    return (
        f"browser-type:https://{site}",
        {"site": site, "access": "type", "how": "type", "text": text, "on": on},
        f"type {len(text)} character(s) into {on} on {site} in Safari",
    )


def _send(site: str, to: str, fields: list[list[str]]) -> Q:
    return (
        f"browser-submit:https://{site}",
        {"site": site, "access": "submit", "to": to, "how": "click", "fields": fields},
        f"send a form on {site} to {to}",
    )


STEPS: dict[str, Q] = {
    # Turn 1: open the check-in page and read it ("please sign in first").
    "open": _read(FORUM),
    # Turns 2-3: sign in the way the person described: the chat app's login,
    # then its quick login inside the chat app's box.
    "sign_in": _click(SIGN_IN, 'a link "Chat app login"'),
    "quick_login": _click(
        SIGN_IN, f'a button "Quick login" inside a box from {CHAT_APP}'
    ),
    # Turns 5-8: pick the mood, write a short line, send it, read the result.
    "mood": _click(FORUM, 'a button "Happy"'),
    "line": _type(FORUM, 'a textbox "Say something today"', "Checking in today."),
    "submit": _click(FORUM, 'a button "Check in"'),
    "result": _read(FORUM),
    # The variants that must still stop or ask.
    "change_email": _send(
        FORUM,
        f"https://{FORUM}/account/email",
        [["email", "alex@example.com"]],
    ),
    "post_elsewhere": _send(
        FORUM, f"https://{FORUM}/thread/42/reply", [["message", "Hello all"]]
    ),
    "pay": _click(FORUM, 'a button "Pay $10 for premium"'),
    # A click whose words hold a key: melo's own detectors find it.
    "key_link": _click(FORUM, f'a link "Use key sk-{"a1B2" * 6}"'),
    "password": _type(FORUM, 'a textbox "Password"', "hunter2-secret"),
    # A click inside a box from another site that melo cannot tell.
    "blind": (
        f"browser-point:https://{SIGN_IN}",
        {
            "site": SIGN_IN,
            "access": "point",
            "how": "click",
            "on": f"inside a box from {CHAT_APP} that melo cannot check",
            "target": "unknown",
        },
        f"click inside a box from {CHAT_APP} that melo cannot check on {SIGN_IN}",
    ),
}

# The person's words over the eight messages (generic, no names).
CHECK_IN = "check in on the forum for me"
THAT_WAY = "you remember how I told you to sign in? do it that way"
# What a chat's runner appends to the message, and the message with it.
NOTES = (
    f"\n\n{BLOCK_HEADER}\n"
    "- the user said “how do I sign in”; melo replied “Pick the chat app login, "
    "wait for the quick login button and click it.”"
)
RECALLED = THAT_WAY + NOTES

# The review replies. On the goal, nothing high-impact: what the prompt asks.
ON_GOAL = verdict()
# The real reply to the sign-in click: filed under passwords and keys.
SIGN_IN_AS_KEYS = verdict(verdict="ask", category="credentials", user_asked="no")
# A reply that stops a step the latest words do not name, though it serves the goal.
NOT_NAMED = verdict(verdict="ask", category="beyond_request", user_asked="no")


class Steps(BaseToolProvider):
    """A tool that asks a browser question by its step's name, as the browser does."""

    def __init__(self) -> None:
        self.answers: list[tuple[str, ApprovalDecision]] = []

    async def start(self) -> list[ProvidedTool]:
        return [
            ProvidedTool(
                definition=ToolDef(
                    name="browser_step",
                    description="Do one step of the check-in in the browser.",
                    parameters={
                        "type": "object",
                        "properties": {"step": {"type": "string"}},
                        "required": ["step"],
                    },
                    annotations={"readOnlyHint": True},
                ),
                function=self._step,
            )
        ]

    async def _step(self, step: str) -> str:
        ask = TOOL_ASK.get()
        assert ask is not None
        name, arguments, preview = STEPS[step]
        decision = await ask(name, dict(arguments), preview)
        self.answers.append((step, decision))
        return "done" if decision.allow else decision.reason

    async def close(self) -> None:
        pass

    def allowed(self) -> dict[str, bool]:
        return {step: decision.allow for step, decision in self.answers}


def _agent(llm: FakeModel, root: Path, tool: Steps) -> Agent:
    root.mkdir(exist_ok=True)
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset(),
            workspace_root=str(root),
            steps_per_attempt=16,
        ),
        tool_providers=[tool],
    )
    agent.chat.sleep = _no_wait
    return agent


def _steps(*names: str, answer: str = "done") -> list[Message]:
    calls = [
        _calls(("browser_step", {"step": name}, f"c{n}"))
        for n, name in enumerate(names, 1)
    ]
    return [*calls, _answer(answer)]


def _review_text(llm: FakeModel, index: int) -> str:
    return llm.review_requests[index][1].content


class Nobody:
    """Nobody has the chat open: Auto cannot show its card."""

    async def watched(self) -> bool:
        return False

    async def ask(self, request: ApprovalRequest, asking: Any) -> ApprovalDecision:
        raise AssertionError("nobody watches: no card is shown")


# ── The check-in runs in Auto with no card ──


@pytest.mark.asyncio
async def test_the_check_in_runs_in_auto_without_a_card_over_its_eight_messages(
    tmp_path: Path,
) -> None:
    tool = Steps()
    turns: list[tuple[str, list[str], list[Message]]] = [
        (CHECK_IN, ["open"], [ON_GOAL]),
        (RECALLED, ["sign_in"], [SIGN_IN_AS_KEYS]),
        ("continue", ["quick_login"], [SIGN_IN_AS_KEYS]),
        ("signed in", ["mood"], [NOT_NAMED]),
        ("go on", ["line", "submit"], [ON_GOAL, NOT_NAMED]),
        ("the time looks wrong, check in again", ["open"], []),
        ("just try it", ["mood"], []),
        ("1", ["result"], []),
    ]
    plans = [m for _, names, _ in turns for m in _steps(*names)]
    reviews = [r for _, _, replies in turns for r in replies]
    llm = FakeModel(plans, reviews)
    agent = _agent(llm, tmp_path / "ws", tool)
    card = Card()
    policy = ChatPolicy(card, "auto")
    recorder = Recorder()

    for words, _, _ in turns:
        notes = NOTES if words == RECALLED else ""  # as a runner gives them
        await agent.run_task(
            words,
            own_chars=len(words) - len(notes),
            recalled=notes,
            on_event=recorder,
            approval=policy,
        )

    assert card.asked == []
    assert all(decision.allow for _, decision in tool.answers), tool.answers
    assert [step for step, _ in tool.answers] == [
        "open",
        "sign_in",
        "quick_login",
        "mood",
        "line",
        "submit",
        "open",
        "mood",
        "result",
    ]
    for finished in recorder.of("run.finished"):
        assert finished.data["reviews"]["blocked"] == []
    # A step judged once is not judged again in the chat.
    assert len(llm.review_requests) == len(reviews)


@pytest.mark.asyncio
async def test_a_sign_in_click_is_no_credential_use(tmp_path: Path) -> None:
    """The real reply: "uses passwords or keys" for a click on a sign-in link."""
    tool = Steps()
    llm = FakeModel(_steps("sign_in"), [SIGN_IN_AS_KEYS])
    agent = _agent(llm, tmp_path / "ws", tool)
    card = Card()

    await agent.run_task(THAT_WAY, approval=ChatPolicy(card, "auto"))

    assert tool.allowed() == {"sign_in": True} and card.asked == []
    # What goes in a box is another matter: a secret typed stays blocked.
    typing = Steps()
    llm = FakeModel(_steps("password"), [SIGN_IN_AS_KEYS])
    agent = _agent(llm, tmp_path / "ws2", typing)
    card = Card()
    recorder = Recorder()

    await agent.run_task(CHECK_IN, on_event=recorder, approval=ChatPolicy(card, "auto"))

    assert typing.allowed() == {"password": False} and card.asked == []
    note = typing.answers[0][1].reason
    assert note.startswith("Auto mode blocked this step, so it did not run.")
    # melo's own words for the model's finding read as a sentence.
    assert "It uses passwords or keys, as your model's review says." in note
    assert "It your model's review says" not in note


@pytest.mark.asyncio
async def test_a_failing_review_refuses_the_question_and_keeps_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def down(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the review is down")

    monkeypatch.setattr(StepReviewer, "review", down)
    tool = Steps()
    agent = _agent(FakeModel(_steps("mood"), []), tmp_path / "ws", tool)
    card = Card()
    recorder = Recorder()

    result = await agent.run_task(
        CHECK_IN, on_event=recorder, approval=ChatPolicy(card, "auto")
    )

    # The question is refused, nobody is asked, and the run goes on.
    assert (result.status, tool.answers, card.asked) == ("completed", [], [])
    [finished] = recorder.of("tool.finished")
    [kept] = finished.data["tool_asks"]
    assert (kept["tool"], kept["allowed"], kept["origin"], kept["reason"]) == (
        STEPS["mood"][0],
        False,
        "error",
        "RuntimeError: the review is down",
    )


@pytest.mark.asyncio
async def test_a_step_the_review_finds_on_the_goal_runs_whatever_the_words_name(
    tmp_path: Path,
) -> None:
    """The person's words only ever add: they never stop a step the goal needs."""
    tool = Steps()
    llm = FakeModel(_steps("mood"), [NOT_NAMED])
    agent = _agent(llm, tmp_path / "ws", tool)
    card = Card()

    await agent.run_task("signed in", approval=ChatPolicy(card, "auto"))

    assert tool.allowed() == {"mood": True} and card.asked == []
    # Off the goal, it still asks.
    off = Steps()
    llm = FakeModel(
        _steps("mood"), [verdict(verdict="ask", category="none", fits_request="no")]
    )
    agent = _agent(llm, tmp_path / "ws2", off)
    card = Card(ApprovalDecision(False))

    await agent.run_task("what is new on the forum?", approval=ChatPolicy(card, "auto"))

    assert off.allowed() == {"mood": False} and len(card.asked) == 1
    [asked] = card.asked
    assert asked.review is not None
    assert asked.review.categories == ("beyond_request",)


# ── What the review reads: the whole chat, and the notes the person points to ──


@pytest.mark.asyncio
async def test_the_review_reads_the_chat_s_first_message_after_many_more(
    tmp_path: Path,
) -> None:
    tool = Steps()
    words = [CHECK_IN, "continue", "signed in", "go on", "hm", "ok"]
    plans = [m for text in words[:-1] for m in [_answer(text)]] + _steps("mood")
    llm = FakeModel(plans, [ON_GOAL])
    agent = _agent(llm, tmp_path / "ws", tool)
    policy = ChatPolicy(Card(), "auto")

    for text in words:
        await agent.run_task(text, approval=policy)

    text = _review_text(llm, 0)
    # The chat's first message is there though five more came after it, and
    # the newest three are the recent requests, oldest first.
    assert CHECK_IN in text
    assert (
        text.index(CHECK_IN)
        < text.index("\ngo on\n")
        < text.index("\nhm\n")
        < text.index("\nok\n")
    )
    assert agent._reviewer.requests == ("go on", "hm", "ok")


@pytest.mark.asyncio
async def test_the_review_reads_the_recalled_way_the_person_points_to(
    tmp_path: Path,
) -> None:
    tool = Steps()
    llm = FakeModel(_steps("sign_in"), [ON_GOAL])
    agent = _agent(llm, tmp_path / "ws", tool)

    await agent.run_task(
        RECALLED,
        user_words=THAT_WAY,
        own_chars=len(THAT_WAY),
        recalled=NOTES,
        approval=ChatPolicy(Card(), "auto"),
    )

    text = _review_text(llm, 0)
    assert "melo's notes recalled for the newest request" in text
    assert "wait for the quick login button and click it" in text
    # The notes are a block of their own, never the person's words.
    assert text.index("<recalled") > text.index(THAT_WAY)
    assert agent._reviewer.requests == (THAT_WAY,)


@pytest.mark.asyncio
async def test_a_consent_reads_only_the_newest_words(tmp_path: Path) -> None:
    """A high-impact step the words may allow keeps the window it had."""
    from localmelo.tests.agent.test_auto_mode import VENV, _plans
    from localmelo.tests.agent.test_auto_mode import _agent as python_agent

    llm = FakeModel(
        [_answer("a"), _answer("b"), _answer("c"), *_plans([VENV], answer="d")],
        [verdict(category="install", user_asked="yes")],
    )
    agent, _ = python_agent(llm, tmp_path / "ws")
    policy = AutoPolicy()
    for text in (
        "chart the matplotlib data",
        "two",
        "three",
        "chart it with matplotlib",
    ):
        await agent.run_task(text, approval=policy)

    text = _review_text(llm, 0)
    assert "chart the matplotlib data" not in text and "<recalled" not in text


# ── Steps off the goal or of a high-impact kind still stop or ask ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("step", "reply", "kind"),
    [
        # Changing the account's email: an account's sign-in settings.
        (
            "change_email",
            verdict(verdict="ask", category="account", fits_request="no"),
            "account",
        ),
        # Posting in another thread: a message to other people.
        (
            "post_elsewhere",
            verdict(verdict="ask", category="send_out", fits_request="no"),
            "send_out",
        ),
        # Paying.
        ("pay", verdict(verdict="ask", category="money", fits_request="no"), "money"),
    ],
)
async def test_a_high_impact_variant_asks_on_the_card(
    tmp_path: Path, step: str, reply: Message, kind: str
) -> None:
    tool = Steps()
    llm = FakeModel(_steps(step), [reply])
    agent = _agent(llm, tmp_path / "ws", tool)
    card = Card(ApprovalDecision(False))

    await agent.run_task(CHECK_IN, approval=ChatPolicy(card, "auto"))

    assert tool.allowed() == {step: False}
    [asked] = card.asked
    assert asked.review is not None and asked.review.categories == (kind,)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["money", "account"])
async def test_paying_or_an_account_change_asks_even_when_asked_for_and_on_the_goal(
    tmp_path: Path, kind: str
) -> None:
    """Never words, the goal or a grant of the chat: only a person's answer."""
    tool = Steps()
    asked_for = verdict(verdict="routine", category=kind, user_asked="yes")
    llm = FakeModel(_steps("pay"), [asked_for])
    agent = _agent(llm, tmp_path / "ws", tool)
    card = Card(ApprovalDecision(True, remember=True))
    policy = ChatPolicy(card, "auto")
    # Clicking on the forum was allowed for the chat before.
    policy.ask.grants.grant_now(f"browser-act:https://{FORUM}")

    await agent.run_task(f"buy premium on {FORUM}", approval=policy)

    [asked] = card.asked
    assert asked.review is not None and asked.review.categories == (kind,)
    # The person's Allow ran it once; it granted nothing new.
    assert tool.allowed() == {"pay": True}
    assert policy.allowed_tools == frozenset({f"browser-act:https://{FORUM}"})


@pytest.mark.asyncio
async def test_a_click_on_a_named_site_is_judged_by_what_it_is_on(
    tmp_path: Path,
) -> None:
    """Naming the forum lets melo open it, not pay on it unreviewed."""
    tool = Steps()
    llm = FakeModel(
        _steps("open", "pay"),
        [verdict(verdict="ask", category="money", fits_request="no")],
    )
    agent = _agent(llm, tmp_path / "ws", tool)
    card = Card(ApprovalDecision(False))

    await agent.run_task(
        f"check in on {FORUM} for me", approval=ChatPolicy(card, "auto")
    )

    # Opening the named site needs no review; the click on "Pay" does.
    assert tool.allowed() == {"open": True, "pay": False}
    [request] = llm.review_requests
    assert "Pay $10 for premium" in request[1].content
    [asked] = card.asked
    assert asked.review is not None and asked.review.categories == ("money",)


@pytest.mark.asyncio
async def test_a_step_nobody_can_be_asked_about_is_blocked_and_needs_the_person(
    tmp_path: Path,
) -> None:
    tool = Steps()
    llm = FakeModel(
        _steps("pay", answer="explained"),
        [verdict(verdict="ask", category="money", fits_request="no")],
    )
    agent = _agent(llm, tmp_path / "ws", tool)
    recorder = Recorder()

    await agent.run_task(
        CHECK_IN,
        on_event=recorder,
        approval=ChatPolicy(Card(), "auto", watch=Nobody()),
    )

    assert tool.allowed() == {"pay": False}
    note = tool.answers[0][1].reason
    # The note says the step needs the person, and never to ask for typed words.
    assert "this step needs them and nobody could be asked now" in note
    assert "keep the chat open so that melo asks them on a card" in note
    assert "Never ask the user to type a permission" in note


@pytest.mark.asyncio
async def test_don_t_allow_on_one_button_leaves_the_check_in_button_alone(
    tmp_path: Path,
) -> None:
    tool = Steps()
    llm = FakeModel(
        _steps("pay", "submit"),
        [verdict(verdict="ask", category="money", fits_request="no"), ON_GOAL],
    )
    agent = _agent(llm, tmp_path / "ws", tool)
    card = Card(ApprovalDecision(False, "not that"))

    await agent.run_task(CHECK_IN, approval=ChatPolicy(card, "auto"))

    assert tool.allowed() == {"pay": False, "submit": True}
    assert len(card.asked) == 1


# ── A point melo cannot tell, and a site Auto let the chat use ──


@pytest.mark.asyncio
async def test_a_click_melo_cannot_tell_asks_the_person_without_a_review(
    tmp_path: Path,
) -> None:
    tool = Steps()
    llm = FakeModel(_steps("blind"))
    agent = _agent(llm, tmp_path / "ws", tool)
    card = Card(ApprovalDecision(True))

    await agent.run_task(THAT_WAY, approval=ChatPolicy(card, "auto"))

    assert tool.allowed() == {"blind": True} and llm.review_requests == []
    [asked] = card.asked
    assert asked.review is not None and asked.review.categories == ("opaque",)


@pytest.mark.asyncio
async def test_a_site_auto_let_the_chat_read_stays_allowed_in_ask(
    tmp_path: Path,
) -> None:
    tool = Steps()
    llm = FakeModel([*_steps("open"), *_steps("open"), *_steps("open")], [ON_GOAL])
    agent = _agent(llm, tmp_path / "ws", tool)
    card = Card(ApprovalDecision(False))
    policy = ChatPolicy(card, "auto")
    recorder = Recorder()

    await agent.run_task(CHECK_IN, approval=policy)
    policy.mode = "ask"  # the person switched the chat to Ask
    await agent.run_task("continue", on_event=recorder, approval=policy)

    # The read Auto let run in turn 1 is not asked again.
    assert card.asked == [] and tool.allowed() == {"open": True}
    assert len(tool.answers) == 2 and tool.answers[1][1].allow
    assert tool.answers[1][1].origin == "session"
    assert tool.answers[1][1].reason == PASSED_IN_AUTO
    # It is listed with the chat's grants, and Remove takes it back.
    name = STEPS["open"][0]
    assert name in policy.allowed_tools
    policy.forget(name)
    await agent.run_task("and again", approval=policy)
    assert len(card.asked) == 1 and not tool.answers[2][1].allow


@pytest.mark.asyncio
async def test_a_site_auto_let_run_never_covers_a_step_auto_stops(
    tmp_path: Path,
) -> None:
    tool = Steps()
    llm = FakeModel(
        [*_steps("mood"), *_steps("pay")],
        [ON_GOAL, verdict(verdict="ask", category="none", fits_request="no")],
    )
    agent = _agent(llm, tmp_path / "ws", tool)
    card = Card(ApprovalDecision(False))
    policy = ChatPolicy(card, "auto")

    await agent.run_task(CHECK_IN, approval=policy)
    await agent.run_task("and buy premium", approval=policy)

    assert tool.allowed() == {"mood": True, "pay": False}
    [asked] = card.asked
    assert asked.review is not None and asked.review.categories == ("beyond_request",)


def test_the_window_keeps_the_chat_s_first_message_and_the_newest_ones() -> None:
    reviewer = StepReviewer(lambda: None)  # type: ignore[arg-type,return-value]
    words = [CHECK_IN, *(f"message {n}" for n in range(2, 13))]
    for text in words:
        reviewer.user_message(text)
    assert reviewer.requests == ("message 10", "message 11", "message 12")
    assert reviewer.earlier[0] == CHECK_IN
    shown = user_message(
        ReviewInputs(
            workspace="/ws",
            requests=reviewer.requests,
            tool="browser-act:https://x",
            action="{}",
            earlier=reviewer.earlier,
        ),
        nonce="n0",
    )
    asked = [
        line
        for line in shown.split("\n")
        if line and not line.startswith(("<", "Workspace", "The action", "What", "{"))
    ]
    # The first, then the newest older ones, then the recent requests.
    assert asked == [
        CHECK_IN,
        *(f"message {n}" for n in range(5, 13)),
    ]
    assert len(asked) == CHAT_REQUESTS + EARLIER_REQUESTS + 1


def test_the_recalled_notes_are_the_newest_message_s() -> None:
    reviewer = StepReviewer(lambda: None)  # type: ignore[arg-type,return-value]
    reviewer.user_message(THAT_WAY, recalled=NOTES)
    assert reviewer.requests == (THAT_WAY,)
    assert reviewer._recalled == NOTES
    # A message read mid-run says nothing of notes: those of the run stay.
    reviewer.user_message("go on")
    assert reviewer._recalled == NOTES
    # A new run's message says which notes came with it, none included.
    reviewer.user_message("next", recalled="")
    assert reviewer._recalled == ""


@pytest.mark.asyncio
async def test_after_a_reload_the_review_still_reads_the_chat_s_first_task(
    tmp_path: Path,
) -> None:
    tool = Steps()
    llm = FakeModel(_steps("mood"), [ON_GOAL])
    agent = _agent(llm, tmp_path / "ws", tool)
    agent.load_conversation([])  # the chat opened again: the window starts over

    await agent.run_task(
        "signed in",
        earlier_turns=[("continue", "1"), ("go on", "2"), ("hm", "3")],
        chat_first=CHECK_IN,
        approval=ChatPolicy(Card(), "auto"),
    )

    assert tool.allowed() == {"mood": True}
    text = _review_text(llm, 0)
    assert CHECK_IN in text and text.index(CHECK_IN) < text.index("\ngo on\n")
    assert agent._reviewer.earlier == (CHECK_IN,)
    # The recent requests are as before: the newest two earlier words and this one.
    assert agent._reviewer.requests == ("go on", "hm", "signed in")


def test_the_first_task_is_not_put_in_twice() -> None:
    reviewer = StepReviewer(lambda: None)  # type: ignore[arg-type,return-value]
    reviewer.earlier_words([CHECK_IN, "continue"], first=CHECK_IN)
    reviewer.user_message("signed in")
    assert reviewer.requests == (CHECK_IN, "continue", "signed in")
    assert reviewer.earlier == ()


def test_a_blank_earlier_word_takes_no_place_among_the_newest() -> None:
    reviewer = StepReviewer(lambda: None)  # type: ignore[arg-type,return-value]
    reviewer.earlier_words(["one", "two", ""])  # the third is a go-ahead turn's
    reviewer.user_message("now")
    assert reviewer.requests == ("one", "two", "now")


@pytest.mark.asyncio
async def test_a_first_task_that_is_not_text_is_refused(tmp_path: Path) -> None:
    llm = FakeModel([], [])
    agent = _agent(llm, tmp_path / "ws", Steps())
    with pytest.raises(TypeError, match="chat_first must be a str"):
        await agent.run_task("signed in", chat_first=b"check in")  # type: ignore[arg-type]
    assert llm.requests == [] and llm.review_requests == []


# ── What an allowance from Auto covers in Ask, and what counts as a secret ──


@pytest.mark.asyncio
async def test_a_click_auto_let_run_is_asked_about_again_in_ask(
    tmp_path: Path,
) -> None:
    """In Ask nobody reviews a click, so none rides Auto's run: "Pay" asks."""
    tool = Steps()
    llm = FakeModel([*_steps("mood"), *_steps("pay"), *_steps("mood")], [ON_GOAL])
    agent = _agent(llm, tmp_path / "ws", tool)
    card = Card(ApprovalDecision(False), ApprovalDecision(True))
    policy = ChatPolicy(card, "auto")

    await agent.run_task(CHECK_IN, approval=policy)
    policy.mode = "ask"  # the person switched the chat to Ask
    await agent.run_task("and buy premium", approval=policy)
    await agent.run_task("pick happy again", approval=policy)

    assert [(step, d.allow) for step, d in tool.answers] == [
        ("mood", True),
        ("pay", False),
        ("mood", True),
    ]
    act = STEPS["mood"][0]
    assert [r.tool_name for r in card.asked] == [act, act]
    assert all(r.review is None for r in card.asked)
    assert policy.allowed_tools == frozenset()


def _browse(name: str, **arguments: str) -> Message:
    call = ToolCall(tool_name=name, arguments=arguments)
    return Message(role="assistant", tool_call=call)


@pytest.mark.asyncio
async def test_a_buy_button_on_a_site_auto_used_does_nothing_in_ask_until_allowed(
    tmp_path: Path,
) -> None:
    """Through the browser: the read of the site stays allowed, the click asks."""
    home = "https://example.com/"
    driver = FakeDriver(
        page(home, "Welcome.", button("Happy"), button("Buy premium"), title="Shop")
    )
    llm = FakeModel(
        [
            _browse("browser_open", url=home),
            _browse("browser_click", ref="e1"),
            _answer("done"),
            _browse("browser_open", url=home),
            _browse("browser_click", ref="e4"),  # Happy showed no change: no reading
            _answer("not bought"),
        ],
        [ON_GOAL],
    )
    root = tmp_path / "ws"
    root.mkdir()
    agent = Agent(
        llm=llm,
        options=AgentOptions(builtins=frozenset(), workspace_root=str(root)),
        tool_providers=[BrowserToolProvider(driver)],
    )
    agent.chat.sleep = _no_wait
    card = Card(ApprovalDecision(False))
    policy = ChatPolicy(card, "auto")
    try:
        await agent.run_task("pick happy on example.com", approval=policy)
        assert [c for c in driver.calls if c[0] == "act"] == [("act", "click:Happy")]
        policy.mode = "ask"
        await agent.run_task("and buy premium", approval=policy)
    finally:
        await agent.close()

    # The read was not asked again; the click on "Buy premium" was, and did nothing.
    [asked] = card.asked
    assert asked.tool_name == "browser-act:https://example.com"
    assert asked.arguments["on"] == 'a button "Buy premium"'
    assert [c for c in driver.calls if c[0] == "act"] == [("act", "click:Happy")]
    assert policy.allowed_tools == frozenset({"browser:https://example.com"})


def test_only_a_plain_read_stays_allowed_from_auto() -> None:
    from localmelo.melo.contracts.approvals import ApprovalRequest, stays_for_chat
    from localmelo.melo.contracts.review import ActionReview

    def request(q: Q, review: ActionReview | None = None, **more: str) -> Any:
        name, arguments, preview = q
        return ApprovalRequest(
            request_id="r",
            run_id="t",
            call_id="c",
            tool_name=name,
            arguments={**arguments, **more},
            preview=preview,
            workspace_root=None,
            tier="network",
            review=review,
        )

    assert stays_for_chat(request(STEPS["open"]))
    plain = f"https://{FORUM}/today"
    assert stays_for_chat(request(STEPS["open"], url=plain, via="web_fetch"))
    for url in (f"{plain}?token=abc", f"{plain}#k", f"https://a@{FORUM}/", plain + "…"):
        assert not stays_for_chat(request(STEPS["open"], url=url, via="web_fetch"))
    for step in ("mood", "line", "submit", "change_email", "blind"):
        assert not stays_for_chat(request(STEPS[step]))
    for kind in ("money", "account"):
        review = ActionReview("routine", (kind,), ())
        assert not stays_for_chat(request(STEPS["open"], review))
    # The asking policy keeps only a site's read, whatever it is told.
    policy = RememberingPolicy(Card())
    policy.passed_in_auto(STEPS["mood"][0])
    policy.passed_in_auto(STEPS["open"][0])
    assert policy.allowed_tools == frozenset({STEPS["open"][0]})


@pytest.mark.asyncio
async def test_a_click_whose_words_hold_a_key_keeps_the_credential_finding(
    tmp_path: Path,
) -> None:
    tool = Steps()
    llm = FakeModel(_steps("key_link"), [SIGN_IN_AS_KEYS])
    agent = _agent(llm, tmp_path / "ws", tool)
    card = Card(ApprovalDecision(False))

    await agent.run_task(CHECK_IN, approval=ChatPolicy(card, "auto"))

    # The finding stands: a credential use is blocked, as typing a password is.
    assert tool.allowed() == {"key_link": False} and card.asked == []
    assert "It uses passwords or keys" in tool.answers[0][1].reason


def test_melo_s_detectors_find_secrets_not_the_words_for_them() -> None:
    key = "sk-" + "a1B2" * 6
    for found in (
        {"on": f'a link "Use key {key}"'},
        {"url": "https://files.example/report?access_token=abc123"},
        {"url": "https://user:hunter2@files.example/"},
        {"fields": [["auth", "Bearer abcdefghijklmnop1234"]]},
    ):
        assert holds_secret(found), found
    assert holds_secret({}, f"open {key}")
    for plain in (
        {"on": 'a link "Chat app login"'},
        {"on": 'a button "Quick login"'},
        {"on": 'a textbox "Password"'},
        {"url": "https://files.example/report"},
    ):
        assert not holds_secret(plain, "click it"), plain
