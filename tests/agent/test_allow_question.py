"""A question a tool asked itself that Auto refused, as a blocked step Allow once can run (#617).

A task's Auto with nobody to ask: a typing question the review does not find
routine is blocked. The record keeps what the gate's own block keeps, and the
key of Allow once only when a chat's card would have asked about the step.
``browser_type`` asks through the browser's own ``require_type``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import (
    ApprovalRequest,
    AutoPolicy,
    EarlierBlocks,
    allowance_key,
    block_note,
)
from localmelo.melo.contracts.review import ActionReview
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.browser import sites
from localmelo.support.browser.driver import BrowserError
from localmelo.support.gateway.auto_blocks import allowable
from localmelo.tests.agent.test_approvals import Recorder, _answer, _calls, _no_wait
from localmelo.tests.agent.test_auto_mode import FakeModel, verdict

SITE = "https://notes.example"
NOTE = "Checked in for today, all is well here."
# The review of a typing step the request does not ask for: it sends data out.
SENDS = verdict(verdict="ask", category="send_out", reason="Posts a note.")
# A text with a key in it, by melo's own detectors (test data, not a real key).
SECRET = "Checked in. token=abcdef0123456789abcdef0123456789"


class Types(BaseToolProvider):
    """``browser_type`` as the browser asks it: the question, then the text typed."""

    def __init__(self) -> None:
        self.typed: list[str] = []

    async def start(self) -> list[ProvidedTool]:
        return [
            ProvidedTool(
                definition=ToolDef(
                    name="browser_type",
                    description="Type into a text box.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "site": {"type": "string"},
                            "kind": {"type": "string"},
                        },
                        "required": ["text"],
                    },
                ),
                function=self._type,
                asks_itself=True,
            )
        ]

    async def _type(self, text: str, site: str = SITE, kind: str = "type") -> str:
        """Ask as the browser asks typing (*kind* ``type``), a form sent, a click or a read."""
        asks = {
            "type": lambda: sites.require_type(site, text, 'a textbox "Note"'),
            "submit": lambda: sites.require_submit(
                site, f"{site}/post", "click", [("note", text)]
            ),
            "act": lambda: sites.require_act(site, "click", 'a button "Post"'),
            "read": lambda: sites.require_site(site),
        }
        try:
            await asks[kind]()
        except BrowserError as error:
            return str(error)
        self.typed.append(text)
        return "typed"

    async def close(self) -> None:
        pass


def typing(text: str = NOTE, call_id: str = "call_1", site: str = SITE) -> Message:
    return _calls(("browser_type", {"text": text, "site": site}, call_id))


def task_agent(llm: FakeModel, root: Path, types: Types) -> Agent:
    root.mkdir(exist_ok=True)
    agent = Agent(
        llm=llm,
        options=AgentOptions(builtins=frozenset(), workspace_root=str(root)),
        tool_providers=[types],
    )
    agent.chat.sleep = _no_wait
    return agent


async def kept_ask(
    tmp_path: Path,
    policy: Any,
    reviews: list[Any],
    plans: list[Message] | None = None,
    read: list[str] | None = None,
) -> dict[str, Any]:
    """The kept record of the run's one refused typing question; *read* gets what the model read of it."""
    llm = FakeModel(plans or [typing(), _answer("It needs the note.")], reviews)
    recorder = Recorder()
    await task_agent(llm, tmp_path / "ws", Types()).run_task(
        "do the daily check-in", on_event=recorder, approval=policy
    )
    [kept] = recorder.of("tool.finished")[0].data["tool_asks"]
    if read is not None:
        read.extend(llm.tool_messages())
    return dict(kept)


def _refused_site(inner: AutoPolicy) -> EarlierBlocks:
    """A task its chat continued, where the site was refused earlier."""
    return EarlierBlocks(
        inner, {"k0": ("Auto mode", "")}, {"k0": ["ask:site:notes.example"]}
    )


# ── Which refused questions Allow once may run (rules 5 and 6) ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "reviews", "refused", "can_allow"),
    [
        # A chat's card would have asked: sending data the request did not ask for.
        ("sends data out", [SENDS], False, True),
        # A review that failed is the person's to decide on a card too.
        ("review failed", ["not json", "not json"], False, True),
        # Only a click allows passwords and keys, and a chat shows no card for
        # one the person's words do not name: Auto blocks it there too.
        (
            "credentials",
            [verdict(verdict="ask", category="credentials")],
            False,
            False,
        ),
        # A site refused earlier in this task stands over a later step on it.
        ("refused site", [SENDS], True, False),
        # A text melo's detectors find a key in keeps no text, whatever the review says.
        ("holds a key", [SENDS], False, False),
    ],
)
async def test_a_refused_question_is_allowable_only_where_a_chat_would_ask(
    tmp_path: Path, case: str, reviews: list[Any], refused: bool, can_allow: bool
) -> None:
    auto = AutoPolicy(task=True)
    reviews = [
        Message(role="assistant", content=r) if isinstance(r, str) else r
        for r in reviews
    ]
    plans = [typing(SECRET), _answer("Not typed.")] if case == "holds a key" else None
    kept = await kept_ask(
        tmp_path, _refused_site(auto) if refused else auto, reviews, plans
    )

    assert (kept["origin"], kept["allowed"], kept["mode"]) == ("auto", False, "auto")
    # The whole text only with the key: a block nobody can allow keeps none.
    assert kept.get("arguments", {}).get("text") == (NOTE if can_allow else None)
    assert SECRET not in repr(kept)
    assert ("step_key" in kept) is can_allow, case
    assert (allowable(kept) is not None) is can_allow, case


@pytest.mark.parametrize(
    ("case", "record"),
    [
        # A record from before this change: no review, no key.
        (
            "old record",
            {"origin": "auto", "allowed": False, "tool": "browser-type:x", "call_key": "c"},
        ),
        ("no key", {"origin": "auto", "allowed": False, "review": {"categories": ["send_out"]}}),
        ("unreadable review", {"origin": "auto", "allowed": False, "step_key": "k", "review": "x"}),
        ("unknown kind", {"origin": "auto", "allowed": False, "step_key": "k", "review": {"categories": ["made_up"]}}),
        ("a question without its text", {"origin": "auto", "allowed": False, "tool": "browser-type:x", "step_key": "k", "review": {"categories": ["send_out"]}}),
        ("the person's own no", {"origin": "user", "allowed": False, "step_key": "k", "review": {"categories": ["send_out"]}}),
        ("ran already", {"origin": "user", "allowed": True, "allowance": "once", "step_key": "k"}),
    ],
)  # fmt: skip
def test_a_record_missing_or_off_anything_is_not_allowable(
    case: str, record: dict[str, Any]
) -> None:
    assert allowable(record) is None, case


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "can_allow"),
    [
        # The card shows all of it and an allowed retry runs: typing, a form sent.
        ("type", True),
        ("submit", True),
        # The card shows only a preview of what a click or a page does.
        ("act", False),
        ("read", False),
    ],
)
async def test_only_a_question_that_holds_its_whole_step_gets_allow_once(
    tmp_path: Path, kind: str, can_allow: bool
) -> None:
    blocked = verdict(verdict="ask", category="beyond_request", fits_request="no")
    plans = [
        _calls(("browser_type", {"text": NOTE, "kind": kind}, "call_1")),
        _answer("Not done."),
    ]
    read: list[str] = []
    kept = await kept_ask(tmp_path, AutoPolicy(task=True), [blocked], plans, read)

    assert ("step_key" in kept, "arguments" in kept) == (can_allow, can_allow), kind
    # The model reads what is true: the person can allow it on the task's page, or
    # cannot there (a task nobody can be asked for, #619).
    [note] = read
    assert ("can allow it there" in note) is can_allow
    assert ("cannot be allowed from the task" in note) is not can_allow
    # In a chat too: Allow once only for a block that has its key.
    review = ActionReview("high", ("beyond_request",), ("Posts a note.",))
    chat_note = block_note(review, allowable=can_allow)
    assert ("allow this step once from the chat" in chat_note) is can_allow


# ── One exact step, once (rule 3 and 4) ──


def _question(**changes: Any) -> ApprovalRequest:
    arguments = {"site": "notes.example", "access": "type", "how": "type", "text": NOTE}
    arguments.update(changes.pop("arguments", {}))
    return ApprovalRequest(
        request_id="r",
        run_id="t",
        call_id="c",
        tool_name=changes.pop("tool_name", f"browser-type:{SITE}"),
        arguments=arguments,
        preview="type",
        workspace_root="/w",
        tier="network",
        review=ActionReview("high", ("send_out",)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "retry"),
    [
        ("other text", _question(arguments={"text": NOTE + "!"})),
        ("other site", _question(tool_name="browser-type:https://other.example")),
        ("other box", _question(arguments={"on": 'a textbox "Title"'})),
        ("a form sent", _question(tool_name=f"browser-submit:{SITE}")),
    ],
)
async def test_an_allowance_covers_its_exact_question_and_nothing_else(
    case: str, retry: ApprovalRequest
) -> None:
    auto = AutoPolicy(task=True)
    allowed = _question()
    auto.allow_once(allowance_key(allowed), ["send_out"], "browser_type")

    assert not (await auto.decide(retry)).allow, case
    assert (await auto.decide(allowed)).allow
    assert not (await auto.decide(allowed)).allow, "used once"


@pytest.mark.asyncio
async def test_the_allowed_text_runs_once_unreviewed_and_its_repeat_is_reviewed(
    tmp_path: Path,
) -> None:
    auto = AutoPolicy(task=True)
    kept = await kept_ask(tmp_path, auto, [SENDS])
    auto.allow_once(kept["step_key"], ["send_out"], "browser_type")
    other = "Checked in for today."
    llm = FakeModel(
        [
            typing(other, "call_2"),
            typing(NOTE, "call_3"),
            typing(NOTE, "call_4"),
            _answer("Done."),
        ],
        [SENDS, SENDS],
    )
    types, recorder = Types(), Recorder()

    await task_agent(llm, tmp_path / "ws", types).run_task(
        "go ahead", on_event=recorder, approval=auto
    )

    # Another text is reviewed and blocked, told it is not the step allowed;
    # the allowed one runs with no review; the same text again is reviewed.
    assert types.typed == [NOTE]
    assert len(llm.review_requests) == 2
    first, ran, again = (e.data for e in recorder.of("tool.finished"))
    assert "The user allowed one exact step; this one differs." in first["output"]
    assert ran["tool_asks"] == [
        {
            "request_id": ran["tool_asks"][0]["request_id"],
            "tool": f"browser-type:{SITE}",
            "allowed": True,
            "origin": "user",
            "allowance": "once",
            "step_key": kept["step_key"],
        }
    ]
    assert again["tool_asks"][0]["origin"] == "auto"
    assert not auto.allowed_once(kept["step_key"])


@pytest.mark.asyncio
async def test_a_continued_task_still_refuses_a_refused_question_not_allowed(
    tmp_path: Path,
) -> None:
    """``EarlierBlocks`` yields only to an allowance of that exact question."""
    auto = AutoPolicy(task=True)
    kept = await kept_ask(tmp_path, auto, [SENDS])
    earlier = EarlierBlocks(
        auto,
        {kept["call_key"]: ("Auto mode", kept["step_key"])},
        {kept["call_key"]: kept["loose_keys"]},
    )
    llm = FakeModel([typing(NOTE, "call_2"), _answer("Not done.")], [])
    types, recorder = Types(), Recorder()

    await task_agent(llm, tmp_path / "ws", types).run_task(
        "go on", on_event=recorder, approval=earlier
    )

    assert types.typed == []
    [refused] = recorder.of("tool.finished")[0].data["tool_asks"]
    assert refused["earlier"] == "Auto mode"
    # Allowed once, the same question runs, unreviewed.
    auto.allow_once(kept["step_key"], ["send_out"], "browser_type")
    llm = FakeModel([typing(NOTE, "call_3"), _answer("Done.")], [])
    await task_agent(llm, tmp_path / "ws", types).run_task("go ahead", approval=earlier)
    assert types.typed == [NOTE]


@pytest.mark.asyncio
async def test_an_allowance_on_a_site_refused_since_lifts_nothing(
    tmp_path: Path,
) -> None:
    """Allow once runs one question; it never stands over a site refused in the task."""
    auto = AutoPolicy(task=True)
    kept = await kept_ask(tmp_path, auto, [SENDS])
    auto.allow_once(kept["step_key"], ["send_out"], "browser_type")
    llm = FakeModel(
        [typing(NOTE, "call_2"), typing("Short note.", "call_3"), _answer("Done.")],
        [verdict()],
    )
    types, recorder = Types(), Recorder()

    await task_agent(llm, tmp_path / "ws", types).run_task(
        "go ahead", on_event=recorder, approval=_refused_site(auto)
    )

    # Neither the allowed text nor a later one Auto found routine is typed there,
    # and nothing says the allowed step ran.
    assert types.typed == []
    asked = [a for e in recorder.of("tool.finished") for a in e.data["tool_asks"]]
    assert not any(a.get("allowance") for a in asked)


@pytest.mark.asyncio
async def test_an_allowed_text_runs_though_another_text_on_its_site_stays_refused(
    tmp_path: Path,
) -> None:
    """Two texts refused on one site; the person allows A: A runs once, B stays refused."""
    auto = AutoPolicy(task=True)
    other = "Checked in for today."
    llm = FakeModel(
        [typing(NOTE, "call_1"), typing(other, "call_2"), _answer("Blocked.")],
        [SENDS, SENDS],
    )
    recorder = Recorder()
    await task_agent(llm, tmp_path / "ws", Types()).run_task(
        "do the daily check-in", on_event=recorder, approval=auto
    )
    a, b = (e.data["tool_asks"][0] for e in recorder.of("tool.finished"))
    auto.allow_once(a["step_key"], ["send_out"], "browser_type")
    # The chat continues the task on its own: both blocks are refused before review.
    earlier = EarlierBlocks(
        auto,
        {k["call_key"]: ("Auto mode", k["step_key"]) for k in (a, b)},
        {k["call_key"]: k["loose_keys"] for k in (a, b)},
    )
    llm = FakeModel(
        [typing(NOTE, "call_3"), typing(other, "call_4"), _answer("Done.")], []
    )
    types = Types()

    await task_agent(llm, tmp_path / "ws", types).run_task("go on", approval=earlier)

    assert types.typed == [NOTE]
