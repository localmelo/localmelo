"""What a scheduled task's kept answer covers at the agent's gate (#619, rules K3, K5, K7, K8).

A run of a scheduled task in Auto with nobody to ask, its policy holding the
person's kept answer for one site (``AutoPolicy.kept``, set by trusted code).
Every step outside the rule is decided exactly as it is without the answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import (
    AutoPolicy,
    DenyAllPolicy,
    ScheduledKeep,
    block_note,
)
from localmelo.melo.contracts.review import ActionReview
from localmelo.melo.schema import Message
from localmelo.support.browser import sites
from localmelo.tests.agent.test_allow_question import (
    NOTE,
    SECRET,
    SENDS,
    SITE,
    Types,
    _refused_site,
    task_agent,
)
from localmelo.tests.agent.test_approvals import Recorder, _answer, _calls
from localmelo.tests.agent.test_auto_mode import FakeModel, verdict

OTHER = "https://other.example"


class Sends(Types):
    """:class:`Types`, and a form sent elsewhere (``away``), where the page decides
    (``page``), or typed then sent in one call (``both``)."""

    async def _type(self, text: str, site: str = SITE, kind: str = "type") -> str:
        if kind in ("away", "page"):
            to = f"{OTHER}/post" if kind == "away" else ""
            return await self._asked(
                text, sites.require_submit(site, to, "enter", [("n", text)])
            )
        if kind == "both":
            typed = await super()._type(text, site, "type")
            return (
                typed if typed != "typed" else await super()._type(text, site, "submit")
            )
        return await super()._type(text, site, kind)

    async def _asked(self, text: str, asked: Any) -> str:
        try:
            await asked
        except sites.BrowserError as error:
            return str(error)
        self.typed.append(text)
        return "typed"


def kept_policy(keep: bool = True, asking: Any = None) -> AutoPolicy:
    auto = AutoPolicy(task=True, asking=asking)
    if keep:
        auto.kept = ScheduledKeep("c1", {SITE: frozenset({"type", "submit"})})
    return auto


async def one_step(
    tmp_path: Path, auto: Any, reviews: list[Any], **call: str
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """What one call typed, the questions it recorded, and what the model read of it."""
    arguments = {"text": NOTE, **call}
    llm = FakeModel(
        [_calls(("browser_type", arguments, "call_1")), _answer("Done.")],
        [
            Message(role="assistant", content=r) if isinstance(r, str) else r
            for r in reviews
        ],
    )
    types, recorder = Sends(), Recorder()
    tmp_path.mkdir(parents=True, exist_ok=True)
    await task_agent(llm, tmp_path / "ws", types).run_task(
        "do the daily check-in", on_event=recorder, approval=auto
    )
    [finished] = recorder.of("tool.finished")
    return types.typed, finished.data["tool_asks"], llm.tool_messages()


def _same(asked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records without what differs between two runs of one step: ids, the
    keys that hold the run's folder (whether there is one stays), the review's time."""
    out = []
    for one in asked:
        same = {k: v for k, v in one.items() if k not in ("request_id", "call_key")}
        same["step_key"] = "step_key" in one
        review = same.get("review")
        if isinstance(review, dict) and isinstance(review.get("model"), dict):
            same["review"] = {**review, "model": {**review["model"], "ms": 0}}
        out.append(same)
    return out


# ── K3: covered only when every part holds; anything else as without it ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "reviews", "call", "policy"),
    [
        ("another site", [SENDS], {"site": OTHER}, "kept"),
        ("a form sent to another site", [SENDS], {"kind": "away"}, "kept"),
        ("a form whose target the page decides", [SENDS], {"kind": "page"}, "kept"),
        ("credentials", [verdict(verdict="ask", category="credentials")], {}, "kept"),
        ("not asked for", [verdict(verdict="ask", category="beyond_request", fits_request="no")], {}, "kept"),
        ("does not fit the task", [verdict(verdict="ask", category="send_out", fits_request="no")], {}, "kept"),
        ("could not be reviewed", ["not json", "not json"], {}, "kept"),
        ("a secret in the text", [SENDS], {"text": SECRET}, "kept"),
        ("a phone number in the text", [SENDS], {"text": "Call me at 415-555-0134 today."}, "kept"),
        ("a site refused in the task", [SENDS], {}, "refused"),
        ("someone can be asked (a chat)", [SENDS], {}, "asking"),
    ],
)  # fmt: skip
async def test_a_step_outside_the_kept_answer_is_decided_as_without_it(
    tmp_path: Path, case: str, reviews: list[Any], call: dict[str, str], policy: str
) -> None:
    def build(keep: bool) -> Any:
        if policy == "refused":
            return _refused_site(kept_policy(keep))
        return kept_policy(
            keep, asking=DenyAllPolicy("no") if policy == "asking" else None
        )

    typed, asked, read = await one_step(tmp_path / "a", build(True), reviews, **call)
    plain, plain_asked, plain_read = await one_step(
        tmp_path / "b", build(False), list(reviews), **call
    )

    assert typed == plain == [], case
    assert not any(a.get("allowance") for a in asked), case
    assert _same(asked) == _same(plain_asked), case
    assert read == plain_read, case


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["type", "submit"])
async def test_the_kept_answer_runs_its_step_and_the_run_says_so(
    tmp_path: Path, kind: str
) -> None:
    typed, asked, _ = await one_step(tmp_path, kept_policy(), [SENDS], kind=kind)

    assert typed == [NOTE]
    [used] = asked
    # K5: the run's record says what went where, on the kept answer; no block.
    assert (used["tool"], used["allowed"], used["allowance"]) == (
        f"browser-{kind}:{SITE}",
        True,
        "kept",
    )
    assert NOTE in repr(used["arguments"]) and "step_key" not in used


@pytest.mark.asyncio
async def test_a_kind_not_kept_on_the_site_is_not_covered(tmp_path: Path) -> None:
    auto = AutoPolicy(task=True)
    auto.kept = ScheduledKeep("c1", {SITE: frozenset({"type"})})

    typed, asked, _ = await one_step(tmp_path, auto, [SENDS], kind="submit")

    assert typed == [] and asked[0]["origin"] == "auto"


# ── K8: typing and sending in one call ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reviews", "runs"),
    [
        # Both questions pass: the text is typed and the form sent.
        ([SENDS, SENDS], True),
        # The send is filed under something else: the call stops at the send.
        ([SENDS, verdict(verdict="ask", category="credentials")], False),
    ],
)
async def test_a_call_that_types_then_sends_needs_both_questions_covered(
    tmp_path: Path, reviews: list[Any], runs: bool
) -> None:
    typed, asked, _ = await one_step(tmp_path, kept_policy(), reviews, kind="both")

    assert typed == ([NOTE, NOTE] if runs else [NOTE])
    assert asked[0]["allowance"] == "kept"
    assert (asked[1].get("allowance") == "kept") is runs


# ── K7: what the model reads for a block nobody can be asked about ──


@pytest.mark.asyncio
async def test_a_block_with_nobody_to_ask_tells_the_model_what_fits_that_run(
    tmp_path: Path,
) -> None:
    _, _, [note] = await one_step(tmp_path, kept_policy(keep=False), [SENDS])

    assert "Nobody can be asked during this run" in note
    assert "The user will see this step on the task's page" in note
    assert "Do not try this step again" in note and "Ending: blocked" in note
    assert "try the step again" not in note and "on a card" not in note
    # A task someone can be asked for, and a chat, read the notes they read before.
    review = ActionReview("high", ("send_out",), ("Posts a note.",), by="model")
    for note in (block_note(review, task=True), block_note(review)):
        assert "Nobody can be asked during this run" not in note
