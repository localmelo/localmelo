"""Answer selection and clipboard feedback use durable, redacted answers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pytest
from prompt_toolkit.output import DummyOutput

from localmelo.melo.schema import INTERRUPT_LABEL, NO_ANSWER_LABEL
from localmelo.support.cli import terminal
from localmelo.support.cli.output import TerminalOut

from .test_session import Setup
from .test_session import setup as setup
from .test_session_navigation import seed


@pytest.fixture
def copied(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    values: list[str] = []

    async def copy_text(text: str, control: Callable[[str], bool]) -> Literal["pbcopy"]:
        values.append(text)
        return "pbcopy"

    monkeypatch.setattr(terminal, "copy_text", copy_text)
    return values


@pytest.mark.parametrize("argument", ["0", "-1", "+1", "１", "1 2", "x", "9" * 5_000])
async def test_invalid_copy_reference_never_touches_the_clipboard(
    setup: Setup,
    copied: list[str],
    argument: str,
) -> None:
    with pytest.raises(ValueError, match=r"Use /copy \[turn\]"):
        await setup.session.session_tools.copy(argument)
    assert copied == []


async def test_no_answer_or_missing_turn_is_explicit(
    setup: Setup,
    copied: list[str],
) -> None:
    with pytest.raises(ValueError, match="No answer to copy yet"):
        await setup.session.session_tools.copy("")
    with pytest.raises(ValueError, match="Turn 9 has no answer"):
        await setup.session.session_tools.copy("9")
    assert copied == []


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "<think>only reasoning</think>",
        INTERRUPT_LABEL,
        NO_ANSWER_LABEL.format(status="cancelled"),
    ],
)
async def test_labels_and_empty_answers_are_not_copied(
    setup: Setup,
    copied: list[str],
    answer: str,
) -> None:
    sid = await seed(setup)
    info = await setup.store.begin_turn(sid, "no answer", holder="other")
    await setup.store.finish_turn(
        sid,
        info.turn,
        holder="other",
        status="cancelled",
        cause="user_stop",
        result=None,
        answer=answer,
    )
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    with pytest.raises(ValueError, match="Turn 2 has no answer"):
        await setup.session.session_tools.copy("")
    assert copied == []


async def test_default_skips_undone_but_an_explicit_turn_can_copy_it(
    setup: Setup,
    copied: list[str],
) -> None:
    sid = await seed(setup)
    info = await setup.store.begin_turn(sid, "later answer", holder="other")
    await setup.store.finish_turn(
        sid,
        info.turn,
        holder="other",
        status="completed",
        cause="run",
        result=None,
        answer="undone Markdown **kept**",
    )
    await setup.store.undo_last_turn(sid)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    await setup.session.session_tools.copy("")
    await setup.session.session_tools.copy("2")
    assert copied == ["stored **answer**", "undone Markdown **kept**"]


async def test_copy_during_a_turn_uses_the_last_finished_answer(
    setup: Setup,
    copied: list[str],
) -> None:
    sid = await seed(setup)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()
    await setup.store.begin_turn(sid, "running elsewhere", holder="other")
    await setup.session.session_tools.copy("")
    assert copied == ["stored **answer**"]
    with pytest.raises(ValueError, match="Turn 2 has no answer"):
        await setup.session.session_tools.copy("2")
    assert len(copied) == 1


async def test_osc52_feedback_claims_a_request_not_confirmed_copy(
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = await seed(setup)
    setup.session.enter(f"/resume {sid}")
    await setup.idle()

    async def request(text: str, control: Callable[[str], bool]) -> Literal["osc52"]:
        assert text == "stored **answer**"
        return "osc52"

    monkeypatch.setattr(terminal, "copy_text", request)
    await setup.session.session_tools.copy("")
    text = " ".join(setup.out.text().split())
    assert "Sent the answer of turn 1" in text
    assert "OSC 52; support varies" in text
    assert "Copied the answer" not in text


class ControlOutput(DummyOutput):
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.flushed = 0
        self.failure: Exception | None = None

    def write_raw(self, data: str) -> None:
        if self.failure is not None:
            raise self.failure
        self.writes.append(data)

    def flush(self) -> None:
        self.flushed += 1


def test_control_request_is_raw_zero_width_and_not_recorded_as_scroll_text() -> None:
    output = ControlOutput()
    out = TerminalOut(output, no_color=True)
    sequence = "\x1b]52;c;YW5zd2Vy\x07"
    assert out.write_control(sequence)
    assert output.writes == [sequence] and output.flushed == 1
    out.detach()
    assert not out.write_control(sequence)
    assert output.writes == [sequence]


@pytest.mark.parametrize("error", [OSError("closed"), ValueError("closed")])
def test_control_request_failure_is_reported(error: Exception) -> None:
    output = ControlOutput()
    output.failure = error
    assert not TerminalOut(output, no_color=True).write_control("\x1b]52;c;eA==\x07")
