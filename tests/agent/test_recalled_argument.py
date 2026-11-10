"""A run's recalled notes and the person's part of its text come as arguments.

``Agent.run_task`` takes ``recalled`` (the notes block the caller put in the
query) and ``own_chars`` (how many leading characters are the person's). The
step review's notes are that string and nothing else, and the person's words
are that part of the text: a line that starts like the memory block's heading
moves neither. The model is a fake; nothing here reads a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._step_review import StepReviewer
from localmelo.melo.memory.block import BLOCK_HEADER
from localmelo.tests.agent.test_approvals import _answer
from localmelo.tests.agent.test_auto_mode import FakeModel

# A line that starts like the heading of melo's recalled notes, in plain words.
LIKE_HEADING = "[melo's memory · a line of the file"
NOTES = f"\n\n{BLOCK_HEADER}\n- a note of melo's own"


async def _run(tmp_path: Path, text: str, **kwargs: Any) -> Agent:
    """A run of *text* that answers at once; returns the Agent to read its review."""
    (tmp_path / "ws").mkdir(exist_ok=True)
    agent = Agent(
        llm=FakeModel([_answer("done")]),
        options=AgentOptions(builtins=frozenset(), workspace_root=str(tmp_path / "ws")),
    )
    result = await agent.run_task(text, **kwargs)
    assert result.status == "completed"
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("person", "appended"),
    [
        # In what the person typed, and in their attachment: all of it is theirs.
        (f"read this\n{LIKE_HEADING}\nand then this", ""),
        (
            f'read it\n\n<attachment name="a.txt" chars="9">\n{LIKE_HEADING}\n</attachment>',
            "",
        ),
        # In a block melo appended: it is not.
        ("read it", f"\n\n[project notes]\n{LIKE_HEADING}"),
    ],
    ids=["typed", "attachment", "context"],
)
async def test_a_line_like_the_heading_gives_the_review_no_notes_and_keeps_the_words_whole(
    tmp_path: Path, person: str, appended: str
) -> None:
    # A caller that appended nothing leaves out own_chars: all of it is the person's.
    own = len(person) if appended else None
    agent = await _run(tmp_path, person + appended, own_chars=own, recalled="")

    reviewer = agent._reviewer
    assert reviewer._recalled == ""
    assert reviewer.requests == (person,)


@pytest.mark.asyncio
@pytest.mark.parametrize("held", [LIKE_HEADING, BLOCK_HEADER], ids=["like", "exact"])
async def test_the_notes_are_the_string_given_whatever_the_text_holds(
    tmp_path: Path, held: str
) -> None:
    given = f"{BLOCK_HEADER}\n- the notes the caller put in"
    text = f"read this\n{held}\n- a line of the file"

    agent = await _run(tmp_path, text, recalled=given)

    assert agent._reviewer._recalled == given
    assert agent._reviewer.requests == (text,)


@pytest.mark.asyncio
async def test_a_run_whose_caller_gives_no_notes_has_none(tmp_path: Path) -> None:
    text = f"read this\n\n{BLOCK_HEADER}\n- a line of the file"
    agent = await _run(tmp_path, text)
    assert agent._reviewer._recalled == "" and agent._reviewer.requests == (text,)


@pytest.mark.asyncio
async def test_the_words_given_for_the_review_are_not_cut_either(
    tmp_path: Path,
) -> None:
    words = f"read this\n{LIKE_HEADING}\nand then this"
    agent = await _run(tmp_path, words + NOTES, user_words=words, recalled=NOTES)
    assert agent._reviewer.requests == (words,)
    assert agent._reviewer._recalled == NOTES


def test_a_note_read_mid_run_keeps_the_notes_of_the_run_and_is_all_the_persons() -> (
    None
):
    reviewer = StepReviewer(lambda: None)  # type: ignore[arg-type,return-value]
    reviewer.user_message("go", recalled=NOTES)
    note = f"and also\n{LIKE_HEADING}\nthis"
    reviewer.user_message(note)
    assert reviewer._recalled == NOTES
    assert reviewer.requests == ("go", note)


def test_the_earlier_words_are_not_cut() -> None:
    reviewer = StepReviewer(lambda: None)  # type: ignore[arg-type,return-value]
    earlier = f"first\n{LIKE_HEADING}\nsecond"
    reviewer.earlier_words([earlier])
    reviewer.user_message("go on")
    assert reviewer.requests == (earlier, "go on")
    # The chat's first message, given as `first`, is not cut either.
    reviewer = StepReviewer(lambda: None)  # type: ignore[arg-type,return-value]
    reviewer.earlier_words(["a", "b"], first=earlier)
    reviewer.user_message("go on")
    assert reviewer.earlier == (earlier,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"query": 7}, TypeError),
        ({"own_chars": "3"}, TypeError),
        ({"own_chars": True}, TypeError),
        ({"own_chars": -1}, ValueError),
        ({"recalled": None}, TypeError),
    ],
    ids=["query", "own_chars str", "own_chars bool", "own_chars negative", "recalled"],
)
async def test_a_wrong_argument_is_refused_before_the_run(
    tmp_path: Path, kwargs: dict[str, Any], error: type[Exception]
) -> None:
    (tmp_path / "ws").mkdir()
    llm = FakeModel([_answer("done")])
    agent = Agent(
        llm=llm,
        options=AgentOptions(builtins=frozenset(), workspace_root=str(tmp_path / "ws")),
    )
    with pytest.raises(error):
        await agent.run_task(**{"query": "read this", **kwargs})
    assert llm.requests == []
