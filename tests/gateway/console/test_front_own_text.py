"""The main chat's agent takes the person's words from what its caller says is theirs.

A caller that appended notes after its own text says where that text ends
(``own_chars``) and which block it put in (``recalled``): the review reads the
person's part as their words and that block as the notes, and the lines the main
chat adds itself stay out of both. The model is scripted; nothing is served.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.melo.memory.block import BLOCK_HEADER
from localmelo.support.gateway.console.front import FrontAgent
from localmelo.support.sessions.identity import ACTING_NOTE
from localmelo.tests.sessions.helpers import ScriptedLLM, answer

TYPED = "what is the plan for today?"
NOTES = f"\n\n{BLOCK_HEADER}\n- a note of melo's own"
CONTEXT = "\n\n[project notes]\na line of the file"


@pytest.mark.asyncio
async def test_the_review_holds_the_persons_part_and_exactly_the_notes_given(
    tmp_path: Path,
) -> None:
    (tmp_path / "ws").mkdir()
    llm = ScriptedLLM([answer("done")])
    agent = FrontAgent(
        llm=llm,
        options=AgentOptions(builtins=frozenset(), workspace_root=str(tmp_path / "ws")),
    )

    # No `user_words`: the person's words are the part `own_chars` says, not the
    # notes, the context or the line the main chat adds after them.
    result = await agent.run_task(
        TYPED + NOTES + CONTEXT, own_chars=len(TYPED), recalled=NOTES
    )

    assert result.status == "completed"
    assert agent._reviewer.requests == (TYPED,)
    assert agent._reviewer._recalled == NOTES
    sent = next(m.content for m in reversed(llm.requests[0]) if m.role == "user")
    assert sent == f"{TYPED}{NOTES}{CONTEXT}\n\n{ACTING_NOTE}"
