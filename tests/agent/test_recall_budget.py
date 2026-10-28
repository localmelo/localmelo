"""R5's recall budgets: the boundary cap, the live allowance and the query."""

from __future__ import annotations

from localmelo.melo.agent.agent import _RunState
from localmelo.melo.memory.notes import RECALL_END
from localmelo.melo.memory.recall import (
    RECALL_BOUNDARY_CHARS,
    RECALL_LIVE_CHARS,
    RECALL_TURN_CHARS,
    recall_span_chars,
    recall_spans,
)
from localmelo.melo.schema import Message

from .test_recall_loop import (
    FakeEmbedding,
    FakeLLM,
    _agent,
    _answer,
    _block,
    _call,
    _entry,
    _options,
)


async def test_the_boundary_block_fits_four_thousand_characters() -> None:
    llm = FakeLLM(_answer())
    agent = _agent(
        llm,
        _options(reasoning="compress", retriever="embedding", max_steps=1),
        embedding=FakeEmbedding(),
    )
    await agent.hippo.long.add(
        "the old answer " * 600,
        [1.0, 0.0],
        {"task_id": "t-old"},
    )
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    spans = recall_spans(llm.requests[0])
    assert len(spans) == 1
    assert spans[0].endswith(RECALL_END)
    assert len(spans[0]) <= RECALL_BOUNDARY_CHARS


async def test_the_live_allowance_stops_a_new_block() -> None:
    agent = _agent(
        FakeLLM(_answer()),
        _options(reasoning="compress", retriever="bm25", max_steps=1),
    )
    assert (
        agent.load_notes([_entry("n1", "plan: the answer is 42 and tests pass")]) == 1
    )
    run = _RunState(task_id="t")
    agent.hippo.working.append(
        Message(role="tool", content=_block("x" * (RECALL_LIVE_CHARS + 100)))
    )
    assert recall_span_chars(agent.hippo.working.get_window()) >= RECALL_LIVE_CHARS
    try:
        _record, block, _hits = await agent._run_recall("step", "the answer 42", run)
    finally:
        await agent.close()
    assert block == ""
    assert run.recall_skipped == 1


async def test_a_partly_used_live_allowance_shortens_the_limit() -> None:
    agent = _agent(
        FakeLLM(_answer()),
        _options(reasoning="compress", retriever="bm25", max_steps=1),
    )
    used = RECALL_LIVE_CHARS - 500
    agent.hippo.working.append(Message(role="tool", content=_block("x" * used)))
    assert (
        agent.load_notes(
            [
                _entry(
                    "n1",
                    "plan: the answer is 42 and tests pass\nfound: " + "y" * 2_000,
                )
            ]
        )
        == 1
    )
    run = _RunState(task_id="t")
    try:
        _record, block, _hits = await agent._run_recall(
            "step", "the answer 42 tests pass", run
        )
    finally:
        await agent.close()
    assert 0 < len(block) <= 500
    assert recall_span_chars([Message(role="tool", content=block)]) <= 500


async def test_the_turn_cap_is_the_library_budget() -> None:

    assert RECALL_TURN_CHARS == 9_500
    assert RECALL_LIVE_CHARS == 12_000


async def test_six_thought_chunks_per_task_at_most() -> None:
    thinking = "\n\n".join(f"Thought number {i}. " * 30 for i in range(10))
    llm = FakeLLM(
        _call("<note>\nplan: think about it\n</note>", thinking=thinking),
        _call("<note>\nplan: think again\n</note>", thinking=thinking),
        _answer(),
    )
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="off",
            recall_kinds=frozenset({"note", "draft", "thought"}),
            max_steps=3,
        ),
    )
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    thoughts = [entry for entry in agent.notes() if entry.kind == "thought"]
    assert 0 < len(thoughts) <= 6
    assert result.memory_stats is not None
    assert result.memory_stats["thought_chunks"] == len(thoughts)


async def test_a_thirty_step_run_stays_inside_every_budget() -> None:
    filler = "the answer is 42 and the tests pass " * 40  # about 1,200 chars
    replies = [
        _call(f"<note>\nplan: {filler}\n</note>", arguments={"text": str(i)})
        for i in range(30)
    ]
    replies.append(_answer())
    llm = FakeLLM(*replies)
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="bm25",
            max_steps=31,
            recall_k=3,
        ),
    )
    assert (
        agent.load_notes([_entry(f"n{i}", f"plan: {filler}") for i in range(30)]) == 30
    )
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    store = agent.hippo.working.notes
    assert len(store.entries()) <= 300
    assert store.chars() <= 300_000
    window = agent.hippo.working.get_window()
    assert recall_span_chars(window) <= RECALL_LIVE_CHARS
    for request in llm.requests:
        assert sum(len(message.content) for message in request) < 100_000
    blocks = recall_spans(window)
    assert blocks
    assert all(block.endswith(RECALL_END) for block in blocks)
    # Each block is under the per-turn cap on its own, not only the live
    # total of all of them.
    assert all(len(block) <= RECALL_TURN_CHARS for block in blocks)
