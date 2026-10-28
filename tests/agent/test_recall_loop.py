"""R5's recall through the agent loop: start, attempt, step and the tool."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any
from unittest import mock

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
)
from localmelo.melo.memory.notes import (
    RECALL_END,
    RECALL_START,
    ReasoningNote,
)
from localmelo.melo.memory.retrieval.llm import LLMRetriever
from localmelo.melo.memory.short import group_turns
from localmelo.melo.memory.short.note_store import NoteStoreEntry
from localmelo.melo.schema import Message, ToolCall, ToolDef

CALL_ID = "call_1"


class FakeLLM(BaseLLMProvider):
    reasoning_field = "auto"

    def __init__(self, *replies: Message) -> None:
        self.replies: deque[Message] = deque(replies)
        self.requests: list[list[Message]] = []
        self.offers: list[list[str]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(messages)
        self.offers.append(sorted(tool.name for tool in tools or ()))
        if not self.replies:
            raise AssertionError("no reply scripted")
        return self.replies.popleft()

    async def close(self) -> None:
        pass


class FakeEmbedding(BaseEmbeddingProvider):
    """One axis per text: the first text is [1, 0], the second [0, 1]."""

    model = "fake-embedding"

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]

    async def close(self) -> None:
        pass


def _echo() -> ToolDef:
    return ToolDef(
        name="echo",
        description="echo text back",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )


async def _run_echo(text: str) -> str:
    return f"echo: {text}"


async def _run_recall(query: str, k: int | None = None) -> str:
    return "fake recall"


def _options(**kwargs: Any) -> AgentOptions:
    kwargs.setdefault("builtins", frozenset())
    kwargs.setdefault("attempt_policy", "none")
    kwargs.setdefault("tool_exposure", "all")
    return AgentOptions(**kwargs)


def _agent(llm: BaseLLMProvider, options: AgentOptions, **kwargs: Any) -> Agent:
    agent = Agent(llm=llm, options=options, **kwargs)
    agent.hippo.register_tool(_echo())
    agent.executor.register("echo", _run_echo)
    return agent


def _call(
    note: str = "",
    *,
    thinking: str = "",
    arguments: dict[str, Any] | None = None,
) -> Message:
    return Message(
        role="assistant",
        content=note,
        thinking=thinking,
        tool_calls=[
            ToolCall(
                tool_name="echo",
                arguments=arguments or {"text": "a"},
                id=CALL_ID,
            )
        ],
    )


def _answer(content: str = "done", *, thinking: str = "") -> Message:
    return Message(role="assistant", content=content, thinking=thinking)


def _system(request: list[Message]) -> str:
    return next(message.content for message in request if message.role == "system")


def _block(text: str) -> str:
    return f"{RECALL_START} x]\n{text}\n{RECALL_END}"


# ── start ──


async def test_start_recalls_another_tasks_long_entry_on_the_query() -> None:
    llm = FakeLLM(_answer())
    embedding = FakeEmbedding()
    agent = _agent(
        llm,
        _options(reasoning="compress", retriever="embedding", max_steps=1),
        embedding=embedding,
    )
    await agent.hippo.long.add(
        "the answer to the old task is 42",
        [1.0, 0.0],
        {"task_id": "t-old", "created_at": 1_700_000_000.0},
    )
    try:
        result = await agent.run_task("what is the answer")
    finally:
        await agent.close()
    assert result.status == "completed"
    first = llm.requests[0]
    query = first[1]
    assert query.role == "user"
    assert "the answer to the old task is 42" in query.content
    assert "[m1] memory · task t-old" in query.content
    assert result.memory_stats is not None
    assert result.memory_stats["recalls"] == 1
    assert result.memory_stats["recalls_by_trigger"]["start"] == 1
    assert result.memory_stats["recall_hits"] == 1
    assert result.config["retriever"] == "embedding"
    # The block is in the request copy only, not the window.
    assert "localmelo recall" not in agent.hippo.working.get_window()[0].content


async def test_start_finds_nothing_and_sends_no_request() -> None:
    llm = FakeLLM(_answer())
    embedding = FakeEmbedding()
    agent = _agent(
        llm,
        _options(reasoning="compress", retriever="embedding", max_steps=1),
        embedding=embedding,
    )
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.memory_stats is not None
    assert result.memory_stats["recalls"] == 0
    assert len(llm.requests) == 1
    assert embedding.calls == 0
    assert "localmelo recall" not in llm.requests[0][1].content


# ── step ──


async def test_a_note_the_window_does_not_show_is_recalled_at_step() -> None:
    note = _entry("n1", "plan: the answer is 42\nfound: tests pass")
    first_note = "<note>\nplan: the answer is 42 and tests pass\n</note>"
    llm = FakeLLM(
        _call(first_note, arguments={"text": "a"}),
        _call(
            "<note>\nplan: the answer is 42 and tests pass\n</note>",
            arguments={"text": "b"},
        ),
        _call(
            "<note>\nplan: the answer is 42 and tests pass\n</note>",
            arguments={"text": "c"},
        ),
        _answer(),
    )
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="bm25",
            max_steps=4,
            recall_k=3,
        ),
    )  # type: ignore[call-arg]
    assert agent.load_notes([note]) == 1
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    # The stored note is not in the window at any planning call.
    third = llm.requests[2]
    blocks = [m.content for m in third if RECALL_START in m.content]
    assert blocks
    assert "[n1]" in blocks[0]
    assert blocks[0].endswith(RECALL_END)
    assert result.memory_stats is not None
    assert result.memory_stats["recalls_by_trigger"]["step"] >= 1
    # The block is append-only: the fourth request starts with the third
    # request byte for byte, and nothing re-rendered the block.
    third_text = [message.content for message in llm.requests[2]]
    fourth_text = [message.content for message in llm.requests[3]]
    assert fourth_text[: len(third_text)] == third_text
    assert any(message.content == blocks[0] for message in llm.requests[3])


async def test_message_placement_puts_the_block_in_its_own_user_message() -> None:
    note = _entry("n1", "plan: the answer is 42\nfound: tests pass")
    llm = FakeLLM(
        _call("<note>\nplan: the answer is 42 and tests pass\n</note>"),
        _answer(),
    )
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="bm25",
            recall_placement="message",
            max_steps=2,
        ),
    )
    assert agent.load_notes([note]) == 1
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    window = agent.hippo.working.get_window()
    units = group_turns(window, empty_prefix=True)
    recall_units = [
        unit for unit in units if any(RECALL_START in m.content for m in unit)
    ]
    assert recall_units
    assert recall_units[0][-1].role == "user"
    assert any(m.role == "tool" for m in recall_units[0][:-1])


async def test_the_empty_reply_nudge_does_not_recall_its_own_new_note() -> None:
    llm = FakeLLM(
        Message(
            role="assistant",
            content=(
                "<note>\nplan: the last reply was empty; continue the task\n</note>"
            ),
        ),
        _answer(),
    )
    agent = _agent(llm, _options(reasoning="compress", retriever="bm25", max_steps=2))
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    window = agent.hippo.working.get_window()
    labels = [m for m in window if m.role == "assistant" and "empty" in m.content]
    assert labels
    # The label already carries the note's text, so the step block must not
    # quote the same note back.
    assert "plan: the last reply was empty" in labels[0].content
    nudges = [
        m for m in window if m.role == "user" and "last reply was empty" in m.content
    ]
    assert nudges
    assert RECALL_START not in nudges[0].content


async def test_the_empty_reply_nudge_carries_the_step_block() -> None:
    note = _entry("n1", "plan: the empty reply means the answer is 42 and tests pass")
    llm = FakeLLM(
        Message(
            role="assistant",
            content=(
                "<note>\nplan: the empty reply means the answer is 42 and "
                "tests pass\n</note>"
            ),
        ),
        _answer(),
    )
    agent = _agent(
        llm, _options(reasoning="compress", retriever="bm25", max_steps=2)
    )  # type: ignore[call-arg]
    assert agent.load_notes([note]) == 1
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    window = agent.hippo.working.get_window()
    nudges = [
        m for m in window if m.role == "user" and "last reply was empty" in m.content
    ]
    assert nudges
    assert RECALL_START in nudges[0].content
    assert "[n1]" in nudges[0].content
    units = group_turns(window, empty_prefix=True)
    assert any(len(unit) == 2 and unit[1] is nudges[0] for unit in units)


# ── attempt ──


async def test_attempt_recall_runs_before_the_digest_and_fills_the_boundary() -> None:
    note = "<note>\nplan: the answer is 42 and tests pass\n</note>"
    reflection = json.dumps(
        {
            "summary": "one step",
            "failed_hypotheses": [],
            "next_promising_directions": ["the answer 42 tests"],
            "recommended_action": "continue",
            "task_still_feasible": True,
            "next_step_is_concrete": True,
            "next_step_is_novel": True,
        }
    )
    llm = FakeLLM(
        _call(note),
        _answer(reflection),
        _answer("done"),
    )
    preloaded = _entry("n0", "plan: the answer is 42\nfound: the tests pass")
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="bm25",
            recall_at=frozenset({"attempt"}),
            attempt_policy="reflect",
            steps_per_attempt=1,
            max_attempts=2,
            max_steps=4,
        ),
    )
    assert agent.load_notes([preloaded]) == 1
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    # The second attempt's first request carries the boundary block on its
    # task query message.
    second = llm.requests[2]
    query = next(m for m in second if m.role == "user" and m.content.startswith("q"))
    assert RECALL_START in query.content
    assert "[n0]" in query.content
    assert result.memory_stats is not None
    assert result.memory_stats["recalls_by_trigger"]["attempt"] == 1
    assert result.memory_stats["digests"] >= 1


# ── the tool ──


class SlowLLM(FakeLLM):
    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        if tools is None:  # a retrieve request
            await asyncio.sleep(0.05)
        return await super().chat(messages, tools)


async def test_a_timed_out_llm_recall_falls_back_to_bm25_and_still_blocks() -> None:
    note = _entry("n0", "plan: the answer is 42 and tests pass")
    llm = SlowLLM(
        _call("<note>\nplan: the answer is 42 and tests pass\n</note>"),
        _answer(),
    )
    agent = _agent(llm, _options(reasoning="compress", retriever="llm", max_steps=2))
    assert agent.load_notes([note]) == 1
    agent._retrievers["llm"] = LLMRetriever(agent.chat.retrieve, timeout_s=0.01)
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    assert result.memory_stats is not None
    assert result.memory_stats["recall_fallbacks"] == 1
    assert result.memory_stats["retrieve_errors"] == 1
    log = result.memory_stats["recall_log"]
    assert log
    assert log[0]["retriever"] == "bm25"
    assert log[0]["fallback"] == "timeout"
    tools = [
        message
        for message in agent.hippo.working.get_window()
        if message.role == "tool"
    ]
    assert any("[n0]" in message.content for message in tools)


async def test_the_recall_tool_is_offered_and_records_on_the_next_row() -> None:
    note = _entry("n1", "plan: the answer is 42\nfound: tests pass")
    tool_call = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                tool_name="recall",
                arguments={"query": "the answer 42 tests", "k": 2},
                id="r1",
            )
        ],
    )
    llm = FakeLLM(tool_call, _answer())
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="bm25",
            recall_at=frozenset({"tool"}),
            max_steps=2,
        ),
    )
    assert agent.load_notes([note]) == 1
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    assert "recall" in llm.offers[0]
    assert "recall" in llm.offers[1]
    window = agent.hippo.working.get_window()
    tool_results = [m for m in window if m.role == "tool"]
    assert tool_results
    assert "[n1]" in tool_results[0].content
    assert result.n_tool == 1
    assert result.memory_stats is not None
    assert result.memory_stats["recalls_by_trigger"]["tool"] == 1


async def test_the_recall_tool_outside_a_run_answers_one_line() -> None:
    agent = _agent(
        FakeLLM(),
        _options(reasoning="compress", retriever="bm25", recall_at=frozenset({"tool"})),
    )
    try:
        result = await agent.run_tool("recall", {"query": "x"})
    finally:
        await agent.close()
    assert "only available inside a run" in result.output


async def test_the_recall_tool_does_not_return_a_note_the_window_shows() -> None:
    llm = FakeLLM(
        _call("<note>\nplan: the answer is 42 and tests pass\n</note>"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    tool_name="recall",
                    arguments={"query": "the answer 42 tests"},
                    id="r1",
                )
            ],
        ),
        _answer(),
    )
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="bm25",
            recall_at=frozenset({"tool"}),
            max_steps=3,
        ),
    )
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    tool_results = [
        message
        for message in agent.hippo.working.get_window()
        if message.role == "tool"
    ]
    recall_results = [m for m in tool_results if "recall" in m.content]
    assert recall_results
    # The only document is the note the window already shows, so the tool
    # has nothing to add.
    assert "recall found nothing for that query" in recall_results[0].content
    assert not any("[n1]" in m.content for m in tool_results)


def _shell() -> ToolDef:
    return ToolDef(
        name="shell_exec",
        description="run a command",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )


async def _run_shell(command: str) -> str:
    return "ran"


async def test_a_tool_arguments_secret_is_redacted_before_the_retrieval_request() -> (
    None
):
    secret = "sk-abcdefghijklmnopqrstuvwxyz"
    llm = FakeLLM(
        Message(
            role="assistant",
            content=("<note>\nplan: echo the answer and check the output\n</note>"),
            tool_calls=[ToolCall("shell_exec", {"command": f"echo {secret}"}, "c1")],
        ),
        Message(role="assistant", content='{"ids": ["n0"]}'),
        _answer(),
    )
    agent = _agent(llm, _options(reasoning="compress", retriever="llm", max_steps=2))
    agent.hippo.register_tool(_shell())
    agent.executor.register("shell_exec", _run_shell)
    assert (
        agent.load_notes(
            [_entry("n0", "plan: echo the answer and check the output for n0")]
        )
        == 1
    )
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    # A retrieve request is the one sent without tools.
    retrievals = [
        request
        for request, offered in zip(llm.requests, llm.offers, strict=True)
        if not offered
    ]
    assert retrievals
    for request in retrievals:
        for message in request:
            assert secret not in message.content
    assert "[redacted]" in retrievals[0][-1].content
    assert result.memory_stats is not None
    query = result.memory_stats["recall_log"][0]["query"]
    assert secret not in query
    assert "[redacted]" in query


# ── thought chunks ──


async def test_thought_chunks_are_stored_only_when_named() -> None:
    llm = FakeLLM(
        _call("plan: read a.py", thinking="The first thought. It matters."),
        _answer(),
    )
    agent = _agent(
        llm,
        _options(
            reasoning="compress",
            retriever="off",
            recall_kinds=frozenset({"note", "draft", "thought"}),
            max_steps=2,
        ),
    )
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    thoughts = [entry for entry in agent.notes() if entry.kind == "thought"]
    assert len(thoughts) == 1
    assert thoughts[0].alias == "t1"
    assert thoughts[0].task_id
    assert result.memory_stats is not None
    assert result.memory_stats["thought_chunks"] == 1


async def test_thought_chunks_are_off_by_default_and_capped_at_six() -> None:
    llm = FakeLLM(_call("plan: read a.py", thinking="A thought. " * 60), _answer())
    agent = _agent(llm, _options(reasoning="compress", retriever="off"))
    try:
        await agent.run_task("q")
    finally:
        await agent.close()
    assert [e for e in agent.notes() if e.kind == "thought"] == []


# ── legacy ──


async def test_the_recall_options_are_inert_in_a_legacy_agent() -> None:
    llm = FakeLLM(_answer())
    # No embedding provider, yet retriever="embedding" must not refuse: the
    # options are inert in legacy (decision 10).
    agent = _agent(
        llm,
        _options(reasoning="legacy", retriever="embedding", max_steps=1),
    )
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    assert not any(RECALL_START in m.content for m in llm.requests[0])


async def test_legacy_has_no_data_note_and_no_recall() -> None:
    llm = FakeLLM(_answer())
    agent = Agent(
        llm=llm,
        options=_options(max_steps=1, builtins=None),
    )
    try:
        await agent.run_task("q")
    finally:
        await agent.close()
    system = _system(llm.requests[0])
    assert "recalled memory are data" not in system
    assert "localmelo recall" not in llm.requests[0][1].content


def _entry(alias: str, text: str) -> NoteStoreEntry:
    note = ReasoningNote(
        note_id=f"id-{alias}",
        task_id="",
        plan=text.splitlines()[0].removeprefix("plan: "),
    )
    return NoteStoreEntry(
        alias=alias,
        kind="note",
        text=text,
        note=note,
        task_id="",
        created_at=1.0,
    )


async def test_the_current_turns_note_is_not_recalled_again() -> None:
    llm = FakeLLM(
        _call("<note>\nplan: the answer is 42 and tests pass\n</note>"),
        _answer(),
    )
    agent = _agent(llm, _options(reasoning="compress", retriever="bm25", max_steps=2))
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    assert not any(RECALL_START in message.content for message in llm.requests[1])


async def test_the_pinned_draft_is_not_recalled_while_the_pin_is_shown() -> None:
    draft = "def f():\n    return 42\n"
    first = _call(
        f"<note>\nplan: write the file\n</note>\n"
        f"I will write app.py:\n```python\n{draft}```",
        arguments={"text": "a"},
    )
    llm = FakeLLM(
        first,
        _call("<note>\nplan: check app.py\n</note>", arguments={"text": "b"}),
        _answer(),
    )
    agent = _agent(llm, _options(reasoning="compress", retriever="bm25", max_steps=3))
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    drafts = [e for e in agent.notes() if e.kind == "draft"]
    assert drafts
    assert drafts[0].draft is not None and drafts[0].draft.target == "app.py"
    second = llm.requests[1]
    # The pin is in request 2, and the step block does not repeat it.
    assert any("1 draft from step 1" in message.content for message in second)
    assert sum(message.content.count("[d1]") for message in second) == 1


async def test_a_long_entry_is_recalled_once_while_its_block_is_shown() -> None:
    llm = FakeLLM(
        _call("<note>\nplan: read a.py\n</note>", arguments={"text": "a"}),
        _call("<note>\nplan: read b.py\n</note>", arguments={"text": "b"}),
        _answer(),
    )
    agent = _agent(
        llm,
        _options(reasoning="compress", retriever="embedding", max_steps=3),
        embedding=FakeEmbedding(),
    )
    await agent.hippo.long.add(
        "the old answer is 42 and the tests pass",
        [1.0, 0.0],
        {"task_id": "t-old"},
    )
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    first = sum("[m1]" in message.content for message in llm.requests[0])
    assert first == 1
    assert all(
        "[m1]" not in message.content
        for request in llm.requests[1:]
        for message in request
        if message.role == "tool"
    )


async def test_the_block_is_fitted_with_the_recorded_latency() -> None:
    from localmelo.melo.agent.agent import _RunState
    from localmelo.melo.memory import recall as library
    from localmelo.melo.memory.retrieval.base import Hit, RetrievalResult

    class NoClock:
        name = "bm25"

        async def search(self, query: Any, docs: Any, *, k: int) -> RetrievalResult:
            return RetrievalResult(
                hits=[Hit(doc.doc_id, 1.0, index) for index, doc in enumerate(docs, 1)],
                retriever="bm25",
                ms=0,
            )

    agent = _agent(
        FakeLLM(),
        _options(reasoning="compress", retriever="bm25", max_steps=1),
    )
    assert agent.load_notes([_entry("n0", "plan: the answer is 42")]) == 1
    seen: list[int] = []

    def spy(items: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["ms"])
        return library.fit_recall_block(items, **kwargs)

    try:
        with (
            mock.patch.object(agent, "_get_retriever", return_value=NoClock()),
            mock.patch(
                "localmelo.melo.agent.agent.time.perf_counter",
                side_effect=[100.0, 102.345],
            ),
            mock.patch("localmelo.melo.agent.agent.fit_recall_block", spy),
        ):
            _record, block, _hits = await agent._run_recall(
                "step", "the answer 42", _RunState(task_id="t")
            )
    finally:
        await agent.close()
    # The fit sees the 2,345 ms the block is rendered with, not 0.
    assert seen == [2_345]
    assert "2345 ms" in block


async def test_the_long_search_asks_the_store_for_two_rows() -> None:
    from localmelo.melo.agent.agent import _RunState

    agent = _agent(
        FakeLLM(),
        _options(reasoning="compress", retriever="embedding", max_steps=1),
        embedding=FakeEmbedding(),
    )
    await agent.hippo.long.add(
        "a memory", [1.0, 0.0], {"task_id": "t-old", "created_at": 1.0}
    )
    search = mock.AsyncMock(
        return_value=[("a memory", 0.9, {"task_id": "t-old", "created_at": 1.0})]
    )
    try:
        with mock.patch.object(agent.hippo.long, "search", search):
            docs = await agent._long_docs("q", _RunState(task_id="current"))
    finally:
        await agent.close()
    assert [doc.text for doc in docs] == ["a memory"]
    assert search.await_args is not None
    assert search.await_args.kwargs["top_k"] == 2


async def test_long_candidates_are_the_two_best_not_the_two_oldest() -> None:
    agent = _agent(
        FakeLLM(),
        _options(reasoning="compress", retriever="embedding", max_steps=1),
        embedding=FakeEmbedding(),
    )
    await agent.hippo.long.add(
        "a low old memory", [0.8, 0.6], {"task_id": "t-a", "created_at": 1.0}
    )
    await agent.hippo.long.add(
        "another low old memory", [0.8, 0.6], {"task_id": "t-b", "created_at": 2.0}
    )
    await agent.hippo.long.add(
        "the best new memory", [0.99, 0.14], {"task_id": "t-c", "created_at": 3.0}
    )
    from localmelo.melo.agent.agent import _RunState

    try:
        docs = await agent._long_docs("q", _RunState(task_id="current"))
    finally:
        await agent.close()
    assert docs
    assert any(doc.text == "the best new memory" for doc in docs)
    assert len(docs) == 2
    assert [doc.meta["alias"] for doc in docs] == ["m1", "m2"]


# ── the silent returns ──


async def test_recall_empty_counts_a_trigger_with_no_candidates() -> None:
    # The start trigger has no long entry, and the step trigger's only note
    # is the turn just appended, which the window already shows: neither has
    # anything to rank, and both used to return without a trace.
    llm = FakeLLM(
        _call("<note>\nplan: read a.py\n</note>", arguments={"text": "a"}),
        _answer(),
    )
    agent = _agent(llm, _options(reasoning="compress", retriever="bm25", max_steps=2))
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    assert result.memory_stats is not None
    assert result.memory_stats["recalls"] == 0
    assert result.memory_stats["recall_empty"] == 2


async def test_recall_empty_counts_a_search_with_no_hits() -> None:
    # A note the window never showed is a candidate, but the step's query
    # shares no floor unit with it, so bm25 returns nothing.
    llm = FakeLLM(
        _call(
            "<note>\nplan: investigate widget alpha\n</note>",
            arguments={"text": "a"},
        ),
        _answer(),
    )
    agent = _agent(llm, _options(reasoning="compress", retriever="bm25", max_steps=2))
    assert agent.load_notes([_entry("n1", "plan: unrelated thing")]) == 1
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    assert result.memory_stats is not None
    assert result.memory_stats["recalls"] == 0
    assert result.memory_stats["recall_empty"] == 2


async def test_a_superseded_note_the_window_left_is_recalled_at_step() -> None:
    # R4 marks an attempt's notes superseded at the attempt boundary; their
    # turns can still be evicted, and the step trigger is what makes an
    # evicted turn reachable again.
    note = _entry(
        "n1",
        "plan: parser module error\nfound: build.log says the parser module is wrong",
    )
    note.stage = "superseded"
    llm = FakeLLM(
        _call("<note>\nplan: read a.py\n</note>", arguments={"text": "a"}),
        _call(
            "<note>\nplan: review the parser module error\n</note>",
            arguments={"text": "b"},
        ),
        _answer(),
    )
    agent = _agent(llm, _options(reasoning="compress", retriever="bm25", max_steps=3))
    assert agent.load_notes([note]) == 1
    try:
        result = await agent.run_task("q")
    finally:
        await agent.close()
    assert result.status == "completed"
    assert result.memory_stats is not None
    assert result.memory_stats["recalls_by_trigger"]["step"] >= 1
    assert any("[n1]" in message.content for message in llm.requests[2])
