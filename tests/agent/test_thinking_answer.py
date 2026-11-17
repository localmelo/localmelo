"""A thinking model's answer is never lost (#469).

A model that ends its turn with the answer only in its thinking and an empty
reply is asked once more, with thinking off where the service has that switch;
if the row still ends empty, the last lines of the thinking are the answer,
marked as taken from it. A reply with no thinking at all still fails.
The stub model server plays the replies, as for unfinished thinking (#278).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import thinking_disabled
from localmelo.melo.schema import (
    MAX_THINKING_ANSWER_CHARS,
    THINKING_ANSWER_NOTE,
    UNFINISHED_ANSWER_ERROR,
    RunResult,
    thinking_answer,
)
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.fixtures.stub_llm import SCENARIOS_DIR, StubServer, load_scenarios

OFF = {"thinking": {"type": "disabled"}}
TAIL = "So I will tell them the folder is empty."

# ── The words taken from the thinking ──


def test_the_last_lines_of_the_thinking_are_the_answer_and_they_are_marked() -> None:
    lines = [
        "The user wants the folder's contents, so I will list the files.",
        "The folder is empty, which means there is nothing to read here.",
        TAIL,
    ]
    words = thinking_answer("First idea.\n" + "\n".join(lines))
    assert THINKING_ANSWER_NOTE == "(from melo's thinking)"
    # The last lines, until there is enough to say something; earlier ones stay out.
    assert words == "\n".join(lines) + f"\n\n{THINKING_ANSWER_NOTE}"


def test_one_long_last_line_is_enough_and_a_huge_one_is_cut_at_a_word() -> None:
    assert thinking_answer("Early.\n" + "word " * 60 + "end.").startswith("word word")
    huge = thinking_answer("Start.\n" + "alpha " * 1000 + "omega.")
    body = huge.removesuffix(f"\n\n{THINKING_ANSWER_NOTE}")
    assert body.startswith("…alpha") and body.endswith("omega.")
    assert len(body) <= MAX_THINKING_ANSWER_CHARS + 1


@pytest.mark.parametrize("thinking", ["", "  \n\n ", "...", "\n-- \n"])
def test_thinking_with_no_words_gives_no_answer(thinking: str) -> None:
    assert thinking_answer(thinking) == ""


def test_the_switch_of_a_hosted_service_counts_as_thinking_off() -> None:
    assert thinking_disabled(OFF)
    assert not thinking_disabled({"thinking": {"type": "enabled"}})
    assert not thinking_disabled({"thinking": "disabled"})


# ── A run on the stub ──


@pytest.fixture
def stub() -> Iterator[StubServer]:
    server = StubServer(load_scenarios([SCENARIOS_DIR])).start()
    try:
        yield server
    finally:
        server.close()


async def _run(
    stub: StubServer,
    scenario: str,
    tmp_path: Path,
    *,
    switch: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    fallback: str = "on",
) -> tuple[RunResult, list[AgentEvent]]:
    stub.default = scenario
    events: list[AgentEvent] = []
    agent = Agent(
        llm=OpenAICompatLLM(
            base_url=stub.base_url,
            model="stub-model",
            thinking_off_body=switch,
            extra_body=extra_body,
        ),
        options=AgentOptions(
            builtins=frozenset({"list_files"}),
            attempt_policy="none",
            retriever="off",
            max_steps=6,
            workspace_root=str(tmp_path),
            empty_reply_fallback=fallback,  # type: ignore[arg-type]
        ),
    )
    try:
        result = await agent.run_task("What is in this folder?", on_event=events.append)
    finally:
        await agent.close()
    return result, events


def _sent_thinking(stub: StubServer) -> list[Any]:
    """The ``thinking`` field of each request the model server got, in order."""
    return [body.get("thinking") for body in stub.requests]


async def test_a_thinking_only_row_is_asked_once_more_with_thinking_off(
    stub: StubServer, tmp_path: Path
) -> None:
    result, _ = await _run(stub, "thinking_only_then_answer", tmp_path, switch=OFF)
    assert (result.status, result.result) == ("completed", "The folder is empty.")
    assert result.n_llm == 3
    # Only the retry goes out with the switch, and only that one request.
    assert _sent_thinking(stub) == [None, None, OFF["thinking"]]


async def test_a_row_still_empty_after_the_retry_ends_with_the_marked_thinking(
    stub: StubServer, tmp_path: Path
) -> None:
    result, events = await _run(stub, "thinking_only_always", tmp_path, switch=OFF)
    assert result.status == "completed" and result.error == ""
    assert result.result == (
        "The user wants the folder's contents.\nThe folder is empty.\n"
        f"{TAIL}\n\n{THINKING_ANSWER_NOTE}"
    )
    assert result.n_llm == 3
    assert _sent_thinking(stub) == [None, None, OFF["thinking"]]
    assert events[-1].type == "run.finished"


async def test_without_a_switch_the_second_empty_reply_ends_with_the_marked_thinking(
    stub: StubServer, tmp_path: Path
) -> None:
    result, _ = await _run(stub, "thinking_only_always", tmp_path)
    assert result.status == "completed"
    assert result.result.endswith(f"{TAIL}\n\n{THINKING_ANSWER_NOTE}")
    assert result.n_llm == 2  # nothing to retry with
    assert _sent_thinking(stub) == [None, None]


async def test_no_retry_when_the_requests_already_ask_for_no_thinking(
    stub: StubServer, tmp_path: Path
) -> None:
    result, _ = await _run(
        stub, "thinking_only_always", tmp_path, switch=OFF, extra_body=OFF
    )
    assert result.status == "completed"
    assert result.result.endswith(THINKING_ANSWER_NOTE)
    assert result.n_llm == 2
    assert _sent_thinking(stub) == [OFF["thinking"]] * 2


async def test_a_reply_with_no_thinking_at_all_still_fails(
    stub: StubServer, tmp_path: Path
) -> None:
    result, _ = await _run(stub, "no_thinking_text", tmp_path, switch=OFF)
    assert (result.status, result.error, result.result) == (
        "failed",
        UNFINISHED_ANSWER_ERROR,
        "",
    )
    assert result.n_llm == 2  # asked once more, no retry and no answer from nothing
    assert _sent_thinking(stub) == [None, None]


# ── empty_reply_fallback: off is the run from before, on says what it did ──


def _notices(events: list[AgentEvent]) -> list[str]:
    return [event.data["kind"] for event in events if event.type == "notice"]


@pytest.mark.parametrize(
    "scenario", ["thinking_only_always", "thinking_only_then_answer"]
)
@pytest.mark.parametrize("switch", [OFF, None])
async def test_off_ends_the_run_as_it_did_before_the_fallback_existed(
    stub: StubServer, tmp_path: Path, scenario: str, switch: dict[str, Any] | None
) -> None:
    """Pinned to the run of the version before the fallback, switch or not: the
    same two requests, the same notice, the same failed result."""
    result, events = await _run(stub, scenario, tmp_path, switch=switch, fallback="off")
    assert (result.status, result.error, result.result) == (
        "failed",
        UNFINISHED_ANSWER_ERROR,
        "",
    )
    assert (result.steps, result.n_llm) == (2, 2)
    assert _sent_thinking(stub) == [None, None]  # no retry with thinking off
    assert _notices(events) == ["unfinished_answer"]
    assert result.empty_reply is None and "empty_reply" not in result.to_dict()
    assert "empty_reply" not in events[-1].data["result"]
    assert result.config["empty_reply_fallback"] == "off"


async def test_on_leaves_no_trace_in_a_run_that_needed_no_fallback(
    stub: StubServer, tmp_path: Path
) -> None:
    result, events = await _run(stub, "answer", tmp_path, switch=OFF)
    assert result.status == "completed"
    assert result.empty_reply is None and "empty_reply" not in result.to_dict()
    assert "empty_reply_fallback" not in result.config  # the default is not listed
    assert not {"thinking_off_retry", "answer_from_thinking"} & set(_notices(events))


async def test_on_marks_the_retry_that_gave_the_answer(
    stub: StubServer, tmp_path: Path
) -> None:
    result, events = await _run(stub, "thinking_only_then_answer", tmp_path, switch=OFF)
    assert result.result == "The folder is empty."
    flags = {"thinking_off_retry": True, "from_thinking": False}
    assert result.empty_reply == flags
    assert result.to_dict()["empty_reply"] == flags
    assert events[-1].data["result"]["empty_reply"] == flags  # run.finished
    # The retry's notice follows the usual one of the reply that led to it.
    assert _notices(events) == [
        "unfinished_answer",
        "unfinished_answer",
        "thinking_off_retry",
    ]
    notice = next(e for e in events if e.data.get("kind") == "thinking_off_retry")
    assert notice.data["message"] == (
        "The model ended its turn with only thinking; asking again with thinking off."
    )
    assert notice.data["detail"] == {}


async def test_on_marks_the_retry_and_the_answer_taken_from_the_thinking(
    stub: StubServer, tmp_path: Path
) -> None:
    result, events = await _run(stub, "thinking_only_always", tmp_path, switch=OFF)
    flags = {"thinking_off_retry": True, "from_thinking": True}
    assert result.empty_reply == flags
    assert events[-1].data["result"]["empty_reply"] == flags
    assert _notices(events) == [
        "unfinished_answer",
        "unfinished_answer",
        "thinking_off_retry",
        "answer_from_thinking",
    ]
    assert events[-1].type == "run.finished"  # the notice comes before it


async def test_on_marks_only_the_answer_where_the_service_has_no_switch(
    stub: StubServer, tmp_path: Path
) -> None:
    result, events = await _run(stub, "thinking_only_always", tmp_path)
    assert result.empty_reply == {"thinking_off_retry": False, "from_thinking": True}
    assert _notices(events) == ["unfinished_answer", "answer_from_thinking"]


async def test_a_failed_run_with_no_thinking_has_nothing_to_mark(
    stub: StubServer, tmp_path: Path
) -> None:
    result, events = await _run(stub, "no_thinking_text", tmp_path, switch=OFF)
    assert result.status == "failed"
    assert result.empty_reply is None
    assert not {"thinking_off_retry", "answer_from_thinking"} & set(_notices(events))


def test_a_result_with_the_flags_rebuilds_from_its_dict() -> None:
    """``run.finished`` carries ``result.to_dict()``, and the other end rebuilds it."""
    flags = {"thinking_off_retry": True, "from_thinking": False}
    result = RunResult(task_id="t", status="completed", empty_reply=flags)
    assert RunResult(**result.to_dict()) == result
    assert "empty_reply" not in RunResult(task_id="t", status="completed").to_dict()
