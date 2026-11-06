from __future__ import annotations

import copy

import pytest

from localmelo.melo.agent._chat import PREFIX_RUN, Chat
from localmelo.melo.agent._context_fit import ContextFitError, FitCut, fit_request
from localmelo.melo.agent._request_accounting import request_proxy
from localmelo.melo.checker.checker import PROMPT_CAP_CHARS, Checker, prompt_chars
from localmelo.melo.schema import (
    EMPTY_REPLY_LABEL,
    EMPTY_REPLY_NUDGE,
    REFLECTIONS_HEADER,
    ImagePart,
    Message,
    ToolCall,
    ToolDef,
    ToolResult,
)
from localmelo.tests.agent.test_reflection_gate import FakeLLM


def estimate(messages: list[Message], tools: list[ToolDef] | None) -> int:
    return request_proxy(messages, tools, replay=True)[0]


def pair(number: int, output: str, *, thinking: str = "") -> list[Message]:
    call = ToolCall("shell", {"i": number}, id=f"id{number}")
    return [
        Message("assistant", tool_calls=[call], thinking=thinking),
        Message("tool", output, tool_call_id=call.id),
    ]


def fit(
    messages: list[Message],
    limit: int,
    *,
    tools: list[ToolDef] | None = None,
    replay: bool = False,
    threshold: int | None = None,
) -> tuple[list[Message], dict[str, int]]:
    return fit_request(
        messages,
        tools,
        limit=limit,
        estimate=estimate,
        requires_replay=replay,
        force_smaller_than=threshold,
    )


def test_preview_unicode_pairs_and_copy() -> None:
    tools = [ToolDef("shell", "schema", {"type": "object"})]
    messages = [Message("system", "safe"), Message("user", "任务🙂")]
    messages += pair(1, "汉🙂x" * 1000 + "\n[exit code 0]")
    messages += pair(2, "new evidence")
    original = copy.deepcopy(messages)
    target = estimate(messages, tools) - 1000
    result, counts = fit(messages, target, tools=tools)
    assert estimate(result, tools) <= target
    assert counts == {"tool_results": 1, "thinking": 0, "units": 0}
    assert "characters omitted" in result[3].content
    assert "[exit code 0]" in result[3].content
    assert [m.tool_call_id for m in result if m.role == "tool"] == ["id1", "id2"]
    assert messages == original
    result[2].tool_calls[0].arguments["i"] = 99
    assert messages[2].tool_calls[0].arguments["i"] == 1


def test_many_calls_keep_order_and_pairing() -> None:
    calls = [ToolCall("shell", {"n": n}, id=f"c{n}") for n in range(12)]
    messages = [Message("user", "task"), Message("assistant", tool_calls=calls)]
    messages.extend(
        Message("tool", "abc" * 800, tool_call_id=call.id) for call in calls
    )
    result, counts = fit(messages, estimate(messages, None) - 500)
    assert [call.id for call in result[1].tool_calls] == [call.id for call in calls]
    assert [m.tool_call_id for m in result[2:]] == [call.id for call in calls]
    assert counts["tool_results"] > 0


def test_evicts_whole_old_units_and_nudge_but_not_user() -> None:
    messages = [
        Message("user", "task"),
        *pair(1, "small"),
        Message("assistant", EMPTY_REPLY_LABEL),
        Message("user", EMPTY_REPLY_NUDGE),
        *pair(2, "latest"),
    ]
    limit = estimate([messages[0], *messages[-2:]], None)
    result, counts = fit(messages, limit)
    assert result == [messages[0], *messages[-2:]]
    assert counts["units"] == 2


def test_thinking_only_optional_and_old() -> None:
    messages = [
        Message("user", "task"),
        *pair(1, "result", thinking="t" * 2000),
        *pair(2, "new", thinking="new thought"),
    ]
    limit = estimate(messages, None) - 300
    result, counts = fit(messages, limit)
    assert counts["thinking"] == 1
    assert result[1].thinking == "" and result[3].thinking == "new thought"
    result, counts = fit(messages, limit, replay=True)
    assert counts["thinking"] == 0 and counts["units"] == 1
    assert result[-2].thinking == "new thought"


def test_failure_pinned_and_inflight_not_truncated() -> None:
    messages = [
        Message("user", "task"),
        *pair(1, "x" * 1000 + "\n[exit code 1]"),
        *pair(2, "latest"),
    ]
    with pytest.raises(ContextFitError):
        fit(messages, 10)
    pending = [
        Message("user", "task"),
        Message(
            "assistant",
            tool_calls=[ToolCall("shell", {}, id="a"), ToolCall("shell", {}, id="b")],
        ),
        Message("tool", "x" * 2000, tool_call_id="a"),
    ]
    with pytest.raises(ContextFitError):
        fit(pending, estimate(pending, None) - 100)


def test_images_and_old_failures_go_only_when_nothing_else_is_left() -> None:
    image = ImagePart.from_bytes(b"image", "image/png")
    messages = [
        Message("user", "task"),
        *pair(1, "error: old"),
        *pair(2, "short"),
        *pair(3, "plain"),
        *pair(4, "error: new"),
        *pair(5, "latest"),
    ]
    messages[4].images = [image]
    head, old, pictured, _plain, failing, newest = (
        messages[:1],
        messages[1:3],
        messages[3:5],
        messages[5:7],
        messages[7:9],
        messages[9:],
    )
    kept = head + old + pictured + failing + newest
    result, counts = fit(messages, estimate(kept, None))
    assert result == kept and counts["units"] == 1
    # The latest failing check stays to the end.
    kept = head + failing + newest
    result, counts = fit(messages, estimate(kept, None))
    assert result == kept and counts["units"] == 3
    assert messages[4].images == [image]


def test_earlier_turns_go_whole_before_the_task() -> None:
    earlier = [
        Message("user", "old task " + "o" * 2000),
        Message("assistant", "old answer"),
        *pair(1, "error: " + "e" * 2000),
    ]
    task = [Message("user", "task"), *pair(2, "latest")]
    messages = [Message("system", "s"), *earlier, *task]
    limit = estimate([messages[0], *task], None)
    result, counts = fit_request(
        messages,
        None,
        limit=limit,
        estimate=estimate,
        requires_replay=False,
        history=earlier,
    )
    assert result == [messages[0], *task] and counts["units"] == 3
    # Without them named, the old user turn and answer stay.
    with pytest.raises(ContextFitError):
        fit(messages, limit)


def test_retry_strict_and_impossible_schemas() -> None:
    messages = [Message("user", "task"), *pair(1, "x" * 2000), *pair(2, "new")]
    old = estimate(messages, None)
    result, _ = fit(messages, old + 100, threshold=old)
    assert estimate(result, None) < old
    tools = [ToolDef("x", "y" * 2000, {})]
    with pytest.raises(ContextFitError, match="increase the input budget"):
        fit([Message("user", "task")], 1, tools=tools)


def test_user_text_that_looks_like_recall_goes_only_whole_and_last() -> None:
    from localmelo.melo.memory.notes import RECALL_START

    messages = [
        Message("user", "original"),
        *pair(0, "old " * 100),
        *pair(1, "short"),
        Message("user", RECALL_START + " important instruction"),
        *pair(2, "latest"),
    ]
    roomy, _ = fit(messages, estimate(messages, None) - 1)
    assert messages[5].content in [m.content for m in roomy]
    tight, _ = fit(messages, estimate([messages[0], *messages[-2:]], None))
    assert [m.content for m in tight] == ["original", "", "latest"]


def test_two_fits_do_not_repreview_or_recount() -> None:
    messages = [Message("user", "task"), *pair(1, "x" * 5000), *pair(2, "y" * 5000)]
    first, counts = fit(messages, estimate(messages, None) - 1000)
    assert counts["tool_results"] == 1 and "4744 characters omitted" in first[2].content
    second, counts = fit(
        first, estimate(first, None) + 100, threshold=estimate(first, None)
    )
    assert "4744 characters omitted" in second[2].content
    assert (
        counts["tool_results"] == 1 and "4744 characters omitted" in second[4].content
    )


def test_preview_keeps_exit_code_across_boundary() -> None:
    messages = [
        Message("user", "task"),
        *pair(1, "x" * 250 + "[exit code 0]" + "y" * 237),
    ]
    result, _ = fit(messages, estimate(messages, None) - 30)
    assert "[exit code 0]" in result[2].content
    assert "237 characters omitted" in result[2].content


def test_preview_oldest_before_thinking() -> None:
    messages = [
        Message("user", "task"),
        *pair(1, "a" * 3000, thinking="t" * 3000),
        *pair(2, "b" * 3000),
    ]
    result, counts = fit(messages, estimate(messages, None) - 400)
    assert "characters omitted" in result[2].content
    assert (
        result[4].content == messages[4].content
        and result[1].thinking == messages[1].thinking
    )
    assert counts == {"tool_results": 1, "thinking": 0, "units": 0}


def test_old_recall_evicted_whole_current_recall_kept() -> None:
    from localmelo.melo.memory.notes import RECALL_END, RECALL_START

    old = RECALL_START + " old]\n" + "r" * 2000 + "\n" + RECALL_END
    current = RECALL_START + " current]\n" + "s" * 2000 + "\n" + RECALL_END
    messages = [
        Message("user", "task"),
        *pair(1, "old output"),
        Message("user", old),
        *pair(2, "new output\n" + current),
    ]
    expected = [messages[0], *messages[1:3], *messages[-2:]]
    result, counts = fit(messages, estimate(expected, None))
    assert result == expected and counts["units"] == 1
    assert current in result[-1].content
    with pytest.raises(ContextFitError):
        fit(messages, 1)


def test_old_embedded_recall_removed_without_changing_tool_pair() -> None:
    from localmelo.melo.memory.notes import RECALL_END, RECALL_START

    recall = RECALL_START + " old]\n" + "r" * 1000 + "\n" + RECALL_END
    messages = [
        Message("user", "task"),
        *pair(1, "old output\n" + recall),
        *pair(2, "new output"),
    ]
    expected = [Message("user", "task"), *pair(1, "old output"), *pair(2, "new output")]
    result, counts = fit(messages, estimate(expected, None))
    assert result[2].content == "old output" and result[2].tool_call_id == "id1"
    assert counts["units"] == 1 and counts["tool_results"] == 0


def test_preview_preserves_embedded_current_recall() -> None:
    from localmelo.melo.memory.notes import RECALL_END, RECALL_START

    recall = RECALL_START + " current]\n" + "r" * 1000 + "\n" + RECALL_END
    messages = [Message("user", "task"), *pair(1, "x" * 2000 + "\n" + recall)]
    result, _ = fit(messages, estimate(messages, None) - 200)
    assert result[-1].content.endswith(recall)


def test_error_metadata_preserved() -> None:
    messages = [Message("user", "task"), *pair(1, "a" * 2000), *pair(2, "latest")]
    messages[2].tool_result = ToolResult("shell", error="important")
    # A failure is never previewed: it goes whole, and only at the last level.
    result, counts = fit(messages, estimate(messages, None) - 100)
    assert counts == {"tool_results": 0, "thinking": 0, "units": 1}
    assert [m.content for m in result] == ["task", "", "latest"]


# The shapes a 20,000-character prompt cap refused while the earlier fit sent
# them: each request is as Chat built it, the size as the cap counts it.


def chars(messages: list[Message], tools: list[ToolDef] | None) -> int:
    return prompt_chars(messages)


def cap_fit(
    messages: list[Message], hold: FitCut | None = None
) -> tuple[list[Message], dict[str, int]]:
    return fit_request(
        messages,
        None,
        limit=20_000,
        estimate=chars,
        requires_replay=False,
        drop_target=0.7,
        hold=hold,
    )


def test_a_finished_attempt_goes_when_the_task_and_the_newest_need_the_room() -> None:
    task = "t" * 4_300
    messages = [
        Message("system", "s" * 400),
        Message("user", task),
        *pair(1, "x" * 2_000),
        Message("user", "The attempt has ended. Reflect on it.\n" + task),
        Message("assistant", '{"summary": "' + "y" * 1_800 + '"}'),
        Message("user", f"{REFLECTIONS_HEADER}\n" + "z" * 1_100),
        *pair(2, "Traceback (most recent call last):\n" + "f" * 9_500),
    ]
    cut = FitCut()
    first, _ = cap_fit(messages, cut)
    assert chars(first, None) <= 20_000
    kept = [m.content for m in first]
    assert kept[:2] == ["s" * 400, task] and kept[-1] == messages[-1].content
    assert messages[6].content in kept  # the newest reflections block
    assert messages[4].content not in kept  # the attempt's reflect prompt
    # The next request repeats the cut byte for byte.
    second, _ = cap_fit([*messages, *pair(3, "ok")], cut)
    assert second[: len(first)] == first


def test_the_latest_failing_check_goes_when_the_newest_call_needs_the_room() -> None:
    write = ToolCall("file_write", {"path": "game.py", "content": "c" * 10_000}, id="w")
    messages = [
        Message("system", "s" * 400),
        Message("user", "t" * 3_900),
        *pair(1, "a" * 3_000),
        *pair(2, "Traceback (most recent call last):\n" + "f" * 9_900),
        Message("assistant", tool_calls=[write]),
        Message("tool", "Wrote 10000 characters", tool_call_id="w"),
    ]
    result, _ = cap_fit(messages)
    assert chars(result, None) <= 20_000
    assert [m.content for m in result][:2] == ["s" * 400, "t" * 3_900]
    assert result[-2].tool_calls == [write]


async def test_the_answer_on_stop_is_sent_over_the_prompt_cap() -> None:
    # Its evidence has its own budget; the task alone fills most of the cap.
    llm = FakeLLM([Message("assistant", "the answer")])
    checker = Checker()
    chat = Chat(llm, prompt_check=checker.pre_plan, fit_target=0.7)
    turns = [m for n in range(30) for m in pair(n, "o" * 1_000)]
    query = "q" * (PROMPT_CAP_CHARS - 10_000)
    token = chat.begin_prefix()
    try:
        reply = await chat.final_answer(turns, query)
    finally:
        PREFIX_RUN.reset(token)
    assert reply.content == "the answer"
    assert prompt_chars(llm.calls[0]["messages"]) > PROMPT_CAP_CHARS
