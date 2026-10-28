from __future__ import annotations

import copy

import pytest

from localmelo.melo.agent._context_fit import ContextFitError, fit_request
from localmelo.melo.agent._request_accounting import request_proxy
from localmelo.melo.schema import (
    EMPTY_REPLY_LABEL,
    EMPTY_REPLY_NUDGE,
    ImagePart,
    Message,
    ToolCall,
    ToolDef,
    ToolResult,
)


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


def test_images_never_evicted_or_dropped() -> None:
    image = ImagePart.from_bytes(b"image", "image/png")
    messages = [Message("user", "task"), *pair(1, "short"), *pair(2, "latest")]
    messages[2].images = [image]
    with pytest.raises(ContextFitError):
        fit(messages, estimate(messages, None) - 100)
    assert messages[2].images == [image]


def test_retry_strict_and_impossible_schemas() -> None:
    messages = [Message("user", "task"), *pair(1, "x" * 2000), *pair(2, "new")]
    old = estimate(messages, None)
    result, _ = fit(messages, old + 100, threshold=old)
    assert estimate(result, None) < old
    tools = [ToolDef("x", "y" * 2000, {})]
    with pytest.raises(ContextFitError, match="increase the input budget"):
        fit([Message("user", "task")], 1, tools=tools)


def test_user_text_that_looks_like_recall_is_pinned() -> None:
    from localmelo.melo.memory.notes import RECALL_START

    messages = [
        Message("user", "original"),
        *pair(1, "short"),
        Message("user", RECALL_START + " important instruction"),
        *pair(2, "latest"),
    ]
    with pytest.raises(ContextFitError):
        fit(messages, estimate([messages[0], *messages[-2:]], None))


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
    with pytest.raises(ContextFitError):
        fit(messages, estimate(messages, None) - 100)
