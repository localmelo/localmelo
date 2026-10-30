"""Stored usage counters and working-memory rows are displayed honestly."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest

from localmelo.melo.schema import EMPTY_REPLY_NUDGE, SHORT_TERM_MAX, Message, ToolCall
from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.session_views import memory_view, usage_view
from localmelo.support.cli.theme import ASCII, UNICODE
from localmelo.support.cli.views import BackendInfo
from localmelo.support.sessions.runner import ResumeInfo
from localmelo.support.sessions.store import TurnInfo


def turn(result: dict[str, Any] | None, **kwargs: Any) -> TurnInfo:
    base = TurnInfo(
        id="turn",
        session_id="session",
        turn=1,
        run_id="run",
        user_text="question",
        status="completed",
        cause="run",
        answer="answer",
        result=result,
        holder="cli",
        undone=False,
        started_at=1,
        finished_at=2,
        client_request_id=None,
        typed_text="question",
    )
    return replace(base, **kwargs)


def render(value: Any, width: int = 160) -> str:
    out = RecordingOut(width)
    out.print(value)
    return out.text()


def result(**kwargs: Any) -> dict[str, Any]:
    return dict(
        steps=3,
        n_tool=2,
        usage={"prompt_tokens": 12, "completion_tokens": 8},
        duration_s=1.2,
        **kwargs,
    )


@pytest.mark.parametrize(
    "bad", [True, -1, float("nan"), float("inf"), "12", [], None, 1.5]
)
def test_non_integer_counters_are_unknown_not_truncated(bad: Any) -> None:
    data = result()
    data["steps"] = bad
    data["n_tool"] = bad
    data["usage"] = {"prompt_tokens": bad, "completion_tokens": bad}
    text = render(usage_view([turn(data)]))
    assert text.count("?") >= 9
    assert "0+?" in text


def test_integer_counters_keep_precision_above_float_mantissa() -> None:
    exact = 2**53 + 1
    data = result()
    data["steps"] = exact
    text = render(usage_view([turn(data)]))
    assert str(exact) in text


def test_running_usage_is_unknown_even_if_a_partial_result_exists() -> None:
    text = render(usage_view([turn(result(), status="running", cause="")]))
    assert "running" in text and text.count("?") >= 11
    assert "1.2s" not in text


def test_known_undone_work_contributes_to_totals() -> None:
    text = render(usage_view([turn(result(), undone=True), turn(result(), turn=2)]))
    assert "Done (undone)" in text
    totals = next(
        line for line in text.splitlines() if line.lstrip().startswith("Totals ")
    )
    assert totals.split() == ["Totals", "6", "4", "24", "16", "2.4s"]


def test_empty_usage_is_explicit() -> None:
    assert render(usage_view([])).strip() == "No turns yet."


@pytest.mark.parametrize(
    ("mode", "fallback", "count", "checkpoint", "note"),
    [
        ("pairs", False, 2, 0, "Earlier tool steps are not reloaded."),
        ("pairs", True, 2, 0, "Earlier tool steps are not reloaded."),
        (
            "window",
            False,
            3,
            1,
            "Tool steps of the 2 newest resumed turns are not reloaded.",
        ),
        (
            "window",
            False,
            3,
            2,
            "Tool steps of the newest resumed turn are not reloaded.",
        ),
        ("window", False, 3, 3, ""),
    ],
)
def test_memory_resume_explanation_comes_from_resume_info(
    mode: Any, fallback: bool, count: int, checkpoint: int, note: str
) -> None:
    loaded = ResumeInfo(mode, fallback, count, checkpoint, 2, 20)
    text = render(
        memory_view(
            [Message("user", "only the visible tail"), Message("assistant", "answer")],
            BackendInfo("vllm", "vLLM", "stub", "http://127.0.0.1:9/v1"),
            window_max=SHORT_TERM_MAX,
            memory=True,
            loaded=loaded,
            width=100,
            symbols=UNICODE,
        )
    )
    assert "1 turn, 0 with tool steps" in text
    if note:
        assert note in text
    else:
        assert "not reloaded" not in text


def test_memory_sanitizes_rows_and_does_not_mutate_the_copied_window() -> None:
    secret = "sk-" + "f00dfeed12" * 3
    messages = [
        Message("user", "界" * 100),
        Message(
            "assistant",
            "<think>private reasoning</think>",
            tool_calls=[
                ToolCall(id="call", tool_name="file_read", arguments={}),
            ],
        ),
        Message("tool", f"answer {secret}\x1b]52;c;bad\x07", tool_call_id="call"),
        Message("user", EMPTY_REPLY_NUDGE),
        Message("assistant", "done"),
    ]
    before = copy.deepcopy(messages)
    text = render(
        memory_view(
            messages,
            BackendInfo("vllm", "vLLM", "stub", "http://127.0.0.1:9/v1"),
            window_max=SHORT_TERM_MAX,
            memory=False,
            loaded=None,
            width=40,
            symbols=ASCII,
        )
    )
    assert "1 turn, 1 with tool steps" in text
    assert "file_read" in text and "[redacted]" in text
    assert "private reasoning" not in text and secret not in text and "\x1b" not in text
    assert "(tool calls)" in text
    assert "off (--memory off)" in text
    assert before == messages
