"""The live region's streamed text, its labels and the one final print."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from localmelo.melo.schema import RunResult
from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.render import (
    STREAM_BUFFERED,
    STREAM_LIVE_LINES,
    LiveState,
    TurnRenderer,
    live_rows,
)
from localmelo.support.cli.theme import UNICODE
from localmelo.support.cli.views import BackendInfo
from localmelo.support.sessions.store import StoredEvent

URL = "http://127.0.0.1:18431/v1"


def _backend() -> BackendInfo:
    return BackendInfo(
        key="openai-compatible",
        label="OpenAI-compatible",
        model="stub-model",
        base_url=URL,
    )


def _renderer(
    tags: Mapping[str, str] | None = None, width: int = 80
) -> tuple[RecordingOut, TurnRenderer]:
    out = RecordingOut(width)
    renderer = TurnRenderer(out, symbols=UNICODE, backend=_backend, reasoning_tags=tags)
    renderer.begin_turn("What is in this folder?")
    return out, renderer


def _event(renderer: TurnRenderer, type_: str, **data: Any) -> None:
    renderer.on_event(
        StoredEvent(id=1, session_id="s", turn=1, event={"type": type_, "data": data})
    )


def _finish(
    renderer: TurnRenderer, status: str = "completed", result: str = ""
) -> None:
    _event(
        renderer,
        "run.finished",
        result=RunResult(
            task_id="t", status=status, result=result, n_tool=0, duration_s=1.0
        ).to_dict(),
    )


def _rows(live: LiveState) -> str:
    return "\n".join(
        "".join(text for _, text in row) for row in live_rows(live, 100.0, 80, UNICODE)
    )


def test_released_text_shows_in_the_live_region() -> None:
    _, renderer = _renderer({"stub-model": "none"})
    feed = renderer.stream_feed()
    feed.opened()
    feed.held()
    assert "thinking" in _rows(renderer.live)  # type: ignore[arg-type]
    feed.content("Hello\n")
    assert "Hello" in _rows(renderer.live)  # type: ignore[arg-type]
    assert "thinking" not in _rows(renderer.live)  # type: ignore[arg-type]


def test_the_live_region_keeps_its_row_budget_and_shows_the_tail() -> None:
    _, renderer = _renderer({"stub-model": "none"})
    feed = renderer.stream_feed()
    feed.opened()
    feed.content("\n".join(f"line {n}" for n in range(40)))
    rows = live_rows(renderer.live, 100.0, 80, UNICODE)  # type: ignore[arg-type]
    assert len(rows) == 1 + 1 + STREAM_LIVE_LINES
    assert rows[1][0][1] == "… +32 lines"
    assert rows[-1][0][1] == "line 39"


def test_a_short_terminal_lowers_the_stream_rows_and_zero_hides_them() -> None:
    _, renderer = _renderer({"stub-model": "none"})
    feed = renderer.stream_feed()
    feed.opened()
    feed.content("\n".join(f"line {n}" for n in range(40)))
    live = renderer.live
    assert live is not None
    assert len(live_rows(live, 100.0, 80, UNICODE, stream_lines=2)) == 1 + 1 + 2
    assert len(live_rows(live, 100.0, 80, UNICODE, stream_lines=0)) == 1


def test_a_completed_stream_prints_the_final_answer_once() -> None:
    out, renderer = _renderer({"stub-model": "none"})
    feed = renderer.stream_feed()
    feed.opened()
    feed.content("Hello there\n")
    feed.closed()
    _event(renderer, "assistant.message", content="Hello there", tool_calls=[])
    _finish(renderer, result="Hello there")
    text = out.text()
    assert text.count("● Hello there") == 1
    assert "partial reply" not in text


def test_a_tool_call_preamble_is_closed_and_labelled() -> None:
    out, renderer = _renderer({"stub-model": "none"})
    feed = renderer.stream_feed()
    feed.opened()
    feed.content("Let me read it.\n")
    feed.closed()
    _event(
        renderer,
        "assistant.message",
        content="Let me read it.",
        tool_calls=[{"id": "c1", "name": "file_read", "preview": "app.py"}],
    )
    text = out.text()
    assert "partial reply (tool call)" in text
    assert text.count("Let me read it.") == 1


def test_a_retry_closes_and_labels_the_partial_text() -> None:
    out, renderer = _renderer({"stub-model": "none"})
    feed = renderer.stream_feed()
    feed.opened()
    feed.content("First try\n")
    feed.closed()
    _event(renderer, "notice", kind="retry", message="retrying")
    text = out.text()
    assert "partial reply (retrying)" in text and "First try" in text


def test_a_step_end_drops_the_held_copy_of_the_same_text() -> None:
    out, renderer = _renderer({"stub-model": "none"})
    feed = renderer.stream_feed()
    feed.opened()
    feed.content("<note>internal state</note>\n")
    feed.closed()
    _event(
        renderer,
        "assistant.message",
        content="<note>internal state</note>",
        tool_calls=[],
    )
    _event(renderer, "step.started", step=2, attempt=1)
    text = out.text()
    assert text.count("internal state") == 1
    assert "partial reply (not the final answer)" in text


def test_a_failed_stream_and_the_held_copy_print_once() -> None:
    out, renderer = _renderer({"stub-model": "none"})
    feed = renderer.stream_feed()
    feed.opened()
    feed.content("A preamble that fails validation.\n")
    feed.closed()
    _event(
        renderer,
        "assistant.message",
        content="A preamble that fails validation.",
        tool_calls=[],
    )
    _finish(renderer, status="failed", result="")
    text = out.text()
    assert text.count("A preamble that fails validation.") == 1
    assert "partial reply (not the final answer)" in text


def test_end_turn_without_run_finished_prints_a_completed_answer_once() -> None:
    out, renderer = _renderer({"stub-model": "none"})
    feed = renderer.stream_feed()
    feed.opened()
    feed.content("Hello there\n")
    feed.closed()
    renderer.end_turn(
        RunResult(task_id="t", status="completed", result="Hello there"), None
    )
    text = out.text()
    assert text.count("● Hello there") == 1
    assert "partial reply" not in text


def test_a_cancelled_stream_is_labelled_and_printed_once() -> None:
    out, renderer = _renderer({"stub-model": "none"})
    feed = renderer.stream_feed()
    feed.opened()
    feed.content("Half an answer\n")
    feed.closed()
    _finish(renderer, status="cancelled")
    renderer.end_turn(
        RunResult(task_id="t", status="cancelled", result=""), None, cause="user_stop"
    )
    text = out.text()
    assert text.count("partial reply (stopped)") == 1
    assert text.count("Half an answer") == 1


def test_the_model_hint_selects_the_stream_and_whole_reply_rule() -> None:
    _, renderer = _renderer({"stub-model": "closing_only"})
    assert renderer.stream_hint() == "closing_only"
    assert renderer.stream_feed().reasoning_tags == "closing_only"
    _, undeclared = _renderer()
    assert undeclared.stream_hint() is None


def test_a_first_delta_before_its_own_step_started_is_not_stale() -> None:
    """Events lag the stream: step.started may follow the step's first delta."""
    out, renderer = _renderer({"stub-model": "none"})
    feed = renderer.stream_feed()
    feed.step(1)
    feed.opened()
    feed.content("Hello there\n")
    _event(renderer, "step.started", step=1, attempt=1)
    assert "partial reply" not in out.text()
    assert "Hello there" in _rows(renderer.live)  # type: ignore[arg-type]
    feed.closed()
    _event(renderer, "assistant.message", step=1, content="Hello there")
    _finish(renderer, result="Hello there")
    assert out.text().count("● Hello there") == 1
    assert "partial reply" not in out.text()


def test_a_fast_tool_lets_the_next_step_stream_before_the_last_reply_event() -> None:
    out, renderer = _renderer({"stub-model": "none"})
    feed = renderer.stream_feed()
    feed.step(1)
    feed.opened()
    feed.content("Let me read it.\n")
    feed.closed()
    feed.step(2)  # the tool finished at once; step 1's events are still queued
    feed.opened()
    feed.content("The file prints hello.\n")
    _event(renderer, "step.started", step=1, attempt=1)
    _event(
        renderer,
        "assistant.message",
        step=1,
        content="Let me read it.",
        tool_calls=[{"id": "c1", "name": "file_read", "preview": "app.py"}],
    )
    _event(renderer, "step.started", step=2, attempt=1)
    text = out.text()
    assert text.count("Let me read it.") == 1
    assert "partial reply (tool call)" in text
    assert "partial reply (not the final answer)" not in text
    assert "The file prints hello." in _rows(renderer.live)  # type: ignore[arg-type]
    assert "Let me read it." not in _rows(renderer.live)  # type: ignore[arg-type]


def test_a_late_retry_notice_does_not_cut_the_next_attempt() -> None:
    out, renderer = _renderer({"stub-model": "none"})
    feed = renderer.stream_feed()
    feed.step(1)
    feed.opened()
    feed.content("First try\n")
    feed.closed()
    feed.opened()  # the retry has begun; its notice has not arrived yet
    feed.content("Second try\n")
    _event(renderer, "notice", kind="retry", message="retrying")
    text = out.text()
    assert text.count("partial reply (retrying)") == 1
    assert "First try" in text and "Second try" not in text
    assert "Second try" in _rows(renderer.live)  # type: ignore[arg-type]


def _reply(renderer: TurnRenderer, *chunks: str) -> None:
    """One planning stream that releases *chunks*, then the turn's end."""
    feed = renderer.stream_feed()
    feed.opened()
    feed.held()
    for chunk in chunks:
        feed.content(chunk)
    feed.closed()
    text = "".join(chunks).strip()
    _event(renderer, "assistant.message", content=text, tool_calls=[])
    _finish(renderer, result=text)


def test_a_whole_reply_without_a_hint_is_noted_once_per_model() -> None:
    out, renderer = _renderer()
    _reply(renderer, "Hello there\n")
    assert " ".join(out.text().split()).count(STREAM_BUFFERED) == 1
    renderer.begin_turn("again")
    _reply(renderer, "Hello again\n")
    assert " ".join(out.text().split()).count(STREAM_BUFFERED) == 1


def test_a_reply_that_streamed_or_has_a_hint_is_not_noted() -> None:
    out, renderer = _renderer()
    _reply(renderer, "line one\n", "line two\n")  # released in two parts
    assert STREAM_BUFFERED not in " ".join(out.text().split())
    out, renderer = _renderer({"stub-model": "none"})
    _reply(renderer, "Hello there\n")
    assert STREAM_BUFFERED not in " ".join(out.text().split())
