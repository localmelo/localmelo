"""The live region's streamed text, its labels and the one final print."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from localmelo.melo.schema import RunResult
from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.render import (
    SERVER_WHOLE,
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
    tags: Mapping[str, str] | None = None, width: int = 80, developer: bool = True
) -> tuple[RecordingOut, TurnRenderer]:
    """A renderer in developer mode, where the labels and notes under test are shown."""
    out = RecordingOut(width)
    renderer = TurnRenderer(
        out,
        symbols=UNICODE,
        backend=_backend,
        reasoning_tags=tags,
        developer=developer,
    )
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
    assert "Hello" not in _rows(renderer.live)  # type: ignore[arg-type]
    feed.content("Hello\n")
    assert "Hello" in _rows(renderer.live)  # type: ignore[arg-type]
    assert "thinking" not in _rows(renderer.live)  # type: ignore[arg-type]


def test_waiting_for_the_model_is_one_row_not_two() -> None:
    """The spinner row says Thinking; held text adds no row that says it again."""
    _, renderer = _renderer()  # a model with no hint: the filter holds its text
    feed = renderer.stream_feed()
    feed.opened()
    assert _rows(renderer.live).count("\n") == 0  # type: ignore[arg-type]
    feed.held()
    feed.held()
    rows = live_rows(renderer.live, 100.0, 80, UNICODE, frame=0)  # type: ignore[arg-type]
    assert len(rows) == 1
    assert "".join(text for _, text in rows[0]).startswith("· Thinking…")
    assert _rows(renderer.live).lower().count("thinking") == 1  # type: ignore[arg-type]


def test_no_activity_repeats_itself_in_a_second_row() -> None:
    """Held text under any activity draws only the spinner row (Replying, Retrying, ...)."""
    _, renderer = _renderer()
    feed = renderer.stream_feed()
    feed.opened()
    live = renderer.live
    assert live is not None
    for activity in ("Thinking", "Replying", "Reflecting", "Stopping", "Starting"):
        live.activity = activity
        feed.held()
        assert len(live_rows(live, 100.0, 80, UNICODE, frame=0)) == 1, activity
    live.retry_until = 130.0
    assert len(live_rows(live, 100.0, 80, UNICODE, frame=0)) == 1
    live.waiting = True
    feed.held()
    assert len(live_rows(live, 100.0, 80, UNICODE, frame=0)) == 1
    # Rows that add something stay: streamed text, the tool row.
    live.waiting, live.retry_until, live.activity = False, 0.0, "Running"
    live.tool, live.tool_started = "shell_exec", 99.0
    feed.content("Let me look.")
    rows = live_rows(live, 100.0, 80, UNICODE, frame=0)
    assert [row[0][1].split()[0] for row in rows[1:]] == ["Let", "└"]
    assert "shell_exec" in rows[-1][0][1]


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


def test_outside_developer_mode_the_reasoning_format_is_never_mentioned() -> None:
    out, renderer = _renderer(developer=False)
    _reply(renderer, "Hello there\n")  # a whole reply from a model with no hint
    assert "Hello there" in out.text()
    assert STREAM_BUFFERED not in " ".join(out.text().split())
    assert "reasoning format" not in out.text()


def test_outside_developer_mode_partial_text_prints_without_a_label() -> None:
    out, renderer = _renderer({"stub-model": "none"}, developer=False)
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
    feed.opened()  # the next step starts, as a retry of a broken stream would
    feed.content("First try\n")
    feed.closed()
    renderer.stream_opened()
    text = out.text()
    assert text.count("Let me read it.") == 1 and text.count("First try") == 1
    assert "partial reply" not in text


def test_a_reply_that_streamed_or_has_a_hint_is_not_noted() -> None:
    out, renderer = _renderer()
    _reply(renderer, "line one\n", "line two\n")  # released in two parts
    assert STREAM_BUFFERED not in " ".join(out.text().split())
    out, renderer = _renderer({"stub-model": "none"})
    _reply(renderer, "Hello there\n")
    assert STREAM_BUFFERED not in " ".join(out.text().split())


def _whole(renderer: TurnRenderer, text: str = "Hello there") -> None:
    """One planning request that the server answered whole, then the turn's end."""
    feed = renderer.stream_feed()
    feed.opened()
    feed.whole()
    feed.closed()
    _event(renderer, "assistant.message", content=text, tool_calls=[])
    _finish(renderer, result=text)


def _said(out: RecordingOut, words: str) -> int:
    return " ".join(out.text().split()).count(words)


def test_a_server_that_sends_replies_whole_is_noted_once() -> None:
    for tags in (None, {"stub-model": "none"}):  # the model's hint does not matter
        out, renderer = _renderer(tags)
        _whole(renderer)
        assert _said(out, SERVER_WHOLE) == 1
        renderer.begin_turn("again")
        _whole(renderer, "Hello again")
        assert _said(out, SERVER_WHOLE) == 1
        assert out.text().count("● Hello again") == 1


def test_the_whole_reply_note_is_one_dim_line_after_the_footer() -> None:
    out, renderer = _renderer()
    _whole(renderer)
    lines = out.text().splitlines()
    note = [n for n, line in enumerate(lines) if SERVER_WHOLE in line]
    assert len(note) == 1
    assert lines[note[0] - 1].startswith("✻ Worked for")
    assert lines[note[0]].startswith("· ") and lines[note[0]].count("·") == 1
    assert "Worked for" not in lines[note[0]]


def test_the_note_that_a_model_has_no_hint_is_not_added_to_it() -> None:
    out, renderer = _renderer()  # no hint: a reply released whole would say so too
    _whole(renderer)
    assert _said(out, STREAM_BUFFERED) == 0


def test_a_reply_that_streamed_does_not_say_the_server_sends_replies_whole() -> None:
    out, renderer = _renderer({"stub-model": "none"})
    _reply(renderer, "line one\n", "line two\n")
    assert _said(out, SERVER_WHOLE) == 0


def test_the_whole_reply_note_waits_for_a_turn_that_completes() -> None:
    out, renderer = _renderer()
    feed = renderer.stream_feed()
    feed.opened()
    feed.whole()
    feed.closed()
    _finish(renderer, status="failed")
    assert _said(out, SERVER_WHOLE) == 0
    renderer.begin_turn("again")
    _whole(renderer)
    assert _said(out, SERVER_WHOLE) == 1


def test_a_whole_reply_shows_nothing_live_before_it_is_complete() -> None:
    _, renderer = _renderer()
    feed = renderer.stream_feed()
    feed.opened()
    feed.whole()
    assert "thinking" in _rows(renderer.live).lower()  # type: ignore[arg-type]
    assert SERVER_WHOLE not in _rows(renderer.live)  # type: ignore[arg-type]
