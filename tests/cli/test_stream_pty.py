"""Live streamed replies in the real terminal, against the streaming stub.

These need a pseudo-terminal, which the headless sandbox does not allow; they
run where ``os.openpty`` does. Each test starts ``python -m localmelo`` with a
temporary HOME, ``[cli] stream = true`` and a per-model framing hint, and a
stub that sends incremental SSE. Nothing reaches a model server.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pyte
import pytest

from localmelo.support.cli.render import SERVER_WHOLE
from localmelo.tests.cli.test_e2e_pty import Melo
from localmelo.tests.fixtures.stub_llm import StubServer, load_scenarios

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")

SCENARIOS = Path(__file__).resolve().parents[1] / "fixtures" / "stub_scenarios"


def _configure(
    home: Path, url: str, *, hint: str | None = "none", stream: bool | None = True
) -> None:
    """Write the config; ``stream=None`` leaves ``[cli] stream`` unset (its default)."""
    folder = home / ".cache" / "localmelo"
    folder.mkdir(parents=True, exist_ok=True)
    tags = f'\n[cli.reasoning_tags]\n"stub-model" = "{hint}"\n' if hint else ""
    key = "" if stream is None else f"[cli]\nstream = {'true' if stream else 'false'}\n"
    (folder / "config.toml").write_text(
        f'chat_backend = "vllm"\nembedding_backend = "none"\n\n[vllm]\n'
        f'chat_url = "{url}"\nchat_model = "stub-model"\n\n{key}{tags}',
        encoding="utf-8",
    )


def screen(term: Melo, columns: int = 100) -> str:
    """What the terminal shows, scrollback included.

    ``Melo.text()`` is the byte stream with its escapes removed, so every
    redraw of the live region repeats there; counts of printed lines need the
    screen the redraws leave.
    """
    display = pyte.HistoryScreen(columns, 30, history=500)
    pyte.Stream(display).feed(term.output.decode("utf-8", "replace"))
    rows = [*display.history.top, *display.history.bottom]
    lines = ["".join(row[c].data for c in range(columns)) for row in rows]
    return "\n".join([*(line.rstrip() for line in lines), *display.display])


@pytest.fixture
def servers() -> Iterator[list[StubServer]]:
    items: list[StubServer] = []
    yield items
    for server in items:
        server.close()


@pytest.fixture
def streamed(
    tmp_path: Path, servers: list[StubServer]
) -> Iterator[Callable[..., Melo]]:
    terms: list[Melo] = []

    def start(
        scenario: str,
        *,
        hint: str | None = "none",
        stream: bool | None = True,
        argv: list[str] | None = None,
    ) -> Melo:
        server = StubServer(load_scenarios([SCENARIOS]), default=scenario).start()
        servers.append(server)
        _configure(tmp_path, server.base_url, hint=hint, stream=stream)
        term = Melo(tmp_path, argv or [])
        terms.append(term)
        return term

    yield start
    for term in terms:
        term.close()


def test_a_streamed_reply_shows_live_then_prints_the_final_answer_once(
    streamed: Callable[..., Melo],
) -> None:
    term = streamed("stream_live")
    term.see("Ask melo")
    term.type("stream please\r")
    term.see("The first line of the answer.")
    assert "Worked for" not in term.text()  # it is live, not the final print
    term.see("Worked for")
    text = screen(term)
    assert text.count("● The first line of the answer.") == 1
    assert "partial reply" not in text
    assert term.quit() == 0


def test_split_utf8_and_closing_tag_release_only_the_answer(
    streamed: Callable[..., Melo],
) -> None:
    term = streamed("stream_utf8", hint="closing_only")
    term.see("Ask melo")
    term.type("/thinking off\r")
    term.see("Thinking excerpts off.")
    term.type("utf8 please\r")
    term.see("Still streaming")
    assert "Worked for" not in term.text()  # it is live, not the final print
    term.see("Worked for")
    text = screen(term)
    assert "secret plan" not in text
    assert text.count("● 你好，世界") == 1
    assert term.quit() == 0


def test_a_bearer_token_split_across_chunks_is_never_drawn(
    streamed: Callable[..., Melo],
) -> None:
    term = streamed("stream_bearer", hint="none")
    term.see("Ask melo")
    term.type("bearer please\r")
    term.see("[redacted]")
    term.see("Still streaming here.")
    assert "Worked for" not in term.text()  # it is live, not the final print
    term.see("Worked for")
    assert "abcdefghijklmnopqrstuvwxyz0123456789" not in term.text()
    assert term.quit() == 0


def test_a_tool_call_preamble_is_labelled_before_the_tool_block(
    streamed: Callable[..., Melo],
) -> None:
    term = streamed("stream_tool", argv=["--developer"])
    (term.workspace / "app.py").write_text("print('hello')\n", encoding="utf-8")
    term.see("Ask melo")
    term.type("read app.py\r")
    term.see("partial reply (tool call)")
    term.see("Let me read it.")
    term.see("The file prints hello.")
    assert screen(term).count("Let me read it.") == 1
    assert term.quit() == 0


def test_a_tool_call_preamble_outside_developer_mode_has_no_label(
    streamed: Callable[..., Melo],
) -> None:
    term = streamed("stream_tool")
    (term.workspace / "app.py").write_text("print('hello')\n", encoding="utf-8")
    term.see("Ask melo")
    term.type("read app.py\r")
    term.see("The file prints hello.")
    text = screen(term)
    assert text.count("Let me read it.") == 1
    assert "partial reply" not in text
    assert term.quit() == 0


def test_esc_mid_stream_closes_and_labels_the_partial_once(
    streamed: Callable[..., Melo],
) -> None:
    term = streamed("stream_slow", argv=["--developer"])
    term.see("Ask melo")
    term.type("stream please\r")
    term.see("Streaming line one.")
    term.type(b"\x1b")
    term.see("partial reply (stopped)")
    term.see("Stopped")
    assert screen(term).count("partial reply (stopped)") == 1
    assert term.quit() == 0


def test_separate_reasoning_is_never_content_and_thinking_off_hides_it(
    streamed: Callable[..., Melo],
) -> None:
    term = streamed("stream_thinking", hint="none")
    term.see("Ask melo")
    term.type("/thinking off\r")
    term.see("Thinking excerpts off.")
    term.type("think please\r")
    term.see("The streamed answer begins here.")
    term.see("Worked for")
    assert "Comparing the options" not in term.text()
    assert term.quit() == 0


def test_a_stream_that_breaks_before_the_end_is_labelled(
    streamed: Callable[..., Melo],
) -> None:
    term = streamed("stream_drop", argv=["--developer"])
    term.see("Ask melo")
    term.type("drop please\r")
    term.see("partial reply (not the final answer)")
    term.see("Could not get an answer from the model server: Streaming response")
    assert term.quit() == 0


def test_opted_out_turns_send_no_stream_and_print_the_answer_once(
    streamed: Callable[..., Melo], servers: list[StubServer]
) -> None:
    """``[cli] stream = false``: the ordinary complete-reply turn, untouched."""
    term = streamed("answer", stream=False)
    term.see("Ask melo")
    term.type("stream please\r")
    term.see("Worked for")
    text = screen(term)
    assert text.count("● Hello! I am melo, running on a stub model.") == 1
    assert "partial reply" not in text
    assert not servers[0].requests[-1].get("stream")
    assert term.quit() == 0


@pytest.mark.parametrize(
    ("argv", "notes"),
    [(["--developer"], 1), ([], 0)],
    ids=["developer mode", "default mode"],
)
def test_with_no_config_for_it_the_reply_streams_and_developer_mode_notes_a_whole_one(
    streamed: Callable[..., Melo],
    servers: list[StubServer],
    argv: list[str],
    notes: int,
) -> None:
    """Streaming is on by default; a model with no hint says why its reply waited,
    once, in developer mode: a new user has no use for the reasoning format."""
    term = streamed("answer", hint=None, stream=None, argv=argv)
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Worked for")
    assert servers[0].requests[-1].get("stream") is True
    term.type("hello again\r")
    term.see("Worked for", timeout_s=30)
    term.read_until(lambda out: out.count(b"Worked for") >= 2)
    text = " ".join(screen(term).split())
    assert (
        text.count("replies appear when complete until its reasoning format is set")
        == notes
    )
    assert text.count("● Hello! I am melo, running on a stub model.") == 2
    assert term.quit() == 0


def test_x_cli_stream_false_turns_streaming_off_for_one_run(
    streamed: Callable[..., Melo], servers: list[StubServer]
) -> None:
    term = streamed("answer", hint=None, stream=None, argv=["-X", "cli.stream=false"])
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Worked for")
    assert not servers[0].requests[-1].get("stream")
    assert "reasoning format" not in term.text()
    assert term.quit() == 0


def test_an_undeclared_model_holds_its_text_until_the_reply_ends(
    streamed: Callable[..., Melo],
) -> None:
    """The conservative rule: no ``[cli.reasoning_tags]`` entry, no live text."""
    term = streamed("stream_slow", hint=None)
    term.see("Ask melo")
    term.type("stream please\r")
    term.see("✻ thinking…")
    assert "Streaming line one." not in term.text()  # held, not drawn
    term.see("Worked for")
    text = screen(term)
    assert text.count("● Streaming line one.") == 1
    assert "partial reply" not in text
    assert term.quit() == 0


def _asked_to_stream(server: StubServer) -> list[bool]:
    """Per planning request the server saw, whether it asked for a stream."""
    return [request.get("stream") is True for request in server.requests]


@pytest.mark.parametrize(
    "scenario,reply,asked",
    [
        # One JSON body answers the first request; the second turn does not ask.
        ("stream_ignored", "This reply came whole, in one body.", [True, False]),
        # The first request is asked again at once; the second turn does not ask.
        (
            "stream_closed",
            "This reply needed a second request.",
            [True, False, False],
        ),
    ],
    ids=["ignores stream", "closes the stream at once"],
)
def test_a_server_that_does_not_stream_still_answers_every_turn_and_says_so_once(
    streamed: Callable[..., Melo],
    servers: list[StubServer],
    scenario: str,
    reply: str,
    asked: list[bool],
) -> None:
    """With streaming on by default, neither server fails a turn."""
    term = streamed(scenario, hint=None, stream=None)
    term.see("Ask melo")
    term.type("plain please\r")
    term.see("Worked for")
    first = " ".join(screen(term).split())
    assert first.count(f"● {reply}") == 1
    assert first.count(SERVER_WHOLE) == 1
    assert "Could not get an answer" not in first
    assert "reasoning format is set" not in first  # the other whole-reply note
    term.type("plain again\r")
    term.read_until(lambda out: out.count(b"Worked for") >= 2)
    text = " ".join(screen(term).split())
    assert text.count(f"● {reply}") == 2
    assert text.count(SERVER_WHOLE) == 1  # once, not once per turn
    assert _asked_to_stream(servers[0]) == asked
    assert term.quit() == 0


def test_a_stream_that_breaks_after_part_of_a_reply_is_not_asked_again(
    streamed: Callable[..., Melo], servers: list[StubServer]
) -> None:
    term = streamed("stream_drop")
    term.see("Ask melo")
    term.type("drop please\r")
    term.see("Could not get an answer from the model server: Streaming response")
    assert SERVER_WHOLE not in " ".join(screen(term).split())
    assert all(_asked_to_stream(servers[0]))  # no request without streaming
    assert term.quit() == 0
