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

from localmelo.tests.cli.test_e2e_pty import Melo
from localmelo.tests.fixtures.stub_llm import StubServer, load_scenarios

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")

SCENARIOS = Path(__file__).resolve().parents[1] / "fixtures" / "stub_scenarios"


def _configure(
    home: Path, url: str, *, hint: str | None = "none", stream: bool = True
) -> None:
    folder = home / ".cache" / "localmelo"
    folder.mkdir(parents=True, exist_ok=True)
    tags = f'\n[cli.reasoning_tags]\n"stub-model" = "{hint}"\n' if hint else ""
    (folder / "config.toml").write_text(
        f'chat_backend = "vllm"\nembedding_backend = "none"\n\n[vllm]\n'
        f'chat_url = "{url}"\nchat_model = "stub-model"\n\n'
        f"[cli]\nstream = {'true' if stream else 'false'}\n{tags}",
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
def streamed(tmp_path: Path) -> Iterator[Callable[..., Melo]]:
    terms: list[Melo] = []
    servers: list[StubServer] = []

    def start(scenario: str, *, hint: str | None = "none", stream: bool = True) -> Melo:
        server = StubServer(load_scenarios([SCENARIOS]), default=scenario).start()
        servers.append(server)
        _configure(tmp_path, server.base_url, hint=hint, stream=stream)
        term = Melo(tmp_path, [])
        terms.append(term)
        return term

    yield start
    for term in terms:
        term.close()
    for server in servers:
        server.close()


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
    term = streamed("stream_tool")
    (term.workspace / "app.py").write_text("print('hello')\n", encoding="utf-8")
    term.see("Ask melo")
    term.type("read app.py\r")
    term.see("partial reply (tool call)")
    term.see("Let me read it.")
    term.see("The file prints hello.")
    assert screen(term).count("Let me read it.") == 1
    assert term.quit() == 0


def test_esc_mid_stream_closes_and_labels_the_partial_once(
    streamed: Callable[..., Melo],
) -> None:
    term = streamed("stream_slow")
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
    term = streamed("stream_drop")
    term.see("Ask melo")
    term.type("drop please\r")
    term.see("partial reply (not the final answer)")
    term.see("provider_error")
    assert term.quit() == 0
