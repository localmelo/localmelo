"""The status bar says the session's usage on a real terminal, with and without colors."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path

import pyte
import pytest

from localmelo.tests.fixtures.stub_llm import StubServer

from .test_e2e_pty import Melo
from .test_e2e_pty import melo as melo
from .test_e2e_pty import stub as stub

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")

USAGE = re.compile(r"↑ \d+ ↓ \d+ · ctx \d+")


def _bar(term: Melo, columns: int) -> str:
    """The bottom row of the screen, which is the status bar, once it has settled."""
    term.read_until(lambda out: False, 1.0)  # what the turn's last event draws
    screen = pyte.Screen(columns, 30)
    pyte.Stream(screen).feed(term.output.decode("utf-8", "replace"))
    return [row.rstrip() for row in screen.display if row.strip()][-1]


@pytest.mark.parametrize("no_color", ["1", ""], ids=["NO_COLOR", "colors"])
@pytest.mark.parametrize("columns", [100, 80])
def test_the_bar_says_the_tokens_and_the_context(
    melo: Callable[..., Melo], columns: int, no_color: str
) -> None:
    term = melo(columns=columns, env={"NO_COLOR": no_color})
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Hello! I am melo, running on a stub model.")
    term.see("Worked for")
    term.see("ctx ")
    bar = _bar(term, columns)
    assert len(bar) <= columns
    assert USAGE.search(bar) and "│ hello" in bar, bar
    assert bar.endswith("ask · shift+tab"), bar  # the mode and its key, always
    # The welcome and /status name the model and the folder; a cache the
    # server did not report is not shown at all.
    assert "stub-model" not in bar and "demo" not in bar and "cache" not in bar
    assert term.quit() == 0


def test_a_narrow_bar_drops_the_title_and_keeps_the_figures_whole(
    melo: Callable[..., Melo],
) -> None:
    term = melo(columns=28)
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Worked for")  # the answer wraps at this width
    term.see("ctx ")
    bar = _bar(term, 28)
    assert len(bar) <= 28 and USAGE.search(bar), bar
    assert "hello" not in bar
    assert "…" not in bar and "..." not in bar, bar  # a figure is never cut
    assert term.quit() == 0


def test_a_cloud_bar_says_the_cost_and_the_limit_its_response_reported(
    melo: Callable[..., Melo], stub: StubServer, tmp_path: Path
) -> None:
    folder = tmp_path / ".cache" / "localmelo"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.toml").write_text(
        'chat_backend = "openai"\nembedding_backend = "none"\n\n[openai]\n'
        f'api_key_env = "STUB_KEY"\nchat_model = "stub-model"\n'
        f'base_url = "{stub.base_url}"\n\n'
        '[pricing."stub-model"]\ninput = 3.0\noutput = 15.0\ncache_read = 0.3\n',
        encoding="utf-8",
    )
    term = melo(configured=False, env={"STUB_KEY": "stub", "NO_COLOR": ""})
    term.see("Ask melo")
    term.type("What did this cost?\r")
    term.see("Worked for")
    term.see("limit ")
    bar = _bar(term, 100)
    # 3k uncached at $3, 9k cached at $0.30, 800 out at $15, per million tokens.
    assert bar.startswith("  $0.02 · ctx 12k"), bar
    assert re.search(r"limit 42% ▮▮▯▯▯ ↻[56]m", bar), bar
    assert "↑" not in bar  # priced: the cost, not the tokens
    assert term.quit() == 0


@pytest.mark.parametrize("no_color", ["1", ""], ids=["NO_COLOR", "colors"])
def test_shift_tab_switches_ask_and_auto_and_the_bar_follows(
    melo: Callable[..., Melo], no_color: str
) -> None:
    term = melo(columns=100, env={"NO_COLOR": no_color})
    term.see("Ask melo")
    assert _bar(term, 100).endswith("ask · shift+tab")
    term.type(b"\x1b[Z")  # Shift+Tab
    term.see("Auto: routine steps run; risky ones are blocked with the reason.")
    assert _bar(term, 100).endswith("auto · shift+tab")
    term.type(b"\x1b[Z")
    term.see("Ask: melo asks before running commands or changing files.")
    assert _bar(term, 100).endswith("ask · shift+tab")
    # Tab still completes: the key did not type or complete anything.
    term.type("/hel\t")
    term.see("/help")
    term.type(b"\x03")  # Ctrl+C clears the draft; Ctrl+D quits an empty one
    assert term.quit() == 0


def test_a_narrow_bar_keeps_the_mode_and_drops_the_key(
    melo: Callable[..., Melo],
) -> None:
    term = melo(columns=50)
    term.see("Ask melo")
    term.type(b"\x1b[Z")
    term.see("Auto: routine steps run")
    bar = _bar(term, 50)
    assert bar.endswith("auto") and "shift+tab" not in bar, bar
    assert term.quit() == 0


def test_ascii_draws_the_mode_and_its_key_in_ascii(melo: Callable[..., Melo]) -> None:
    term = melo(columns=100, env={"LOCALMELO_ASCII": "1"})
    term.see("Ask melo")
    assert _bar(term, 100).endswith("ask - shift+tab")
    term.type(b"\x1b[Z")
    term.see("Auto: routine steps run")
    assert _bar(term, 100).endswith("auto - shift+tab")
    assert term.quit() == 0
    term.output.decode("ascii")  # nothing else reached the terminal
