"""The first-run list and the progress of the setup job (support/server_setup/terminal.py).

Both are text only: the list is checked on a plan made from data, the progress on
a stream in memory.
"""

from __future__ import annotations

import io
import threading
from typing import Any

import pytest

from localmelo.support import progress_line
from localmelo.support.cli.theme import ASCII
from localmelo.support.server_setup import run
from localmelo.support.server_setup.listing import size_left
from localmelo.support.server_setup.terminal import Progress, consent_lines

CHAT, EMBEDDING = "model:qwen3-8b-omlx", "model:qwen3-embedding-0.6b-omlx"


def _plan(**states: str) -> run.Plan:
    """A plan of a Mac with 83 GB free; *states* sets an item's state by id."""
    items = [
        {"id": "backend:omlx", "kind": "backend", "name": "oMLX 0.7.3", "via": "dmg"}
        | {"size_bytes": 881_000_000, "dest": "~/Applications/oMLX.app"},
        {"id": CHAT, "kind": "model", "name": "Qwen3 8B", "role": "chat"}
        | {"via": "huggingface", "size_bytes": 4_600_000_000}
        | {"dest": "~/models/huggingface", "backend": "omlx"},
        {"id": EMBEDDING, "kind": "model", "name": "Qwen3 Embedding 0.6B"}
        | {"via": "huggingface", "size_bytes": 1_200_000_000, "role": "embedding"}
        | {"dest": "~/models/huggingface", "backend": "omlx"},
    ]
    for item in items:
        item["state"] = states.get(str(item["id"]), "todo")
    return run._parse(
        {
            "schema": "melo_serve.plan/1",
            "supported": True,
            "machine": {"chip": "Apple M5 Max", "ram_gb": 128, "macos": "27.2"}
            | {"free_gb": 83.0},
            "pick": {"id": "qwen3-8b-omlx", "name": "Qwen3 8B", "backend": "omlx"},
            "items": items,
            "download_bytes": 6_681_000_000,
            "alternatives": [],
        }
    )


def test_the_list_is_one_aligned_table_with_sizes_and_places() -> None:
    assert consent_lines(_plan(), set(), "") == [
        "This Mac: Apple M5 Max · 128 GB · macOS 27.2 · 83 GB free",
        "Recommended: Qwen3 8B on oMLX. Nothing is installed until you say yes.",
        "",
        "  melo_serve            model service    about 30 MB   uv tool",
        "  oMLX 0.7.3            chat backend          881 MB   pinned download → ~/Applications/oMLX.app",
        "  Qwen3 8B              chat model            4.6 GB   Hugging Face → ~/models/huggingface",
        "  Qwen3 Embedding 0.6B  memory model          1.2 GB   same place",
        '  Start at login        melo_serve                     macOS shows a "Login Items" notice',
        "  Start at login        the gateway                    the same notice",
        "Download about 6.7 GB. 83 GB free.",
    ]


def test_the_list_says_what_is_here_what_was_left_out_and_what_that_leads_to() -> None:
    plan = _plan(**{"backend:omlx": "done", CHAT: "partial"})
    items = tuple(
        run.Item(**{**i.__dict__, "cached_bytes": 3_000_000_000}) if i.id == CHAT else i
        for i in plan.items
    )
    plan = run.Plan(
        **{**plan.__dict__, "items": items, "download_bytes": 2_800_000_000}
    )
    skip = {EMBEDDING, run.SERVICE_MELO_SERVE}
    assert (
        size_left(plan, skip) == 1_600_000_000
    )  # what is here and what is skipped is not counted
    assert consent_lines(plan, skip, "0.4.0", chosen=True)[1:] == [
        "Chosen: Qwen3 8B on oMLX. Nothing is installed until you say yes.",
        "",
        "  melo_serve            model service             ✓ already here",
        "  oMLX 0.7.3            chat backend              ✓ already here",
        "  Qwen3 8B              chat model       1.6 GB   3.0 of 4.6 GB already here",
        "  Qwen3 Embedding 0.6B  memory model              skipped",
        "  Start at login        melo_serve                skipped",
        "  Start at login        the gateway               the same notice",
        "Download about 1.6 GB. 83 GB free.",
        "Without Qwen3 Embedding 0.6B, memory goes by words.",
        "Without the login item, melo serve starts melo_serve when needed.",
    ]
    assert consent_lines(plan, {CHAT}, "")[-1] == (
        "Without Qwen3 8B, melo_serve cannot chat yet; "
        "melo-serve get qwen3-8b-omlx finishes it later."
    )


def test_a_narrow_terminal_puts_where_it_goes_under_each_row_and_ascii_stays_ascii() -> (
    None
):
    lines = consent_lines(_plan(), (), "", width=80, symbols=ASCII)
    assert "  oMLX 0.7.3            chat backend          881 MB" in lines
    assert (" " * 24 + "pinned download > ~/Applications/oMLX.app") in lines
    assert all(line.isascii() for line in lines)


# ── Progress ──


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _events(progress: Progress) -> None:
    progress(
        run.ItemEvent(
            "backend:omlx", "running", "downloading", 310_000_000, 881_000_000
        )
    )
    progress(run.ItemEvent(CHAT, "running", "", 1_200_000_000, 4_600_000_000))
    progress(
        run.ItemEvent(CHAT, "done", "downloaded Qwen3 8B into ~/models/huggingface")
    )
    progress(run.ItemEvent(EMBEDDING, "skipped", run.SKIPPED))
    progress(run.ItemEvent(run.LOAD_CHAT, "running"))
    progress(run.ItemEvent(run.LOAD_CHAT, "failed", "failed: it did not answer"))


def test_a_terminal_gets_a_live_line_for_a_row_and_one_closing_line_each() -> None:
    out = _Tty()
    _events(Progress(_plan(), out))
    assert out.getvalue().split(progress_line.ERASE) == [
        "  oMLX  ▓▓▓▓░░░░░░  310 / 881 MB",
        "  Qwen3 8B  ▓▓▓░░░░░░░  1.2 / 4.6 GB",
        "✓ Qwen3 8B  downloaded Qwen3 8B into ~/models/huggingface\n"
        "· Qwen3 Embedding 0.6B  skipped (you chose to skip it)\n"
        "  Loading Qwen3 8B…",
        "✗ Loading Qwen3 8B  failed: it did not answer\n",
    ]


def test_a_live_line_is_cut_to_the_width(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(progress_line, "terminal_columns", lambda stream: 20)
    out = _Tty()
    Progress(_plan(), out)(run.ItemEvent(CHAT, "running", "", 1, 2))
    drawn = out.getvalue().removeprefix(progress_line.ERASE)
    assert len(drawn) == 19 and drawn.endswith("…")


def test_a_pipe_gets_closing_lines_only() -> None:
    out = io.StringIO()
    progress = Progress(_plan(), out)
    _events(progress)
    progress(run.Finished("failed", "failed: it did not answer"))  # shown already
    progress(run.Finished("failed", "failed: something else"))
    assert out.getvalue() == (
        "✓ Qwen3 8B  downloaded Qwen3 8B into ~/models/huggingface\n"
        "· Qwen3 Embedding 0.6B  skipped (you chose to skip it)\n"
        "✗ Loading Qwen3 8B  failed: it did not answer\n"
        "✗ failed: something else\n"
    )


def test_the_rows_of_the_job_have_plain_names_and_an_unknown_one_shows_its_id() -> None:
    out = io.StringIO()
    progress = Progress(_plan(), out)
    for row in (
        run.MELO_SERVE,
        run.SERVICE_MELO_SERVE,
        run.SAVE,
        "config:hf_home",
        "odd:thing",
    ):
        progress(run.ItemEvent(row, "done", "ok"))
    assert out.getvalue().splitlines() == [
        "✓ melo_serve  ok",
        "✓ Start at login  ok",
        "✓ Saving the settings  ok",
        "✓ Model folder  ok",
        "✓ thing  ok",
    ]


def test_events_from_another_thread_are_whole_and_a_broken_stream_raises_nothing() -> (
    None
):
    out = io.StringIO()
    progress = Progress(_plan(), out)
    threads = [
        threading.Thread(target=progress, args=(run.ItemEvent(f"x:{n}", "done", "ok"),))
        for n in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(out.getvalue().splitlines()) == sorted(
        f"✓ {n}  ok" for n in map(str, range(20))
    )

    class Broken(io.StringIO):
        def write(self, text: str) -> Any:
            raise OSError("closed")

    Progress(_plan(), Broken())(run.ItemEvent("x", "done", "ok"))
