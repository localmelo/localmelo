"""file_read by line range, and the note a cut read ends with.

A read that shows only a part of a file says how many lines the file has and
which line to go on from, so the model reads the rest with ``start_line``
instead of slicing the file through Python.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import localmelo.melo.executor.builtins as builtin_tools
from localmelo.melo.checker import Checker
from localmelo.melo.checker.validators import MAX_OUTPUT_LEN
from localmelo.melo.executor import (
    ErrorCategory,
    ExecutionOutcome,
    ExecutionRequest,
    ExecutionStatus,
    Executor,
    builtin_names,
    register_builtins,
)
from localmelo.melo.memory.coordinator import Hippo

LINES = 1_240


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "root"
    path.mkdir()
    return Path(os.path.realpath(path))


def _executor(root: Path | None, *, rooted_tools: bool = True) -> Executor:
    hippo = Hippo()
    workspace_root = None if root is None else str(root)
    executor = Executor(
        hippo, Checker(mode="interactive"), workspace_root=workspace_root
    )
    register_builtins(
        executor,
        hippo,
        names=builtin_names(),
        mode="interactive",
        workspace_root=workspace_root if rooted_tools else None,
    )
    return executor


async def _read(executor: Executor, path: str, **arguments: Any) -> ExecutionOutcome:
    return await executor.execute_structured(
        ExecutionRequest("file_read", {"path": path, **arguments})
    )


def _numbered(count: int = LINES) -> str:
    """*count* lines of 62 characters: ``line 0001 xxx…``."""
    return "".join(f"line {i:04d} {'x' * 51}\n" for i in range(1, count + 1))


@pytest.fixture
def big(root: Path) -> Path:
    path = root / "big.txt"
    path.write_text(_numbered())
    return path


def _shown_lines(output: str) -> list[int]:
    """The numbers of the file's lines in *output*."""
    return [
        int(line.split()[1]) for line in output.splitlines() if line.startswith("line ")
    ]


async def test_a_long_file_without_a_range_says_where_to_go_on(
    root: Path, big: Path
) -> None:
    outcome = await _read(_executor(root), "big.txt")

    assert outcome.status == ExecutionStatus.SUCCESS
    shown = _shown_lines(outcome.output)
    assert shown == list(range(1, len(shown) + 1)) and 0 < len(shown) < LINES
    assert len(outcome.output) <= MAX_OUTPUT_LEN
    assert outcome.output.endswith(
        f"... [truncated: the file has 1,240 lines; showing 1-{len(shown)}; "
        f"read more with start_line={len(shown) + 1}]"
    )
    # Going on from the line it names reads the next line of the file.
    more = await _read(_executor(root), "big.txt", start_line=len(shown) + 1)
    assert _shown_lines(more.output)[0] == len(shown) + 1


async def test_a_range_shows_those_lines_and_where_the_file_goes_on(
    root: Path, big: Path
) -> None:
    outcome = await _read(_executor(root), "big.txt", start_line=1, end_line=400)

    assert _shown_lines(outcome.output) == list(range(1, 401))
    assert outcome.output.endswith(
        "... [the file has 1,240 lines; showing 1-400; read more with start_line=401]"
    )


async def test_a_range_in_the_middle_and_one_line(root: Path, big: Path) -> None:
    executor = _executor(root)

    middle = await _read(executor, "big.txt", start_line=600, end_line=605)
    one = await _read(executor, "big.txt", start_line=7, end_line=7)

    assert _shown_lines(middle.output) == [600, 601, 602, 603, 604, 605]
    assert middle.output.endswith(
        "... [the file has 1,240 lines; showing 600-605; read more with start_line=606]"
    )
    assert one.output == (
        f"line 0007 {'x' * 51}\n"
        "... [the file has 1,240 lines; showing 7-7; read more with start_line=8]"
    )


async def test_a_range_to_the_end_has_nothing_to_go_on_to(
    root: Path, big: Path
) -> None:
    executor = _executor(root)

    tail = await _read(executor, "big.txt", start_line=1_201)
    past = await _read(executor, "big.txt", start_line=1_201, end_line=9_999)

    for outcome in (tail, past):
        assert _shown_lines(outcome.output) == list(range(1_201, 1_241))
        assert outcome.output.endswith(
            "... [the file has 1,240 lines; showing 1201-1240]"
        )


async def test_a_range_over_a_short_file_returns_the_text_as_it_is(root: Path) -> None:
    (root / "small.txt").write_text(_numbered(30))
    executor = _executor(root)

    whole = await _read(executor, "small.txt")
    ranged = await _read(executor, "small.txt", start_line=1, end_line=100)

    assert whole.output == ranged.output == _numbered(30)


async def test_a_line_past_the_end_is_an_error_that_counts_the_lines(
    root: Path, big: Path
) -> None:
    outcome = await _read(_executor(root), "big.txt", start_line=2_000)

    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error == "start_line 2000 is past the end: the file has 1,240 lines"


@pytest.mark.parametrize(("start", "end"), [(0, None), (5, 4), (-1, 3)])
async def test_a_range_that_counts_wrong_is_an_error(
    root: Path, big: Path, start: int, end: int | None
) -> None:
    arguments: dict[str, Any] = {"start_line": start}
    if end is not None:
        arguments["end_line"] = end

    outcome = await _read(_executor(root), "big.txt", **arguments)

    assert outcome.status == ExecutionStatus.ERROR
    assert outcome.error_category == ErrorCategory.RUNTIME_ERROR
    assert "count from 1" in outcome.error


async def test_a_file_with_one_long_line_keeps_the_byte_note(root: Path) -> None:
    (root / "one.txt").write_text("x" * 300_000)

    outcome = await _read(_executor(root), "one.txt")

    assert outcome.output.endswith("\n... [truncated: the file has 300000 bytes]")


async def test_a_long_first_line_is_cut_and_the_next_line_named(root: Path) -> None:
    (root / "wide.txt").write_text("y" * 60_000 + "\nsecond\n")
    executor = _executor(root)

    outcome = await _read(executor, "wide.txt")
    second = await _read(executor, "wide.txt", start_line=2)

    assert len(outcome.output) == MAX_OUTPUT_LEN
    assert outcome.output.endswith(
        "line 1 is too long, showing its start; read more with start_line=2]"
    )
    assert second.output == "second\n... [the file has 2 lines; showing 2-2]"


async def test_the_lines_of_a_file_over_the_scan_cap_are_counted_as_at_least(
    root: Path, big: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builtin_tools, "MAX_LINE_SCAN_BYTES", 10_000)

    outcome = await _read(_executor(root), "big.txt", start_line=1, end_line=5)

    assert outcome.output.endswith(
        "the file has 162+ lines; showing 1-5; read more with start_line=6]"
    )


async def test_a_file_without_a_final_newline_and_an_empty_file(root: Path) -> None:
    (root / "tail.txt").write_text("a\nb\nc")
    (root / "empty.txt").write_text("")
    executor = _executor(root)

    last = await _read(executor, "tail.txt", start_line=3)
    first = await _read(executor, "tail.txt", start_line=1, end_line=1)
    empty = await _read(executor, "empty.txt")

    assert last.output == "c\n... [the file has 3 lines; showing 3-3]"
    assert (
        first.output
        == "a\n... [the file has 3 lines; showing 1-1; read more with start_line=2]"
    )
    assert empty.output == ""


async def test_the_read_without_a_root_takes_a_range_too(tmp_path: Path) -> None:
    path = tmp_path / "plain.txt"
    path.write_text(_numbered())
    executor = _executor(None)

    ranged = await _read(executor, str(path), start_line=10, end_line=12)
    cut = await _read(executor, str(path))

    assert _shown_lines(ranged.output) == [10, 11, 12]
    assert ranged.output.endswith(
        "... [the file has 1,240 lines; showing 10-12; read more with start_line=13]"
    )
    assert "the file has 1,240 lines; showing 1-" in cut.output
    assert len(cut.output) <= MAX_OUTPUT_LEN
