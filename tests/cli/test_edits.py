"""Reading a file_write target and the approval panel's diff (support/cli/edits.py)."""

from __future__ import annotations

import os
from pathlib import Path

from localmelo.support.cli import edits


def test_a_missing_file_is_a_new_file(tmp_path: Path) -> None:
    target = edits.read_target(str(tmp_path / "missing.txt"))
    assert (target.exists, target.diffable, target.problem) == (False, False, "")


def test_a_text_file_is_read(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("one\ntwo\n", encoding="utf-8")
    target = edits.read_target(str(path))
    assert target.exists and target.diffable
    assert target.text == "one\ntwo\n"
    assert target.size_bytes == len("one\ntwo\n")


def test_a_file_over_the_limit_is_not_read(tmp_path: Path) -> None:
    path = tmp_path / "big"
    path.write_text("x" * (edits.DIFF_MAX_BYTES + 1), encoding="utf-8")
    target = edits.read_target(str(path))
    assert target.exists and not target.diffable
    assert target.problem == "overwrites a 200 KB file"


def test_a_binary_file_is_not_text(tmp_path: Path) -> None:
    path = tmp_path / "blob"
    path.write_bytes(b"\xff\xfe\x00")
    target = edits.read_target(str(path))
    assert target.exists and not target.diffable
    assert target.problem == "is not text"


def test_a_final_symlink_is_not_followed(tmp_path: Path) -> None:
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    target = edits.read_target(str(link))
    assert target.exists and not target.diffable
    assert target.problem == "is not a regular file"


def test_a_fifo_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "pipe"
    os.mkfifo(path)
    target = edits.read_target(str(path))
    assert target.exists and not target.diffable
    assert target.problem == "is not a regular file"


def test_a_directory_is_not_a_regular_file(tmp_path: Path) -> None:
    target = edits.read_target(str(tmp_path))
    assert target.exists and target.problem == "is not a regular file"


def test_the_diff_numbers_lines_and_counts_changes() -> None:
    rows, added, removed = edits.diff("a\nb\nc\nd\ne\n", "a\nB\nc\nd\ne\nf\n")
    assert (added, removed) == (2, 1)
    assert [(row.kind, row.old, row.new, row.text) for row in rows] == [
        (" ", 1, 1, "a"),
        ("-", 2, None, "b"),
        ("+", None, 2, "B"),
        (" ", 3, 3, "c"),
        (" ", 4, 4, "d"),
        (" ", 5, 5, "e"),
        ("+", None, 6, "f"),
    ]


def test_hunks_are_separated_by_an_ellipsis_row() -> None:
    old = "".join(f"{number}\n" for number in range(1, 21))
    new = old.replace("1\n", "one\n", 1).replace("20\n", "twenty\n")
    rows, added, removed = edits.diff(old, new)
    assert (added, removed) == (2, 2)
    assert sum(1 for row in rows if row.kind == "…") == 1
    assert any(row.kind == " " and row.new == 2 for row in rows)


def test_a_missing_file_differs_from_an_empty_one() -> None:
    rows, added, removed = edits.diff("", "new\n")
    assert (added, removed) == (1, 0)
    assert [(row.kind, row.new) for row in rows] == [("+", 1)]


def test_changed_lines_that_look_like_diff_headers_are_kept() -> None:
    rows, added, removed = edits.diff("-- x\nkeep\n", "keep\n")
    assert (added, removed) == (0, 1)
    assert ("-", 1, None, "-- x") in [(r.kind, r.old, r.new, r.text) for r in rows]
    rows, added, removed = edits.diff("keep\n", "++ y\nkeep\n")
    assert (added, removed) == (1, 0)
    assert ("+", None, 1, "++ y") in [(r.kind, r.old, r.new, r.text) for r in rows]


def test_a_line_ending_or_final_newline_change_is_a_change() -> None:
    assert edits.diff("a", "a\n")[1:] == (1, 1)
    assert edits.diff("a\r\nb\r\n", "a\nb\n")[1:] == (2, 2)
    assert edits.diff("keep\n", "keep\n")[1:] == (0, 0)


def test_a_file_that_cannot_be_read_says_so(tmp_path: Path) -> None:
    from unittest import mock

    path = tmp_path / "notes.txt"
    path.write_text("buy milk\n", encoding="utf-8")
    with (
        mock.patch("localmelo.support.cli.edits.file_facts", return_value=(True, 9)),
        mock.patch("os.open", side_effect=PermissionError(13, "Permission denied")),
    ):
        target = edits.read_target(str(path))
    assert target.exists and not target.diffable
    assert target.problem == "cannot be read"
