"""Reading a file_write target and the approval panels' diff (support/sessions/edits.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from localmelo.support.sessions import edits


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
        mock.patch(
            "localmelo.support.sessions.edits.file_facts", return_value=(True, 9)
        ),
        mock.patch("os.open", side_effect=PermissionError(13, "Permission denied")),
    ):
        target = edits.read_target(str(path))
    assert target.exists and not target.diffable
    assert target.problem == "cannot be read"


# ── The console's diff: approval_diff ──

KEY = "sk-" + "a1b2c3d4e5f6g7h8i9j0k1l2"
PEM = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ\n"
    "AAAAB3NzaC1yc2EAAAADAQABAAABgQDXXXXXXXXXXXX\n"
    "-----END OPENSSH PRIVATE KEY-----\n"
)


def _replace(tmp_path: Path, old: str | bytes, new: str) -> dict:
    path = tmp_path / "notes.txt"
    if isinstance(old, bytes):
        path.write_bytes(old)
    else:
        path.write_text(old, encoding="utf-8")
    return edits.approval_diff(str(path), new)


def test_the_console_diff_gives_kinds_in_words_and_line_numbers(
    tmp_path: Path,
) -> None:
    old = "".join(f"line {number}\n" for number in range(1, 21))
    new = old.replace("line 2\n", "two\n").replace("line 19\n", "nineteen\n")
    result = _replace(tmp_path, old, new)
    assert (result["added"], result["removed"], result["note"]) == (2, 2, "")
    rows = [(row["kind"], row["line"], row["text"]) for row in result["rows"]]
    assert rows[:5] == [
        ("context", 1, "line 1"),
        ("removed", 2, "line 2"),
        ("added", 2, "two"),
        ("context", 3, "line 3"),
        ("context", 4, "line 4"),
    ]
    assert ("gap", None, "") in rows


def test_text_the_file_already_holds_has_no_rows(tmp_path: Path) -> None:
    assert _replace(tmp_path, "same\n", "same\n") == {
        "added": 0,
        "removed": 0,
        "rows": [],
        "note": "",
    }


def test_a_secret_is_hidden_in_every_row_and_a_changed_secret_still_shows(
    tmp_path: Path,
) -> None:
    other = "sk-" + "z9y8x7w6v5u4t3s2r1q0p9o8"
    old = f"name = a\nkey = {KEY}\nAuthorization: Bearer abcdefghijklmnopqrstuvwxyz\n"
    new = f"name = b\nkey = {other}\nAuthorization: Bearer abcdefghijklmnopqrstuvwxyz\n"
    result = _replace(tmp_path, old, new)
    shown = json.dumps(result)
    assert KEY not in shown and other not in shown and "abcdefghijklmnop" not in shown
    texts = [(row["kind"], row["text"]) for row in result["rows"]]
    assert ("removed", "key = [redacted]") in texts
    assert ("added", "key = [redacted]") in texts
    assert ("context", "Authorization: Bearer [redacted]") in texts
    assert (result["added"], result["removed"]) == (2, 2)


@pytest.mark.parametrize("where", ["file", "text"])
def test_a_private_key_is_never_shown(tmp_path: Path, where: str) -> None:
    old, new = ("notes\n" + PEM, "x\n") if where == "file" else ("x\n", "y\n" + PEM)
    result = _replace(tmp_path, old, new)
    assert result["rows"] == []
    assert "private key" in result["note"]
    assert "b3BlbnNzaC1" not in json.dumps(result)
    # A block with one body line keeps its lines, so its rows show it hidden.
    short = PEM.replace("AAAAB3NzaC1yc2EAAAADAQABAAABgQDXXXXXXXXXXXX\n", "")
    rows = _replace(tmp_path, "notes\n" + short, "x\n")["rows"]
    assert "[redacted]" in [row["text"] for row in rows]
    assert "b3BlbnNzaC1" not in json.dumps(rows)


def test_what_cannot_be_compared_is_told_in_words(tmp_path: Path) -> None:
    said = "melo cannot show what this replaces: "
    assert _replace(tmp_path, b"\xff\xfe\x00", "x")["note"] == said + (
        "the file is not text."
    )
    huge = "x" * (edits.DIFF_MAX_BYTES + 1)
    assert _replace(tmp_path, huge, "x")["note"] == said + "the file is too large."
    assert _replace(tmp_path, "x", huge)["note"] == said + "the new text is too large."
    assert "too large" not in _replace(tmp_path, "x", "y")["note"]
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    assert edits.approval_diff(str(fifo), "x")["note"] == said + (
        "it is not a regular file."
    )
    for problem, why in (
        ("cannot be read", "the file cannot be read."),
        ("", "the file cannot be compared."),
    ):
        target = edits.Target(path="x", exists=True, problem=problem)
        with mock.patch.object(edits, "read_target", return_value=target):
            assert edits.approval_diff("x", "y")["note"] == said + why


def test_a_change_of_many_lines_skips_the_diff(tmp_path: Path) -> None:
    old = "".join(f"old {number}\n" for number in range(600))
    new = "".join(f"new {number}\n" for number in range(600))
    with mock.patch.object(edits, "diff", side_effect=AssertionError("ran")):
        result = _replace(tmp_path, old, new)
    assert result["rows"] == [] and result["note"].endswith("too many lines change.")


def test_a_reordering_has_too_many_rows_though_every_line_is_in_both(
    tmp_path: Path,
) -> None:
    lines = [f"line {number}\n" for number in range(700)]
    result = _replace(tmp_path, "".join(lines), "".join(reversed(lines)))
    assert result["rows"] == [] and result["note"].endswith("too many lines change.")


def test_a_long_line_is_shown_whole(tmp_path: Path) -> None:
    line = "x" * 5_000
    result = _replace(tmp_path, "short\n", line + "\n")
    assert [row["text"] for row in result["rows"] if row["kind"] == "added"] == [line]
