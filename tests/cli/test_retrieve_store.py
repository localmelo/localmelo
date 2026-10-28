"""``melo-retrieve``'s store exports and the offline ``--route`` mode."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from localmelo.support import retrieve

CANDIDATES = [
    {
        "candidate_id": "c1",
        "source": "user",
        "text": "I live in Berlin near the river and work at Acme",
        "signal": "user_statement",
        "task_id": "t1",
    },
    {
        "candidate_id": "c2",
        "source": "user",
        "text": "I live in Berlin near the park and work at Acme",
        "signal": "user_statement",
        "task_id": "t1",
    },
    {
        "candidate_id": "c3",
        "source": "user",
        "text": "I live in Paris in a large house",
        "signal": "user_statement",
        "task_id": "t1",
    },
    {
        "candidate_id": "c4",
        "source": "user",
        "text": "Alice's address is 12 Main Street in Springfield",
        "signal": "user_statement",
        "task_id": "t2",
    },
    {
        "candidate_id": "c5",
        "source": "user",
        "text": "Bob's address is 12 Main Street in Springfield",
        "signal": "user_statement",
        "task_id": "t2",
    },
    {
        "candidate_id": "c6",
        "source": "user",
        "text": "Bob's address is 12 Main Street in Springfield",
        "signal": "user_statement",
        "task_id": "t2",
    },
    {
        "candidate_id": "c7",
        "source": "user",
        "text": "\x00",
        "signal": "user_statement",
        "task_id": "t2",
    },
]


def _candidates(path: Path, rows: list[dict[str, Any]]) -> str:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return str(path)


def _lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return _lines_from(capsys.readouterr())


def _lines_from(out: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.out.splitlines() if line]


def test_route_prints_one_decision_per_candidate_in_input_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    candidates = _candidates(tmp_path / "c.jsonl", CANDIDATES)
    assert (
        retrieve.main(["--route", candidates, "--memory-dir", str(tmp_path / "m")]) == 0
    )
    rows = _lines(capsys)
    assert [row["candidate_id"] for row in rows] == [
        f"c{index}" for index in range(1, 8)
    ]
    assert [row["action"] for row in rows] == [
        "NEW",
        "UPDATE",
        "CONFLICT",
        "NEW",
        "RELATED",
        "IGNORE",
        "REFUSED",
    ]
    assert rows[-1]["reason"]


def test_the_route_scratch_store_exports_what_it_wrote(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "m"
    candidates = _candidates(tmp_path / "c.jsonl", CANDIDATES[:2])
    assert retrieve.main(["--route", candidates, "--memory-dir", str(directory)]) == 0
    capsys.readouterr()
    assert retrieve.main(["--export-store", str(directory)]) == 0
    rows = _lines(capsys)
    assert [row["text"] for row in rows] == [
        "I live in Berlin near the park and work at Acme"
    ]
    assert rows[0]["status"] == "active"
    assert rows[0]["supersedes"] == 1


def test_route_redacts_the_provenance_and_the_export_shows_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key = "sk-" + "a" * 24
    directory = tmp_path / "m"
    row = {**CANDIDATES[0], "task_text": f"deploy with {key}"}
    candidates = _candidates(tmp_path / "c.jsonl", [row])
    assert retrieve.main(["--route", candidates, "--memory-dir", str(directory)]) == 0
    capsys.readouterr()
    assert retrieve.main(["--export-store", str(directory)]) == 0
    out = capsys.readouterr()
    assert key not in out.out
    rows = _lines_from(out)
    assert rows[0]["provenance"]["task_text"] == "deploy with [redacted]"


def test_export_decisions_prints_seq_order_and_text_null_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "m"
    candidates = _candidates(tmp_path / "c.jsonl", CANDIDATES[:2])
    assert retrieve.main(["--route", candidates, "--memory-dir", str(directory)]) == 0
    capsys.readouterr()
    assert retrieve.main(["--export-decisions", str(directory)]) == 0
    rows = _lines(capsys)
    assert [row["seq"] for row in rows] == [1, 2]
    assert [row["action"] for row in rows] == ["NEW", "UPDATE"]
    assert [row["text"] for row in rows] == [None, None]
    assert all(row["text_sha256"] for row in rows)
    assert isinstance(rows[0]["source_ids"], list)


def test_export_store_reads_a_v0_file_without_migrating_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "long_term.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE long_term (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "text TEXT NOT NULL, vector BLOB NOT NULL, "
        "metadata TEXT NOT NULL DEFAULT '{}')"
    )
    conn.execute(
        "INSERT INTO long_term (text, vector, metadata) VALUES (?, ?, ?)",
        ("an old reflection", b"\x00\x00\x00\x00", '{"type": "reflection"}'),
    )
    conn.commit()
    conn.close()
    assert retrieve.main(["--export-store", str(path)]) == 0
    rows = _lines(capsys)
    assert rows[0]["kind"] == "reflection"
    assert rows[0]["source"] == "reflection"
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0


def test_a_newer_file_is_one_line_and_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "long_term.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE long_term (id INTEGER PRIMARY KEY, text TEXT, vector BLOB, "
        "metadata TEXT)"
    )
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()
    assert retrieve.main(["--export-store", str(path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("melo-retrieve: error: ")
    assert "version 2" in err


def test_a_missing_file_is_one_line_and_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert retrieve.main(["--export-store", str(tmp_path / "missing.db")]) == 2
    assert "No such file" in capsys.readouterr().err
    assert retrieve.main(["--export-decisions", str(tmp_path / "missing.db")]) == 2
    assert "No such file" in capsys.readouterr().err


def test_a_directory_without_a_store_exports_zero_and_exit_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # T59: a directory with no long_term.db is an empty store, not a usage
    # error; the normal output shape is zero lines plus the count on stderr.
    directory = tmp_path / "no-store"
    directory.mkdir()
    assert retrieve.main(["--export-store", str(directory)]) == 0
    out = capsys.readouterr()
    assert out.out == ""
    assert f"exported 0 active entries from {directory / 'long_term.db'}" in out.err
    assert retrieve.main(["--export-decisions", str(directory)]) == 0
    out = capsys.readouterr()
    assert out.out == ""
    assert f"exported 0 decisions from {directory / 'long_term.db'}" in out.err
    # A path that is not a directory keeps the usage error.
    assert retrieve.main(["--export-store", str(tmp_path / "no-store.db")]) == 2
    assert "No such file" in capsys.readouterr().err


def test_a_directory_with_a_store_still_exports_its_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "m"
    candidates = _candidates(tmp_path / "c.jsonl", CANDIDATES[:1])
    assert retrieve.main(["--route", candidates, "--memory-dir", str(directory)]) == 0
    capsys.readouterr()
    assert retrieve.main(["--export-store", str(directory)]) == 0
    assert len(_lines(capsys)) == 1


def test_the_three_modes_are_mutually_exclusive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    candidates = _candidates(tmp_path / "c.jsonl", CANDIDATES[:1])
    assert (
        retrieve.main(
            [
                "--export-store",
                str(tmp_path),
                "--export-decisions",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert retrieve.main(["--export-store", str(tmp_path), "--retriever", "bm25"]) == 2
    assert (
        retrieve.main(
            [
                "--export-run",
                str(tmp_path),
                "--route",
                candidates,
                "--memory-dir",
                str(tmp_path / "m"),
            ]
        )
        == 2
    )
    assert "cannot be combined" in capsys.readouterr().err


def test_the_new_modes_accept_and_ignore_model_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # T53R finding 9 / Requirement 12: no model flag is needed or refused.
    directory = tmp_path / "m"
    candidates = _candidates(tmp_path / "c.jsonl", CANDIDATES[:1])
    model_flags = [
        "--base-url",
        "http://127.0.0.1:8000/v1",
        "--chat-model",
        "qwen",
        "--embedding-base-url",
        "http://127.0.0.1:8000/v1",
        "--embedding-model",
        "embed",
        "--retrieval-timeout",
        "5",
    ]
    assert (
        retrieve.main(
            [
                "--route",
                candidates,
                "--memory-dir",
                str(directory),
                *model_flags,
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert retrieve.main(["--export-store", str(directory), *model_flags]) == 0
    capsys.readouterr()
    assert retrieve.main(["--export-decisions", str(directory), *model_flags]) == 0


def test_export_run_refuses_memory_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        retrieve.main(["--export-run", str(tmp_path), "--memory-dir", str(tmp_path)])
        == 2
    )
    assert "--memory-dir applies only to --route" in capsys.readouterr().err


def test_route_needs_an_explicit_scratch_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    candidates = _candidates(tmp_path / "c.jsonl", CANDIDATES[:1])
    assert retrieve.main(["--route", candidates]) == 2
    assert "--route needs --memory-dir" in capsys.readouterr().err


def test_route_refuses_the_user_s_memory_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    candidates = _candidates(tmp_path / "c.jsonl", CANDIDATES[:1])
    monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(tmp_path / "user-memory"))
    assert (
        retrieve.main(
            ["--route", candidates, "--memory-dir", str(tmp_path / "user-memory")]
        )
        == 2
    )
    assert "must not be the user's memory directory" in capsys.readouterr().err


def test_an_unknown_signal_or_source_is_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_signal = _candidates(
        tmp_path / "bad-signal.jsonl",
        [{**CANDIDATES[0], "signal": "because"}],
    )
    assert (
        retrieve.main(["--route", bad_signal, "--memory-dir", str(tmp_path / "m")]) == 2
    )
    assert "unknown signal" in capsys.readouterr().err
    bad_source = _candidates(
        tmp_path / "bad-source.jsonl",
        [{**CANDIDATES[0], "source": "step_summary"}],
    )
    assert (
        retrieve.main(["--route", bad_source, "--memory-dir", str(tmp_path / "m")]) == 2
    )
    assert "unknown source" in capsys.readouterr().err


def test_a_route_without_a_model_flag_never_needs_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    candidates = _candidates(tmp_path / "c.jsonl", CANDIDATES[:1])
    assert (
        retrieve.main(["--route", candidates, "--memory-dir", str(tmp_path / "m")]) == 0
    )
    assert "routed 1 candidates" in capsys.readouterr().err


def test_the_export_store_matches_the_bench_reader_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    candidates = _candidates(tmp_path / "c.jsonl", CANDIDATES[:1])
    assert (
        retrieve.main(["--route", candidates, "--memory-dir", str(tmp_path / "m")]) == 0
    )
    capsys.readouterr()
    assert retrieve.main(["--export-store", str(tmp_path / "m")]) == 0
    row = _lines(capsys)[0]
    # The T4 reader reads these two keys (eval/memory/native_t4_and_d11.md).
    assert isinstance(row["text"], str)
    assert isinstance(row["provenance"]["source_ids"], list)
