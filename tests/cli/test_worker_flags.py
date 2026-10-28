"""Worker flags are opt-in and restricted to direct or import mode."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from localmelo.__main__ import _build_parser, _check_args, main


@pytest.mark.parametrize(
    "argv",
    [
        ["--save-session"],
        ["--save-session", "--serve"],
        ["--parent-run", "abc", "--", "task"],
        ["--import-session", "wrong", "log"],
        ["--import-session", "dsh-jsonl", "log", "--", "task"],
        ["--title", "name"],
    ],
)
def test_invalid_worker_modes_are_usage_errors(argv: list[str]) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        _check_args(parser, parser.parse_args(argv))
    assert exc.value.code == 2


def test_worker_flags_are_explicit() -> None:
    parser = _build_parser()
    direct = parser.parse_args(
        ["--save-session", "--parent-run", "run-1", "--", "task"]
    )
    _check_args(parser, direct)
    assert direct.save_session == "" and direct.parent_run == "run-1"
    imported = parser.parse_args(
        ["--import-session", "dsh-jsonl", "log", "--title", "Example"]
    )
    _check_args(parser, imported)
    assert imported.import_session == ["dsh-jsonl", "log"]


def test_import_mode_runs_without_configuration(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"type": "final", "text": "done"}) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["melo", "--import-session", "dsh-jsonl", str(log), "--title", "Example"],
    )
    main()
    assert "imported" in capsys.readouterr().out
    main()
    assert "already imported" in capsys.readouterr().out
