"""The files a task was asked to write, and the ones missing from its folder (#440)."""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.support.gateway.task_files import asked_files, missing_files


@pytest.mark.parametrize(
    ("text", "names"),
    [
        (
            "Read my site, then write the advice to career-advice.md",
            ["career-advice.md"],
        ),
        ("save it as jobs.csv", ["jobs.csv"]),
        ("Create notes/summary.md with the key points", ["notes/summary.md"]),
        ("export the table into out/table.csv and tell me", ["out/table.csv"]),
        ("把建议写到 career-advice.md", ["career-advice.md"]),
        ("整理一下，保存为 report.md", ["report.md"]),
        ("结果存到「results.json」里", ["results.json"]),
        # Names read, fixed or forbidden are not asked for.
        ("summarize notes.md", []),
        ("fix the bug in app.py", []),
        ("write tests for parser.py", []),
        ("don't write anything to notes.md", []),
        ("不要写到 notes.md", []),
        ("go to my website and tell me which jobs fit me", []),
    ],
)
def test_the_files_the_words_ask_for(text: str, names: list[str]) -> None:
    assert asked_files([text]) == names


def test_each_name_once_across_the_messages() -> None:
    texts = ["write it to a.md", "then save the table as b.csv", "and update a.md"]
    assert asked_files(texts) == ["a.md", "b.csv"]


def test_a_file_is_missing_unless_it_is_in_the_folder(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "table.csv").write_text("a,b\n")
    (tmp_path / "deep" / "er").mkdir(parents=True)
    (tmp_path / "deep" / "er" / "advice.md").write_text("# Advice\n")
    names = ["out/table.csv", "advice.md", "career-advice.md", "out/advice.md"]
    # A bare name is found anywhere under the folder; a path only where written.
    assert missing_files(str(tmp_path), names) == ["career-advice.md", "out/advice.md"]
    assert missing_files(None, ["career-advice.md"]) == ["career-advice.md"]
