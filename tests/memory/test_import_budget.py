"""A memory folder's budget: every note file's summary first, round by round, and what did not fit is counted.

The folders here are small invented ones (no real file is read); the home is built under *tmp_path*
and passed to :func:`build_plan` with no notes of the user's own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.support.memory.importer import build_plan
from localmelo.support.memory.importer.plan import PER_SOURCE, ImportPlan
from localmelo.support.memory.importer.rules import FEEDBACK_TOPIC, person, topic_for
from localmelo.support.memory.importer.sources import (
    Statement,
    discover,
    read_source,
)

from .import_home import OWN, SECRET, put


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    found = tmp_path.resolve() / "fake-home"
    found.mkdir()
    put(found / ".gitconfig", f"[user]\n\temail = {OWN}\n")
    monkeypatch.setenv("HOME", str(found))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    return found


def folder(home: Path) -> Path:
    return home / ".claude" / "projects" / "p" / "memory"


def write_note(
    home: Path,
    name: str,
    description: str,
    body: list[str],
    kind: str = "feedback",
) -> Path:
    """A note file as an assistant writes it: front matter, then the fact and its reasons."""
    head = f"---\nname: {name}\ndescription: {description}\nmetadata:\n  type: {kind}\n---\n"
    text = head + "\n".join(body) + "\n**Why:** told directly.\n"
    return put(folder(home) / f"{name}.md", text)


def unique(tag: str, n: int) -> str:
    """A statement that shares no words worth counting with any other."""
    return f"Remember zeta{tag}x{n} handles qq{tag}y{n} near ww{tag}z{n}."


def scan(home: Path) -> tuple[ImportPlan, dict[str, Any]]:
    plan = build_plan(home, [], timeout_s=30)
    (source,) = plan.sources
    return plan, source


def left_out(source: dict[str, Any]) -> dict[str, int]:
    return {k: v for k, v in source["filtered"].items() if v}


def note_texts(plan: ImportPlan, source: dict[str, Any]) -> list[str]:
    return [n.text for n in plan.notes[source["id"]]]


def test_a_folder_of_files_takes_every_summary_first_and_counts_what_did_not_fit(
    home: Path,
) -> None:
    for tag in range(30):
        write_note(
            home,
            f"file{tag:02d}",
            unique(f"d{tag}", 0),
            [unique(f"b{tag}", n) for n in range(5)],
        )
    plan, source = scan(home)
    kept = plan.notes[source["id"]]
    assert source["items_found"] == 30 * 6
    assert source["items_kept"] == len(kept) == PER_SOURCE
    assert left_out(source) == {"capped": 30 * 6 - PER_SOURCE}
    # Every statement is counted once: kept, or one reason.
    assert source["items_found"] == source["items_kept"] + sum(
        source["filtered"].values()
    )
    texts = {n.text for n in kept}
    assert all(unique(f"d{tag}", 0) in texts for tag in range(30)), "every summary"
    # Round two is the first body statement of every file; nothing deeper fits.
    assert all(unique(f"b{tag}", 0) in texts for tag in range(30))
    assert not any(unique(f"b{tag}", 1) in texts for tag in range(30))
    # Files go in sorted order within a round; the summary of a file comes first.
    assert [n.provenance["path"] for n in kept[:30]] == sorted(
        n.provenance["path"] for n in kept[:30]
    )
    assert all(n.provenance["line"] == 3 for n in kept[:30]), "the front matter line"


def test_a_file_that_sorts_last_still_gets_its_summary_in(home: Path) -> None:
    for tag in range(40):
        write_note(
            home,
            f"a{tag:02d}",
            unique(f"d{tag}", 0),
            [unique(f"b{tag}", n) for n in range(10)],
        )
    write_note(home, "zzz-last", unique("last", 0), [unique("lastb", 0)])
    plan, source = scan(home)
    assert unique("last", 0) in note_texts(plan, source)
    assert source["items_kept"] == PER_SOURCE


def test_a_long_file_cannot_crowd_out_the_others(home: Path) -> None:
    write_note(home, "a-long", unique("long", 0), [unique("lb", n) for n in range(100)])
    for tag in range(10):
        write_note(
            home,
            f"small{tag}",
            unique(f"sd{tag}", 0),
            [unique(f"sb{tag}", n) for n in range(3)],
        )
    plan, source = scan(home)
    kept = plan.notes[source["id"]]
    by_file: dict[str, int] = {}
    for n in kept:
        name = Path(n.provenance["path"]).stem
        by_file[name] = by_file.get(name, 0) + 1
    assert [by_file[f"small{tag}"] for tag in range(10)] == [4] * 10, "all of each"
    assert by_file["a-long"] == PER_SOURCE - 40
    assert left_out(source) == {"capped": 101 - (PER_SOURCE - 40)}


def test_a_summary_passes_the_secret_and_duplicate_filters_like_any_statement(
    home: Path,
) -> None:
    write_note(home, "a-key", f"Deploy with {SECRET} on Fridays", [unique("k", 0)])
    write_note(home, "b-one", "Keep answers short and plain", [unique("one", 0)])
    # Same summary as b-one, and a first statement that repeats its own summary.
    write_note(
        home, "c-two", "Keep answers short and plain", ["Keep answers short and plain."]
    )
    plan, source = scan(home)
    texts = note_texts(plan, source)
    assert not any(SECRET in t for t in texts)
    assert texts.count("Keep answers short and plain.") == 1
    assert left_out(source) == {"secret": 1, "duplicate": 2}
    assert source["items_found"] == 3 * 2


def test_a_summary_is_read_with_its_quotes_line_and_the_type_of_its_file(
    home: Path,
) -> None:
    put(
        folder(home) / "q.md",
        "---\nname: q\ndescription: \"Reply in plain words: no jargon\"\nmetadata:\n  type: 'user'\n---\nThe body fact stays here.\n",
    )
    put(
        folder(home) / "r.md",
        "---\nname: r\ndescription: >-\n  folded text\ntype: feedback\n---\nAnother body fact.\n",
    )
    put(
        folder(home) / "s.md",
        "---\nname: s\ndescription: Short\ntype: feedback\n---\nA third body fact.\n",
    )
    (src,) = discover(home)
    got = [(s.text, s.kind, s.line, s.heading) for s in read_source(src, home, set())]
    assert got == [
        ("Reply in plain words: no jargon", "user", 3, "q"),
        ("The body fact stays here.", "user", 7, "q"),
        ("Another body fact.", "feedback", 7, "r"),  # a block scalar is no summary
        ("Short", "feedback", 3, "s"),
        ("A third body fact.", "feedback", 6, "s"),
    ]


def test_only_a_memory_folder_has_summaries(home: Path) -> None:
    md = home / ".claude" / "CLAUDE.md"
    put(md, "---\ndescription: A summary line here\n---\n- Use uv for envs.\n")
    (src,) = discover(home)
    assert src.kind == "md"
    assert [s.text for s in read_source(src, home, set())] == ["Use uv for envs."]


def test_a_single_file_source_keeps_its_order_and_counts_the_rest(home: Path) -> None:
    put(
        home / ".claude" / "CLAUDE.md",
        "\n".join(f"- {unique('m', n)}" for n in range(70)) + "\n",
    )
    plan, source = scan(home)
    texts = note_texts(plan, source)
    assert texts == [unique("m", n) for n in range(PER_SOURCE)]  # file order
    assert (source["items_found"], source["items_kept"]) == (70, PER_SOURCE)
    assert left_out(source) == {"capped": 10}


def test_a_single_note_file_keeps_its_order_after_its_summary(home: Path) -> None:
    write_note(
        home, "only", unique("s", 0), [unique("o", n) for n in range(70)], "feedback"
    )
    plan, source = scan(home)
    assert note_texts(plan, source) == [unique("s", 0)] + [
        unique("o", n) for n in range(PER_SOURCE - 1)
    ]
    assert (source["items_found"], left_out(source)) == (71, {"capped": 11})


def test_a_statement_left_out_by_the_limit_does_not_hide_a_later_one_as_a_repeat(
    home: Path,
) -> None:
    # Two sources: the second holds what the first had to leave out.
    write_note(home, "big", unique("g", 0), [unique("gb", n) for n in range(70)])
    other = home / ".claude" / "projects" / "q" / "memory"
    put(
        other / "again.md",
        "---\nname: again\ndescription: x\ntype: feedback\n---\n" + unique("gb", 69),
    )
    plan = build_plan(home, [], timeout_s=30)
    texts = [n.text for notes in plan.notes.values() for n in notes]
    assert unique("gb", 69) in texts, "capped in one source, kept in the other"


# ── Who a possessive belongs to ──


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (
            "Put project environment variables in the project's .envrc (direnv); do not edit ~/.zshrc.",
            "feedback",
        ),
        ("Write code, comments and commit messages in English.", "feedback"),
        (
            "项目需要的环境变量写进项目根目录的 .envrc（direnv），不要改 ~/.zshrc。",
            "feedback",
        ),
        ("开发的代码和仓库里的文件一律用英文：代码、注释、文档。", "feedback"),
        ("高效的代码要写测试。", ""),
        ("文件的名字用英文。", ""),
        ("Python's virtualenvs live under .venv.", ""),
        ("Git's default branch should be main.", ""),
        ("GitHub's API needs a token.", ""),
        ("English's spelling is American.", ""),
        ("Makefile's targets stay phony.", ""),
        ("README's title is fixed.", ""),
    ],
)
def test_a_rule_about_a_file_a_language_or_a_tool_is_not_about_a_person(
    text: str, kind: str
) -> None:
    topic = topic_for(Statement(text, "p", 1, "", kind))
    assert topic != "people"
    if kind == "feedback":
        assert topic == FEEDBACK_TOPIC


@pytest.mark.parametrize(
    "text",
    [
        "Marco's studio is in Lisbon.",
        "Dr. Priya Rao's office is in room 9C.",
        "Anne-Marie's birthday is in May.",
        "McKay's team meets on Friday.",
        "José's studio is in Lisbon.",
        "马可的工作室在里斯本。",
        "小宇的教练是老王。",
        "王老师的办公室在九楼。",
        "妈妈的生日是五月。",
    ],
)
def test_a_fact_about_a_named_person_is_people(text: str) -> None:
    assert topic_for(Statement(text, "p", 1, "", "")) == "people"


def test_a_sentence_about_the_users_relative_is_unchanged() -> None:
    # "my" makes it the user's own sentence (owner "user"): about, as before.
    text = "My sister Mei lives in Osaka."
    assert topic_for(Statement(text, "p", 1, "", "")) == "about"


@pytest.mark.parametrize(
    ("owner", "expected"),
    [
        ("Marco", True),
        ("Priya Rao", True),
        ("Python", False),
        ("English", False),
        ("Git", False),
        ("GitHub", False),
        ("README", False),
        ("iPhone", False),
        ("Makefile", False),
        ("MacBook", False),
        ("Google", False),
        ("McKay", True),
        ("东京", False),
        ("项目需要", False),
        ("马可", True),
    ],
)
def test_person_tells_a_name_from_a_word_that_is_not_one(
    owner: str, expected: bool
) -> None:
    assert person(owner) is expected
