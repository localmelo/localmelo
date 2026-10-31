"""Bringing in what other assistants remember: scan, filters, notes, undo.

Every test runs over a synthetic HOME (:mod:`.import_home`) and passes it
explicitly; ``Path.home`` and the environment are pointed at it too, so nothing
can read the real one. No model is involved.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.memory import ChatMemory, records
from localmelo.support.memory.importer import build_plan, select
from localmelo.support.memory.importer.rules import reject, topic_for
from localmelo.support.memory.importer.sources import (
    Statement,
    decode_workspace,
    markdown,
)

from .import_home import CHAIN, LEAK, OWN, SECRET, build, put

NEVER = (
    SECRET,
    "hunter2",
    "other.test",
    "412 555",
    "Apple ID",
    "4821937",
    "LEAK-LINE",
    "CODEX-TOKEN",
    "CODEX-HISTORY",
    "Padding line",
    "Edit tool",
    "/compact",
    "MCP",
)


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tmp = tmp_path.resolve()
    found = build(tmp)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: found))
    monkeypatch.setenv("HOME", str(found))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    return found


@pytest.fixture()
async def memory(tmp_path: Path) -> AsyncIterator[ChatMemory]:
    found = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        yield found
    finally:
        await found.aclose()


def by_name(plan: Any, home: Path) -> dict[str, dict[str, Any]]:
    """The scan's sources by their path under HOME."""
    return {str(Path(s["path"]).relative_to(home)): s for s in plan.sources}


def texts(plan: Any) -> list[str]:
    return [n.text for notes in plan.notes.values() for n in notes]


async def scan(memory: ChatMemory, home: Path) -> Any:
    return await memory.import_scan(home=home)


async def test_the_scan_counts_each_source_and_what_was_filtered(
    memory: ChatMemory, home: Path
) -> None:
    found = by_name(await scan(memory, home), home)
    summary = {
        name: (
            s["harness"],
            s["items_found"],
            s["items_kept"],
            {k: v for k, v in s["filtered"].items() if v},
        )
        for name, s in found.items()
    }
    (project,) = [
        n for n in summary if n.startswith(".claude/projects/") and "vanished" not in n
    ]
    assert summary == {
        ".claude/CLAUDE.md": (
            "claude",
            18,
            9,
            {"secret": 2, "third_party": 4, "mechanics": 3},
        ),
        project: ("claude", 5, 4, {"duplicate": 1}),
        ".cc-switch/claude-accounts/a1/config/CLAUDE.md": ("claude", 1, 1, {}),
        ".codex/AGENTS.md": ("codex", 3, 1, {"duplicate": 1, "mechanics": 1}),
        ".gemini/GEMINI.md": ("gemini", 3, 3, {}),
        ".codex/memories_1.sqlite": ("codex", 2, 1, {"mechanics": 1}),
        ".hermes/memories/USER.md": (
            "hermes",
            3,
            1,
            {"third_party": 1, "duplicate": 1},
        ),
        ".hermes/memories/MEMORY.md": ("hermes", 2, 1, {"mechanics": 1}),
    }
    assert found[".claude/CLAUDE.md"]["sample"][0] == "Prefer tabs in Makefiles."
    assert all(len(s["sample"]) <= 3 and s["scope"] == "user" for s in found.values())


async def test_nothing_private_or_about_the_other_tool_is_kept(
    memory: ChatMemory, home: Path
) -> None:
    plan = await scan(memory, home)
    kept = "\n".join(texts(plan))
    assert [word for word in NEVER if word in kept] == []
    assert f"Write to {OWN} for questions about the lab." in texts(plan)
    assert "和你对话可以用中文。" in texts(plan)  # the language is kept
    assert not any(
        "vanished" in n.provenance["path"] for ns in plan.notes.values() for n in ns
    )


async def test_without_a_git_email_every_address_is_dropped(
    memory: ChatMemory, home: Path
) -> None:
    (home / ".gitconfig").unlink()
    assert not any("@" in text for text in texts(await scan(memory, home)))


async def test_a_git_config_linked_from_outside_home_is_not_read(
    memory: ChatMemory, home: Path
) -> None:
    put(home.parent / "outside" / "gitconfig", f"[user]\n\temail = {OWN}\n")
    (home / ".gitconfig").unlink()
    (home / ".gitconfig").symlink_to(home.parent / "outside" / "gitconfig")
    assert not any("@" in text for text in texts(await scan(memory, home)))


async def test_imports_follow_links_four_deep_and_never_leave_home(
    memory: ChatMemory, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    real_read, real_connect = Path.read_text, sqlite3.connect

    def spy_read(self: Path, *args: Any, **kwargs: Any) -> str:
        opened.append(str(self.resolve()))
        return real_read(self, *args, **kwargs)

    def spy_connect(target: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(target).removeprefix("file://").split("?")[0])
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy_read)
    monkeypatch.setattr(sqlite3, "connect", spy_connect)
    got = texts(await scan(memory, home))
    assert CHAIN[:4] == [t for t in got if t in CHAIN]  # the fifth is one link too deep
    assert LEAK not in got
    mine = (str(home), memory.directory)  # the memory folder is melo's own
    assert opened and [p for p in opened if not p.startswith(mine)] == []
    banned = ("leak.md", "auth.json", "history.jsonl", "big.md")
    assert [p for p in opened if p.endswith(banned)] == []


async def test_auto_memory_files_keep_their_type_and_their_folder(
    memory: ChatMemory, home: Path
) -> None:
    plan = await scan(memory, home)
    (project,) = [s for s in plan.sources if s["workspace"]]
    assert project["workspace"] == str(home / "work" / "app")
    kept = {n.text: n for n in plan.notes[project["id"]]}
    assert kept["The app keeps its fixtures under tests/data."].topic == "projects"
    assert (
        kept["The app keeps its fixtures under tests/data."].provenance["workspace"]
        == project["workspace"]
    )
    assert kept["Open bugs are tracked on the team board."].topic == "projects"
    assert kept["The user is a robotics researcher."].topic == "about"
    assert "workspace" not in kept["Keep answers short and plain."].provenance
    assert not any("](user_role.md)" in n.provenance["quote"] for n in kept.values())


async def test_the_environment_names_more_config_and_hermes_folders(
    memory: ChatMemory, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    put(home / "custom" / "CLAUDE.md", "- Prefer small pull requests.\n")
    put(home / "alt" / "memories" / "USER.md", "Zed likes tea.")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / "custom"))
    monkeypatch.setenv("HERMES_HOME", str(home / "alt"))
    got = texts(await scan(memory, home))
    assert "Prefer small pull requests." in got and "Zed likes tea." in got
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home.parent / "outside"))
    assert LEAK not in texts(await scan(memory, home))


async def test_a_codex_database_is_read_without_being_touched(
    memory: ChatMemory, home: Path
) -> None:
    db = home / ".codex" / "memories_1.sqlite"
    before = (db.stat().st_mtime_ns, db.stat().st_size)
    plan = await scan(memory, home)
    assert "Prefers pytest over unittest." in texts(plan)
    assert (db.stat().st_mtime_ns, db.stat().st_size) == before
    assert sorted(p.name for p in db.parent.iterdir() if "sqlite" in p.name) == [
        "memories_1.sqlite"
    ]


async def test_apply_writes_notes_with_their_provenance_and_the_event(
    memory: ChatMemory, home: Path
) -> None:
    plan = await scan(memory, home)
    done = await memory.import_apply(plan, [s["id"] for s in plan.sources])
    notes = await memory.records()
    assert len(notes) == len(done["written_ids"]) == len(texts(plan)) == 21
    assert done["skipped"] == 0 and done["batch_id"] == plan.batch_id
    assert done["event"]["type"] == "memory.import"
    assert [s["written"] for s in done["event"]["data"]["sources"]] == [
        s["items_kept"] for s in plan.sources
    ]
    note = next(n for n in notes if n.text == "Use uv for new Python projects.")
    assert note.admitted_by == "import" and note.session_id == ""
    assert {
        k: note.provenance[k]
        for k in ("source", "harness", "heading", "batch_id", "topic", "quote")
    } == {
        "source": "import",
        "harness": "claude",
        "heading": "Machine rules › Environment",
        "batch_id": plan.batch_id,
        "topic": "preferences",
        "quote": "Use uv for new Python projects.",
    }
    assert (
        note.provenance["path"].endswith(".claude/CLAUDE.md")
        and note.provenance["line"] > 1
    )
    assert len(note.provenance["line_sha"]) == 16
    assert (
        next(n for n in notes if n.text.startswith("The app keeps")).kind == "episodic"
    )


async def test_only_the_chosen_sources_are_written_and_an_unknown_one_is_refused(
    memory: ChatMemory, home: Path
) -> None:
    plan = await scan(memory, home)
    gemini = by_name(plan, home)[".gemini/GEMINI.md"]["id"]
    done = await memory.import_apply(plan, [gemini])
    assert len(done["written_ids"]) == 3
    assert {n.provenance["harness"] for n in await memory.records()} == {"gemini"}
    with pytest.raises(ValueError, match="Unknown source"):
        await memory.import_apply(plan, ["claude-nope"])
    again = await memory.import_apply(plan, [gemini])  # a source is written once
    assert (again["written_ids"], again["skipped"]) == ([], 3)


async def test_a_rescan_adds_only_new_lines(memory: ChatMemory, home: Path) -> None:
    first = await scan(memory, home)
    await memory.import_apply(first, [s["id"] for s in first.sources])
    second = await scan(memory, home)
    assert texts(second) == []
    assert sum(s["filtered"]["duplicate"] for s in second.sources) >= 21
    put(
        home / ".gemini" / "GEMINI.md",
        "# 语言\n- 和我对话可以用中文。\n- 提交信息用英文写。\n",
    )
    third = await scan(memory, home)
    assert texts(third) == ["提交信息用英文写。"]


async def test_a_changed_line_supersedes_its_old_wording_and_undo_restores_it(
    memory: ChatMemory, home: Path
) -> None:
    first = await scan(memory, home)
    await memory.import_apply(first, [s["id"] for s in first.sources])
    old = "Keep commits small and focused on one change."
    put(
        home / ".codex" / "AGENTS.md",
        "# Working rules\n## Git\n- Keep commits small and focused on one change per topic.\n",
    )
    plan = await scan(memory, home)
    assert texts(plan) == ["Keep commits small and focused on one change per topic."]
    (note,) = (n for ns in plan.notes.values() for n in ns)
    assert note.target is not None
    done = await memory.import_apply(plan, [s["id"] for s in plan.sources])
    now = [n.text for n in await memory.records()]
    assert old not in now and note.text in now
    assert await memory.undo_import(done["batch_id"]) == 1
    now = [n.text for n in await memory.records()]
    assert old in now and note.text not in now


async def test_undo_takes_the_whole_batch_and_leaves_the_users_own_notes(
    memory: ChatMemory, home: Path
) -> None:
    own = records.Candidate(
        "You live in Lisbon.", "I live in Lisbon.", "about", "user_statement"
    )
    await records.write(
        memory.stores()[0],
        [own],
        [None],
        source={"session_id": "s1", "turn": 1},
        embed_model="",
    )
    plan = await scan(memory, home)
    await memory.import_apply(plan, [s["id"] for s in plan.sources])
    assert await memory.undo_import(plan.batch_id) == 21
    assert [n.text for n in await memory.records()] == ["You live in Lisbon."]
    assert await memory.undo_import(plan.batch_id) == 0


async def test_undo_leaves_a_note_the_user_reworded_and_counts_it(
    memory: ChatMemory, home: Path
) -> None:
    """An edited note is the person's own words now: undo keeps it, and says how many it kept."""
    plan = await scan(memory, home)
    await memory.import_apply(plan, [s["id"] for s in plan.sources])
    assert await memory.import_kept(plan.batch_id) == 0
    first = (await memory.records())[0]
    await memory.edit_record(first.id, "You like very short answers.")
    assert await memory.import_kept(plan.batch_id) == 1
    assert await memory.undo_import(plan.batch_id) == 20
    assert [n.text for n in await memory.records()] == ["You like very short answers."]
    assert await memory.import_kept(plan.batch_id) == 1
    assert await memory.undo_import(plan.batch_id) == 0


async def test_a_line_the_user_already_has_is_a_duplicate_not_a_replacement(
    memory: ChatMemory, home: Path
) -> None:
    own = records.Candidate(
        "Use uv for new Python projects.", "I use uv.", "preferences", "user_request"
    )
    await records.write(
        memory.stores()[0],
        [own],
        [None],
        source={"session_id": "s1", "turn": 1},
        embed_model="",
    )
    plan = await scan(memory, home)
    assert "Use uv for new Python projects." not in texts(plan)
    await memory.import_apply(plan, [s["id"] for s in plan.sources])
    assert [n.text for n in await memory.records()].count(
        "Use uv for new Python projects."
    ) == 1


async def test_a_source_keeps_60_notes_and_a_batch_300(
    memory: ChatMemory, home: Path
) -> None:
    from .import_home import lines, note

    for tag in range(6):
        put(
            home / ".claude" / "projects" / f"p{tag}" / "memory" / "f.md",
            note("feedback", lines(tag, 70)),
        )
    plan = await scan(memory, home)
    wide = [s for s in plan.sources if s["items_found"] == 70]
    assert [(s["items_found"], s["items_kept"]) for s in wide] == [(70, 60)] * 6
    done = await memory.import_apply(plan, [s["id"] for s in plan.sources])
    assert len(done["written_ids"]) == 300
    assert done["skipped"] == sum(s["items_kept"] for s in plan.sources) - 300


async def test_a_scan_out_of_time_returns_what_it_has_as_partial(
    memory: ChatMemory, home: Path
) -> None:
    plan = await memory.import_scan(home=home, timeout_s=-1)
    assert plan.partial is True and plan.sources == []
    assert (await scan(memory, home)).partial is False
    assert not (await memory.records())  # a scan writes nothing


async def test_imported_notes_are_recalled_but_never_pinned(
    memory: ChatMemory, home: Path
) -> None:
    plan = await scan(memory, home)
    await memory.import_apply(plan, [s["id"] for s in plan.sources])
    assert await memory.pinned_quotes() == ""
    found = await memory.search("where do the models live under Developer")
    assert [i.text for i in found if i.is_record][:1] == [
        "Models live under ~/Developer/ml_source."
    ]
    assert "the file CLAUDE.md" in found[0].prompt


async def test_the_default_home_is_the_users_not_a_fixed_path(
    memory: ChatMemory, home: Path
) -> None:
    assert (
        by_name(await memory.import_scan(), home).keys()
        == by_name(await scan(memory, home), home).keys()
    )


async def test_plans_are_cached_by_batch_id_and_only_the_latest_few(
    memory: ChatMemory, home: Path
) -> None:
    ids = [(await scan(memory, home)).batch_id for _ in range(6)]
    assert (
        memory.import_plan(ids[-1]) is not None and memory.import_plan(ids[0]) is None
    )


@pytest.mark.parametrize(
    "line",
    [
        "Use the Edit tool to change files.",
        "Run /compact when the context is long.",
        "Call SendMessage to reach the other session.",
        "Add MCP servers in settings.json.",
        "Hooks config lives in .claude/hooks.",
        "Never use TaskStop on a running worker.",
    ],
)
def test_lines_about_the_other_tool_are_dropped(line: str) -> None:
    assert reject(line, OWN) == "mechanics"


@pytest.mark.parametrize(
    "line",
    [
        "Mail backup@other.test when a job fails.",
        "Her Apple ID is the team's shared one.",
        "Call 412-555-0187 after hours.",
        "Reach +44 20 7946 0958 for support.",
        "Ask Dr. Priya Rao, student ID 4821937, first.",
    ],
)
def test_private_data_of_someone_else_is_dropped(line: str) -> None:
    assert reject(line, OWN) == "third_party"


@pytest.mark.parametrize(
    "line",
    [
        f"Send questions to {OWN}.",
        "Ports 8400-8799 belong to the benchmark.",
        "Released on 2026-10-31 with python 3.11.",
        "Git hooks run ruff before a commit.",
        "Run the tests with uv run pytest.",
        "A permission denied error means the file is read-only.",
        "Pin dependency versions in uv.lock.",
    ],
)
def test_ordinary_lines_pass_the_filters(line: str) -> None:
    assert reject(line, OWN) == ""


def test_a_secret_is_dropped_before_anything_else() -> None:
    assert reject(f"Use key {SECRET} with the Bash tool.", OWN) == "secret"


@pytest.mark.parametrize(
    ("text", "kind", "topic"),
    [
        ("The user is a robotics researcher.", "", "about"),
        ("I am a PhD student.", "", "about"),
        ("Marco's studio is in Lisbon.", "", "people"),
        ("We decided to use uv for envs.", "", "decisions"),
        ("Always run tests before a push.", "", "preferences"),
        ("The fixtures live in tests/data.", "project", "projects"),
        ("Hobbies include climbing.", "user", "about"),
        ("Models live under ~/Developer/ml_source.", "", "preferences"),
    ],
)
def test_topics_follow_the_words_and_the_note_type(
    text: str, kind: str, topic: str
) -> None:
    assert topic_for(Statement(text, "p", 1, "", kind)) == topic


def test_markdown_keeps_heading_paths_and_skips_what_is_not_a_statement() -> None:
    md = "# A\n## B\n- one thing here\n```\ncode line\n```\n| t | a |\n- Lead-in:\n  - nested point\n# C\ntext line here\n"
    got = [(s.heading, s.text, s.line) for s in markdown(md, "p")]
    assert got == [
        ("A › B", "one thing here", 3),
        ("A › B", "nested point", 9),
        ("C", "text line here", 11),
    ]


def test_a_long_line_splits_into_sentences() -> None:
    long = " ".join(f"Sentence number {n} is here." for n in range(30))
    got = list(markdown(long, "p"))
    assert len(got) == 30 and all(len(s.text) < 300 for s in got)


def test_a_project_folder_name_decodes_to_the_real_folder(home: Path) -> None:
    flat = "".join(c if c.isalnum() else "-" for c in str(home / "work" / "app"))
    assert decode_workspace(flat, home) == str(home / "work" / "app")
    assert decode_workspace(flat + "-x", home) == ""
    assert decode_workspace("-elsewhere-app", home) == ""


def test_select_refuses_a_source_the_scan_did_not_find(home: Path) -> None:
    plan = build_plan(home, [], timeout_s=5)
    with pytest.raises(ValueError, match="Unknown source"):
        select(plan, ["nope"])
