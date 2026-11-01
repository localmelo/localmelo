"""A project's own instruction file: which files apply to a folder, and what is read.

Every case builds a synthetic tree under ``tmp_path`` and names its own fake
``home=``: nothing here reads the real home folder or any real instruction file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from localmelo.support.memory import project_files
from localmelo.support.memory.project_files import (
    CUT_MARK,
    MAX_FILE_BYTES,
    instructions,
)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_the_nearest_file_comes_first_then_the_parents(home: Path) -> None:
    write(home / "repo" / "AGENTS.md", "Root: build with make.")
    write(home / "repo" / "pkg" / "CLAUDE.md", "Package: test with pytest.")
    found = instructions(str(home / "repo" / "pkg"), home=str(home))
    assert found is not None
    assert found.text.index("Package: test with pytest.") < found.text.index(
        "Root: build with make."
    )
    assert [file.path for file in found.files] == ["CLAUDE.md", "../AGENTS.md"]
    assert found.text.startswith("--- CLAUDE.md ---\nPackage: test with pytest.")
    assert "\n\n--- ../AGENTS.md ---\nRoot: build with make." in found.text
    assert found.files[0].chars == len("Package: test with pytest.")
    assert not any(file.cut for file in found.files) and found.skipped == ()


def test_at_each_level_agents_beats_claude_beats_gemini(home: Path) -> None:
    folder = home / "repo"
    for name in ("GEMINI.md", "CLAUDE.md", "AGENTS.md"):
        write(folder / name, f"from {name}")
    found = instructions(str(folder), home=str(home))
    assert found is not None and found.text == "--- AGENTS.md ---\nfrom AGENTS.md"
    (folder / "AGENTS.md").unlink()
    found = instructions(str(folder), home=str(home))
    assert found is not None and found.text == "--- CLAUDE.md ---\nfrom CLAUDE.md"
    (folder / "CLAUDE.md").unlink()
    found = instructions(str(folder), home=str(home))
    assert found is not None and found.text == "--- GEMINI.md ---\nfrom GEMINI.md"


def test_at_most_three_files_are_read(home: Path) -> None:
    deep = home / "a" / "b" / "c" / "d" / "e"
    for level in (deep, deep.parent, deep.parent.parent, home / "a" / "b", home / "a"):
        write(level / "AGENTS.md", f"level {level.name}")
    found = instructions(str(deep), home=str(home))
    assert found is not None
    assert [file.path for file in found.files] == [
        "AGENTS.md",
        "../AGENTS.md",
        "../../AGENTS.md",
    ]
    assert "level b" not in found.text and "level a" not in found.text


def test_nothing_found_is_none(home: Path) -> None:
    (home / "repo").mkdir()
    write(home / "repo" / "README.md", "not an instruction file")
    assert instructions(str(home / "repo"), home=str(home)) is None
    assert instructions(str(home / "missing"), home=str(home)) is None


def test_the_limit_cuts_at_a_paragraph_and_marks_it(home: Path) -> None:
    paragraphs = [f"Paragraph {n}: " + " ".join(["word"] * 20) for n in range(10)]
    write(home / "repo" / "AGENTS.md", "\n\n".join(paragraphs))
    found = instructions(str(home / "repo"), home=str(home), limit=400)
    assert found is not None
    assert len(found.text) <= 400
    assert found.text.endswith(f"\n{CUT_MARK}")
    body = found.text.removeprefix("--- AGENTS.md ---\n").removesuffix(f"\n{CUT_MARK}")
    # Whole paragraphs only: the cut falls on a blank line, not inside a paragraph.
    assert body == "\n\n".join(paragraphs[: body.count("Paragraph")])
    assert 0 < body.count("Paragraph") < 10
    assert found.files[0].cut is True and found.files[0].chars == len(body)


def test_the_limit_prefers_the_end_of_a_section_over_a_heading_alone(
    home: Path,
) -> None:
    text = "# Build\n\nRun make.\n\n## Test\nRun pytest with the flags that the team agreed on."
    write(home / "repo" / "AGENTS.md", text)
    # Room for the first section and most of the second heading's text.
    found = instructions(str(home / "repo"), home=str(home), limit=70)
    assert found is not None
    body = found.text.removeprefix("--- AGENTS.md ---\n").removesuffix(f"\n{CUT_MARK}")
    assert body == "# Build\n\nRun make."
    assert "## Test" not in found.text


def test_a_file_that_fits_whole_is_not_marked_and_a_parent_may_be_cut(
    home: Path,
) -> None:
    write(home / "repo" / "pkg" / "AGENTS.md", "Near.")
    write(home / "repo" / "AGENTS.md", "\n\n".join("Far " + "x" * 80 for _ in range(8)))
    found = instructions(str(home / "repo" / "pkg"), home=str(home), limit=500)
    assert found is not None
    assert [(file.path, file.cut) for file in found.files] == [
        ("AGENTS.md", False),
        ("../AGENTS.md", True),
    ]
    assert len(found.text) <= 500 and found.text.endswith(CUT_MARK)


def test_a_parent_with_no_room_left_is_left_out(home: Path) -> None:
    write(home / "repo" / "pkg" / "AGENTS.md", "Near. " * 40)
    write(home / "repo" / "AGENTS.md", "Far. " * 100)
    found = instructions(str(home / "repo" / "pkg"), home=str(home), limit=290)
    assert found is not None
    assert [file.path for file in found.files] == ["AGENTS.md"]


def test_the_walk_never_goes_above_home(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "AGENTS.md", "above home")  # the folder that holds home
    write(home / "repo" / "AGENTS.md", "in the project")
    found = instructions(str(home / "repo" / "pkg-less"), home=str(home))
    assert found is None  # a folder that does not exist
    (home / "repo" / "pkg").mkdir()
    found = instructions(str(home / "repo" / "pkg"), home=str(home))
    assert found is not None and found.text.endswith("in the project")
    assert "above home" not in found.text


def test_a_folder_outside_home_reads_only_itself(home: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    write(outside / "AGENTS.md", "outside parent")
    write(outside / "proj" / "CLAUDE.md", "outside project")
    found = instructions(str(outside / "proj"), home=str(home))
    assert found is not None
    assert [file.path for file in found.files] == ["CLAUDE.md"]
    assert "outside parent" not in found.text


def test_a_symlink_that_leaves_the_tree_is_ignored(home: Path, tmp_path: Path) -> None:
    secret = write(tmp_path / "private" / "notes.md", "not this project's")
    folder = home / "repo"
    folder.mkdir()
    (folder / "AGENTS.md").symlink_to(secret)
    assert instructions(str(folder), home=str(home)) is None
    # The next name at that level counts instead of the refused link.
    write(folder / "CLAUDE.md", "the real one")
    found = instructions(str(folder), home=str(home))
    assert found is not None and found.text.endswith("the real one")
    # A link to a file of the home folder, outside the walked folders' own, is refused too.
    stray = write(home / "notes" / "todo.md", "a note of the person")
    (folder / "CLAUDE.md").unlink()
    (folder / "CLAUDE.md").symlink_to(stray)
    assert instructions(str(folder), home=str(home)) is None


def test_a_symlink_inside_the_tree_is_followed_once(home: Path) -> None:
    write(home / "repo" / "docs" / "agents.md", "Shared instructions.")
    write(home / "repo" / "AGENTS.md", "Root instructions.")
    pkg = home / "repo" / "pkg"
    pkg.mkdir()
    (pkg / "AGENTS.md").symlink_to(Path("..") / "AGENTS.md")
    (pkg / "CLAUDE.md").symlink_to(Path("..") / "docs" / "agents.md")
    found = instructions(str(pkg), home=str(home))
    assert found is not None
    # The pkg link points at the root file, so the root level adds nothing new.
    assert [file.path for file in found.files] == ["AGENTS.md"]
    assert found.text.count("Root instructions.") == 1


def test_a_folder_named_like_a_file_is_not_one(home: Path) -> None:
    (home / "repo" / "AGENTS.md").mkdir(parents=True)
    write(home / "repo" / "CLAUDE.md", "the file")
    found = instructions(str(home / "repo"), home=str(home))
    assert found is not None and found.text.endswith("the file")


def test_a_file_over_256_kib_is_skipped_with_a_note(home: Path) -> None:
    write(home / "repo" / "AGENTS.md", "x" * (MAX_FILE_BYTES + 1))
    write(home / "repo" / "CLAUDE.md", "never read: AGENTS.md is the one that counts")
    found = instructions(str(home / "repo"), home=str(home))
    assert found is not None
    assert found.text == "" and found.files == ()
    assert found.skipped == ("AGENTS.md",)
    # The size limit is inclusive.
    write(home / "repo" / "AGENTS.md", "y" * MAX_FILE_BYTES)
    found = instructions(str(home / "repo"), home=str(home), limit=10_000)
    assert found is not None and found.skipped == () and found.files[0].cut


def test_a_big_file_does_not_hide_the_parents(home: Path) -> None:
    write(home / "repo" / "AGENTS.md", "Root instructions.")
    write(home / "repo" / "pkg" / "AGENTS.md", "x" * (MAX_FILE_BYTES + 1))
    found = instructions(str(home / "repo" / "pkg"), home=str(home))
    assert found is not None
    assert found.skipped == ("AGENTS.md",)
    assert [file.path for file in found.files] == ["../AGENTS.md"]


def test_secrets_are_redacted_and_flagged_lines_are_dropped(home: Path) -> None:
    key = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2"
    write(
        home / "repo" / "AGENTS.md",
        "Run `uv run pytest`.\n"
        f"Use the key {key} for the sandbox.\n"
        "The admin password is hunter2.\n"
        "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkq\n-----END PRIVATE KEY-----\n"
        "Lint with ruff.\n",
    )
    found = instructions(str(home / "repo"), home=str(home))
    assert found is not None
    assert key not in found.text and "[redacted]" in found.text
    assert "hunter2" not in found.text and "MIIEvQIBADANBgkq" not in found.text
    assert "Run `uv run pytest`." in found.text and "Lint with ruff." in found.text


def test_undecodable_bytes_are_replaced_and_a_bom_is_dropped(home: Path) -> None:
    path = home / "repo" / "AGENTS.md"
    path.parent.mkdir()
    path.write_bytes(b"\xef\xbb\xbfUse tabs \xff\xfe here.\r\nSecond line.\r\n")
    found = instructions(str(home / "repo"), home=str(home))
    assert found is not None
    assert found.text == "--- AGENTS.md ---\nUse tabs �� here.\nSecond line."


def test_the_homes_own_file_and_the_assistants_folders_are_not_a_project(
    home: Path,
) -> None:
    write(home / "AGENTS.md", "the person's own file")
    write(home / "CLAUDE.md", "the person's own file")
    write(home / ".claude" / "CLAUDE.md", "imported as notes")
    write(home / ".codex" / "AGENTS.md", "imported as notes")
    write(home / ".cc-switch" / "claude-accounts" / "a" / "config" / "CLAUDE.md", "x")
    write(home / "repo" / "AGENTS.md", "the project")
    found = instructions(str(home / "repo"), home=str(home))
    assert found is not None and found.text.endswith("the project")
    assert [file.path for file in found.files] == ["AGENTS.md"]
    # A task started in home itself, or in an assistant's folder, has no project file.
    assert instructions(str(home), home=str(home)) is None
    assert instructions(str(home / ".claude"), home=str(home)) is None
    assert instructions(str(home / ".codex"), home=str(home)) is None


def test_a_link_to_the_homes_own_file_is_refused(home: Path) -> None:
    mine = write(home / "AGENTS.md", "the person's own file")
    folder = home / "repo"
    folder.mkdir()
    (folder / "AGENTS.md").symlink_to(mine)
    assert instructions(str(folder), home=str(home)) is None


def test_a_home_given_through_a_link_is_resolved(home: Path, tmp_path: Path) -> None:
    write(home / "repo" / "AGENTS.md", "the project")
    alias = tmp_path / "alias"
    alias.symlink_to(home)
    found = instructions(str(alias / "repo"), home=str(alias))
    assert found is not None and found.text.endswith("the project")


def test_home_defaults_to_the_current_users_home(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(home))
    write(home / "repo" / "sub" / "AGENTS.md", "inner")
    write(home / "repo" / "AGENTS.md", "outer")
    found = instructions(str(home / "repo" / "sub"))
    assert found is not None and len(found.files) == 2
    assert os.path.expanduser("~") == str(home)
    assert project_files.NAMES == ("AGENTS.md", "CLAUDE.md", "GEMINI.md")
