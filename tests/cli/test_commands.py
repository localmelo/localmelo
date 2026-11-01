"""Slash commands: what is a command, and what `/` completes."""

from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.completion import CompleteEvent, Completion
from prompt_toolkit.document import Document

from localmelo.support.cli.commands import (
    COMMANDS,
    HELP_GROUPS,
    find,
    help_text,
    parse_command,
)
from localmelo.support.cli.completion import AttachmentCompletion, SlashCompleter
from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.text import cells
from localmelo.support.cli.theme import ASCII


@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        ("/help", ("help", "help", "")),
        ("/model  qwen3-8b ", ("model", "model", "qwen3-8b")),
        ("/clear", ("clear", "clear", "")),
        ("/quit", ("exit", "quit", "")),
        ("/nope now", (None, "nope", "now")),
    ],
)
def test_commands_parse(text: str, parsed: tuple[str | None, str, str]) -> None:
    result = parse_command(text)
    assert result is not None
    command, name, arguments = result
    assert (None if command is None else command.name, name, arguments) == parsed


@pytest.mark.parametrize(
    "text",
    [
        "/etc/hosts is broken",
        "hello /help",
        "/Help",
        "/",
        "/2fa codes",
        " /help me",
        " /exit",
        "\t/new",
    ],
)
def test_messages_are_not_commands(text: str) -> None:
    assert parse_command(text) is None


def _complete(text: str, models: list[str] | None = None) -> list[tuple[str, str, str]]:
    completer = SlashCompleter(lambda: models or [])
    return [
        (c.text, c.display_text, c.display_meta_text)
        for c in completer.get_completions(Document(text), CompleteEvent())
    ]


def test_attachment_completions_are_workspace_files_only(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "file.txt").write_text("private")
    (root / "dir").mkdir()
    (root / "dir" / "nested.txt").write_text("private")
    (root / "link").symlink_to(tmp_path, target_is_directory=True)
    (root / "file-link").symlink_to(root / "file.txt")
    completer = SlashCompleter()
    completer.workspace = lambda: str(root)

    def complete(text: str) -> list[Completion]:
        return list(completer.get_completions(Document(text), CompleteEvent()))

    assert [
        (c.text, isinstance(c, AttachmentCompletion)) for c in complete("see @")
    ] == [("@dir/", False), ("@file.txt", True)]
    assert [c.text for c in complete("see @dir/n")] == ["@dir/nested.txt"]
    for text in ("email@file", "@../", "@/etc/", "@link/", "@file-link"):
        assert complete(text) == []


def test_slash_lists_the_everyday_commands_first_by_their_bare_names() -> None:
    shown = _complete("/")
    names = [text for text, _, _ in shown]
    everyday = [name for _, group in HELP_GROUPS for name in group]
    assert names[:8] == ["/model", "/new", "/status", "/help", "/exit", "/undo"] + [
        "/copy",
        "/retry",
    ]
    assert {n[1:] for n in names[:8]} == set(everyday)  # the menu shows 8 rows
    assert names[8:12] == ["/clear", "/backend", "/attach", "/attachments"]
    assert names[-1] == "/mcp" and len(names) == len(COMMANDS)
    assert all(
        text == display for text, display, _ in shown
    )  # arguments are in the line
    assert ("/help", "/help", "Show commands; /help all lists every one") in shown


def test_no_menu_description_is_cut_at_80_columns() -> None:
    # The name column is as wide as the longest name, plus 2; the menu has 79 columns,
    # and the description keeps 1 cell on each side of it.
    shown = _complete("/")
    column = max(cells(display) for _, display, _ in shown) + 2
    assert all(cells(meta) <= 79 - column - 2 for _, _, meta in shown)


def test_typing_narrows_the_list() -> None:
    assert _complete("/he") == [
        ("/help", "/help", "Show commands; /help all lists every one")
    ]
    assert _complete("/mo") == [
        ("/model", "/model", "Pick a model, or /model NAME (saved)")
    ]
    assert _complete("hello /he") == []


def test_verbose_and_thinking_complete_on_and_off() -> None:
    assert [text for text, _, _ in _complete("/verbose ")] == ["on", "off"]
    assert [text for text, _, _ in _complete("/thinking of")] == ["off"]
    assert _complete("/verbose on") == []


def test_model_completes_the_listed_models() -> None:
    assert [
        text
        for text, _, _ in _complete("/model qw", ["qwen3-8b", "llama3", "qwen3-32b"])
    ] == [
        "qwen3-8b",
        "qwen3-32b",
    ]


def _help(topic: str = "", *, width: int = 80, everything: bool = False) -> list[str]:
    out = RecordingOut(width)
    out.print(help_text(topic, width=width, everything=everything))
    return out.text().strip("\n").split("\n")


def test_help_starts_with_the_everyday_commands_in_three_groups() -> None:
    rows = _help()
    assert [row for row in rows if not row.startswith(" ")] == [
        "Conversation",
        "Model",
        "Help",
    ]
    assert [row.split()[0] for row in rows if row.startswith("  /")][:-1] == [
        "/new",
        "/retry",
        "/undo",
        "/copy",
        "/model",
        "/status",
        "/help",
        "/exit",
    ]
    text = "\n".join(rows)
    assert "! command runs it in your shell" in text and "esc stop the turn" not in text
    assert "shift+tab switches between Ask and Auto" in text
    assert rows[-1] == "  /help all lists every command and key."
    assert "/backend" not in text and len(rows) <= 14  # one screen


@pytest.mark.parametrize(("topic", "everything"), [("all", False), ("", True)])
def test_help_all_and_developer_mode_list_every_command_and_the_keys(
    topic: str, everything: bool
) -> None:
    text = "\n".join(_help(topic, everything=everything))
    assert all(f"/{command.name} " in text for command in COMMANDS)
    assert "More\n" in text and "Input\n  enter send" in text
    assert "esc stop the turn" in text and "shift+tab switches between" in text


def test_one_commands_details_and_the_unknown_one() -> None:
    assert _help("model") == [
        "/model [name]",
        "  Pick a model, or /model NAME (saved)",
        "  During a turn: no",
    ]
    assert _help("nope") == ["Unknown command /nope. /help lists the commands."]
    assert find("quit") is find("exit")


@pytest.mark.parametrize("width", [80, 60, 44])
def test_help_rows_continue_under_their_description(width: int) -> None:
    for everything in (False, True):
        rows = _help(width=width, everything=everything)
        # Only a heading is at the first column, and no row is wider than the terminal.
        assert all(row.startswith(" ") or row.isalpha() for row in rows)
        assert all(cells(row) <= width for row in rows)
    # A description longer than its room goes under itself: "/approvals" is 14 cells.
    assert any(row.startswith(" " * 16) for row in _help("all", width=44))


def test_help_keeps_ascii_in_ascii_mode() -> None:
    out = RecordingOut(80)
    out.print(help_text("all", ASCII))
    assert out.text().isascii() and "up/down history" in out.text()


def test_the_bare_slash_lists_each_command_once() -> None:
    names = [text for text, _, _ in _complete("/")]
    assert len(names) == len(COMMANDS)
    assert "/quit" not in names and "/clear" in names
    assert find("clear") is not find("new") and find("quit") is find("exit")


def test_new_and_clear_are_two_commands_that_name_each_other() -> None:
    assert _complete("/cl") == [
        ("/clear", "/clear", "Start a new chat and clear the screen")
    ]
    assert _complete("/ne") == [
        ("/new", "/new", "Start a new chat; /clear also clears the screen")
    ]
    new, clear = find("new"), find("clear")
    assert new is not None and clear is not None and new.aliases == clear.aliases == ()
    # The short /help already tells them apart; /help clear has its own details.
    assert any("/clear also clears the screen" in row for row in _help())
    assert _help("clear")[:2] == ["/clear", "  Start a new chat and clear the screen"]
    assert not any(row.lstrip().startswith("/clear") for row in _help())


def test_an_alias_row_names_the_other_spelling() -> None:
    assert _complete("/q") == [("/quit", "/quit", "Quit (also /exit, ctrl+d)")]
    assert _complete("/exi") == [("/exit", "/exit", "Quit (also /quit, ctrl+d)")]
    assert _complete("/ex") == [
        ("/exit", "/exit", "Quit (also /quit, ctrl+d)"),
        ("/expand", "/expand", "Show a tool call's full output"),
    ]
