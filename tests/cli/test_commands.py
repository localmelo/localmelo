"""Slash commands: what is a command, and what `/` completes."""

from __future__ import annotations

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from localmelo.support.cli.commands import COMMANDS, find, help_text, parse_command
from localmelo.support.cli.completion import SlashCompleter
from localmelo.support.cli.output import RecordingOut


@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        ("/help", ("help", "help", "")),
        ("/model  qwen3-8b ", ("model", "model", "qwen3-8b")),
        ("/clear", ("new", "clear", "")),
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


def test_slash_lists_every_command_with_its_line() -> None:
    shown = _complete("/")
    assert [text for text, _, _ in shown] == [
        "/help",
        "/new",
        "/clear",
        "/model",
        "/backend",
        "/exit",
        "/quit",
    ]
    assert ("/help", "/help [command]", "Show commands and keys") in shown


def test_typing_narrows_the_list() -> None:
    assert _complete("/he") == [("/help", "/help [command]", "Show commands and keys")]
    assert _complete("/mo") == [
        ("/model", "/model [name]", "Show, list or switch the model (saved)")
    ]
    assert _complete("hello /he") == []


def test_model_completes_the_listed_models() -> None:
    assert [
        text
        for text, _, _ in _complete("/model qw", ["qwen3-8b", "llama3", "qwen3-32b"])
    ] == [
        "qwen3-8b",
        "qwen3-32b",
    ]


def test_help_lists_the_commands_and_one_commands_details() -> None:
    out = RecordingOut(80)
    out.print(help_text())
    text = out.text()
    assert all(f"/{command.name}" in text for command in COMMANDS)
    assert "esc stop the turn" in text
    out = RecordingOut(80)
    out.print(help_text("model"))
    assert (
        "/model [name]\n  Show, list or switch the model (saved)\n  During a turn: no"
        in out.text()
    )
    assert find("quit") is find("exit")


def test_an_alias_row_names_the_other_spelling() -> None:
    assert _complete("/cl") == [("/clear", "/clear", "Start a new session (also /new)")]
    assert _complete("/ne") == [("/new", "/new", "Start a new session (also /clear)")]
    assert _complete("/q") == [("/quit", "/quit", "Quit (also /exit, ctrl+d)")]
    assert _complete("/ex") == [("/exit", "/exit", "Quit (also /quit, ctrl+d)")]
