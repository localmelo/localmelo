"""Command words: ``melo serve`` and its kin are the options they replace, nothing more.

Each word gives the namespace its old options give. A task that starts with a command
word is a task when it is quoted or follows ``--``; typed bare, it ends in a hint and
exit 2 before any model is built.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.support import command_words, config, task_cli
from localmelo.support.config import Config


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def brief(tmp_path: Path) -> str:
    path = tmp_path / "brief.md"
    path.write_text("Write hello.txt\n", encoding="utf-8")
    return str(path)


def _namespace(argv: list[str]) -> dict[str, Any]:
    return vars(cli._parse(argv)[1])


# (the word and its options, the options it replaces)
EQUIVALENT = [
    ("setup", "--reconfigure"),
    ("serve", "--serve"),
    ("serve -p 9000 --host 0.0.0.0", "--serve --host 0.0.0.0 --port 9000"),
    ("serve --port 0", "--serve --port 0"),
    ("serve --playground", "--serve --enable-playground"),
    # The macOS app's command line.
    (
        "serve --force --host 127.0.0.1 -p 8123",
        "--serve --allow-unconfigured --host 127.0.0.1 --port 8123",
    ),
    ("serve install", "--daemon install"),
    ("serve uninstall", "--daemon uninstall"),
    ("serve status", "--daemon status"),
    ("serve install -p 9001", "--daemon install --port 9001"),
    ("console", "--gateway-login"),
    ("console -p 5000", "--gateway-login --port 5000"),
    ("console --host ::1", "--gateway-login --host ::1"),
    ("console --reset", "--gateway-rotate-token"),
    ("import dsh-jsonl run.jsonl", "--import-session dsh-jsonl run.jsonl"),
    (
        "import dsh-jsonl run.jsonl --title T",
        "--import-session dsh-jsonl run.jsonl --title T",
    ),
    ("worker BRIEF", "--brief BRIEF"),
    (
        "worker BRIEF --parent r1 --title T --output-format stream-json",
        "--brief BRIEF --parent-run r1 --save-session T --output-format stream-json",
    ),
    (
        "worker BRIEF --parent=r1 --inbox-fd 4 --timeout 60",
        "--brief BRIEF --parent-run r1 --inbox-fd 4 --timeout 60",
    ),
]


@pytest.mark.parametrize(("new", "old"), EQUIVALENT, ids=[n for n, _ in EQUIVALENT])
def test_a_word_gives_the_namespace_of_its_old_options(
    new: str, old: str, brief: str
) -> None:
    new_argv = new.replace("BRIEF", brief).split()
    old_argv = old.replace("BRIEF", brief).split()
    assert command_words.expand(new_argv) != new_argv
    assert _namespace(new_argv) == _namespace(old_argv)


def test_title_names_a_task_session_as_save_session_does() -> None:
    assert _namespace(["--title", "T", "--", "q"]) == _namespace(
        ["--save-session", "T", "--", "q"]
    )
    assert _namespace(["--title", "T", "--", "q"])["title"] is None


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--title", "T"], "--title is for a task, melo worker or melo import"),
        (["--title", "T", "--save-session", "U", "q"], "both name the session"),
    ],
)
def test_title_without_a_task_is_a_usage_error(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli._parse(argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


# -- a task that starts with a command word ---------------------------------------


def _run(argv: list[str]) -> tuple[int, str | None, mock.MagicMock]:
    """Run main() with *argv*; a task would be run by the returned fake agent."""
    agent = mock.MagicMock()
    agent.run_task = mock.AsyncMock(return_value=mock.MagicMock(status="completed"))
    build = mock.MagicMock(return_value=agent)
    with (
        mock.patch.object(cli, "_register_backends"),
        mock.patch.object(cli, "_build_direct_mode_agent", build),
        mock.patch.object(cli, "_run_direct", return_value=0) as run_direct,
        mock.patch.object(sys, "argv", ["melo", *argv]),
    ):
        try:
            cli.main()
            code = 0
        except SystemExit as exc:
            assert isinstance(exc.code, int)
            code = exc.code
    task = run_direct.call_args.args[1] if run_direct.call_args else None
    return code, task, build


@pytest.mark.parametrize(
    ("argv", "task"),
    [
        (["setup the project"], "setup the project"),
        (["serve the docs on port 8000"], "serve the docs on port 8000"),
        (["--", "setup", "the", "project"], "setup the project"),
        (["--", "serve", "x"], "serve x"),
        (["--", "config"], "config"),
        (["fix", "setup"], "fix setup"),
        (["--no-save-session", "--", "task", "list"], "task list"),
    ],
)
def test_a_quoted_task_or_one_after_the_separator_stays_a_task(
    argv: list[str], task: str
) -> None:
    code, ran, build = _run(argv)
    assert (code, ran) == (0, task)
    build.assert_called_once()


@pytest.mark.parametrize(
    "words",
    [
        ["setup", "a", "venv", "here"],
        ["serve", "the", "docs"],
        ["serve", "install", "now"],
        ["console", "log", "x"],
        ["config", "the", "nginx", "server"],
        ["task", "fix", "the", "bug"],
        ["worker", "brief.md", "extra"],
    ],
    ids=lambda words: words[0],
)
def test_a_bare_task_that_starts_with_a_word_prints_the_quoting_hint(
    words: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, build = _run(words)
    assert code == 2
    assert capsys.readouterr().err == (
        f'melo: "{words[0]}" is a melo command. '
        f'To run this as a task, quote it: melo "{" ".join(words)}"\n'
    )
    build.assert_not_called()


def test_the_hint_escapes_what_the_shell_would_expand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, _, _ = _run(["setup", 'say "$HOME"'])
    assert code == 2
    assert 'melo "setup say \\"\\$HOME\\""' in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["serve", "--bogus"], "unrecognized arguments: --bogus"),
        (["serve", "install", "--force"], "--playground and --force are for running"),
        (["serve", "-p", "many"], "invalid int value: 'many'"),
        (["setup", "--yolo"], "unrecognized arguments: --yolo"),
        (["worker"], "the brief file comes first"),
        (["worker", "--inbox", "f"], "the brief file comes first"),
        (["import", "dsh-jsonl"], "the following arguments are required: file"),
    ],
)
def test_a_wrong_option_of_a_word_is_a_usage_error(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, build = _run(argv)
    assert code == 2
    assert message in capsys.readouterr().err
    build.assert_not_called()


@pytest.mark.parametrize(
    "word", ["setup", "serve", "console", "worker", "import", "config"]
)
def test_a_word_explains_itself_with_h(
    word: str, capsys: pytest.CaptureFixture[str]
) -> None:
    for flag in ("-h", "--help"):
        assert _run([word, flag])[0] == 0
    out = capsys.readouterr().out
    assert out.count(f"usage: melo {word}") == 2
    assert "Example: melo " in out
    assert max(len(line) for line in out.splitlines()) <= 80


# -- the words that do their own work ---------------------------------------------


def test_task_runs_the_task_client_under_its_own_name() -> None:
    with (
        mock.patch.object(task_cli, "main", return_value=7) as client,
        pytest.raises(SystemExit) as exc,
    ):
        command_words.expand(["task", "--json", "list"])
    assert exc.value.code == 7
    client.assert_called_once_with(["--json", "list"], prog="melo task")


def test_the_task_verbs_are_the_task_clients() -> None:
    (subcommands,) = (
        action
        for action in task_cli.parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert tuple(subcommands.choices) == command_words.TASK_VERBS


def test_config_shows_the_path_and_what_differs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        command_words.expand(["config"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == (
        "~/.cache/localmelo/config.toml (no file yet)\nevery setting is at its default\n"
    )
    cfg = Config(chat_backend="ollama")
    cfg.retrieval.k = 5
    config.save(cfg)
    with pytest.raises(SystemExit):
        command_words.expand(["config"])
    assert capsys.readouterr().out == (
        '~/.cache/localmelo/config.toml\nchat_backend = "ollama"\nretrieval.k = 5\n'
    )


def test_serve_force_starts_the_gateway_like_allow_unconfigured() -> None:
    empty = Config(chat_backend="", embedding_backend="")
    with (
        mock.patch.object(cli, "_register_backends"),
        mock.patch.object(cli, "_on_a_terminal", return_value=False),
        mock.patch.object(sys, "argv", ["melo", "serve", "--force", "-p", "8123"]),
        mock.patch("localmelo.support.config.load", return_value=empty),
        mock.patch.object(cli, "_start_gateway") as start,
    ):
        cli.main()
    start.assert_called_once_with(empty)
    assert empty._allow_unconfigured is True
    assert empty.gateway.port == 8123
