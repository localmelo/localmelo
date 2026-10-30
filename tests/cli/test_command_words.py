"""Command words: ``melo serve`` and its kin are the options they replace, nothing more.

Each word gives the namespace its old options give. A task that starts with a command
word is a task when it is quoted or follows ``--``; typed bare, it ends in a hint and
exit 2 before any model is built.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.support import command_words, config, task_cli
from localmelo.support.config import Config
from localmelo.tests.cli.test_direct_mode_process import FakeChatServer, _env


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


# -- a command word typed after options -------------------------------------------------

URL = "http://x/v1"


def _misplaced(word: str, example: str) -> str:
    return (
        f"melo: {word} is a command: put it first, as in {example}.\n"
        f'To give melo the task "{word}", write melo -- {word}.\n'
    )


@pytest.mark.parametrize(
    ("argv", "example"),
    [
        # The example keeps the user's settings and drops what a command cannot take.
        (["-X", "gateway.port=9000", "serve"], "melo serve -X gateway.port=9000"),
        (["-Xgateway.port=9000", "serve"], "melo serve -X gateway.port=9000"),
        (["--set=gateway.port=9000", "serve"], "melo serve -X gateway.port=9000"),
        (
            ["-X", "cli.stream=on", "-X", "scheduler.policy=fcfs", "console"],
            "melo console -X cli.stream=on -X scheduler.policy=fcfs",
        ),
        (
            ["-X", "scheduler.gpu_lock_file=/tmp/a b", "serve"],
            "melo serve -X 'scheduler.gpu_lock_file=/tmp/a b'",
        ),
        (["-s", "5", "-X", "cli.stream=on", "setup"], "melo setup -X cli.stream=on"),
        # A setting that is a direct-run option (`--recall-k`) is not one the word takes.
        (
            ["-X", "retrieval.k=5", "-X", "cli.stream=on", "serve"],
            "melo serve -X cli.stream=on",
        ),
        (["-X", "retrieval.k=5", "serve"], "melo serve"),
        (["-u", URL, "-m", "x", "serve"], "melo serve"),
        (["--base-url", URL, "--chat-model", "x", "serve"], "melo serve"),
        (["-s", "5", "serve", "-t", "9"], "melo serve"),  # the word between options
        (["-X", "cli.stream=on", "config"], "melo config"),  # it takes no -X
        (["-s", "5", "task"], "melo task"),
        (["-s", "5", "worker"], "melo worker"),
        (["-s", "5", "import"], "melo import"),
    ],
    ids=lambda value: " ".join(value) if isinstance(value, list) else value,
)
def test_a_lone_command_word_after_options_is_refused_with_where_it_goes(
    argv: list[str], example: str, capsys: pytest.CaptureFixture[str]
) -> None:
    word = next(arg for arg in argv if arg in command_words.COMMANDS)
    code, ran, build = _run(argv)
    out, err = capsys.readouterr()
    assert (code, ran, out, err) == (2, None, "", _misplaced(word, example))
    build.assert_not_called()


@pytest.mark.parametrize("word", command_words.COMMANDS)
def test_every_command_word_is_refused_after_options_and_runs_after_the_separator(
    word: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, ran, build = _run(["-s", "5", word])
    assert (code, ran) == (2, None)
    assert capsys.readouterr().err == _misplaced(word, f"melo {word}")
    build.assert_not_called()
    code, ran, build = _run(["-s", "5", "--", word])
    assert (code, ran) == (0, word)
    build.assert_called_once()


@pytest.mark.parametrize("output", ["-o", "--output", "--output-format"])
def test_the_refusal_is_the_same_with_json_output(
    output: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # Every exit-2 usage error writes to stderr only, with or without JSON output: no
    # result line, and no request to the model server.
    code, ran, build = _run([output, "json", "-u", URL, "-m", "x", "serve"])
    out, err = capsys.readouterr()
    assert (code, ran, out) == (2, None, "")
    assert err == _misplaced("serve", "melo serve")
    build.assert_not_called()


def test_the_separator_makes_the_word_a_task_whatever_comes_before_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for argv in (
        ["-u", URL, "-m", "x", "--", "serve"],
        ["-X", "gateway.port=9000", "--", "serve"],
        ["-o", "json", "--", "serve"],
    ):
        code, ran, build = _run(argv)
        assert (code, ran) == (0, "serve"), argv
        build.assert_called_once()
    assert capsys.readouterr().err == ""


def test_an_option_value_or_a_longer_task_that_holds_the_word_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "serve").mkdir()
    monkeypatch.chdir(tmp_path)
    for argv, task in (
        (["-w", "serve", "fix it"], "fix it"),
        (["-m", "serve", "fix it"], "fix it"),
        (["-s", "5", "serve", "the", "docs"], "serve the docs"),
        (["-s", "5", "fix", "setup"], "fix setup"),
        (["-X", "cli.stream=on", "serve", "the", "docs"], "serve the docs"),
    ):
        code, ran, build = _run(argv)
        assert (code, ran) == (0, task), argv
        build.assert_called_once()
    assert _namespace(["-r", "serve"])["resume"] == "serve"  # a session titled so


def test_a_brief_that_says_only_a_command_word_is_still_a_brief(tmp_path: Path) -> None:
    path = tmp_path / "one-word.md"
    path.write_text("serve", encoding="utf-8")
    for argv in (["--brief", str(path)], ["worker", str(path)]):
        assert _namespace(argv)["query"] == ["serve"], argv


def _real_melo(
    tmp_path: Path, server: FakeChatServer, *argv: str
) -> subprocess.CompletedProcess[str]:
    """The real command in a process, on a stub server and a temporary HOME."""
    return subprocess.run(
        [sys.executable, "-I", "-m", "localmelo", "-u", server.base_url, "-m", "x"]
        + list(argv),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        env=_env(tmp_path),
        cwd=tmp_path,
        check=False,
    )


def test_the_real_command_refuses_before_it_asks_the_server_anything(
    tmp_path: Path,
) -> None:
    server = FakeChatServer()
    try:
        done = _real_melo(tmp_path, server, "-o", "json", "serve")
    finally:
        server.close()
    assert (done.returncode, done.stdout) == (2, "")
    assert done.stderr == _misplaced("serve", "melo serve")
    assert server.requests == 0 and server.bodies == []


def test_the_real_command_runs_the_word_as_a_task_after_the_separator(
    tmp_path: Path,
) -> None:
    server = FakeChatServer()
    try:
        done = _real_melo(tmp_path, server, "--", "serve")
    finally:
        server.close()
    assert (done.returncode, done.stdout, done.stderr) == (0, "done\n", "")
    assert server.requests == 1
    users = [m["content"] for m in server.bodies[0]["messages"] if m["role"] == "user"]
    assert any("serve" in content for content in users)


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
