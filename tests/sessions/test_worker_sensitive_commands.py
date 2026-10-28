"""Best-effort detection of simple sensitive tool reads."""

from pathlib import Path

import pytest

from localmelo.support.sessions.worker import sensitive_read


@pytest.mark.parametrize(
    "command",
    [
        "cat .env",
        "head -n 2 .env",
        "tail -c 10 .env",
        "less .env",
        "more .env",
        "sed -n '1,2p' .env",
        "sed -n -e '1p' .env",
        "grep -n example .env",
        "grep -e example .env",
        "cat -- 'folder/.env.local'",
    ],
)
def test_shell_sensitive_reads(command: str, tmp_path: Path) -> None:
    assert sensitive_read("shell_exec", {"command": command}, str(tmp_path))


def test_shell_paths_and_symlinks(tmp_path: Path) -> None:
    (tmp_path / "alias.txt").symlink_to(tmp_path / ".env")
    for path in (str(tmp_path / ".env"), "sub/../.env", "alias.txt"):
        assert (
            sensitive_read("shell_exec", {"command": f"cat {path}"}, str(tmp_path))
            == path
        )


@pytest.mark.parametrize(
    "command",
    [
        "cat normal.txt",
        "grep .env normal.txt",
        "sed -n '.env' normal.txt",
        "echo .env",
        "cat '.env",
        "cat $FILE",
        "cat normal.txt | grep .env",
    ],
)
def test_shell_non_reads_and_unsupported(command: str, tmp_path: Path) -> None:
    assert sensitive_read("shell_exec", {"command": command}, str(tmp_path)) is None


@pytest.mark.parametrize(
    "code",
    [
        "print(open('.env').read())",
        "with open(file='../secrets.txt') as f: print(f.read())",
        "open('normal.txt'); open('.env.local')",
    ],
)
def test_python_literal_reads(code: str, tmp_path: Path) -> None:
    assert sensitive_read("python_exec", {"code": code}, str(tmp_path))


def test_python_literal_absolute_and_symlink(tmp_path: Path) -> None:
    (tmp_path / "alias").symlink_to(tmp_path / ".env")
    for path in (str(tmp_path / ".env"), "alias", "dir/../.env"):
        assert (
            sensitive_read("python_exec", {"code": f"open({path!r})"}, str(tmp_path))
            == path
        )


@pytest.mark.parametrize(
    "code", ["open(path)", "open('normal.txt')", "open('", "print('.env')"]
)
def test_python_unsupported_or_non_sensitive(code: str, tmp_path: Path) -> None:
    assert sensitive_read("python_exec", {"code": code}, str(tmp_path)) is None


def test_direct_read_preserved(tmp_path: Path) -> None:
    assert (
        sensitive_read("mcp__files__read_file", {"path": ".env"}, str(tmp_path))
        == ".env"
    )
    assert sensitive_read("file_read", {"path": "normal.txt"}, str(tmp_path)) is None
