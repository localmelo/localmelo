"""The ``!`` shell: what a draft means, one run on a pseudo-terminal, snapshots.

Runs use no real terminal (no fds): the child still gets a pty of its own, and
the output is only kept.
"""

from __future__ import annotations

import os
import signal
import sys
import termios
from pathlib import Path

import pytest

from localmelo.support.cli import usershell
from localmelo.support.cli.usershell import ShellResult, ShellRun, parse, snapshot

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("! ls -l", ("run", "ls -l")),
        ("!\tls", ("run", "ls")),
        ("!   echo  hi  ", ("run", "echo  hi")),
        ("! echo a\necho b", ("run", "echo a\necho b")),
        ("!", ("help", "")),
        ("!  ", ("help", "")),
        ("\\! hello", ("literal", "! hello")),
        ("\\!", ("literal", "!")),
        ("\\!x", None),
        ("!important", None),
        (" ! ls", None),
        ("hello ! ls", None),
        ("/help", None),
    ],
)
def test_parse(text: str, expected: tuple[str, str] | None) -> None:
    parsed = parse(text)
    assert (None if parsed is None else (parsed.kind, parsed.text)) == expected


def test_the_shell_is_shell_when_usable_else_sh(tmp_path: Path) -> None:
    assert usershell.user_shell({"SHELL": "/bin/sh"}) == "/bin/sh"
    assert usershell.user_shell({"SHELL": "zsh"}) == "/bin/sh"  # not absolute
    assert usershell.user_shell({"SHELL": str(tmp_path / "missing")}) == "/bin/sh"
    assert usershell.user_shell({"SHELL": str(tmp_path)}) == "/bin/sh"
    assert usershell.user_shell({}) == "/bin/sh"


async def _run(command: str, cwd: Path, **kwargs: object) -> ShellResult:
    return await ShellRun(command, str(cwd), shell="/bin/sh", **kwargs).run()  # type: ignore[arg-type]


async def test_a_run_reports_its_exit_code_cwd_and_output(tmp_path: Path) -> None:
    result = await _run("pwd; echo out; echo err >&2; exit 3", tmp_path)
    assert result.returncode == 3 and result.status() == "exit 3"
    assert result.signal is None
    text = result.output.decode()
    assert os.path.realpath(text.splitlines()[0]) == os.path.realpath(tmp_path)
    assert "out" in text and "err" in text  # one stream, as on a terminal
    assert 0 <= result.elapsed_s < 5


async def test_a_signal_is_reported_by_name(tmp_path: Path) -> None:
    result = await _run("kill -TERM $$", tmp_path)
    assert result.returncode == -signal.SIGTERM
    assert result.status() == "killed by SIGTERM"


async def test_a_pipeline_runs_and_the_child_has_a_terminal(tmp_path: Path) -> None:
    result = await _run(
        "printf 'b\\na\\n' | sort | head -1; test -t 0 && echo tty", tmp_path
    )
    assert result.output.decode().split() == ["a", "tty"]


async def test_a_missing_working_directory_is_a_start_failure(tmp_path: Path) -> None:
    result = await _run("true", tmp_path / "gone")
    assert result.returncode is None
    assert result.status().startswith("could not start: ")


async def test_the_tail_of_the_output_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(usershell, "MAX_OUTPUT_BYTES", 1000)
    result = await _run("seq 1 2000", tmp_path)
    assert len(result.output) == 1000 and result.truncated
    assert result.total_bytes == len("".join(f"{n}\r\n" for n in range(1, 2001)))
    item = snapshot(result)
    assert item is not None
    assert item.content.startswith("[earlier output left out: the last 1,000 of ")
    assert item.content.rstrip().endswith("2000")


async def test_stop_ends_the_childs_group_and_nothing_else(tmp_path: Path) -> None:
    import asyncio

    run = ShellRun("echo $$; sleep 60 & sleep 60", str(tmp_path), shell="/bin/sh")
    task = asyncio.ensure_future(run.run())
    await asyncio.sleep(0.5)
    assert run._process is not None
    pid = run._process.pid
    assert os.getpgid(pid) == pid != os.getpgrp()
    run.stop()
    result = await asyncio.wait_for(task, 10)
    assert result.stopped and result.signal == "SIGTERM"
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)  # the backgrounded sleep went with its group


async def test_stop_never_signals_melos_own_group(tmp_path: Path) -> None:
    run = ShellRun("true", str(tmp_path), shell="/bin/sh")
    result = await run.run()
    assert result.returncode == 0
    run.stop()  # a finished run: nothing is signalled (the test is still alive)


def test_readable_text_drops_escapes_and_applies_redraws() -> None:
    raw = (
        b"\x1b[31mred\x1b[0m \x1b]0;title\x07done\r\n"
        b"10%\r50%\r100%\r\n"
        b"bold: b\x08bB\r\n"
        b"caf\xc3\xa9 \xe4\xbd\xa0\xe5\xa5\xbd\r\n"
        b"\x1b[?25l\x1b[2Kgone\x07\r\n"
    )
    assert usershell.readable(raw) == ("red done\n100%\nbold: bB\ncafé 你好\ngone\n")


def test_a_split_utf8_character_at_the_cut_is_dropped_not_shown() -> None:
    assert usershell.readable("你好".encode()[1:]) == "好"


def _result(output: bytes, command: str = "echo x") -> ShellResult:
    return ShellResult(command, "/tmp", 0.1, 0, output, len(output))


def test_a_snapshot_is_redacted_and_says_so() -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    item = snapshot(_result(f"key {secret}\r\n".encode(), f"echo {secret}"))
    assert item is not None
    assert secret not in item.content and secret not in item.path
    assert "[redacted]" in item.content and item.changed


def test_a_plain_snapshot_is_not_flagged_and_an_empty_one_is_none() -> None:
    item = snapshot(_result(b"\x1b[32mok\x1b[0m\r\n"))
    assert item is not None and item.content == "ok\n" and not item.changed
    assert snapshot(_result(b"\x1b[2J\r\n  \r\n")) is None
    assert snapshot(_result(b"")) is None


def test_a_snapshot_name_is_one_short_line() -> None:
    item = snapshot(_result(b"x\r\n", "echo a\n" + "b" * 100))
    assert item is not None
    assert "\n" not in item.path and len(item.path) <= 80 and item.path.endswith("…")


def test_concealed_text_is_not_in_the_snapshot() -> None:
    raw = (
        b"visible\r\n\x1b[8mSECRET INSTRUCTION\x1b[0m\r\nline3\r\n"
        b"\x1b[0;8mmore hidden\x1b[28m shown\r\n"
        b"\x1b[38;5;8mgrey is not concealed\x1b[0m\r\n"
    )
    text = usershell.readable(raw)
    assert text == "visible\n\nline3\n shown\ngrey is not concealed\n"
    item = snapshot(_result(raw))
    assert item is not None and "SECRET" not in item.content
    assert "hidden" not in item.content


def test_text_before_the_last_screen_clear_is_not_in_the_snapshot() -> None:
    raw = b"old\r\n\x1b[2Jmiddle\r\n\x1b[H\x1b[3Jscreen cleared\r\n"
    assert usershell.readable(raw) == "screen cleared\n"
    # Concealment set before a clear still holds after it.
    assert usershell.readable(b"\x1b[8mhid\x1b[2Jstill hid\x1b[0m seen") == " seen"


def test_a_cut_tail_drops_its_partial_first_line_before_redaction() -> None:
    sk = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
    gh = "ghp_" + "a1B2c3D4" * 5
    for key in (sk, gh):
        output = (key[4:] + "\r\nafter\r\n").encode()  # cut four characters in
        item = snapshot(ShellResult("x", "/tmp", 0.1, 0, output, len(output) + 9))
        assert item is not None and key[4:] not in item.content
        assert item.content.endswith("\nafter\n")
        whole = snapshot(_result(output))  # not cut: nothing is dropped
        assert whole is not None and key[4:] in whole.content
    one_line = ShellResult("x", "/tmp", 0.1, 0, b"no newline", 99)
    assert snapshot(one_line) is None


async def _modes_after(stdin_fd: int, command: str, tmp_path: Path) -> list[int]:
    run = ShellRun(command, str(tmp_path), shell="/bin/sh", stdin_fd=stdin_fd)
    await run.run()
    return termios.tcgetattr(stdin_fd)


async def test_the_terminal_mode_is_restored_after_a_run(tmp_path: Path) -> None:
    master, slave = os.openpty()
    try:
        before = termios.tcgetattr(slave)
        assert await _modes_after(slave, "true", tmp_path) == before
        assert await _modes_after(slave, "exit 4", tmp_path) == before  # failed
        assert (
            await _modes_after(slave, "echo $((1+1)); kill -9 $$", tmp_path) == before
        )
    finally:
        os.close(master)
        os.close(slave)


async def test_the_terminal_mode_is_restored_after_a_cancelled_run(
    tmp_path: Path,
) -> None:
    import asyncio

    master, slave = os.openpty()
    try:
        before = termios.tcgetattr(slave)
        run = ShellRun("sleep 30", str(tmp_path), shell="/bin/sh", stdin_fd=slave)
        task = asyncio.ensure_future(run.run())
        for _ in range(500):
            if run._process is not None and termios.tcgetattr(slave) != before:
                break  # raw mode is on
            await asyncio.sleep(0.01)
        assert termios.tcgetattr(slave) != before
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert termios.tcgetattr(slave) == before
    finally:
        os.close(master)
        os.close(slave)
