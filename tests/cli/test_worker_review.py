"""Regression tests for worker result evidence and stream privacy."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.events import AgentEvent
from localmelo.support.cli.worker_contract import (
    WorkerObserver,
    read_brief,
    workspace_head,
)


def emit(observer: WorkerObserver, kind: str, **data: Any) -> None:
    observer(AgentEvent(type=kind, run_id="r", seq=1, ts=0, data=data))


def test_output_format_help_and_invalid_choice(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import localmelo.__main__ as cli
    from localmelo.support import command_words

    parser = cli._build_parser()
    assert "stream-json" in command_words.help_text("worker")
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--output-format", "bogus", "task"])
    assert exc.value.code == 2
    assert "choose from 'text', 'json', 'stream-json'" in capsys.readouterr().err


def test_directory_brief_error_names_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import localmelo.__main__ as cli

    monkeypatch.setattr(sys, "argv", ["melo", "--brief", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert str(tmp_path) in error and "not a regular file" in error
    assert "directory: 3" not in error


def test_assignment_in_assistant_and_tool_arguments_is_masked(tmp_path: Path) -> None:
    out = io.StringIO()
    observer = WorkerObserver(out, str(tmp_path), stream=True)
    emit(observer, "assistant.message", content="echo ACME_CREDENTIAL=privatevalue")
    emit(
        observer,
        "tool.started",
        name="shell_exec",
        call_id="x",
        arguments={"command": "echo ACME_CREDENTIAL=privatevalue"},
    )
    observer.close()
    records = [json.loads(line) for line in out.getvalue().splitlines()]
    assert len(records) == 2
    assert "privatevalue" not in out.getvalue()
    assert all("[redacted]" in json.dumps(record) for record in records)


def test_unreadable_brief_paths_fail_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for path in (tmp_path, fifo, tmp_path / "missing"):
        with pytest.raises(ValueError):
            read_brief(path)


def test_workspace_head_and_dirty_are_independent(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "file").write_text("test")
    assert workspace_head(str(tmp_path)) == (None, True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "Initial",
        ],
        check=True,
    )
    head, _ = workspace_head(str(tmp_path))
    (tmp_path / ".git" / "index").write_bytes(b"invalid")
    assert workspace_head(str(tmp_path)) == (head, None)


def test_stream_masks_sensitive_assignment_lines(tmp_path: Path) -> None:
    out = io.StringIO()
    observer = WorkerObserver(out, str(tmp_path), stream=True)
    emit(
        observer,
        "assistant.message",
        content="API_TOKEN=privatevalue\nPASSWORD: anothersecret\nnormal=yes",
    )
    observer.close()
    text = json.loads(out.getvalue())["event"]["data"]["content"]
    assert "privatevalue" not in text and "anothersecret" not in text
    assert "normal=yes" in text


def test_recorded_check_has_raw_bounded_command(tmp_path: Path) -> None:
    observer = WorkerObserver(io.StringIO(), str(tmp_path), stream=False)
    command = "pytest " + "x" * 2100
    emit(
        observer,
        "tool.started",
        name="shell_exec",
        call_id="a",
        arguments={"command": command},
    )
    emit(
        observer,
        "tool.finished",
        call_id="a",
        status="success",
        exit_code=0,
        output="1 passed",
    )
    item = observer.tests()["items"][0]
    assert item["command"] == command[:2000]
    assert item["command_truncated"] is True and "result" not in item


_FAKE_MAIN = r"""
import sys, httpx
import localmelo.__main__ as cli
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "café"}}]}))
cli._build_direct_mode_agent = lambda *a, **k: Agent(llm=OpenAICompatLLM(base_url="http://test/v1", model="fake", transport=transport), options=AgentOptions(memory=False))
sys.argv = ["melo", "--brief", sys.argv[1], "--output-format", sys.argv[2]]
cli.main()
"""


@pytest.mark.parametrize("fmt", ["json", "stream-json"])
def test_real_cli_ascii_and_closed_pipe(tmp_path: Path, fmt: str) -> None:
    brief = tmp_path / "brief"
    brief.write_text("task")
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONIOENCODING": "ascii"}
    argv = [sys.executable, "-c", _FAKE_MAIN, str(brief), fmt]
    normal = subprocess.run(argv, env=env, capture_output=True, timeout=20)
    assert normal.returncode == 0
    records = [json.loads(line) for line in normal.stdout.splitlines()]
    result = records[-1]["result"] if fmt == "stream-json" else records[-1]
    assert result["run"]["result"] == "café"
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        broken = subprocess.run(
            argv, env=env, stdout=write_fd, stderr=subprocess.PIPE, timeout=20
        )
    finally:
        os.close(write_fd)
    assert broken.returncode == 1
    assert (
        b"worker output failed" in broken.stderr and b"Traceback" not in broken.stderr
    )


def test_sensitive_shell_output_is_omitted(tmp_path: Path) -> None:
    out = io.StringIO()
    observer = WorkerObserver(out, str(tmp_path), stream=True)
    emit(
        observer,
        "tool.started",
        name="shell_exec",
        call_id="a",
        arguments={"command": "cat .env"},
    )
    emit(
        observer,
        "tool.finished",
        call_id="a",
        output="unmatched secret fixture",
        error="",
        images=[],
    )
    observer.close()
    assert "unmatched secret fixture" not in out.getvalue()
    assert "Sensitive file content omitted" in out.getvalue()
