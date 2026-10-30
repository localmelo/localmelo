"""Worker startup and output transport failures."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import localmelo.__main__ as cli


@pytest.mark.parametrize("fmt", ["json", "stream-json"])
def test_startup_error_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fmt: str,
) -> None:
    brief = tmp_path / "brief"
    brief.write_text("Task", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["melo", "--brief", str(brief), "--output-format", fmt]
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(cli, "_build_direct_mode_agent", fail)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert output.out.count("\n") == 1
    record = json.loads(output.out)
    if fmt == "stream-json":
        assert record["type"] == "result"
        record = record["result"]
    assert record["status"] == "startup_error"
    assert record["run"] is None and record["run_id"] is None
    assert record["exit_code"] == 1
    assert record["tests"] == {"state": "not_recorded", "items": []}
    assert "backend unavailable" in output.err


@pytest.mark.parametrize("fmt", ["json", "stream-json"])
def test_config_value_after_brief_reports_startup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fmt: str,
) -> None:
    from localmelo.support import config

    brief = tmp_path / "brief"
    brief.write_text("Task", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["melo", "--brief", str(brief), "--output-format", fmt]
    )
    monkeypatch.setattr(
        config,
        "load",
        lambda: config.Config(agent=config.AgentConfig(context_window=-1)),
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    result = records[-1]["result"] if fmt == "stream-json" else records[-1]
    assert result["status"] == "startup_error" and result["run"] is None
    assert result["exit_code"] == 1
    assert "context_window" in result["error"]


@pytest.mark.parametrize("fmt", ["json", "stream-json"])
def test_fixed_prefix_config_after_brief_reports_startup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fmt: str,
) -> None:
    from localmelo.support import config

    brief = tmp_path / "brief"
    brief.write_text("Task", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["melo", "--brief", str(brief), "--output-format", fmt]
    )
    monkeypatch.setattr(
        config,
        "load",
        lambda: config.Config(agent=config.AgentConfig(fixed_prefix="yes")),  # type: ignore[arg-type]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    result = records[-1]["result"] if fmt == "stream-json" else records[-1]
    assert result["status"] == "startup_error" and result["run"] is None
    assert "fixed_prefix must be a boolean" in result["error"]


@pytest.mark.parametrize("fmt", ["json", "stream-json"])
def test_unconfigured_backend_reports_startup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fmt: str,
) -> None:
    from localmelo.support import config

    brief = tmp_path / "brief"
    brief.write_text("Task", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "load", lambda: config.Config())
    monkeypatch.setattr(
        sys, "argv", ["melo", "--brief", str(brief), "--output-format", fmt]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    record = json.loads(capsys.readouterr().out)
    if fmt == "stream-json":
        record = record["result"]
    assert record["status"] == "startup_error" and record["run"] is None


def test_config_failure_is_redacted_and_usage_stays_resultless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from localmelo.support import config

    brief = tmp_path / "brief"
    brief.write_text("Task", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["melo", "--brief", str(brief), "extra"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2 and capsys.readouterr().out == ""

    def fail() -> config.Config:
        raise ValueError("API_TOKEN=privatevalue")

    monkeypatch.setattr(config, "load", fail)
    monkeypatch.setattr(sys, "argv", ["melo", "--brief", str(brief)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert "privatevalue" not in output.out + output.err
    assert json.loads(output.out)["status"] == "startup_error"


def test_startup_error_shows_home_as_tilde_and_redacts_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from localmelo.support import config

    brief = tmp_path / "brief"
    brief.write_text("Task", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    home_config = tmp_path / ".cache" / "localmelo" / "config.toml"

    def fail() -> config.Config:
        raise OSError(f"cannot read {home_config}: API_TOKEN=privatevalue")

    monkeypatch.setattr(config, "load", fail)
    monkeypatch.setattr(sys, "argv", ["melo", "--brief", str(brief)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    output = capsys.readouterr()
    record = json.loads(output.out)
    assert record["error"] == (
        "cannot read ~/.cache/localmelo/config.toml: API_TOKEN=[redacted]"
    )
    assert str(tmp_path) not in output.out + output.err
    assert "privatevalue" not in output.out + output.err


_FAKE_STARTUP_FAILURE = r"""
import sys
import localmelo.__main__ as cli

def fail(*args, **kwargs):
    raise RuntimeError("backend unavailable")

cli._build_direct_mode_agent = fail
sys.argv = ["melo", "--brief", sys.argv[1], "--output-format", sys.argv[2]]
cli.main()
"""


@pytest.mark.parametrize("fmt", ["json", "stream-json"])
def test_startup_error_on_a_broken_pipe_exits_one(tmp_path: Path, fmt: str) -> None:
    brief = tmp_path / "brief"
    brief.write_text("task", encoding="utf-8")
    env = {**os.environ, "HOME": str(tmp_path)}
    argv = [sys.executable, "-c", _FAKE_STARTUP_FAILURE, str(brief), fmt]
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        broken = subprocess.run(
            argv, env=env, stdout=write_fd, stderr=subprocess.PIPE, timeout=20
        )
    finally:
        os.close(write_fd)
    assert broken.returncode == 1
    assert b"worker output failed" in broken.stderr
    assert b"Exception ignored" not in broken.stderr
    assert b"Traceback" not in broken.stderr


def test_legacy_startup_exception_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["melo", "--", "task"])

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("original")

    monkeypatch.setattr(cli, "_build_direct_mode_agent", fail)
    with pytest.raises(ValueError, match="original"):
        cli.main()


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("text", "Done.\n"),
        (
            "json",
            '{"task_id": "fixed", "status": "completed", "result": "Done.", '
            '"error": "", "steps": 0, "attempts": 0, "n_llm": 0, '
            '"n_main": 0, "n_aux": 0, "n_tool": 0, "llm_errors": 0, '
            '"usage": {}, "duration_s": 0.0, "config": {}, '
            '"localmelo_version": "fixed-version"}\n',
        ),
    ],
)
def test_legacy_direct_output_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fmt: str,
    expected: str,
) -> None:
    from localmelo.melo.schema import RunResult

    calls: list[str] = []

    class FakeAgent:
        async def run_task(self, query: str) -> RunResult:
            calls.append(query)
            return RunResult(task_id="fixed", status="completed", result="Done.")

        async def close(self) -> None:
            pass

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["melo", "--no-save-session", "--output-format", fmt, "--", "task"]
    )
    monkeypatch.setattr(cli, "_build_direct_mode_agent", lambda *a, **kw: FakeAgent())
    monkeypatch.setattr(cli, "_localmelo_version", lambda: "fixed-version")
    cli.main()
    captured = capsys.readouterr()
    assert captured.out.encode("utf-8") == expected.encode("utf-8")
    assert captured.err == "" and calls == ["task"]


@pytest.mark.parametrize("fmt", ["json", "stream-json"])
def test_brief_main_reports_executed_test_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fmt: str,
) -> None:
    from localmelo.tests.agent.procedural_helpers import (
        ScriptedLLM,
        answer,
        make_agent,
        passing_tools,
        shell,
    )

    brief = tmp_path / "brief"
    brief.write_text("Run pytest in the project", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["melo", "--brief", str(brief), "--output-format", fmt]
    )
    monkeypatch.setattr(
        cli,
        "_build_direct_mode_agent",
        lambda *a, **kw: make_agent(
            ScriptedLLM([shell("pytest -q"), answer("Done")]), tools=passing_tools()
        ),
    )
    cli.main()
    out = capsys.readouterr().out
    records = [json.loads(line) for line in out.splitlines()]
    if fmt == "json":
        assert len(records) == 1 and out.count("\n") == 1
    else:
        assert sum(record["type"] == "result" for record in records) == 1
    result = records[-1]["result"] if fmt == "stream-json" else records[-1]
    assert result["tests"]["state"] == "recorded"
    assert result["tests"]["items"][0]["command"] == "pytest -q"
    assert result["tests"]["items"][0]["exit_code"] is None
    assert result["tests"]["items"][0]["source"] == "tool_event"


def test_short_stream_write_is_a_transport_failure(tmp_path: Path) -> None:
    from localmelo.support.cli.worker_contract import WorkerObserver

    class Short(io.StringIO):
        def write(self, text: str) -> int:
            return super().write(text[:-1])

    observer = WorkerObserver(Short(), str(tmp_path), stream=True)
    observer._enqueue({"v": 1})
    observer.close()
    assert isinstance(observer.error, OSError)


async def test_oversized_event_reports_transport_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    from localmelo.melo.agent import Agent, AgentOptions
    from localmelo.support.cli import worker_contract
    from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

    monkeypatch.setattr(worker_contract, "MAX_EVENT_BYTES", 1)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Done"}}]},
        )
    )
    agent = Agent(
        llm=OpenAICompatLLM(
            base_url="http://llm.test/v1", model="stub", transport=transport
        ),
        options=AgentOptions(memory=False),
    )
    out = io.StringIO()
    code = await cli._run(
        agent, "task", output_format="stream-json", out=out, worker_brief=True
    )
    assert code == 1
    records = [json.loads(line) for line in out.getvalue().splitlines()]
    assert sum(record["type"] == "result" for record in records) == 1
    assert records[-1]["type"] == "result"
    assert records[-1]["result"]["status"] == "transport_error"
    assert records[-1]["result"]["run"] is None
