"""Worker startup and output transport failures."""

from __future__ import annotations

import io
import json
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


def test_legacy_startup_exception_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["melo", "task"])

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("original")

    monkeypatch.setattr(cli, "_build_direct_mode_agent", fail)
    with pytest.raises(ValueError, match="original"):
        cli.main()


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
    monkeypatch.setattr(sys, "argv", ["melo", "--brief", str(brief), "--output-format", fmt])
    monkeypatch.setattr(
        cli,
        "_build_direct_mode_agent",
        lambda *a, **kw: make_agent(
            ScriptedLLM([shell("pytest -q"), answer("Done")]), tools=passing_tools()
        ),
    )
    cli.main()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    result = records[-1]["result"] if fmt == "stream-json" else records[-1]
    assert result["tests"]["state"] == "recorded"
    assert result["tests"]["items"][0]["command"] == "pytest -q"
    assert result["tests"]["items"][0]["source"] == "tool_event"


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
    assert records[-1]["type"] == "result"
    assert records[-1]["result"]["status"] == "transport_error"
    assert records[-1]["result"]["run"] is None
