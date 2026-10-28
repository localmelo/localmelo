"""The opt-in direct worker brief and stream contract."""

import io
import json
from pathlib import Path

import pytest

from localmelo.support.cli.worker_contract import WorkerObserver, read_brief


def test_read_brief_preserves_text_and_rejects_empty_and_invalid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "brief.md"
    path.write_text("First line\nSecond line\n", encoding="utf-8")
    assert read_brief(path) == "First line\nSecond line\n"
    path.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_brief(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError):
        read_brief(path)


def test_stream_redacts_identified_sensitive_read(tmp_path: Path) -> None:
    from localmelo.melo.contracts.events import AgentEvent

    out = io.StringIO()
    observer = WorkerObserver(out, str(tmp_path), stream=True)
    observer(
        AgentEvent(
            type="tool.started",
            run_id="r",
            seq=1,
            ts=0,
            data={"name": "file_read", "call_id": "c", "arguments": {"path": ".env"}},
        )
    )
    observer(
        AgentEvent(
            type="tool.finished",
            run_id="r",
            seq=2,
            ts=0,
            data={
                "name": "file_read",
                "call_id": "c",
                "output": "sensitive fixture",
                "output_chars": 17,
                "error": "",
                "images": [],
            },
        )
    )
    observer.close()
    frames = [json.loads(line) for line in out.getvalue().splitlines()]
    assert len(frames) == 2
    assert "sensitive fixture" not in json.dumps(frames)
    assert "Sensitive file content omitted" in frames[1]["event"]["data"]["output"]


def test_test_evidence_uses_latest_check_in_current_attempt(tmp_path: Path) -> None:
    from localmelo.melo.contracts.events import AgentEvent

    observer = WorkerObserver(io.StringIO(), str(tmp_path), stream=False)

    def event(kind: str, **data: object) -> None:
        observer(AgentEvent(type=kind, run_id="r", seq=1, ts=0, data=data))

    event("step.started", attempt=1)
    event(
        "tool.started",
        name="shell_exec",
        call_id="a",
        arguments={"command": "pytest -q"},
    )
    event(
        "tool.finished",
        name="shell_exec",
        call_id="a",
        status="success",
        exit_code=0,
        output="3 passed",
        error="",
    )
    assert observer.tests()["state"] == "recorded"
    event("step.started", attempt=2)
    assert observer.tests()["state"] == "not_recorded"
    event(
        "tool.started",
        name="shell_exec",
        call_id="b",
        arguments={"command": "pytest -q"},
    )
    event(
        "tool.finished",
        name="shell_exec",
        call_id="b",
        status="success",
        exit_code=1,
        output="1 failed",
        error="",
    )
    assert observer.tests()["state"] == "not_recorded"


@pytest.mark.parametrize(
    "fmt,save", [("json", False), ("stream-json", False), ("stream-json", True)]
)
def test_brief_main_with_fake_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fmt: str,
    save: bool,
) -> None:
    import sys

    import httpx

    import localmelo.__main__ as cli
    from localmelo.melo.agent import Agent, AgentOptions
    from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

    brief = tmp_path / "brief.md"
    brief.write_text("Explain the example.\n", encoding="utf-8")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Done"}}]},
        )
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "_build_direct_mode_agent",
        lambda *args, **kwargs: Agent(
            llm=OpenAICompatLLM(
                base_url="http://llm.test/v1", model="stub", transport=transport
            ),
            options=AgentOptions(memory=False),
        ),
    )
    args = ["melo", "--brief", str(brief), "--output-format", fmt]
    if save:
        args += ["--save-session"]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    result = records[-1]["result"] if fmt == "stream-json" else records[-1]
    assert result["v"] == 1 and result["status"] == "completed"
    assert result["run"]["result"] == "Done"
    assert result["tests"]["state"] == "not_recorded"
    if fmt == "stream-json":
        assert records[-1]["type"] == "result"
        assert records[-2]["event"]["type"] == "run.finished"


@pytest.mark.parametrize(
    "args",
    [
        ["--brief", "missing.md"],
        ["--brief", "brief.md", "task"],
        ["--brief", "brief.md", "--serve"],
        ["--brief", "brief.md", "--output-format", "text"],
        ["--output-format", "stream-json", "task"],
        ["--br", "brief.md"],
    ],
)
def test_invalid_modes_fail_before_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    import sys

    import localmelo.__main__ as cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "brief.md").write_text("Task", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["melo", *args])
    monkeypatch.setattr(
        cli,
        "_build_direct_mode_agent",
        lambda *a, **kw: pytest.fail("agent constructed"),
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


async def test_broken_stream_returns_nonzero_without_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import httpx

    import localmelo.__main__ as cli
    from localmelo.melo.agent import Agent, AgentOptions
    from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

    class Broken(io.StringIO):
        def write(self, value: str) -> int:
            raise BrokenPipeError("closed")

    monkeypatch.setenv("HOME", str(tmp_path))
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
    code = await cli._run(
        agent, "task", output_format="stream-json", out=Broken(), worker_brief=True
    )
    assert code == 1
