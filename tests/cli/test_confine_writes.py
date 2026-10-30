"""Confined writes (#261): the defaults, the switch, the run's record, fail-closed.

On for ``--brief`` runs and saved tasks, off for chats and other direct runs;
``[agent] confine_writes = false`` (or ``-X agent.confine_writes=false``) turns
it off. The executor side is in ``tests/executor/test_sandbox.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.executor import ConfinementError, sandbox
from localmelo.melo.schema import Message, ToolCall
from localmelo.support import config, config_cli
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.agent.test_reflection_gate import FakeLLM

URL = ["--base-url", "http://127.0.0.1:1/v1", "--chat-model", "m"]


def _shell(command: str) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_call=ToolCall(tool_name="shell_exec", arguments={"command": command}),
    )


def _agent(root: Path | None, home: Path, **options: Any) -> Agent:
    return Agent(
        llm=FakeLLM([_shell("touch ~/outside.txt"), Message("assistant", "Done")]),
        options=AgentOptions(
            memory=False,
            builtins=frozenset({"shell_exec"}),
            workspace_root=None if root is None else str(root),
            tool_env={"HOME": str(home)},
            **options,
        ),
    )


class TestOptionsAndRecord:
    def test_the_option_is_off_by_default_and_a_bool(self) -> None:
        assert AgentOptions().confine_writes is False
        with pytest.raises(ValueError, match="confine_writes must be a bool"):
            AgentOptions(confine_writes="yes")  # type: ignore[arg-type]

    @pytest.mark.parametrize("rooted", [True, False])
    async def test_a_confined_run_counts_its_refused_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rooted: bool
    ) -> None:
        if sys.platform != "darwin":
            pytest.skip("writes are confined only on macOS")
        (tmp_path / "ws").mkdir()
        (tmp_path / "home").mkdir()
        # Without a root, the folder melo runs in is the workspace.
        monkeypatch.chdir(tmp_path / "ws")
        root = tmp_path / "ws" if rooted else None
        try:
            agent = _agent(root, tmp_path / "home", confine_writes=True)
        except ConfinementError as exc:
            pytest.skip(str(exc))
        try:
            outcome = await agent.run_task("make a file")
        finally:
            await agent.close()
        assert outcome.to_dict()["confinement"] == {
            "writes": "confined",
            "refused_writes": 1,
        }
        assert not (tmp_path / "home" / "outside.txt").exists()
        assert (tmp_path / "ws" / sandbox.TMP_DIR / ".gitignore").is_file()

    async def test_off_macos_the_record_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sandbox, "sys", SimpleNamespace(platform="linux"))
        agent = _agent(tmp_path, tmp_path, confine_writes=True)
        try:
            outcome = await agent.run_task("make a file")
        finally:
            await agent.close()
        assert outcome.to_dict()["confinement"] == {
            "writes": "not_confined",
            "reason": "writes are confined only on macOS, not linux",
        }

    async def test_off_the_record_has_no_key(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path, tmp_path / "home")
        try:
            outcome = await agent.run_task("make a file")
        finally:
            await agent.close()
        assert outcome.confinement is None and "confinement" not in outcome.to_dict()
        assert not (tmp_path / sandbox.TMP_DIR).exists()

    def test_the_agent_fails_closed_before_any_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sandbox, "sys", SimpleNamespace(platform="darwin"))
        monkeypatch.setattr(sandbox, "SANDBOX_EXEC", "/nonexistent/sandbox-exec")
        with pytest.raises(ConfinementError, match="agent.confine_writes"):
            _agent(tmp_path, tmp_path, confine_writes=True)


class TestConfig:
    def test_the_key_defaults_on_and_is_written_only_when_off(self) -> None:
        assert config.AgentConfig().confine_writes is True
        config.save(config.Config())
        path = Path(config.config_path())
        assert "confine_writes" not in path.read_text()
        config.save(config.Config(agent=config.AgentConfig(confine_writes=False)))
        assert "confine_writes = false" in path.read_text()
        assert config.load().agent.confine_writes is False
        bad = config.Config()
        bad.agent.confine_writes = "no"  # type: ignore[assignment]
        assert any("confine_writes must be a boolean" in e for e in bad.validate())

    def test_x_turns_it_off_for_one_run(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli._parse(["-X", "agent.confine_writes=false", *URL, "fix it"])
        assert config.overrides() == {"agent.confine_writes": False}
        assert config.load().agent.confine_writes is False
        assert (
            config_cli.main(["keys", "confine", "-X", "agent.confine_writes=off"]) == 0
        )
        row = capsys.readouterr().out.splitlines()[1].split()
        assert row[:3] == ["agent.confine_writes", "false", "true"]


class TestCli:
    def _main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        argv: list[str],
    ) -> tuple[str, str, list[AgentOptions]]:
        built: list[AgentOptions] = []

        def handler(request: httpx.Request) -> httpx.Response:
            message = {"role": "assistant", "content": "Done"}
            return httpx.Response(200, json={"choices": [{"message": message}]})

        def build(*args: Any, options: AgentOptions, **kwargs: Any) -> Agent:
            built.append(options)
            return Agent(
                llm=OpenAICompatLLM(
                    base_url="http://llm.test/v1",
                    model="stub",
                    transport=httpx.MockTransport(handler),
                ),
                options=dataclasses.replace(options, memory=False),
            )

        (tmp_path / "work").mkdir()
        (tmp_path / "brief.md").write_text("Explain.\n", encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli, "_build_direct_mode_agent", build)
        monkeypatch.setattr(sys, "argv", ["melo", *argv])
        with contextlib.suppress(SystemExit):
            cli.main()
        captured = capsys.readouterr()
        return captured.out, captured.err, built

    @pytest.mark.parametrize(
        ("argv", "on"),
        [
            (["--brief", "brief.md", "-w", "work"], True),
            (["--brief", "brief.md", "-w", "work", "-X", "agent.confine_writes=0"], None),
            (["-X", "agent.confine_writes=false", "--brief", "brief.md", "-w", "work"], False),
            (["-w", "work", "-o", "json", "--", "say hello"], False),
        ],
    )  # fmt: skip
    def test_brief_runs_are_confined_unless_turned_off(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        argv: list[str],
        on: bool | None,
    ) -> None:
        if on is None:  # not a boolean: refused as the other [agent] keys are
            with pytest.raises(SystemExit) as exc:
                monkeypatch.setenv("HOME", str(tmp_path))
                cli._parse(argv)
            assert exc.value.code == 2
            return
        out, _err, built = self._main(tmp_path, monkeypatch, capsys, argv)
        assert built[0].confine_writes is on
        record = json.loads(out)
        run = record.get("run", record)
        if not on:
            assert "confinement" not in run
        elif sys.platform == "darwin":
            assert run["confinement"]["writes"] in ("confined", "not_confined")

    def test_a_brief_fails_closed_with_one_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sandbox, "sys", SimpleNamespace(platform="darwin"))
        monkeypatch.setattr(sandbox, "SANDBOX_EXEC", "/nonexistent/sandbox-exec")
        (tmp_path / "work").mkdir()
        (tmp_path / "brief.md").write_text("Explain.\n", encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        argv = ["melo", "--brief", "brief.md", "-w", "work", *URL]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        record = json.loads(captured.out)
        assert record["status"] == "startup_error" and record["run"] is None
        assert record["error"] == (
            "writes cannot be confined to the workspace: /nonexistent/sandbox-exec "
            "is missing; set agent.confine_writes to false to run without it "
            "(-X agent.confine_writes=false for one run)"
        )
        assert captured.err == f"melo: worker startup failed: {record['error']}\n"

    def test_a_bad_value_in_the_file_stops_a_brief(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            config,
            "load",
            lambda: config.Config(agent=config.AgentConfig(confine_writes="no")),  # type: ignore[arg-type]
        )
        out, _err, built = self._main(
            tmp_path, monkeypatch, capsys, ["--brief", "brief.md", "-w", "work"]
        )
        assert built == []
        record = json.loads(out)
        assert record["status"] == "startup_error"
        assert record["error"] == "[agent] confine_writes must be a boolean"


@pytest.mark.parametrize("on", [True, False])
async def test_a_saved_task_follows_the_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, on: bool
) -> None:
    from localmelo.support.gateway.chat_gate import ChatGate
    from localmelo.support.gateway.task_policy import TaskSpec
    from localmelo.support.gateway.workers import TaskWorkers
    from localmelo.support.sessions.hub import SessionHub
    from localmelo.support.sessions.store import SessionStore

    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off")
    hub = SessionHub(store)
    cfg = config.Config()
    cfg.gateway.workspace = str(workspace)
    cfg.agent.confine_writes = on
    workers = TaskWorkers(store, gate, lambda: cfg, hub)
    seen: list[AgentOptions] = []
    entered = asyncio.Event()

    async def build(**kwargs: Any) -> Any:
        seen.append(kwargs["options"])
        entered.set()
        raise RuntimeError("stop test worker")

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    try:
        await workers.tasks.open()
        await workers.tasks.create_task(
            TaskSpec.from_payload({"objective": "work"}),
            origin="user",
            workspace=str(workspace),
            _recovery_proof=True,
        )
        await workers.open()
        await asyncio.wait_for(entered.wait(), 3)
        assert seen[0].confine_writes is on
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()
