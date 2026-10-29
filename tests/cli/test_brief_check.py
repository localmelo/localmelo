"""The ``--brief`` check gate on the worker side: the callable that runs a check
as a ``shell_exec`` call, the ``tests`` record it feeds, and the CLI wiring."""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.executor import Executor
from localmelo.support.cli.worker_contract import WorkerObserver, run_brief_check
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.agent.test_reflection_gate import FakeLLM


async def _executor(
    root: Path | None, builtins: frozenset[str] | None, **options: Any
) -> Executor:
    agent = Agent(
        llm=FakeLLM([]),
        options=AgentOptions(
            memory=False,
            mode="direct",
            builtins=builtins,
            workspace_root=None if root is None else str(root),
            **options,
        ),
    )
    await agent._ensure_tools()
    return agent.executor


SHELL = frozenset({"shell_exec"})


class TestRunBriefCheck:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("rooted", [True, False])
    async def test_the_exit_status_decides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rooted: bool
    ) -> None:
        monkeypatch.chdir(tmp_path)
        executor = await _executor(tmp_path if rooted else None, SHELL)
        passed = await run_brief_check(executor, "echo fine")
        assert (passed.passed, passed.exit_code, passed.refused) == (True, 0, "")
        assert passed.output.strip() == "fine" and passed.duration_s >= 0
        failed = await run_brief_check(executor, "echo broken; exit 3")
        assert (failed.passed, failed.exit_code) == (False, 3)
        assert failed.output.strip() == "broken"  # no marker, no [exit code 3] line

    @pytest.mark.asyncio
    async def test_the_command_runs_in_the_workspace(self, tmp_path: Path) -> None:
        executor = await _executor(tmp_path, SHELL)
        result = await run_brief_check(executor, "pwd")
        assert Path(result.output.strip()).resolve() == tmp_path.resolve()

    @pytest.mark.asyncio
    async def test_the_output_is_the_last_lines_and_redacted(
        self, tmp_path: Path
    ) -> None:
        executor = await _executor(tmp_path, SHELL)
        result = await run_brief_check(
            executor, "echo API_KEY=hunter2; seq 1 100; exit 1"
        )
        lines = result.output.splitlines()
        assert (lines[0], lines[-1], len(lines)) == ("41", "100", 60)
        secret = await run_brief_check(executor, "echo API_KEY=hunter2; exit 1")
        assert "hunter2" not in secret.output and "[redacted]" in secret.output
        named = await run_brief_check(executor, "echo ok; exit 1 # TOKEN=abc")
        assert "abc" not in named.command

    @pytest.mark.asyncio
    async def test_a_policy_refusal_is_not_run(self, tmp_path: Path) -> None:
        # The default direct set offers no shell_exec: the call is refused.
        executor = await _executor(tmp_path, None)
        marker = tmp_path / "ran"
        result = await run_brief_check(executor, f"touch {marker}")
        assert not marker.exists()
        assert result.passed is False and result.exit_code is None
        assert "shell_exec is not available" in result.refused

    @pytest.mark.asyncio
    async def test_a_timeout_is_a_failure_that_is_not_refused(
        self, tmp_path: Path
    ) -> None:
        executor = await _executor(tmp_path, SHELL, tool_timeout_s=0.5)
        result = await run_brief_check(executor, "sleep 10")
        assert result.passed is False and result.exit_code is None
        assert result.refused == "" and "timed out" in result.output
        assert result.duration_s < 5


def _event(type_: str, data: dict[str, Any]) -> AgentEvent:
    return AgentEvent(type_, "run", 1, 0.0, data)


def _notice(round_: int, command: str, exit_code: int | None, **detail: Any) -> Any:
    return _event(
        "notice",
        {
            "kind": "brief_check",
            "message": "m",
            "step": 1,
            "attempt": 1,
            "detail": {
                "round": round_,
                "command": command,
                "exit_code": exit_code,
                "duration_s": 0.5,
                "refused": "",
                **detail,
            },
        },
    )


class TestTestsRecord:
    def _observer(self, tmp_path: Path) -> WorkerObserver:
        return WorkerObserver(None, str(tmp_path), stream=False)  # type: ignore[arg-type]

    def test_no_gate_leaves_the_record_as_before(self, tmp_path: Path) -> None:
        assert self._observer(tmp_path).tests() == {
            "state": "not_recorded",
            "items": [],
        }

    def test_the_last_round_decides(self, tmp_path: Path) -> None:
        observer = self._observer(tmp_path)
        observer(_notice(1, "pytest -q", 1))
        assert observer.tests()["state"] == "failed"
        observer(_notice(2, "pytest -q", 0))
        assert observer.tests() == {
            "state": "passed",
            "items": [
                {
                    "command": "pytest -q",
                    "command_truncated": False,
                    "exit_code": 0,
                    "duration_s": 0.5,
                    "source": "brief_check",
                }
            ],
        }

    def test_a_failure_or_a_refusal_in_the_last_round_is_failed(
        self, tmp_path: Path
    ) -> None:
        observer = self._observer(tmp_path)
        observer(_notice(1, "make a", 0))
        observer(_notice(1, "make b", 2))
        failed = observer.tests()
        assert failed["state"] == "failed"
        assert [item["exit_code"] for item in failed["items"]] == [0, 2]
        refused = self._observer(tmp_path)
        refused(_notice(1, "touch x", None, refused="Blocked: no shell"))
        [item] = refused.tests()["items"]
        assert refused.tests()["state"] == "failed"
        assert (item["exit_code"], item["refused"]) == (None, "Blocked: no shell")

    def test_a_long_command_is_capped(self, tmp_path: Path) -> None:
        observer = self._observer(tmp_path)
        observer(_notice(1, "x" * 2500, 0))
        [item] = observer.tests()["items"]
        assert len(item["command"]) == 2000 and item["command_truncated"] is True


class TestCli:
    def _main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        brief: str,
        replies: list[str],
        on_request: Callable[[int], None] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[AgentOptions]]:
        bodies: list[dict[str, Any]] = []
        built: list[AgentOptions] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            if on_request is not None:
                on_request(len(bodies))
            reply = replies[min(len(bodies), len(replies)) - 1]
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": reply}}]
                },
            )

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

        workspace = tmp_path / "work"
        workspace.mkdir()
        file = tmp_path / "brief.md"
        file.write_text(brief, encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli, "_build_direct_mode_agent", build)
        monkeypatch.setattr(
            sys,
            "argv",
            ["melo", "--brief", str(file), "--workspace", str(workspace)]
            + ["--builtins", "all"],
        )
        try:
            cli.main()
        except SystemExit as exc:
            assert exc.code in (0, None)
        return json.loads(capsys.readouterr().out), bodies, built

    def test_a_brief_run_hands_the_agent_the_gate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _record, _bodies, built = self._main(
            tmp_path, monkeypatch, capsys, "Explain.\n", ["Done"]
        )
        assert built[0].brief_check is run_brief_check

    def test_a_failing_check_is_told_fixed_and_recorded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The second model request fixes the workspace, as a tool call would.
        marker = tmp_path / "work" / "fixed.txt"

        def fix(request: int) -> None:
            if request == 2:
                marker.write_text("fixed\n")

        record, bodies, _built = self._main(
            tmp_path,
            monkeypatch,
            capsys,
            "Make fixed.txt.\nCheck: test -f fixed.txt\n",
            ["Done", "Fixed"],
            on_request=fix,
        )
        assert record["status"] == "completed" and record["exit_code"] == 0
        assert record["tests"]["state"] == "passed"
        [item] = record["tests"]["items"]
        assert (item["command"], item["exit_code"]) == ("test -f fixed.txt", 0)
        assert item["source"] == "brief_check"
        assert len(bodies) == 2
        assert "Exit code: 1" in bodies[1]["messages"][-1]["content"]

    def test_a_check_that_keeps_failing_is_recorded_failed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        record, bodies, _built = self._main(
            tmp_path, monkeypatch, capsys, "Do it.\nCheck: exit 4\n", ["Done"]
        )
        assert (record["status"], record["exit_code"]) == ("completed", 0)
        assert record["tests"]["state"] == "failed"
        assert [i["exit_code"] for i in record["tests"]["items"]] == [4]
        assert len(bodies) == 3  # the answer, and two retries

    def test_a_brief_without_a_check_line_is_not_recorded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        record, bodies, _built = self._main(
            tmp_path, monkeypatch, capsys, "Explain.\n", ["Done"]
        )
        assert record["tests"] == {"state": "not_recorded", "items": []}
        assert len(bodies) == 1
