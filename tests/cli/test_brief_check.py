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
from localmelo.melo.checker.checker import Checker
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.executor import (
    ErrorCategory,
    ExecutionRequest,
    ExecutionStatus,
    Executor,
)
from localmelo.melo.schema import CheckResult, ToolCall, ToolDef
from localmelo.support.cli.worker_contract import WorkerObserver, run_brief_check
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.agent.test_reflection_gate import FakeLLM


async def _agent(
    root: Path | None,
    builtins: frozenset[str] | None,
    *,
    gate: bool = True,
    **options: Any,
) -> Agent:
    agent = Agent(
        llm=FakeLLM([]),
        options=AgentOptions(
            memory=False,
            mode="direct",
            builtins=builtins,
            workspace_root=None if root is None else str(root),
            brief_check=run_brief_check if gate else None,
            **options,
        ),
    )
    await agent._ensure_tools()
    return agent


async def _executor(
    root: Path | None,
    builtins: frozenset[str] | None,
    *,
    gate: bool = True,
    **options: Any,
) -> Executor:
    return (await _agent(root, builtins, gate=gate, **options)).executor


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
    @pytest.mark.parametrize("builtins", [None, frozenset(), SHELL])
    async def test_the_check_runs_in_every_builtins_setting(
        self, tmp_path: Path, builtins: frozenset[str] | None
    ) -> None:
        # The default direct set offers no shell_exec, and --builtins none
        # nothing at all: the harness's own call still runs.
        executor = await _executor(tmp_path, builtins)
        passed = await run_brief_check(executor, "echo fine")
        assert (passed.passed, passed.exit_code, passed.refused) == (True, 0, "")
        failed = await run_brief_check(executor, "echo broken; exit 3")
        assert (failed.passed, failed.exit_code, failed.refused) == (False, 3, "")

    @pytest.mark.asyncio
    async def test_the_model_still_cannot_call_shell_exec(self, tmp_path: Path) -> None:
        agent = await _agent(tmp_path, None)
        marker = tmp_path / "ran"
        assert "shell_exec" not in [t.name for t in agent.hippo.list_tools()]
        outcome = await agent.executor.execute_structured(
            ExecutionRequest(
                tool_name="shell_exec", arguments={"command": f"touch {marker}"}
            )
        )
        assert outcome.status is ExecutionStatus.ERROR
        assert outcome.error_category is ErrorCategory.TOOL_NOT_FOUND
        assert "shell_exec is not available" in outcome.error
        assert not marker.exists()

    @pytest.mark.asyncio
    async def test_a_run_without_a_gate_has_no_check_to_run(
        self, tmp_path: Path
    ) -> None:
        executor = await _executor(tmp_path, None, gate=False)
        marker = tmp_path / "ran"
        result = await run_brief_check(executor, f"touch {marker}")
        assert not marker.exists()
        assert result.passed is False and result.exit_code is None
        assert "shell_exec is not available" in result.refused

    @pytest.mark.asyncio
    async def test_the_checker_can_still_block_a_check(self, tmp_path: Path) -> None:
        # The check goes through pre_execute like any call.
        agent = await _agent(tmp_path, None)
        marker = tmp_path / "ran"

        async def block(call: ToolCall, tool_def: ToolDef | None) -> CheckResult:
            return CheckResult(allowed=False, reason="not this one")

        agent.checker.pre_execute = block  # type: ignore[assignment,method-assign]
        result = await run_brief_check(agent.executor, f"touch {marker}")
        assert not marker.exists()
        assert result.passed is False and result.exit_code is None
        assert result.refused == "Blocked: not this one"

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

    def test_the_output_tail_is_kept_redacted_and_bounded(self, tmp_path: Path) -> None:
        observer = self._observer(tmp_path)
        observer(_notice(1, "pytest -q", 1, output_tail="API_KEY=hunter2\nlast"))
        [item] = observer.tests()["items"]
        assert "hunter2" not in item["output_tail"]
        assert item["output_tail"].endswith("\nlast")
        observer(_notice(2, "pytest -q", 1, output_tail="y" * 9000))
        [item] = observer.tests()["items"]
        assert len(item["output_tail"]) == 4000

    def test_an_old_notice_without_a_tail_gives_an_item_without_one(
        self, tmp_path: Path
    ) -> None:
        observer = self._observer(tmp_path)
        observer(_notice(1, "pytest -q", 1))
        [item] = observer.tests()["items"]
        assert "output_tail" not in item


class TestCli:
    def _main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        brief: str,
        replies: list[Any],
        on_request: Callable[[int], None] | None = None,
        builtins: tuple[str, ...] = ("--builtins", "all"),
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[AgentOptions]]:
        bodies: list[dict[str, Any]] = []
        built: list[AgentOptions] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            if on_request is not None:
                on_request(len(bodies))
            reply = replies[min(len(bodies), len(replies)) - 1]
            message = (
                reply
                if isinstance(reply, dict)
                else {"role": "assistant", "content": reply}
            )
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
            + list(builtins),
        )
        self.exit_code = 0
        try:
            cli.main()
        except SystemExit as exc:
            self.exit_code = int(exc.code or 0)
        captured = capsys.readouterr()
        self.stderr = captured.err
        return json.loads(captured.out), bodies, built

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

    def test_a_failing_check_records_the_tail_of_its_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        record, _bodies, _built = self._main(
            tmp_path,
            monkeypatch,
            capsys,
            "Do it.\nCheck: seq 1 30; echo API_KEY=hunter2; exit 4\n",
            ["Done"],
        )
        [item] = record["tests"]["items"]
        lines = item["output_tail"].splitlines()
        assert (lines[0], len(lines)) == ("12", 20)
        assert "hunter2" not in item["output_tail"]

    def test_a_check_that_keeps_failing_is_recorded_failed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        record, bodies, _built = self._main(
            tmp_path, monkeypatch, capsys, "Do it.\nCheck: exit 4\n", ["Done"]
        )
        assert (record["status"], record["exit_code"]) == ("completed", 4)
        assert self.exit_code == 4
        assert record["tests"]["state"] == "failed"
        assert [i["exit_code"] for i in record["tests"]["items"]] == [4]
        assert len(bodies) == 3  # the answer, and two retries
        assert self.stderr.splitlines() == [
            "melo: the brief's check failed: exit 4 (exit 4)"
        ]

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

    @pytest.mark.parametrize("builtins", [(), ("--builtins", "none")])
    def test_the_check_runs_by_default_and_a_pass_exits_0(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        builtins: tuple[str, ...],
    ) -> None:
        record, bodies, built = self._main(
            tmp_path,
            monkeypatch,
            capsys,
            "Do it.\nCheck: echo fine\nCheck: test -d .\n",
            ["Done"],
            builtins=builtins,
        )
        assert built[0].mode == "direct" and "shell_exec" not in (
            built[0].builtins or ()
        )
        assert (record["status"], record["exit_code"], self.exit_code) == (
            "completed",
            0,
            0,
        )
        assert record["tests"]["state"] == "passed"
        assert [i["exit_code"] for i in record["tests"]["items"]] == [0, 0]
        assert not any("shell_exec" in str(b.get("tools")) for b in bodies)
        assert self.stderr == ""

    def test_a_failed_check_exits_4_by_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        record, _bodies, _built = self._main(
            tmp_path,
            monkeypatch,
            capsys,
            "Do it.\nCheck: echo no; exit 7\n",
            ["Done"],
            builtins=(),
        )
        assert (record["status"], record["exit_code"], self.exit_code) == (
            "completed",
            4,
            4,
        )
        assert record["tests"]["state"] == "failed"
        assert self.stderr.splitlines() == [
            "melo: the brief's check failed: echo no; exit 7 (exit 7)"
        ]

    def test_a_refused_check_exits_4(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        async def block(*args: Any, **kwargs: Any) -> CheckResult:
            return CheckResult(allowed=False, reason="not this one")

        monkeypatch.setattr(Checker, "pre_execute", block)
        record, _bodies, _built = self._main(
            tmp_path, monkeypatch, capsys, "Do it.\nCheck: touch ran\n", ["Done"]
        )
        assert (record["status"], record["exit_code"], self.exit_code) == (
            "completed",
            4,
            4,
        )
        [item] = record["tests"]["items"]
        assert item["refused"] == "Blocked: not this one"
        assert not (tmp_path / "work" / "ran").exists()
        assert self.stderr.splitlines() == [
            "melo: the brief's check failed: touch ran (not run)"
        ]

    def test_the_model_still_cannot_call_shell_exec_by_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "shell_exec",
                        "arguments": json.dumps({"command": "touch by_model"}),
                    },
                }
            ],
        }
        record, bodies, _built = self._main(
            tmp_path,
            monkeypatch,
            capsys,
            "Do it.\nCheck: touch by_check\n",
            [call, "Done"],
            builtins=(),
        )
        assert record["tests"]["state"] == "passed" and self.exit_code == 0
        assert (tmp_path / "work" / "by_check").exists()
        assert not (tmp_path / "work" / "by_model").exists()
        tool_reply = next(m for m in bodies[1]["messages"] if m["role"] == "tool")
        assert "shell_exec is not available" in tool_reply["content"]

    def test_a_brief_without_a_check_never_exits_4(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        record, _bodies, _built = self._main(
            tmp_path, monkeypatch, capsys, "Explain.\n", ["Done"], builtins=()
        )
        assert (record["exit_code"], self.exit_code) == (0, 0)
        assert self.stderr == ""
