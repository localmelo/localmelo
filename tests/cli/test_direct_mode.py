"""Direct mode of the CLI, in process: flags, exit codes, output and signals.

The agent is a mock, so nothing here talks to a model server or starts
a process. tests/cli/test_direct_mode_process.py runs the real CLI.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.metadata
import io
import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.executor import builtin_names
from localmelo.melo.schema import Message, RunResult, ToolDef
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.mcp import McpToolProvider

# Every status but completed and crashed, which have exit codes of their own.
NOT_COMPLETED = [
    "failed",
    "gate_stopped",
    "budget_exhausted",
    "stuck",
    "provider_error",
    "setup_error",
    "timeout",
]

PACKAGE_DIR = Path(cli.__file__).resolve().parent


# ── Helpers ─────────────────────────────────────────────────────────────────


def _outcome(status: str, *, result: str = "", error: str = "") -> RunResult:
    return RunResult(task_id="0" * 32, status=status, result=result, error=error)


def _agent(*outcomes: RunResult) -> mock.MagicMock:
    """A mock Agent whose run_task() returns *outcomes* in turn."""
    agent = mock.MagicMock()
    agent.run_task = mock.AsyncMock(side_effect=list(outcomes))
    agent.close = mock.AsyncMock()
    agent.options = AgentOptions(mode="direct")
    return agent


@dataclasses.dataclass
class Ran:
    """What one call of main() did."""

    code: int
    build: mock.MagicMock  # the _build_direct_mode_agent stand-in
    agent: mock.MagicMock

    @property
    def options(self) -> AgentOptions:
        options = self.build.call_args.kwargs["options"]
        assert isinstance(options, AgentOptions)
        return options

    @property
    def tool_providers(self) -> list[Any]:
        return list(self.build.call_args.kwargs["tool_providers"])


def _main(*argv: str, outcome: RunResult | None = None) -> Ran:
    """Run main() with *argv*; the agent's run ends with *outcome*."""
    agent = _agent(outcome or _outcome("completed", result="42"))
    build = mock.MagicMock(return_value=agent)
    with (
        mock.patch.object(cli, "_register_backends"),
        mock.patch.object(cli, "_build_direct_mode_agent", build),
        mock.patch.object(cli, "_localmelo_version", return_value="v-test"),
        mock.patch("sys.argv", ["melo", *argv]),
    ):
        try:
            cli.main()
        except SystemExit as exc:
            assert isinstance(exc.code, int)
            return Ran(exc.code, build, agent)
    return Ran(0, build, agent)


def _mcp_config(tmp_path: Path) -> Path:
    """A valid MCP config; nothing in these tests starts its server."""
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"mcpServers": {"fake": {"command": "fake-mcp-server"}}}),
        encoding="utf-8",
    )
    return path


# ── Flags ───────────────────────────────────────────────────────────────────


class TestFlags:
    def test_no_flags_keep_the_defaults(self) -> None:
        ran = _main("what", "is", "6*7")
        assert ran.code == 0
        assert ran.options == AgentOptions(mode="direct")
        assert ran.tool_providers == []
        ran.agent.run_task.assert_awaited_once_with("what is 6*7")

    def test_workspace_sets_the_root(self, tmp_path: Path) -> None:
        ran = _main("--workspace", str(tmp_path), "q")
        assert ran.code == 0
        assert ran.options == AgentOptions(
            mode="direct", workspace_root=os.path.realpath(tmp_path)
        )

    def test_workspace_is_resolved_like_the_ui(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        root = _main("--workspace", ".", "q").options.workspace_root
        assert root == os.path.realpath(tmp_path)

    def test_no_workspace_keeps_the_current_directory(self) -> None:
        # Q10: without the flag a direct run stays where it always ran.
        assert _main("q").options.workspace_root is None

    def test_each_flag_sets_its_option(self) -> None:
        ran = _main(
            "--max-steps",
            "7",
            "--attempt-policy",
            "none",
            "--reflection-gate",
            "utility",
            "--tool-exposure",
            "all",
            "--builtins",
            "file_read, shell_exec",
            "--memory",
            "off",
            "--memory-dir",
            "~/melo-memory",
            "--timeout",
            "12.5",
            "q",
        )
        assert ran.code == 0
        assert ran.options == AgentOptions(
            mode="direct",
            max_steps=7,
            attempt_policy="none",
            reflection_gate="utility",
            tool_exposure="all",
            builtins=frozenset({"file_read", "shell_exec"}),
            memory=False,
            memory_dir="~/melo-memory",  # the Agent expands it
            timeout_s=12.5,
        )

    @pytest.mark.parametrize(
        ("value", "builtins"),
        [("all", frozenset(builtin_names())), ("none", frozenset())],
    )
    def test_builtins_all_and_none(
        self, value: str, builtins: frozenset[str] | None
    ) -> None:
        assert _main("--builtins", value, "q").options.builtins == builtins

    def test_memory_on(self) -> None:
        assert _main("--memory", "on", "q").options.memory is True

    @pytest.mark.parametrize("gate", ["flags", "utility"])
    def test_reflection_gate(self, gate: str) -> None:
        assert _main("--reflection-gate", gate, "q").options.reflection_gate == gate

    def test_reflection_gate_is_accepted_with_attempt_policy_none(self) -> None:
        # It has no effect there: such a run makes no reflection.
        ran = _main("--attempt-policy", "none", "--reflection-gate", "utility", "q")
        assert ran.code == 0
        assert ran.options == AgentOptions(
            mode="direct", attempt_policy="none", reflection_gate="utility"
        )

    def test_reflection_gate_help(self) -> None:
        [action] = [
            action
            for action in cli._build_parser()._actions
            if action.dest == "reflection_gate"
        ]
        assert action.option_strings == ["--reflection-gate"]
        assert action.choices == ["flags", "utility"]
        assert action.default is None  # AgentOptions decides: "flags"
        assert action.help == (
            "After a reflection, start the next attempt when it chooses continue "
            "and its three checks hold (flags, default), or also require its "
            "utility score to reach 0.1 (utility, the behavior before this option)"
        )

    @pytest.mark.parametrize(("value", "seconds"), [("600", 600.0), ("0.5", 0.5)])
    def test_request_timeout_goes_to_the_provider_not_the_options(
        self, value: str, seconds: float
    ) -> None:
        ran = _main("--request-timeout", value, "q")
        assert ran.code == 0
        assert ran.options == AgentOptions(mode="direct")  # not a run option
        [args] = ran.build.call_args.args
        assert args.request_timeout == seconds

    def test_no_request_timeout_by_default(self) -> None:
        [args] = _main("q").build.call_args.args
        assert args.request_timeout is None

    def test_mcp_config_gives_one_provider_that_is_not_started(
        self, tmp_path: Path
    ) -> None:
        ran = _main("--mcp-config", str(_mcp_config(tmp_path)), "q")
        assert ran.code == 0
        assert ran.options == AgentOptions(mode="direct")
        [provider] = ran.tool_providers
        assert isinstance(provider, McpToolProvider)
        assert provider.clients == []  # the agent starts it in its first run

    def test_mcp_only_offers_only_the_mcp_tools(self, tmp_path: Path) -> None:
        ran = _main("--mcp-only", "--mcp-config", str(_mcp_config(tmp_path)), "q")
        assert ran.code == 0
        assert ran.options == AgentOptions(
            mode="direct", builtins=frozenset(), tool_exposure="all"
        )
        [provider] = ran.tool_providers
        assert isinstance(provider, McpToolProvider)


class TestBuildDirectModeAgent:
    """The options, tool providers and embedding reach the Agent."""

    def _build(self, *argv: str) -> dict[str, Any]:
        parser = cli._build_parser()
        args = parser.parse_args([*argv, "q"])
        options = AgentOptions(max_steps=3)
        providers = [mock.MagicMock()]
        with (
            mock.patch.object(cli, "Agent") as agent_cls,
            mock.patch(
                "localmelo.support.providers.embedding.openai_compat."
                "OpenAICompatEmbedding"
            ) as embedding_cls,
        ):
            cli._build_direct_mode_agent(
                args, options=options, tool_providers=providers
            )
        kwargs = dict(agent_cls.call_args.kwargs)
        assert kwargs.pop("options") is options
        assert kwargs.pop("tool_providers") is providers
        kwargs["embedding_cls"] = embedding_cls
        return kwargs

    def test_no_embedding_without_the_embedding_flags(self) -> None:
        kwargs = self._build("--base-url", "http://chat:8000/v1")
        assert kwargs["embedding"] is None
        kwargs["embedding_cls"].assert_not_called()

    def test_embedding_base_url_and_model(self) -> None:
        kwargs = self._build(
            "--base-url",
            "http://chat:8000/v1",
            "--embedding-base-url",
            "http://embed:9000/v1",
            "--embedding-model",
            "nomic-embed",
        )
        kwargs["embedding_cls"].assert_called_once_with(
            base_url="http://embed:9000/v1", model="nomic-embed"
        )
        assert kwargs["embedding"] is kwargs["embedding_cls"].return_value

    def test_embedding_model_defaults_to_default(self) -> None:
        kwargs = self._build(
            "--base-url", "http://chat:8000/v1", "--embedding-base-url", "http://e/v1"
        )
        kwargs["embedding_cls"].assert_called_once_with(
            base_url="http://e/v1", model="default"
        )

    def test_the_configured_agent_gets_them_too(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400/v1",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400/v1",
                embedding_model="nomic-embed",
            ),
        )
        with mock.patch("localmelo.support.config.load", return_value=cfg):
            kwargs = self._build()
        assert kwargs == {"config": cfg, "embedding_cls": kwargs["embedding_cls"]}


# ── Usage errors: exit code 2 before any run ────────────────────────────────


class TestUsageErrors:
    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["--mcp-only", "q"], "--mcp-only requires --mcp-config"),
            (
                ["--mcp-only", "--mcp-config", "m.json", "--builtins", "all", "q"],
                "--mcp-only sets --builtins; do not give both",
            ),
            (
                ["--mcp-only", "--mcp-config", "m.json", "--tool-exposure", "all", "q"],
                "--mcp-only sets --tool-exposure; do not give both",
            ),
            (["--max-steps", "0", "q"], "argument --max-steps: must be at least 1"),
            (["--max-steps", "many", "q"], "argument --max-steps: not an integer"),
            (
                ["--timeout", "0", "q"],
                "argument --timeout: must be a finite number of seconds > 0, got 0",
            ),
            (["--timeout", "-5", "q"], "argument --timeout: must be a finite"),
            (["--timeout", "nan", "q"], "argument --timeout: must be a finite"),
            (["--timeout", "inf", "q"], "argument --timeout: must be a finite"),
            (["--timeout", "soon", "q"], "argument --timeout: not a number: 'soon'"),
            (
                ["--request-timeout", "0", "q"],
                "argument --request-timeout: must be a finite number of seconds > 0, "
                "got 0",
            ),
            (
                ["--request-timeout", "-1", "q"],
                "argument --request-timeout: must be a finite number of seconds > 0, "
                "got -1",
            ),
            (
                ["--request-timeout", "nan", "q"],
                "argument --request-timeout: must be a finite number of seconds > 0, "
                "got nan",
            ),
            (
                ["--request-timeout", "inf", "q"],
                "argument --request-timeout: must be a finite number of seconds > 0, "
                "got inf",
            ),
            (
                ["--request-timeout", "abc", "q"],
                "argument --request-timeout: not a number: 'abc'",
            ),
            (["--memory-dir", "", "q"], "argument --memory-dir: must not be empty"),
            (
                ["--attempt-policy", "retry", "q"],
                "argument --attempt-policy: invalid choice: 'retry'",
            ),
            (
                ["--reflection-gate", "legacy", "q"],
                "argument --reflection-gate: invalid choice: 'legacy'",
            ),
            (
                ["--tool-exposure", "some", "q"],
                "argument --tool-exposure: invalid choice: 'some'",
            ),
            (["--memory", "maybe", "q"], "argument --memory: invalid choice: 'maybe'"),
            (
                ["--output-format", "yaml", "q"],
                "argument --output-format: invalid choice: 'yaml'",
            ),
            (
                ["--builtins", "file_read,", "q"],
                "argument --builtins: expected all, none or built-in tool names "
                "separated by commas, got 'file_read,'",
            ),
            (
                ["--embedding-model", "m", "--base-url", "http://x/v1", "q"],
                "--embedding-model requires --embedding-base-url",
            ),
            (
                ["--embedding-base-url", "http://e/v1", "q"],
                "--embedding-base-url requires --base-url",
            ),
            (
                ["--serve", "--max-steps", "3"],
                "only for direct mode (a query without --serve): --max-steps",
            ),
            (
                # An empty query runs no task, so there is no run to report.
                ["--output-format", "json", ""],
                "only for direct mode (a query without --serve): --output-format",
            ),
            (
                ["--serve", "--timeout", "5", "--output-format", "json", "q"],
                "only for direct mode (a query without --serve): --timeout, "
                "--output-format",
            ),
            (
                ["--serve", "--request-timeout", "600"],
                "only for direct mode (a query without --serve): --request-timeout",
            ),
            (
                ["--serve", "--request-timeout", "600", "q"],
                "only for direct mode (a query without --serve): --request-timeout",
            ),
            (
                ["--serve", "--reflection-gate", "utility"],
                "only for direct mode (a query without --serve): --reflection-gate",
            ),
            (
                ["--serve", "--reflection-gate", "flags", "q"],
                "only for direct mode (a query without --serve): --reflection-gate",
            ),
            (
                ["--serve", "--attempt-policy", "none", "--reflection-gate", "flags"],
                "only for direct mode (a query without --serve): --attempt-policy, "
                "--reflection-gate",
            ),
            (
                ["--serve", "--workspace", "/tmp"],
                "only for direct mode (a query without --serve): --workspace",
            ),
            (
                # `melo ""` runs no task, so the folder would apply to nothing.
                ["--workspace", "/tmp", ""],
                "only for direct mode (a query without --serve): --workspace",
            ),
        ],
    )
    def test_rejected_before_the_run(
        self, argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ran = _main(*argv)
        assert ran.code == 2
        ran.build.assert_not_called()
        out, err = capsys.readouterr()
        assert out == ""
        assert f"melo: error: {message}" in err

    @pytest.mark.parametrize("kind", ["missing", "file"])
    def test_a_workspace_that_is_not_a_directory_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
    ) -> None:
        target = tmp_path / kind
        if kind == "file":
            target.write_text("", encoding="utf-8")
        ran = _main("--workspace", str(target), "q")
        assert ran.code == 2
        ran.build.assert_not_called()
        assert capsys.readouterr().err == (
            f"melo: --workspace: not a directory: {target}\n"
        )

    def test_an_unknown_builtin_lists_the_known_ones(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ran = _main("--builtins", "file_read,teleport", "q")
        assert ran.code == 2
        known = ", ".join(sorted(builtin_names() | {"recall"}))
        assert (
            f"unknown built-in tools teleport; known: {known}"
            in capsys.readouterr().err
        )

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ("{not json", "is not valid JSON"),
            ('{"mcpServers": []}', "expected an object under 'mcpServers'"),
            (None, "Cannot read MCP config"),
            # Not UTF-8: a byte that starts no character, and UTF-16.
            (b'{"mcpServers": {"x": {"command": "\xff"}}}', "Cannot read MCP config"),
            ('{"mcpServers": {}}'.encode("utf-16"), "Cannot read MCP config"),
            # Nested deeper than the JSON decoder can follow.
            ("[" * 100_000, "is not valid JSON"),
            # An integer too long for Python to convert.
            (
                '{"mcpServers": {"x": {"command": "c", "timeout": '
                + "9" * 5_000
                + "}}}",
                "is not valid JSON",
            ),
        ],
        ids=[
            "not-json",
            "not-a-config",
            "missing",
            "not-utf-8",
            "utf-16",
            "too-deep",
            "huge-integer",
        ],
    )
    def test_a_bad_mcp_config_is_reported_before_the_run(
        self,
        tmp_path: Path,
        content: str | bytes | None,
        message: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = tmp_path / "mcp.json"
        if isinstance(content, bytes):
            path.write_bytes(content)
        elif content is not None:
            path.write_text(content, encoding="utf-8")
        ran = _main("--mcp-config", str(path), "q")
        assert ran.code == 2
        ran.build.assert_not_called()
        out, err = capsys.readouterr()
        assert out == ""
        assert err.startswith("melo: error: ")
        assert err.count("\n") == 1
        assert message in err
        assert str(path) in err

    @pytest.mark.parametrize(
        ("below_a_file", "reason"),
        [(False, "not a directory"), (True, "Not a directory")],
        ids=["a-file", "below-a-file"],
    )
    def test_a_memory_dir_that_cannot_be_made_is_reported_before_the_run(
        self,
        tmp_path: Path,
        below_a_file: bool,
        reason: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        a_file = tmp_path / "file"
        a_file.write_text("", encoding="utf-8")
        memory_dir = a_file / "memory" if below_a_file else a_file
        self._assert_memory_dir_refused(memory_dir, reason, capsys)

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0,
        reason="POSIX permissions, which root bypasses",
    )
    def test_a_memory_dir_without_permission_is_reported_before_the_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o555)
        try:
            self._assert_memory_dir_refused(
                locked / "memory", "Permission denied", capsys
            )
        finally:
            locked.chmod(0o755)

    def _assert_memory_dir_refused(
        self, memory_dir: Path, reason: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() refuses --memory-dir *memory_dir* for *reason* (exit code 2)."""
        argv = ["--base-url", "http://127.0.0.1:9/v1", "--memory-dir", str(memory_dir)]
        with (
            mock.patch.object(cli, "_register_backends"),
            mock.patch.object(cli, "Agent") as agent_cls,
            mock.patch("sys.argv", ["melo", *argv, "q"]),
            pytest.raises(SystemExit) as exited,
        ):
            cli.main()
        assert exited.value.code == 2
        agent_cls.assert_not_called()
        assert capsys.readouterr() == (
            "",
            f"melo: error: argument --memory-dir: cannot use {memory_dir}: "
            f"{reason}\n",
        )


# ── Exit codes ──────────────────────────────────────────────────────────────


class TestExitCodes:
    def test_completed_exits_0(self) -> None:
        assert _main("q", outcome=_outcome("completed", result="42")).code == 0

    @pytest.mark.parametrize("status", NOT_COMPLETED)
    def test_any_other_status_exits_2(self, status: str) -> None:
        assert _main("q", outcome=_outcome(status, error="why")).code == 2

    def test_crashed_exits_1(self) -> None:
        crashed = _outcome("crashed", error="RuntimeError: boom")
        assert _main("q", outcome=crashed).code == 1


# ── Output ──────────────────────────────────────────────────────────────────


class TestTextOutput:
    def test_a_completed_run_prints_only_its_result(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _main("q", outcome=_outcome("completed", result="42"))
        assert capsys.readouterr() == ("42\n", "")

    def test_a_run_that_did_not_complete_adds_the_status_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        outcome = _outcome(
            "budget_exhausted", result="best effort", error="Max steps reached"
        )
        _main("q", outcome=outcome)
        out, err = capsys.readouterr()
        assert out == "best effort\n"
        assert err == (
            "melo: run ended with status budget_exhausted: Max steps reached\n"
        )

    def test_an_empty_result_prints_nothing_on_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _main("q", outcome=_outcome("provider_error", error="HTTP 500"))
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "melo: run ended with status provider_error: HTTP 500\n"

    def test_the_status_line_is_one_line_of_at_most_500_error_characters(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        page = "<html>\n<head>\t<title>502 Bad Gateway</title>\r\n</head>\n"
        error = f"POST /chat/completions failed with HTTP 502: {page}" + "x" * 600
        _main("q", outcome=_outcome("provider_error", error=error))
        err = capsys.readouterr().err
        assert err.count("\n") == 1 and err.endswith("\n")
        prefix = "melo: run ended with status provider_error: "
        assert err.startswith(
            prefix + "POST /chat/completions failed with HTTP 502: <html> <head> "
            "<title>502 Bad Gateway</title> </head> xxx"
        )
        reason = err[len(prefix) : -1]
        assert len(reason) == cli.STATUS_LINE_ERROR_CHARS == 500
        assert reason.endswith("x...")

    def test_a_blank_error_still_gives_a_reason(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _main("q", outcome=_outcome("failed", error=" \n\t "))
        assert capsys.readouterr().err == (
            "melo: run ended with status failed: no error message\n"
        )


class TestJsonOutput:
    def _record(self, out: str) -> dict[str, Any]:
        assert out.count("\n") == 1 and out.endswith("\n"), out
        record = json.loads(out)
        assert isinstance(record, dict)
        return record

    def test_exactly_one_line_with_the_record_and_the_version(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        outcome = RunResult(
            task_id="a" * 32,
            status="budget_exhausted",
            result="line one\nline two",
            error="Max steps reached",
            steps=3,
            attempts=1,
            n_llm=3,
            n_main=3,
            n_tool=2,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            duration_s=1.5,
            config={"max_steps": 3, "tools": ["shell_exec"]},
        )
        ran = _main("--output-format", "json", "q", outcome=outcome)
        assert ran.code == 2
        out, err = capsys.readouterr()
        record = self._record(out)
        # A legacy result omits memory_stats from to_dict(), so the line has
        # the fields the dataclass had before R2, plus the version.
        assert list(record) == [
            field.name
            for field in dataclasses.fields(RunResult)
            if field.name != "memory_stats"
        ] + ["localmelo_version"]
        assert record.pop("localmelo_version") == "v-test"
        assert record == outcome.to_dict()
        assert err == ""  # no status line in this mode

    @pytest.mark.parametrize(("status", "code"), [("completed", 0), ("crashed", 1)])
    def test_the_exit_code_follows_the_status(
        self, status: str, code: int, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ran = _main("--output-format", "json", "q", outcome=_outcome(status))
        assert ran.code == code
        assert self._record(capsys.readouterr().out)["status"] == status

    @pytest.mark.parametrize(
        ("argv", "gate"), [([], "flags"), (["--reflection-gate", "utility"], "utility")]
    )
    def test_the_config_names_the_reflection_gate(
        self, argv: list[str], gate: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A real Agent, with the options the flags built and a model that
        # answers at once.
        def build(args: Any, *, options: AgentOptions, tool_providers: Any) -> Agent:
            return Agent(llm=_AnswerLLM("42"), options=options)

        with (
            mock.patch.object(cli, "_register_backends"),
            mock.patch.object(cli, "_build_direct_mode_agent", build),
            mock.patch.object(cli, "_localmelo_version", return_value="v-test"),
            mock.patch("sys.argv", ["melo", "--output-format", "json", *argv, "q"]),
        ):
            cli.main()  # completed: no SystemExit
        record = self._record(capsys.readouterr().out)
        assert (record["status"], record["result"]) == ("completed", "42")
        keys = list(record["config"])
        assert keys.index("reflection_gate") == keys.index("max_attempts") + 1
        assert record["config"]["reflection_gate"] == gate


class _AnswerLLM(BaseLLMProvider):
    """A model that answers every request with *text*."""

    def __init__(self, text: str) -> None:
        self.text = text

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        return Message(role="assistant", content=self.text)

    async def close(self) -> None:
        pass


# ── Stdout is for the report ────────────────────────────────────────────────


class TestStdoutForTheReport:
    def test_what_reaches_fd_1_meanwhile_goes_to_stderr(
        self, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # sys.stdout as melo has it when it runs as a program: on fd 1.
        stdout = io.TextIOWrapper(io.FileIO(1, "w", closefd=False), encoding="utf-8")
        monkeypatch.setattr(sys, "stdout", stdout)
        with cli._stdout_for_the_report() as out:
            os.write(1, b"written to fd 1\n")
            subprocess.run(["echo", "from a child process"], check=True)
            stdout.write("written to sys.stdout\n")  # still buffered
            print("the report", file=out, flush=True)
        os.write(1, b"after the run\n")
        out_text, err_text = capfd.readouterr()
        assert out_text == "the report\nafter the run\n"
        assert err_text == (
            "written to fd 1\nfrom a child process\nwritten to sys.stdout\n"
        )

    def test_a_stdout_that_is_not_fd_1_is_used_as_it_is(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with cli._stdout_for_the_report() as out:
            assert out is sys.stdout
            print("the report", file=out)
        assert capsys.readouterr() == ("the report\n", "")


# ── localmelo_version ───────────────────────────────────────────────────────


def _fake_git(
    responses: dict[str, str | BaseException],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """subprocess.run for git, answering each subcommand from *responses*."""

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd[0] == "git"
        assert kwargs["cwd"] == PACKAGE_DIR
        response = responses[cmd[1]]
        if isinstance(response, BaseException):
            raise response
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{response}\n", stderr="")

    return run


class TestLocalmeloVersion:
    def test_git_describe_of_the_package_checkout(self) -> None:
        git = mock.MagicMock(
            side_effect=_fake_git(
                {"rev-parse": str(PACKAGE_DIR), "describe": "v0.2.0-3-gabc1234-dirty"}
            )
        )
        with mock.patch.object(cli.subprocess, "run", git):
            assert cli._localmelo_version() == "v0.2.0-3-gabc1234-dirty"
        assert git.call_args.args[0] == [
            "git",
            "describe",
            "--tags",
            "--always",
            "--dirty",
        ]
        assert git.call_args.kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"

    @pytest.mark.parametrize(
        "responses",
        [
            {"rev-parse": FileNotFoundError("git")},
            {"rev-parse": subprocess.CalledProcessError(128, ["git"])},
            {"rev-parse": subprocess.TimeoutExpired(["git"], 5)},
            {"rev-parse": str(PACKAGE_DIR), "describe": ""},
            # The package inside another repository: not described.
            {"rev-parse": str(PACKAGE_DIR.parent)},
        ],
        ids=["no-git", "not-a-checkout", "git-hangs", "nothing-described", "other"],
    )
    def test_falls_back_to_the_installed_version(
        self, responses: dict[str, str | BaseException]
    ) -> None:
        with (
            mock.patch.object(cli.subprocess, "run", side_effect=_fake_git(responses)),
            mock.patch.object(
                cli.importlib.metadata, "version", return_value="0.1.0"
            ) as version,
        ):
            assert cli._localmelo_version() == "0.1.0"
        version.assert_called_once_with("localmelo")

    def test_unknown_without_git_or_an_installed_version(self) -> None:
        missing = importlib.metadata.PackageNotFoundError("localmelo")
        with (
            mock.patch.object(
                cli.subprocess, "run", side_effect=FileNotFoundError("git")
            ),
            mock.patch.object(cli.importlib.metadata, "version", side_effect=missing),
        ):
            assert cli._localmelo_version() == "unknown"

    @pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
    def test_this_checkout(self) -> None:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", *args],
                cwd=PACKAGE_DIR,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()

        top = git("rev-parse", "--show-toplevel")
        if top and Path(top).resolve() == PACKAGE_DIR:
            expected = git("describe", "--tags", "--always", "--dirty")
        else:
            expected = importlib.metadata.version("localmelo")
        assert cli._localmelo_version() == expected


# ── Interactive mode ────────────────────────────────────────────────────────


class TestInteractive:
    def test_a_failed_turn_prints_the_status_line_and_the_loop_goes_on(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent = _agent(
            _outcome("provider_error", error="HTTP 500"),
            _outcome("completed", result="second answer"),
        )
        prompts = ["first task", "  ", "second task", "exit"]
        with mock.patch("builtins.input", side_effect=prompts):
            code = asyncio.run(cli._run(agent, None))
        assert code == cli.EXIT_COMPLETED
        assert [c.args for c in agent.run_task.await_args_list] == [
            ("first task",),
            ("second task",),
        ]
        out, err = capsys.readouterr()
        assert out == (
            "localmelo · local agent\n"
            "Enter a task to begin. Type /help for commands or /exit to quit.\n"
            "shell_exec is off in this loop (direct mode); a task run with "
            "--builtins all turns it on.\n"
            "\nsecond answer\n"
        )
        assert err == "melo: run ended with status provider_error: HTTP 500\n"
        agent.close.assert_awaited_once()

    def test_the_end_of_input_ends_the_loop(self) -> None:
        agent = _agent()
        with mock.patch("builtins.input", side_effect=EOFError):
            assert asyncio.run(cli._run(agent, None)) == cli.EXIT_COMPLETED
        agent.run_task.assert_not_awaited()
        agent.close.assert_awaited_once()

    def test_the_notice_is_not_printed_when_shell_exec_is_on(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent = _agent()
        agent.options = AgentOptions(mode="direct", builtins=frozenset({"shell_exec"}))
        with mock.patch("builtins.input", side_effect=EOFError):
            assert asyncio.run(cli._run(agent, None)) == cli.EXIT_COMPLETED
        out, _ = capsys.readouterr()
        assert "shell_exec is off" not in out


# ── SIGTERM and SIGHUP ──────────────────────────────────────────────────────


class _ExitedBySignalError(Exception):
    """Raised by the stand-in for _exit_by_signal."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _raise_exited(signum: int) -> NoReturn:
    raise _ExitedBySignalError(signum)


posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="SIGTERM and SIGHUP handling is POSIX-only"
)


@posix_only
class TestExitSignals:
    def test_the_first_signal_begins_the_exit_and_cancels_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exited: list[int] = []
        monkeypatch.setattr(cli, "_exit_by_signal", exited.append)
        agent = _agent()
        signals = cli._ExitSignals(agent)

        async def scenario() -> None:
            running = asyncio.create_task(asyncio.Event().wait())
            await asyncio.sleep(0)
            signals._receive(signal.SIGTERM, running)
            signals._receive(signal.SIGHUP, running)  # changes nothing
            with pytest.raises(asyncio.CancelledError):
                await running

        asyncio.run(scenario())
        agent.begin_exit.assert_called_once_with(
            "Cancelled by SIGTERM",
            within_s=cli.SIGNAL_CLEANUP_S - cli._SIGNAL_EXIT_MARGIN_S,
        )
        assert signals.received == signal.SIGTERM
        assert exited == []  # the clean-up limit had not passed

    def test_a_clean_up_past_the_limit_kills_the_providers_and_exits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exited: list[int] = []
        monkeypatch.setattr(cli, "_exit_by_signal", exited.append)
        monkeypatch.setattr(cli, "SIGNAL_CLEANUP_S", 0.01)
        broken, other = mock.MagicMock(), mock.MagicMock()
        broken.kill.side_effect = RuntimeError("cannot kill")
        signals = cli._ExitSignals(_agent(), [broken, other])

        async def scenario() -> None:
            signals._receive(signal.SIGHUP, None)
            await asyncio.sleep(0.2)

        asyncio.run(scenario())
        broken.kill.assert_called_once_with()
        other.kill.assert_called_once_with()
        assert exited == [signal.SIGHUP]

    def test_install_and_uninstall_leave_an_ignored_signal_ignored(self) -> None:
        term_before = signal.getsignal(signal.SIGTERM)
        hup_before = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            signals = cli._ExitSignals(_agent())

            async def scenario() -> tuple[Any, Any]:
                signals.install()
                try:
                    return (
                        signal.getsignal(signal.SIGTERM),
                        signal.getsignal(signal.SIGHUP),
                    )
                finally:
                    signals.uninstall()

            term_during, hup_during = asyncio.run(scenario())
            # nohup's SIGHUP stays ignored; SIGTERM goes to the event loop.
            assert hup_during == signal.SIG_IGN
            assert term_during not in (signal.SIG_DFL, signal.SIG_IGN, term_before)
            assert signal.getsignal(signal.SIGTERM) == term_before
            assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
        finally:
            signal.signal(signal.SIGHUP, hup_before)

    def test_a_signal_during_the_run_exits_by_it_and_prints_nothing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "_exit_by_signal", _raise_exited)
        monkeypatch.setattr(cli, "_localmelo_version", lambda: "v-test")
        agent = _agent()
        term_before = signal.getsignal(signal.SIGTERM)

        async def run_task(query: str) -> RunResult:
            # Only once the event loop handles SIGTERM may the test send it.
            handler = signal.getsignal(signal.SIGTERM)
            assert handler not in (signal.SIG_DFL, signal.SIG_IGN, None, term_before)
            signal.raise_signal(signal.SIGTERM)
            await asyncio.Event().wait()
            raise AssertionError("the run was not cancelled")

        agent.run_task.side_effect = run_task
        with pytest.raises(_ExitedBySignalError) as exited:
            cli._run_direct(agent, "q", "json")
        assert exited.value.signum == signal.SIGTERM
        agent.begin_exit.assert_called_once_with(
            "Cancelled by SIGTERM",
            within_s=cli.SIGNAL_CLEANUP_S - cli._SIGNAL_EXIT_MARGIN_S,
        )
        agent.close.assert_awaited_once()
        assert capsys.readouterr() == ("", "")  # no JSON line, no status line
        assert signal.getsignal(signal.SIGTERM) == term_before

    def test_a_failure_while_stopping_is_one_line_and_still_exits_by_the_signal(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        hup_before = signal.getsignal(signal.SIGHUP)
        if hup_before == signal.SIG_IGN:
            pytest.skip("SIGHUP is ignored here (nohup), so melo leaves it be")
        monkeypatch.setattr(cli, "_exit_by_signal", _raise_exited)
        agent = _agent()
        agent.close.side_effect = RuntimeError("database\nis locked")

        async def run_task(query: str) -> RunResult:
            handler = signal.getsignal(signal.SIGHUP)
            assert handler not in (signal.SIG_DFL, signal.SIG_IGN, None, hup_before)
            signal.raise_signal(signal.SIGHUP)
            await asyncio.Event().wait()
            raise AssertionError("the run was not cancelled")

        agent.run_task.side_effect = run_task
        with pytest.raises(_ExitedBySignalError) as exited:
            cli._run_direct(agent, "q", "text")
        assert exited.value.signum == signal.SIGHUP
        assert capsys.readouterr() == (
            "",
            "melo: error while stopping: RuntimeError: database is locked\n",
        )

    def test_a_run_that_returns_after_the_signal_is_not_reported(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "_exit_by_signal", _raise_exited)
        agent = _agent()
        signals = cli._ExitSignals(agent)

        async def run_task(query: str) -> RunResult:
            # The signal arrives as the run ends: nothing is left to cancel.
            signals._receive(signal.SIGTERM, None)
            return _outcome("completed", result="42")

        agent.run_task.side_effect = run_task
        code = asyncio.run(
            cli._run(agent, "q", output_format="json", exit_signals=signals)
        )
        assert code == cli.EXIT_CRASHED
        assert capsys.readouterr() == ("", "")


# ── The memory directory's mode ─────────────────────────────────────────────


class TestMemoryDirMode:
    """--memory-dir: a new directory is made 0700, an existing one is kept."""

    def _run(self, memory_dir: Path) -> None:
        argv = ["--base-url", "http://127.0.0.1:9/v1", "--memory-dir", str(memory_dir)]
        with (
            mock.patch.object(cli, "_register_backends"),
            mock.patch.object(cli, "Agent") as agent_cls,
            mock.patch("sys.argv", ["melo", *argv, "q"]),
        ):
            agent_cls.return_value = _agent(_outcome("completed", result="42"))
            cli.main()
        agent_cls.assert_called_once()

    def test_a_new_directory_is_made_0700(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "new" / "memory"
        previous = os.umask(0o022)
        try:
            self._run(memory_dir)
        finally:
            os.umask(previous)
        assert memory_dir.stat().st_mode & 0o777 == 0o700

    def test_an_existing_directory_keeps_its_mode(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        memory_dir.chmod(0o755)
        self._run(memory_dir)
        assert memory_dir.stat().st_mode & 0o777 == 0o755


# ── LOCALMELO_THINKING_RETENTION_DAYS ───────────────────────────────────────


class TestThinkingRetentionVariable:
    """An invalid value ends melo with one line and exit code 2, before any run."""

    MESSAGE = (
        "melo: error: LOCALMELO_THINKING_RETENTION_DAYS must be a number of days "
        "greater than 0, not 'abc'\n"
    )

    @pytest.mark.parametrize("query", ["q", ""], ids=["a-query", "the-loop"])
    def test_direct_mode_refuses_an_invalid_value(
        self,
        query: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("LOCALMELO_THINKING_RETENTION_DAYS", "abc")
        ran = _main(query)
        assert ran.code == 2
        ran.build.assert_not_called()
        assert capsys.readouterr() == ("", self.MESSAGE)

    def test_the_gateway_refuses_it_before_the_config_is_loaded(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("LOCALMELO_THINKING_RETENTION_DAYS", "abc")
        with (
            mock.patch("localmelo.support.config.load") as load,
            mock.patch.object(cli, "_start_gateway") as start,
        ):
            ran = _main("--serve")
        assert ran.code == 2
        load.assert_not_called()
        start.assert_not_called()
        assert capsys.readouterr() == ("", self.MESSAGE)

    def test_a_valid_value_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCALMELO_THINKING_RETENTION_DAYS", "7")
        ran = _main("q")
        assert ran.code == 0
        ran.build.assert_called_once()
