"""``[agent] confine_direct_runs`` (#183): a developer switch that confines plain direct runs.

Off by default: a direct run is then as it was. On, a task given on the command
line and the ``melo ""`` loop start ``shell_exec`` and ``python_exec`` under the
sandbox of ``--brief`` runs (``tests/cli/test_confine_writes.py``); the chat UI
and the gateway ignore the key, and a ``--brief`` run keeps ``confine_writes``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import Agent
from localmelo.melo.executor import ConfinementError, sandbox
from localmelo.melo.schema import Message, ToolCall
from localmelo.support import config, config_cli
from localmelo.support.providers.llm import openai_compat
from localmelo.support.sessions.agents import configured_session_options
from localmelo.tests.agent.test_reflection_gate import FakeLLM

BASE = ["--base-url", "http://127.0.0.1:1/v1", "--chat-model", "m"]
KEY = "agent.confine_direct_runs"


def _on(via: str, argv: list[str]) -> list[str]:
    """*argv* with the key on, through the file or ``-X``."""
    if via == "file":
        config.save(config.Config(agent=config.AgentConfig(confine_direct_runs=True)))
        return argv
    return ["-X", f"{KEY}=true", *argv]


async def _build(argv: list[str]) -> Agent:
    parser, args = cli._parse(argv)
    options = cli._direct_options(parser, args)
    assert (
        options.confine_writes is False
    )  # no option sets it: the builder reads the key
    return cli._build_direct_mode_agent(args, parser=parser, options=options)


class TestConfig:
    def test_the_key_is_off_by_default_written_only_when_on_and_a_boolean(self) -> None:
        assert config.AgentConfig().confine_direct_runs is False
        config.save(config.Config())
        path = Path(config.config_path())
        assert "confine_direct_runs" not in path.read_text()
        config.save(config.Config(agent=config.AgentConfig(confine_direct_runs=True)))
        assert "confine_direct_runs = true" in path.read_text()
        assert config.load().agent.confine_direct_runs is True
        bad = config.Config()
        bad.agent.confine_direct_runs = "yes"  # type: ignore[assignment]
        assert any("confine_direct_runs must be a boolean" in e for e in bad.validate())

    def test_x_sets_it_for_one_run_and_the_developer_listings_show_it(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli._parse(["-X", f"{KEY}=true", *BASE, "fix it"])
        assert config.overrides() == {KEY: True}
        assert config.load().agent.confine_direct_runs is True
        assert config_cli.main(["keys", "confine_direct", "-X", f"{KEY}=on"]) == 0
        row = capsys.readouterr().out.splitlines()[1].split()
        assert row == [KEY, "true", "false", "-X"]
        text = cli._build_parser(developer=True).format_help()
        assert any(
            line.startswith(f"  {KEY} ") and line.endswith("(#183)")
            for line in text.splitlines()
        )


class TestOptions:
    @pytest.mark.parametrize("via", ["file", "x"])
    @pytest.mark.parametrize("task", ["fix it", ""])
    @pytest.mark.parametrize("workspace", [True, False])
    async def test_a_plain_direct_run_is_confined_when_on(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        via: str,
        task: str,
        workspace: bool,
    ) -> None:
        (tmp_path / "ws").mkdir()
        # Without --workspace, the folder melo runs in is the workspace.
        monkeypatch.chdir(tmp_path / "ws")
        if workspace and not task:
            pytest.skip(
                "the loop has no --workspace: it works in the folder it runs in"
            )
        argv = [*BASE, *(["-w", str(tmp_path / "ws")] if workspace else []), task]
        try:
            agent = await _build(_on(via, argv))
        except ConfinementError as exc:
            pytest.skip(str(exc))
        try:
            assert agent.options.confine_writes is True
            if sys.platform == "darwin":
                assert (tmp_path / "ws" / sandbox.TMP_DIR / ".gitignore").is_file()
        finally:
            await agent.close()

    @pytest.mark.parametrize("task", ["fix it", ""])
    async def test_off_a_direct_run_is_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str
    ) -> None:
        monkeypatch.chdir(tmp_path)
        explicit = ["-X", f"{KEY}=false", *BASE, task]
        for argv in ([*BASE, task], explicit):
            agent = await _build(argv)
            try:
                assert agent.options.confine_writes is False
                assert agent._confinement is None
            finally:
                await agent.close()
        assert not (tmp_path / sandbox.TMP_DIR).exists()

    async def test_a_brief_keeps_confine_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # main() sets a brief's option from confine_writes before the builder
        # runs; the key never changes it, either way.
        (tmp_path / "brief.md").write_text("Explain.\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        argv = ["-X", f"{KEY}=true", *BASE, "--brief", "brief.md", "-w", str(tmp_path)]
        agent = await _build(argv)
        try:
            assert agent.options.confine_writes is False
        finally:
            await agent.close()

    async def test_the_interactive_ui_and_the_gateway_ignore_it(
        self, tmp_path: Path
    ) -> None:
        cfg = config.Config(agent=config.AgentConfig(confine_direct_runs=True))
        assert configured_session_options(cfg).confine_writes is False
        parser, args = cli._parse(["-X", f"{KEY}=true", *BASE])
        assert cli._direct_options(parser, args).confine_writes is False  # bare melo


class TestRuns:
    def _main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        argv: list[str],
        replies: list[Message],
    ) -> tuple[int, str, str]:
        (tmp_path / "ws").mkdir()
        (tmp_path / "home").mkdir()
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        # tmp_path is inside the user's temp folder, which the profile allows.
        monkeypatch.setattr(sandbox, "user_temp_dir", lambda: None)
        monkeypatch.setattr(
            openai_compat, "OpenAICompatLLM", lambda **kwargs: FakeLLM(replies)
        )
        monkeypatch.setattr(sys, "argv", ["melo", *argv])
        code = 0
        try:
            cli.main()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else int(bool(exc.code))
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def _argv(self, tmp_path: Path, *extra: str) -> list[str]:
        return [
            "-w",
            str(tmp_path / "ws"),
            "--builtins",
            "shell_exec",
            "--memory",
            "off",
            "--no-save-session",
            "-o",
            "json",
            *BASE,
            *extra,
            "--",
            "make a file",
        ]

    @staticmethod
    def _writes() -> list[Message]:
        call = ToolCall("shell_exec", {"command": "touch ~/outside.txt"})
        return [Message("assistant", "", tool_call=call), Message("assistant", "Done")]

    @pytest.mark.skipif(sys.platform != "darwin", reason="writes are confined on macOS")
    def test_on_a_write_outside_the_folder_is_refused_and_counted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        argv = self._argv(tmp_path, "-X", f"{KEY}=true")
        code, out, err = self._main(tmp_path, monkeypatch, capsys, argv, self._writes())
        if "cannot be confined" in err:
            pytest.skip(err.strip())
        assert code == 0, err
        record = json.loads(out)
        assert record["confinement"] == {"writes": "confined", "refused_writes": 1}
        assert not (tmp_path / "home" / "outside.txt").exists()

    def test_off_the_same_run_writes_and_has_no_confinement_key(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code, out, err = self._main(
            tmp_path, monkeypatch, capsys, self._argv(tmp_path), self._writes()
        )
        assert code == 0, err
        assert "confinement" not in json.loads(out)
        assert (tmp_path / "home" / "outside.txt").exists()
        assert not (tmp_path / "ws" / sandbox.TMP_DIR).exists()

    def test_a_run_that_cannot_be_confined_stops_with_one_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sandbox, "sys", SimpleNamespace(platform="darwin"))
        monkeypatch.setattr(sandbox, "SANDBOX_EXEC", "/nonexistent/sandbox-exec")
        argv = self._argv(tmp_path, "-X", f"{KEY}=true")
        code, out, err = self._main(tmp_path, monkeypatch, capsys, argv, [])
        assert code == 1 and out == ""
        assert err == (
            "melo: error: writes cannot be confined to the workspace: "
            "/nonexistent/sandbox-exec is missing; set agent.confine_direct_runs "
            "to false to run without it (-X agent.confine_direct_runs=false for "
            "one run)\n"
        )

    def test_a_bad_value_in_the_file_is_a_usage_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        bad: Any = "no"
        monkeypatch.setattr(
            config,
            "load",
            lambda: config.Config(agent=config.AgentConfig(confine_direct_runs=bad)),
        )
        code, out, err = self._main(
            tmp_path, monkeypatch, capsys, self._argv(tmp_path), []
        )
        assert code == 2 and out == ""
        assert "[agent] confine_direct_runs must be a boolean" in err
