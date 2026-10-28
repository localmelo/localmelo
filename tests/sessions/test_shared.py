"""The helpers both session surfaces share: formats, paths, the status line.

The input and output pairs live in ``tests/fixtures/display_vectors.json``,
which the web console's JS mirror reads too; direct mode's own functions
are the reference for the status line and the version.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.melo import schema
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.schema import RunResult
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.progress_line import run_elapsed
from localmelo.support.providers.embedding.openai_compat import OpenAICompatEmbedding
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.sessions import agents, display
from localmelo.support.sessions.agents import (
    SESSION_TOOL_TIMEOUT_S,
    build_session_agent,
)
from localmelo.support.sessions.store import RUN_STATUSES

VECTORS: dict[str, list[list[Any]]] = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "display_vectors.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize(("seconds", "text"), VECTORS["format_duration"])
def test_format_duration(seconds: float, text: str) -> None:
    assert display.format_duration(seconds) == text


@pytest.mark.parametrize("seconds", [10, 10.9, 59.99, 61, 3599, 3600, 86_400])
def test_from_ten_seconds_a_duration_is_the_progress_lines(seconds: float) -> None:
    assert display.format_duration(seconds) == run_elapsed(seconds)


@pytest.mark.parametrize(("n", "text"), VECTORS["format_tokens"])
def test_format_tokens(n: int, text: str) -> None:
    assert display.format_tokens(n) == text


@pytest.mark.parametrize(("status", "error", "line"), VECTORS["status_line"])
def test_the_status_line_is_direct_modes(status: str, error: str, line: str) -> None:
    assert display.status_line(status, error) == line
    outcome = RunResult(task_id="t", status=status, error=error)
    assert cli._status_line(outcome) == line


def test_a_long_reason_is_cut_as_direct_mode_cuts_it() -> None:
    error = "x" * 600
    outcome = RunResult(task_id="t", status="failed", error=error)
    assert display.status_line("failed", error) == cli._status_line(outcome)
    assert display.status_line("failed", error).endswith("x...")


@pytest.mark.parametrize(
    ("status", "cause", "word", "severity", "title"), VECTORS["status_text"]
)
def test_status_text(
    status: str, cause: str, word: str, severity: str, title: str
) -> None:
    text = display.status_text(status, cause)
    assert (text.word, text.severity, text.title) == (word, severity, title)


def test_status_text_falls_back_from_the_cause_to_the_status() -> None:
    assert display.status_text("cancelled") == display.STATUS_TEXT["cancelled"]
    # A cause with no row of its own reads as its status does.
    assert display.status_text("cancelled", "run") == display.STATUS_TEXT["cancelled"]
    assert (
        display.status_text("cancelled", "shutdown")
        == display.STATUS_TEXT[("cancelled", "shutdown")]
    )


def test_every_run_status_has_a_word() -> None:
    assert {key for key in display.STATUS_TEXT if isinstance(key, str)} >= RUN_STATUSES


def test_an_unknown_status_reads_as_its_name() -> None:
    text = display.status_text("from_the_future")
    assert (text.word, text.severity, text.title) == (
        "from_the_future",
        "warn",
        "from_the_future",
    )


def test_split_thinking_is_the_cores() -> None:
    assert display.split_thinking is schema.split_thinking
    assert display.split_thinking("<think>why</think>\nanswer") == ("why", "answer")


@pytest.mark.parametrize(
    ("status", "error_category", "approval", "short", "long"),
    VECTORS["approval_tag"],
)
def test_approval_tag(
    status: str,
    error_category: str,
    approval: dict[str, Any],
    short: str,
    long: str,
) -> None:
    text = display.approval_tag(status, error_category, approval)
    assert text == display.ApprovalText(short, long)


def test_an_unasked_approval_has_no_tag() -> None:
    assert display.approval_tag("success", "none", None) is None
    assert display.approval_tag("success", "none", {}) is None


def test_a_user_denial_needs_the_call_to_have_been_blocked() -> None:
    approval = {"allowed": False, "origin": "user", "remember": False}
    assert display.approval_tag("success", "none", approval) is None
    assert display.approval_tag("cancelled", "cancelled", approval) is None
    assert display.approval_tag("blocked", "none", approval) is not None
    assert display.approval_tag("error", "approval_denied", approval) is not None


def test_a_withdrawal_is_not_run_only_for_a_cancelled_call() -> None:
    approval = {"allowed": False, "origin": "withdrawn", "remember": False}
    assert display.approval_tag("cancelled", "cancelled", approval) == (
        display.ApprovalText("not run", "Not run: the turn was stopped")
    )
    assert display.approval_tag("success", "none", approval) is None


@pytest.mark.parametrize(("text", "clean"), VECTORS["clean_output"])
def test_clean_output(text: str, clean: str) -> None:
    assert display.clean_output(text) == clean


def test_clean_output_shows_an_unterminated_osc_as_text_without_its_escape() -> None:
    cleaned = display.clean_output("a\x1b]52;c;payload")
    assert "\x1b" not in cleaned and cleaned.startswith("a")


def test_the_version_is_direct_modes() -> None:
    display.localmelo_version.cache_clear()
    assert display.localmelo_version() == cli._localmelo_version()


def test_the_version_is_computed_once_a_process() -> None:
    display.localmelo_version.cache_clear()
    with mock.patch.object(
        display.subprocess, "run", side_effect=OSError("no git")
    ) as run:
        first = display.localmelo_version()
        second = display.localmelo_version()
    display.localmelo_version.cache_clear()
    assert first == second
    assert run.call_count == 1


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    real = tmp_path / "real-home"
    real.mkdir()
    link = tmp_path / "home"
    link.symlink_to(real)
    monkeypatch.setenv("HOME", str(link))
    return link


def test_home_label(home: Path) -> None:
    real = os.path.realpath(home)
    assert display.home_label(str(home)) == "~"
    assert display.home_label(f"{home}/demo") == "~/demo"
    # The home's real path reads the same (macOS resolves /var to /private/var).
    assert display.home_label(f"{real}/项目") == "~/项目"
    assert display.home_label(f"{home}-other/x") == f"{home}-other/x"
    assert display.home_label("/etc/hosts") == "/etc/hosts"


def test_display_path(home: Path) -> None:
    root = f"{home}/demo"
    assert display.display_path(f"{root}/notes.txt", root) == "notes.txt"
    assert display.display_path(f"{root}/src/app.py", root + "/") == "src/app.py"
    assert display.display_path(root, root) == "."
    assert display.display_path(f"{home}/other/x", root) == "~/other/x"
    assert display.display_path(f"{root}2/x", root) == "~/demo2/x"
    assert display.display_path("/etc/hosts", None) == "/etc/hosts"
    assert display.display_path("/etc/hosts", "/") == "etc/hosts"


# ── Session Agents ──


def _config() -> Config:
    return Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(chat_url="http://127.0.0.1:9/v1", chat_model="cfg-m"),
    )


async def test_a_session_agent_works_in_its_workspace_with_session_limits(
    tmp_path: Path,
) -> None:
    options = AgentOptions(max_steps=7, memory=False)
    agent = await build_session_agent(
        cfg=None,
        base_url="http://127.0.0.1:9/v1",
        model="qwen3-8b",
        workspace=str(tmp_path),
        options=options,
    )
    try:
        assert agent.options == dataclasses.replace(
            options,
            workspace_root=str(tmp_path),
            record_answers=True,
            tool_exposure="all",
            tool_timeout_s=SESSION_TOOL_TIMEOUT_S,
            replay_scope="window",
        )
        assert agent.workspace_root == os.path.realpath(tmp_path)
        assert agent._llm.model == "qwen3-8b"
        assert agent._embedding is None
    finally:
        await agent.close()
    assert SESSION_TOOL_TIMEOUT_S == 600.0


def test_session_options_keep_the_callers_mode() -> None:
    for mode in ("interactive", "gateway"):
        options = agents.session_options(
            AgentOptions(mode=mode), "/tmp/workspace"  # type: ignore[arg-type]
        )
        assert options.mode == mode
        assert options.tool_exposure == "all"


async def test_a_url_without_a_model_asks_for_default_with_the_timeout_given(
    tmp_path: Path,
) -> None:
    agent = await build_session_agent(
        cfg=None,
        base_url="http://127.0.0.1:9/v1",
        model=None,
        workspace=str(tmp_path),
        options=AgentOptions(),
        embedding_base_url="http://127.0.0.1:9/v1",
        embedding_model="e5",
        request_timeout_s=33.0,
    )
    try:
        llm: Any = agent._llm
        embedding: Any = agent._embedding
        assert llm.model == "default"
        assert llm._http.timeout.read == 33.0
        assert embedding.model == "e5"
    finally:
        await agent.close()


async def test_a_config_is_copied_with_the_model_and_timeout_given(
    tmp_path: Path,
) -> None:
    cfg = _config()
    agent = await build_session_agent(
        cfg=cfg,
        base_url=None,
        model="other-m",
        workspace=str(tmp_path),
        options=AgentOptions(),
        request_timeout_s=44.0,
    )
    try:
        llm: Any = agent._llm
        assert (llm.model, llm._http.timeout.read) == ("other-m", 44.0)
    finally:
        await agent.close()
    assert (cfg.vllm.chat_model, cfg.request_timeout_s) == ("cfg-m", None)


async def test_without_a_url_or_a_config_nothing_is_built(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cfg or a base_url"):
        await build_session_agent(
            cfg=None,
            base_url=None,
            model=None,
            workspace=str(tmp_path),
            options=AgentOptions(),
        )


async def test_an_agent_that_fails_to_start_is_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []
    real_close = agents.Agent.close

    async def start(self: agents.Agent) -> None:
        raise RuntimeError("MCP server exited")

    async def close(self: agents.Agent) -> None:
        closed.append(True)
        await real_close(self)

    monkeypatch.setattr(agents.Agent, "start", start)
    monkeypatch.setattr(agents.Agent, "close", close)
    with pytest.raises(RuntimeError, match="MCP server exited"):
        await build_session_agent(
            cfg=_config(),
            base_url=None,
            model=None,
            workspace=str(tmp_path),
            options=AgentOptions(),
        )
    assert closed == [True]


class _Wrapped:
    """A provider that only delegates close(), to prove what the Agent got."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    async def close(self) -> None:
        await self.inner.close()


class _RecordingAgent(Agent):
    """An Agent that records how its builder made it."""

    built: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        type(self).built.append(dict(kwargs))
        super().__init__(*args, **kwargs)


async def test_the_hook_wraps_the_providers_the_agent_is_built_with(
    tmp_path: Path,
) -> None:
    seen: list[Any] = []

    def wrap_llm(llm: Any) -> Any:
        seen.append(llm)
        return _Wrapped(llm)

    def wrap_embedding(embedding: Any) -> Any:
        seen.append(embedding)
        return _Wrapped(embedding)

    agent = await build_session_agent(
        cfg=None,
        base_url="http://127.0.0.1:9/v1",
        model="qwen3-8b",
        workspace=str(tmp_path),
        options=AgentOptions(),
        embedding_base_url="http://127.0.0.1:9/v1",
        embedding_model="e5",
        wrap_llm=wrap_llm,
        wrap_embedding=wrap_embedding,
    )
    try:
        # The Chat is made from the wrapped provider, so every model call
        # goes through it.
        assert isinstance(agent._llm, _Wrapped)
        assert agent.chat.llm is agent._llm
        assert isinstance(agent._embedding, _Wrapped)
        assert isinstance(seen[0], OpenAICompatLLM)
        assert isinstance(seen[1], OpenAICompatEmbedding)
        assert agent._llm.inner is seen[0]
        assert agent._embedding.inner is seen[1]
    finally:
        await agent.close()


async def test_the_hook_wraps_the_providers_of_a_config_and_no_hook_keeps_the_old_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _RecordingAgent.built.clear()
    built = _RecordingAgent.built
    monkeypatch.setattr(agents, "Agent", _RecordingAgent)
    cfg = _config()
    cfg.embedding_backend = "vllm"
    cfg.vllm.embedding_url = "http://127.0.0.1:9/v1"
    cfg.vllm.embedding_model = "cfg-e"

    # No hook: the Agent is made from the config, as it was before the hook.
    agent = await build_session_agent(
        cfg=cfg,
        base_url=None,
        model=None,
        workspace=str(tmp_path),
        options=AgentOptions(),
    )
    try:
        assert "config" in built[0]
        assert "llm" not in built[0]
        assert isinstance(agent._llm, OpenAICompatLLM)
        assert agent._llm.model == "cfg-m"
        assert isinstance(agent._embedding, OpenAICompatEmbedding)
    finally:
        await agent.close()

    seen: list[Any] = []

    def wrap_llm(llm: Any) -> Any:
        seen.append(llm)
        return llm

    def wrap_embedding(embedding: Any) -> Any:
        seen.append(embedding)
        return embedding

    agent = await build_session_agent(
        cfg=cfg,
        base_url=None,
        model=None,
        workspace=str(tmp_path),
        options=AgentOptions(),
        wrap_llm=wrap_llm,
        wrap_embedding=wrap_embedding,
    )
    try:
        # With a hook the providers exist first, so the Agent is made from
        # them, and they are the ones the wrappers returned.
        assert "config" not in built[1]
        assert built[1]["llm"] is seen[0]
        assert built[1]["embedding"] is seen[1]
        assert agent._llm is seen[0]
        assert agent._embedding is seen[1]
        assert isinstance(agent._llm, OpenAICompatLLM)
        assert isinstance(agent._embedding, OpenAICompatEmbedding)
    finally:
        await agent.close()


async def test_an_embedding_that_was_not_built_is_not_wrapped(
    tmp_path: Path,
) -> None:
    called: list[Any] = []

    def wrap_embedding(embedding: Any) -> Any:
        called.append(embedding)
        return embedding

    agent = await build_session_agent(
        cfg=None,
        base_url="http://127.0.0.1:9/v1",
        model="qwen3-8b",
        workspace=str(tmp_path),
        options=AgentOptions(),
        wrap_embedding=wrap_embedding,
    )
    try:
        assert agent._embedding is None
        assert called == []
    finally:
        await agent.close()


# ── The facts of a file_write approval (approval_target) ──


def test_approval_target_is_none_for_another_tool() -> None:
    assert display.approval_target({"tool_name": "shell_exec", "arguments": {}}) is None


def test_approval_target_reads_a_missing_file_and_a_text_one(tmp_path: Path) -> None:
    missing = display.approval_target(
        {
            "tool_name": "file_write",
            "arguments": {"path": str(tmp_path / "missing.md")},
            "workspace_root": str(tmp_path),
        }
    )
    assert missing == {"path_label": "missing.md", "exists": False, "size_bytes": 0}
    path = tmp_path / "notes.txt"
    path.write_text("buy milk\n", encoding="utf-8")
    found = display.approval_target(
        {
            "tool_name": "file_write",
            "arguments": {"path": str(path)},
            "workspace_root": str(tmp_path),
        }
    )
    assert found == {
        "path_label": "notes.txt",
        "exists": True,
        "size_bytes": path.stat().st_size,
    }


def test_approval_target_does_not_follow_a_final_symlink_or_open_a_fifo(
    tmp_path: Path,
) -> None:
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    linked = display.approval_target(
        {
            "tool_name": "file_write",
            "arguments": {"path": str(link)},
            "workspace_root": str(tmp_path),
        }
    )
    assert linked == {"path_label": "link", "exists": True, "size_bytes": 0}
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    piped = display.approval_target(
        {
            "tool_name": "file_write",
            "arguments": {"path": str(fifo)},
            "workspace_root": str(tmp_path),
        }
    )
    assert piped == {"path_label": "pipe", "exists": True, "size_bytes": 0}
