"""``[tools] web_fetch``: the setting, which Agents get the tool, its tier, prompt line and label."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.executor.preview import call_preview
from localmelo.support import config, settings
from localmelo.support.gateway.console import front, registry
from localmelo.support.sessions import agents as session_agents
from localmelo.support.sessions.display import activity_text, request_words
from localmelo.support.web import NOTE


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(
        "LOCALMELO_SAFARIDRIVER", "/nonexistent/not-the-real-safaridriver"
    )


def _cfg(**tools: Any) -> config.Config:
    cfg = config.Config(chat_backend="vllm", embedding_backend="none")
    cfg.vllm.chat_url = "http://127.0.0.1:9/v1"
    for name, value in tools.items():
        setattr(cfg.tools, name, value)
    return cfg


class _NoTools(BaseToolProvider):
    async def start(self) -> list[ProvidedTool]:
        return []

    async def close(self) -> None:
        pass


def _tools(agent: Any) -> dict[str, str]:
    return {tool.name: tool.tier for tool in agent.list_tools()}


# ── The setting ──


def test_web_fetch_is_on_by_default_and_a_default_config_says_nothing_of_it() -> None:
    cfg = _cfg()
    assert cfg.tools.web_fetch is True and cfg.setting_errors() == []
    config.save(cfg)
    assert "[tools]" not in Path(config.config_path()).read_text()
    key = settings.catalog()["tools.web_fetch"]
    assert (key.default, key.file, key.section) == (True, True, "tools")


def test_turning_it_off_is_saved_loaded_and_checked() -> None:
    cfg = _cfg(web_fetch=False)
    config.save(cfg)
    assert "[tools]\nweb_fetch = false" in Path(config.config_path()).read_text()
    assert config.load().tools.web_fetch is False
    cfg.tools.web_fetch = "no"  # type: ignore[assignment]
    assert any("[tools] web_fetch must be a bool" in e for e in cfg.setting_errors())
    assert any("[tools] web_fetch must be a bool" in e for e in cfg.validate())


def test_x_turns_it_off_for_one_run() -> None:
    cli._parse(["-X", "tools.web_fetch=off", "q"])
    assert config.overrides() == {"tools.web_fetch": False}
    assert config.load().tools.web_fetch is False


# ── Which Agents have it ──


async def _session_agent(**kwargs: Any) -> Any:
    return await session_agents.build_session_agent(
        cfg=_cfg(),
        base_url=None,
        model=None,
        workspace=str(Path.cwd()),
        options=AgentOptions(mode="interactive"),
        **kwargs,
    )


async def test_a_chat_with_the_setting_has_web_fetch_as_a_read_tool_and_its_prompt_line() -> (
    None
):
    agent = await _session_agent(web_fetch=True)
    try:
        assert _tools(agent)["web_fetch"] == "read"
        assert not agent.executor.requires_approval("web_fetch")
        assert NOTE in agent.chat.system_notes
    finally:
        await agent.close()


async def test_without_the_setting_the_tool_and_its_line_are_absent() -> None:
    agent = await _session_agent()
    try:
        assert "web_fetch" not in _tools(agent)
        assert not any("web_fetch" in note for note in agent.chat.system_notes)
    finally:
        await agent.close()


async def test_with_the_browser_on_the_line_points_to_it() -> None:
    agent = await _session_agent(web_fetch=True, browser_driver="safari-automation")
    try:
        assert {"web_fetch", "browser_open"} <= set(_tools(agent))
        assert any(
            "browser_open can try it" in note for note in agent.chat.system_notes
        )
    finally:
        await agent.close()


@pytest.mark.parametrize("on", [True, False])
async def test_the_console_chats_follow_the_setting(on: bool) -> None:
    cfg = _cfg(web_fetch=on)
    side = await registry.default_builder(cfg, str(Path.cwd()), None, "s1")
    main = await front.build_front_agent(cfg, str(Path.cwd()), "s2", tools=_NoTools())
    try:
        for agent in (side, main):
            assert ("web_fetch" in _tools(agent)) is on
    finally:
        await side.close()
        await main.close()


async def test_the_terminal_and_the_tasks_pass_the_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    from localmelo.support.cli import launch
    from localmelo.support.cli.agents import AgentFactory, url_backend
    from localmelo.support.gateway import workers

    seen: list[Any] = []

    async def fake_build(**kwargs: Any) -> Any:
        seen.append(kwargs.get("web_fetch"))
        return object()

    monkeypatch.setattr("localmelo.support.cli.agents.build_session_agent", fake_build)
    factory = AgentFactory(
        backend=url_backend("http://127.0.0.1:9/v1", "m", source="saved"),
        cfg=None,
        options=AgentOptions(),
        workspace="/w",
        web_fetch=True,
    )
    factory._live.clear()
    await factory.build()
    assert seen == [True]
    assert "web_fetch=loaded.tools.web_fetch is True" in inspect.getsource(launch)
    source = inspect.getsource(workers)
    # A task's provider asks the chat that owns it about a site, or refuses.
    assert "[WebFetchProvider(task=True)]" in source
    assert "if self.settings().tools.web_fetch is True" in source


def test_direct_runs_never_get_it() -> None:
    import argparse
    import inspect

    default = inspect.signature(session_agents.build_session_agent).parameters[
        "web_fetch"
    ]
    assert default.default is False
    config.save(_cfg())
    args = argparse.Namespace(mcp_config=None)
    assert cli._direct_tool_providers(argparse.ArgumentParser(), args) == []


# ── How a call shows ──


@pytest.mark.parametrize(
    ("url", "label"),
    [
        ("https://www.example.com/news?id=3", "Reading example.com"),
        ("http://garden.example:8080/", "Reading garden.example"),
        ("not an address", "Opening a page"),
        ("http://[bad", "Opening a page"),
    ],
)
def test_a_call_reads_as_reading_its_host(url: str, label: str) -> None:
    text = activity_text("web_fetch", {"url": url})
    assert (text.text, text.icon) == (label, "web")


def test_the_call_preview_is_the_address() -> None:
    assert (
        call_preview("web_fetch", {"url": " https://example.com/a "})
        == "https://example.com/a"
    )
    assert (
        call_preview("web_fetch", {"url": "https://e.com/", "part": 3})
        == "https://e.com/ (part 3)"
    )
    assert (
        call_preview("web_fetch", {"url": "https://e.com/", "part": 1})
        == "https://e.com/"
    )
    assert (
        request_words("web_fetch", {"url": "https://example.com/"})
        == "Read example.com"
    )
