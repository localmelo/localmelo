"""``[web] search``: the setting, which Agents get the tool and its prompt line, and how a call shows."""

from __future__ import annotations

import json
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
from localmelo.support.web.search_tool import NOTE


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(
        "LOCALMELO_SAFARIDRIVER", "/nonexistent/not-the-real-safaridriver"
    )


def _cfg(**web: Any) -> config.Config:
    cfg = config.Config(chat_backend="vllm", embedding_backend="none")
    cfg.vllm.chat_url = "http://127.0.0.1:9/v1"
    for name, value in web.items():
        setattr(cfg.web, name, value)
    return cfg


class _NoTools(BaseToolProvider):
    async def start(self) -> list[ProvidedTool]:
        return []

    async def close(self) -> None:
        pass


def _tools(agent: Any) -> dict[str, str]:
    return {tool.name: tool.tier for tool in agent.list_tools()}


# ── The setting ──


def test_web_search_is_off_by_default_and_a_default_config_says_nothing_of_it() -> None:
    cfg = _cfg()
    assert cfg.web.search is False and cfg.setting_errors() == []
    config.save(cfg)
    assert "[web]" not in Path(config.config_path()).read_text()
    key = settings.catalog()["web.search"]
    assert (key.default, key.file, key.section) == (False, True, "web")


def test_the_developer_settings_round_trip_and_are_checked() -> None:
    cfg = _cfg(search_order=["baidu", "browser"], search_count=3)
    config.save(cfg)
    loaded = config.load()
    assert (loaded.web.search_order, loaded.web.search_count) == (
        ["baidu", "browser"],
        3,
    )
    assert (
        _cfg(search_order=["nowhere"]).setting_errors()
        == [
            e
            for e in _cfg(search_order=["nowhere"]).setting_errors()
            if "search_order" in e
        ]
        != []
    )
    assert any("twice" in e for e in _cfg(search_order=["baidu", "baidu"]).validate())
    for bad in (0, 9, True, "3"):
        assert any(
            "search_count" in e for e in _cfg(search_count=bad).setting_errors()
        ), bad


def test_x_turns_it_on_for_one_run() -> None:
    cli._parse(["-X", "web.search=true", "q"])
    assert config.overrides() == {"web.search": True}
    assert config.load().web.search is True
    assert config.load().tools.web_fetch is True


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


async def test_a_chat_with_the_setting_has_web_search_as_a_read_tool_and_its_prompt_line() -> (
    None
):
    agent = await _session_agent(web_search=True)
    try:
        assert _tools(agent)["web_search"] == "read"
        assert not agent.executor.requires_approval("web_search")
        assert NOTE in agent.chat.system_notes
    finally:
        await agent.close()


async def test_without_the_setting_there_is_no_tool_no_line() -> None:
    agent = await _session_agent()
    try:
        assert "web_search" not in _tools(agent)
        assert not any("Web search is on" in n for n in agent.chat.system_notes)
        assert not any("web_search" in n for n in agent.chat.system_notes)
    finally:
        await agent.close()


async def test_with_the_browser_on_the_search_may_use_it() -> None:
    agent = await _session_agent(web_search=True, browser_driver="safari-automation")
    try:
        assert {"web_search", "browser_open"} <= set(_tools(agent))
        assert any(
            n.startswith(NOTE) and "Safari" in n for n in agent.chat.system_notes
        )
    finally:
        await agent.close()


async def test_the_search_does_not_need_web_fetch_and_does_not_turn_it_on() -> None:
    agent = await _session_agent(web_search=True, web_fetch=False)
    try:
        assert "web_search" in _tools(agent) and "web_fetch" not in _tools(agent)
    finally:
        await agent.close()


@pytest.mark.parametrize("on", [True, False])
async def test_the_console_chats_follow_the_setting(on: bool) -> None:
    cfg = _cfg(search=on)
    side = await registry.default_builder(cfg, str(Path.cwd()), None, "s1")
    main = await front.build_front_agent(cfg, str(Path.cwd()), "s2", tools=_NoTools())
    try:
        for agent in (side, main):
            assert ("web_search" in _tools(agent)) is on
            assert (NOTE in agent.chat.system_notes) is on
    finally:
        await side.close()
        await main.close()


async def test_the_terminal_passes_the_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    from localmelo.support.cli import launch
    from localmelo.support.cli.agents import AgentFactory, url_backend

    seen: list[Any] = []

    async def fake_build(**kwargs: Any) -> Any:
        seen.append(kwargs.get("web_search"))
        return object()

    monkeypatch.setattr("localmelo.support.cli.agents.build_session_agent", fake_build)
    for on in (True, False):
        factory = AgentFactory(
            backend=url_backend("http://127.0.0.1:9/v1", "m", source="saved"),
            cfg=None,
            options=AgentOptions(),
            workspace="/w",
            web_search=on,
        )
        factory._live.clear()
        await factory.build()
    assert seen == [True, False]
    assert "web_search=session_config.web.search is True" in inspect.getsource(launch)


async def test_with_the_switch_off_nothing_is_built_and_no_request_goes_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from localmelo.support.web import search_tool
    from localmelo.support.web.fetch import WebFetcher

    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("a search was set up or sent with the switch off")

    monkeypatch.setattr(search_tool.WebSearchProvider, "__init__", refuse)
    monkeypatch.setattr(WebFetcher, "fetch", refuse)
    cfg = _cfg()
    agent = await _session_agent()
    side = await registry.default_builder(cfg, str(Path.cwd()), None, "s1")
    main = await front.build_front_agent(cfg, str(Path.cwd()), "s2", tools=_NoTools())
    try:
        for built in (agent, side, main):
            assert "web_search" not in _tools(built)
    finally:
        await agent.close()
        await side.close()
        await main.close()


async def test_with_the_switch_on_building_a_chat_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from localmelo.support.web.fetch import WebFetcher

    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("a request went out while a chat was built")

    monkeypatch.setattr(WebFetcher, "fetch", refuse)
    agent = await _session_agent(web_search=True)
    try:
        assert "web_search" in _tools(agent)
    finally:
        await agent.close()


def test_direct_runs_never_get_it() -> None:
    import argparse
    import inspect

    default = inspect.signature(session_agents.build_session_agent).parameters[
        "web_search"
    ]
    assert default.default is False
    config.save(_cfg(search=True))
    args = argparse.Namespace(mcp_config=None)
    assert cli._direct_tool_providers(argparse.ArgumentParser(), args) == []


# ── How a call shows ──


def test_a_call_reads_as_a_search_for_its_words() -> None:
    text = activity_text("web_search", {"query": "garden tomato blight"})
    assert (text.text, text.icon) == (
        "Searching the web for “garden tomato blight”",
        "web",
    )


def test_a_call_shows_no_secret_and_cuts_long_words() -> None:
    secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"
    shown = activity_text("web_search", {"query": f"why does {secret} fail"}).text
    assert secret not in shown
    long = activity_text("web_search", {"query": "w" * 500}).text
    assert len(long) < 100


def test_a_call_with_no_words_reads_as_a_plain_search() -> None:
    assert activity_text("web_search", {}).text.startswith("Searching the web")
    assert activity_text("web_search", {"query": 5}).text.startswith(
        "Searching the web"
    )


def test_the_call_preview_is_the_words() -> None:
    assert (
        call_preview("web_search", {"query": "  tomato blight \n"}) == "tomato blight"
    )
    assert call_preview("web_search", {}) == "{}"  # the generic fallback
    assert request_words("web_search", {"query": "tomato blight"}) == (
        "Search the web for “tomato blight”"
    )


# ── Every surface says it in both languages ──

STATIC = Path(__file__).resolve().parents[2] / "support" / "gateway" / "static"


@pytest.mark.parametrize("lang", ["en", "zh-Hans"])
def test_the_search_words_exist_in_both_languages(lang: str) -> None:
    table = json.loads((STATIC / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))
    for key in (
        "steps.now.search",
        "steps.done.search",
        "steps.ask.search",
        "settings.webSearch.title",
        "settings.webSearch.hint",
        "settings.webSearch.on",
        "settings.webSearch.off",
    ):
        assert table.get(key), (lang, key)
    for key in ("steps.now.search", "steps.done.search", "steps.ask.search"):
        assert "{what}" in table[key], (lang, key)
    assert (
        "leave" in table["settings.webSearch.hint"]
        or "离开" in table["settings.webSearch.hint"]
    )
