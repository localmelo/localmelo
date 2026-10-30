"""The terminal side of browsing: the site question's panel and ``/browser``."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import ApprovalRequest
from localmelo.support import config
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.cli.approvals import TerminalApprover, panel_text, session_option
from localmelo.support.cli.output import RecordingOut
from localmelo.support.cli.render import TurnRenderer
from localmelo.support.cli.session import CliSession
from localmelo.support.cli.theme import UNICODE
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.browser.fakes import FakeDriver, page
from localmelo.tests.cli.test_session import FakeApp, FakeFactory, _until
from localmelo.tests.sessions.helpers import ScriptedLLM, answer


def _ask(site: str = "example.com") -> ApprovalRequest:
    return ApprovalRequest(
        request_id="r1",
        run_id="run",
        call_id="c1",
        tool_name=f"browser:https://{site}",
        arguments={"site": site, "origin": f"https://{site}", "access": "read"},
        preview=f"open {site} in Safari and read it",
        workspace_root="/home/me/work",
        tier="network",
    )


def test_the_site_question_is_asked_in_plain_words() -> None:
    text = panel_text(_ask(), width=60)
    assert text.title == "Open a website?"
    assert text.header == "example.com · network · read only"
    assert text.body[0] == ("", "melo wants to open example.com in Safari and read it.")
    said = " ".join(" ".join(row for _, row in text.body[1:]).split())
    assert "your sign-ins are not used" in said
    assert (
        panel_text(_ask(), width=60, show_tier=False).header
        == "example.com · read only"
    )
    assert (
        session_option(_ask()) == "Yes, and allow example.com for the rest of this chat"
    )
    other = ApprovalRequest("r", "run", "c", "mcp__x__y", {}, "", None)
    assert (
        session_option(other) == "Yes, and allow every mcp__x__y call in this session"
    )


def test_a_site_name_with_control_characters_is_shown_escaped() -> None:
    text = panel_text(_ask("ex\x1b[31mample.com"), width=60)
    assert "\x1b" not in text.header and text.escaped


class BrowsingFactory(FakeFactory):
    """Builds Agents that carry the browser tools while browsing is on."""

    def __init__(self, root: Path, driver: FakeDriver) -> None:
        super().__init__(root, {"m1": [answer("ok")]})
        self.driver = driver

    async def build(self, model: str | None = None, *, url: str | None = None) -> Agent:
        providers = (
            [] if self.browser_driver == "off" else [BrowserToolProvider(self.driver)]
        )
        agent = Agent(
            llm=ScriptedLLM([answer("ok")]),
            options=AgentOptions(
                workspace_root=str(self.root), record_answers=True, mode="interactive"
            ),
            tool_providers=providers,
        )
        await agent.start()
        self._live.append(agent)
        return agent


@pytest.fixture
async def chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "work").mkdir()
    factory = BrowsingFactory(
        tmp_path / "work", FakeDriver(page("https://example.com/", "x"))
    )
    store = SessionStore(tmp_path / "sessions.db", writer="cli")
    await store.open()
    agent = await factory.build()
    out, approver = RecordingOut(80), TerminalApprover()
    renderer = TurnRenderer(out, symbols=UNICODE, backend=lambda: factory.backend)
    session = CliSession(
        store=store,
        factory=factory,
        agent=agent,
        renderer=renderer,
        approver=approver,
        mode="ask",
        out=out,
    )
    app = FakeApp()
    session.attach(app)  # type: ignore[arg-type]
    yield session, out, app, factory
    await session.shutdown()


def _tools(session: CliSession) -> list[str]:
    return [t.name for t in session.agent.list_tools() if t.name.startswith("browser_")]


async def test_browser_says_it_is_off_and_a_list_turns_it_on(chat: Any) -> None:
    session, out, app, factory = chat
    app.answers = ["y"]  # the first row: turn browsing on
    session.enter("/browser")
    await _until(
        lambda: factory.browser_driver == "safari-automation"
        and session.state == "idle"
    )
    text = out.text()
    assert "Browsing is off. /browser on lets melo read web pages in Safari." in text
    assert app.asked[0][:2] == (
        "Browsing in Safari",
        ["Turn browsing on", "Leave it off"],
    )
    assert "Browsing is on." in text and "Safari's automation helper answers." in text
    assert (
        "Sites allowed in this chat: none yet; melo asks before it opens a site" in text
    )
    assert "(no saved settings)" in text
    assert len(_tools(session)) == 8
    assert factory.live_agents() == [session.agent]  # the old Agent was closed


async def test_browser_on_and_off_save_the_choice_when_there_is_a_config(
    chat: Any,
) -> None:
    session, out, _, factory = chat
    config.save(config.Config(chat_backend="vllm", embedding_backend="none"))
    session.enter("/browser on")
    await _until(lambda: _tools(session) != [] and session.state == "idle")

    def saved() -> dict[str, Any]:
        with open(config.config_path(), "rb") as handle:
            return tomllib.load(handle)

    assert saved()["browser"] == {"driver": "safari-automation"}
    assert config.load().browser.driver == "safari-automation"
    session.enter("/browser off")
    await _until(lambda: _tools(session) == [] and session.state == "idle")
    assert "browser" not in saved() and factory.browser_driver == "off"
    assert "Browsing is off." in out.text()
    session.enter("/browser off")
    await _until(lambda: "Browsing is already off." in out.text())


async def test_browser_names_what_to_turn_on_when_safari_is_not_ready(
    chat: Any,
) -> None:
    session, out, _, factory = chat
    factory.driver.setup_problem = "Turn on Allow remote automation in Safari."
    session.enter("/browser on")
    await _until(lambda: "⚠ Turn on Allow remote automation in Safari." in out.text())
    assert factory.browser_driver == "safari-automation"  # on; the tools say the same


async def test_browser_lists_the_sites_this_chat_allowed(chat: Any) -> None:
    session, out, app, factory = chat
    session.enter("/browser on")
    await _until(lambda: _tools(session) != [] and session.state == "idle")
    session.approvals._policy._granted.update(
        {"browser:https://b.org", "browser:http://a.net:8080"}
    )
    app.answers = ["n"]
    session.enter("/browser")
    await _until(lambda: "Sites allowed in this chat: a.net:8080, b.org" in out.text())
    assert app.asked[-1][1] == ["Turn browsing off", "Leave it on"]


async def test_a_mistyped_browser_command_says_how(chat: Any) -> None:
    session, out, _, _ = chat
    session.enter("/browser maybe")
    await _until(lambda: "✗ Use /browser, /browser on or /browser off." in out.text())


async def test_a_failed_switch_leaves_browsing_as_it_was(chat: Any) -> None:
    session, out, _, factory = chat
    factory.fail.add("m1")
    session.enter("/browser on")
    await _until(
        lambda: "Could not start" in out.text() or factory.browser_driver == "off"
    )
    await _until(lambda: session.state == "idle")
    assert factory.browser_driver == "off"


async def test_auto_mode_allows_everything_but_still_asks_about_a_website() -> None:
    from localmelo.support.cli.approvals import Approvals

    asked: list[str] = []

    async def ask(request: ApprovalRequest) -> Any:
        from localmelo.melo.contracts.approvals import ApprovalDecision

        asked.append(request.tool_name)
        return ApprovalDecision(True, remember=True)

    approvals = Approvals(ask, "auto")
    policy = approvals.policy()
    assert policy is not None
    shell = ApprovalRequest("r", "run", "c", "shell_exec", {"command": "ls"}, "", None)
    assert (await policy.decide(shell)).origin == "policy" and asked == []
    site = _ask()
    assert (await policy.decide(site)).allow and asked == [
        "browser:https://example.com"
    ]
    assert (
        await policy.decide(site)
    ).origin == "session"  # the chat's grant holds in auto mode
    approvals = Approvals(ask, "yolo")
    assert approvals.policy() is None  # --yolo asks about nothing


async def test_a_new_chat_starts_with_no_page_approved(chat: Any) -> None:
    session, out, _, factory = chat
    session.enter("/browser on")
    await _until(lambda: _tools(session) != [] and session.state == "idle")
    provider = next(
        p for p in session.agent._tool_providers if isinstance(p, BrowserToolProvider)
    )
    tools = {t.definition.name: t for t in await provider.start()}
    await tools["browser_open"].function(url="https://example.com/")
    assert provider._origins and provider._snaps
    session.enter("/new")
    await _until(lambda: "New chat" in out.text() and session.state == "idle")
    assert not provider._origins and not provider._snaps and provider._tab is None
