"""``/browser`` in the terminal: whether melo may read web pages in Safari.

``/browser`` shows the state and checks Safari's setup, then offers a list to
turn browsing on or off; ``/browser on`` and ``/browser off`` do it at once.
Switching builds a new Agent (the tools are registered when it starts), keeps
the chat, and saves ``[browser] driver`` in the config file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from localmelo.support import config
from localmelo.support.browser.driver import BrowserError, BrowserSetupError
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.browser.sites import PREFIX

if TYPE_CHECKING:
    from localmelo.support.cli.session import CliSession

ON = "safari-automation"


def _provider(session: CliSession) -> BrowserToolProvider | None:
    for provider in session.agent._tool_providers:
        if isinstance(provider, BrowserToolProvider):
            return provider
    return None


async def status(session: CliSession) -> None:
    """Print whether browsing is on, whether Safari answers, and the sites allowed."""
    out = session.out
    if session.factory.browser_driver == "off":
        out.print(
            session._text(
                "Browsing is off. /browser on lets melo read web pages in Safari."
            )
        )
        return
    out.print(
        session._text(
            "Browsing is on: melo reads pages in Safari's automation windows."
        )
    )
    provider = _provider(session)
    if provider is not None:
        try:
            await provider.check()
        except BrowserSetupError as exc:
            out.print(session._text(f"⚠ {exc}", "warn"))
        except BrowserError as exc:
            out.print(session._text(f"✗ {exc}", "error"))
        else:
            out.print(session._text("Safari's automation helper answers.", "ok"))
    sites = sorted(
        grant[len(PREFIX) :].split("://", 1)[-1]
        for grant in session.approvals.allowed_tools
        if grant.startswith(PREFIX)
    )
    allowed = ", ".join(sites) or "none yet; melo asks before it opens a site"
    out.print(session._text(f"Sites allowed in this chat: {allowed}", "muted"))


async def command(session: CliSession, arguments: str) -> None:
    """``/browser [on|off]``."""
    word = arguments.strip().lower()
    if word in ("on", "off"):
        await switch(session, ON if word == "on" else "off")
        return
    if word:
        session.out.print(
            session._text("✗ Use /browser, /browser on or /browser off.", "error")
        )
        return
    await status(session)
    if session.app is None:
        return
    now_on = session.factory.browser_driver != "off"
    choices = (
        ["Turn browsing off", "Leave it on"]
        if now_on
        else ["Turn browsing on", "Leave it off"]
    )
    if await session.app.pick("Browsing in Safari", choices, default=1, escape=1) == 0:
        await switch(session, "off" if now_on else ON)


async def switch(session: CliSession, driver: str) -> None:
    """Move the chat to a new Agent with browsing on or off, and save the choice."""
    factory = session.factory
    if driver == factory.browser_driver:
        session.out.print(
            session._text(
                "Browsing is already " + ("off." if driver == "off" else "on."), "muted"
            )
        )
        return
    before, factory.browser_driver = factory.browser_driver, driver
    word = "off" if driver == "off" else "on"
    if not await session._switch(
        factory.backend.model, url=None, note=f"Browsing is {word}."
    ):
        factory.browser_driver = before
        return
    saved = ""
    try:
        cfg = config.load()
        if cfg.is_configured:
            cfg.browser.driver = driver
            config.save(cfg)
        else:
            saved = "for this chat only (no saved settings)"
    except (OSError, ValueError, config.ConfigError) as exc:
        saved = f"not saved: {exc}"
    if saved:
        session.out.print(session._text(f"Browsing is {word} {saved}.", "warn"))
    if driver != "off":
        await status(session)
