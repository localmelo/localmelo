"""``/browser`` in the terminal: whether melo may read web pages, and in which browser.

``/browser`` shows the state and checks the browser's setup, then offers a list
to turn browsing on or off or to switch browser; ``/browser on`` and
``/browser off`` do it at once, and ``/browser chrome`` and ``/browser safari``
choose Chrome in the background or Safari's automation windows. ``on`` is
Safari's windows, the default. Switching builds a new Agent (the tools are
registered when it starts), keeps the chat, and saves ``[browser] driver`` in
the config file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from localmelo.support import config
from localmelo.support.browser import chrome
from localmelo.support.browser.driver import BrowserError, BrowserSetupError
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.browser.sites import PREFIX

if TYPE_CHECKING:
    from localmelo.support.cli.session import CliSession

ON = "safari-automation"
CHROME = "chrome"
# The words /browser takes for a browser, and the driver each names.
DRIVER_WORDS = {"chrome": CHROME, "safari": ON}
USAGE = "✗ Use /browser, /browser on, /browser off, /browser chrome or /browser safari."


def _provider(session: CliSession) -> BrowserToolProvider | None:
    for provider in session.agent._tool_providers:
        if isinstance(provider, BrowserToolProvider):
            return provider
    return None


async def forget(session: CliSession) -> None:
    """A new chat starts with no site allowed, and the old chat's tabs are tidied."""
    provider = _provider(session)
    if provider is not None:
        await provider.end_chat()


async def status(session: CliSession) -> None:
    """Print whether browsing is on, whether Safari answers, and the sites allowed."""
    out = session.out
    driver = session.factory.browser_driver
    if driver == "off":
        out.print(
            session._text(
                "Browsing is off. /browser on lets melo read web pages in Safari; "
                "/browser chrome, in Chrome in the background."
            )
        )
        return
    background = driver == CHROME
    out.print(
        session._text(
            "Browsing is on: melo reads pages in Chrome, in the background (no "
            "window comes to the front)."
            if background
            else "Browsing is on: melo reads pages in Safari's automation windows."
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
            out.print(
                session._text(
                    (
                        "Chrome answers."
                        if background
                        else "Safari's automation helper answers."
                    ),
                    "ok",
                )
            )
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
        now = session.factory.browser_driver
        await switch(session, (now if now != "off" else ON) if word == "on" else "off")
        return
    if word in DRIVER_WORDS:
        await switch(session, DRIVER_WORDS[word])
        return
    if word:
        session.out.print(session._text(USAGE, "error"))
        return
    await status(session)
    if session.app is None:
        return
    now = session.factory.browser_driver
    other = ON if now == CHROME else CHROME
    switch_to = (
        "Switch to Chrome in the background"
        if other == CHROME
        else "Switch to Safari windows"
    )
    if now != "off":
        drivers = ["off", other, now]
        choices = ["Turn browsing off", switch_to, "Leave it on"]
    else:
        drivers = [ON, CHROME, "off"]
        choices = [
            "Turn browsing on: Safari windows (they come to the front)",
            "Turn browsing on: Chrome in the background",
            "Leave it off",
        ]
    last = len(choices) - 1
    picked = await session.app.pick("Browsing", choices, default=last, escape=last)
    if picked != last:
        await switch(session, drivers[picked])


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
    if driver == CHROME and not chrome.chrome_installed():
        # Nothing changes: the setting says what to do first.
        session.out.print(session._text(f"⚠ {chrome.MISSING_MESSAGE}", "warn"))
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
