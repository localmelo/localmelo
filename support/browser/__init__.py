"""Browsing for melo: read, follow and act on web pages in a browser, with the person's say.

``provider_for`` builds the tool provider that a session registers when
``[browser] driver`` names a driver (``off``, the default, registers nothing):
``safari-automation`` (Safari's automation windows, which come to the front) or
``chrome`` (Google Chrome in the background, with no window).
``configured_driver`` is the driver a chat passes: the setting, or ``off`` when
it names no driver. ``driver_for_on`` is the one turning browsing on picks.
"""

from __future__ import annotations

import logging
import os
import shlex
from typing import TYPE_CHECKING

from localmelo.support.browser.desktop import watch_for
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.config import BROWSER_DRIVERS as DRIVERS

if TYPE_CHECKING:
    from localmelo.support.config import Config


def configured_driver(cfg: Config) -> str:
    """``[browser] driver`` of *cfg*; ``off``, with a warning, when it names no driver."""
    driver = cfg.browser.driver
    if driver in DRIVERS:
        return driver
    logging.getLogger(__name__).warning(
        "Invalid [browser] driver %r; browsing is off", driver
    )
    return "off"


def driver_for_on() -> str:
    """The driver turning browsing on picks: ``chrome``, which keeps sign-ins, when Chrome is installed; else ``safari-automation``."""
    from localmelo.support.browser import chrome

    return "chrome" if chrome.chrome_installed() else "safari-automation"


def provider_for(
    driver: str,
    *,
    keep_tabs: bool = False,
    app_windows: bool = False,
    opens_apps: bool = True,
    live: bool = True,
    live_fps: float = 15.0,
    chat_id: str | None = None,
) -> BrowserToolProvider | None:
    """The browser tools for *driver*; None for ``off``. ``ValueError`` for an unknown name.

    With *keep_tabs* (``[browser] keep_tabs``) the tabs melo opened are left
    open when the chat ends or goes idle. With *app_windows* (``[browser]
    app_windows``) melo watches for windows other apps open after its acts
    (:mod:`localmelo.support.browser.desktop`). Without *opens_apps* a page that
    asks for another app is told no, and nobody is asked. With *live* (``[browser] live``) the
    person can watch the Chrome browser (:mod:`localmelo.support.browser.live`), its
    pictures held to *live_fps* a second. *chat_id* names the chat's work folder
    (:mod:`localmelo.melo.contracts.work_folder`); without one the folder is the
    run's own and goes with it.
    """
    if driver == "off":
        return None
    if driver == "safari-automation":
        from localmelo.support.browser.safari import SafariAutomationDriver

        # LOCALMELO_SAFARIDRIVER replaces the program's command line, so tests
        # and screenshots can run a fake and never start the real Safari.
        command, *args = shlex.split(os.environ.get("LOCALMELO_SAFARIDRIVER", "")) or [
            "/usr/bin/safaridriver"
        ]
        return BrowserToolProvider(
            SafariAutomationDriver(command=command, args=(*args, "--mcp")),
            keep_tabs=keep_tabs,
            desktop=watch_for(app_windows),
            opens_apps=opens_apps,
            chat_id=chat_id,
        )
    if driver == "chrome":
        from localmelo.support.browser.chrome import ChromeBackgroundDriver
        from localmelo.support.browser.local_apps import LocalRule

        # LOCALMELO_CHROME replaces Chrome's command line (read when the driver
        # starts), so tests run a fake and never the real Chrome. A page reaches
        # a listed sign-in app on this Mac and nothing else there (local_apps).
        return BrowserToolProvider(
            ChromeBackgroundDriver(local=LocalRule()),
            keep_tabs=keep_tabs,
            desktop=watch_for(app_windows, driver),
            opens_apps=opens_apps,
            live=live,
            live_fps=live_fps,
            chat_id=chat_id,
        )
    raise ValueError(f"unknown browser driver {driver!r}; use one of {DRIVERS}")


__all__ = [
    "DRIVERS",
    "BrowserToolProvider",
    "configured_driver",
    "driver_for_on",
    "provider_for",
]
