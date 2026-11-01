"""Browsing for melo: read and follow web pages in a browser, with the person's say.

``provider_for`` builds the tool provider that a session registers when
``[browser] driver`` names a driver (``off``, the default, registers nothing).
``configured_driver`` is the driver a chat passes: the setting, or ``off`` when
it names no driver.
"""

from __future__ import annotations

import logging
import os
import shlex
from typing import TYPE_CHECKING

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


def provider_for(driver: str) -> BrowserToolProvider | None:
    """The browser tools for *driver*; None for ``off``. ``ValueError`` for an unknown name."""
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
            SafariAutomationDriver(command=command, args=(*args, "--mcp"))
        )
    raise ValueError(f"unknown browser driver {driver!r}; use one of {DRIVERS}")


__all__ = ["DRIVERS", "BrowserToolProvider", "configured_driver", "provider_for"]
