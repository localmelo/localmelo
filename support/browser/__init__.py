"""Browsing for melo: read and follow web pages in a browser, with the person's say.

``provider_for`` builds the tool provider that a session registers when
``[browser] driver`` names a driver (``off``, the default, registers nothing).
"""

from __future__ import annotations

from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.config import BROWSER_DRIVERS as DRIVERS


def provider_for(driver: str) -> BrowserToolProvider | None:
    """The browser tools for *driver*; None for ``off``. ``ValueError`` for an unknown name."""
    if driver == "off":
        return None
    if driver == "safari-automation":
        from localmelo.support.browser.safari import SafariAutomationDriver

        return BrowserToolProvider(SafariAutomationDriver())
    raise ValueError(f"unknown browser driver {driver!r}; use one of {DRIVERS}")


__all__ = ["DRIVERS", "BrowserToolProvider", "provider_for"]
