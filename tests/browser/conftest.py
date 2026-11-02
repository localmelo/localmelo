"""No browsing test may start the real ``safaridriver`` (it would open Safari) or look a name up."""

from __future__ import annotations

import socket

import pytest

from localmelo.support.browser import provider
from localmelo.tests.browser.fakes import FakeDns


@pytest.fixture(autouse=True)
def _never_the_real_safaridriver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "LOCALMELO_SAFARIDRIVER", "/nonexistent/not-the-real-safaridriver"
    )


@pytest.fixture(autouse=True)
def dns(monkeypatch: pytest.MonkeyPatch) -> FakeDns:
    """Name lookups answer from a table, never the network: a public address unless a test says."""
    fake = FakeDns()
    monkeypatch.setattr(socket, "getaddrinfo", fake.getaddrinfo)
    return fake


@pytest.fixture(autouse=True)
def _no_wait_for_the_page_to_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake page moves only when a test says so, so melo need not wait to see it stay."""
    monkeypatch.setattr(provider, "SETTLE_S", 0.0)
