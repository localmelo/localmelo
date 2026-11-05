"""No browsing test may start the real ``safaridriver`` (it would open Safari), the real Chrome, or look a name up."""

from __future__ import annotations

import socket

import pytest

from localmelo.support.browser import pagescript, provider
from localmelo.tests.browser.fakes import FakeDns


@pytest.fixture(autouse=True)
def _never_the_real_safaridriver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "LOCALMELO_SAFARIDRIVER", "/nonexistent/not-the-real-safaridriver"
    )


@pytest.fixture(autouse=True)
def _never_the_real_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    """A driver built without a command starts this, which is no Chrome; the real-Chrome smoke test passes its own."""
    monkeypatch.setenv("LOCALMELO_CHROME", "/nonexistent/not-the-real-chrome")


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


@pytest.fixture(autouse=True)
def _no_wait_after_an_act(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake page is quiet at once: melo looks once after an act instead of waiting."""
    monkeypatch.setattr(pagescript, "LOOK_STEP_S", 0.0)
    monkeypatch.setattr(pagescript, "QUIET_S", 0.0)
    monkeypatch.setattr(pagescript, "LOOK_CAP_S", 0.0)


@pytest.fixture(autouse=True)
def _no_wait_for_a_covered_element(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake page does not move under an element, so melo need not give it a moment."""
    monkeypatch.setattr(pagescript, "COVER_AGAIN_S", 0.0)
