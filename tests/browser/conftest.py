"""No browsing test may start the real ``safaridriver``: it would open Safari."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _never_the_real_safaridriver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "LOCALMELO_SAFARIDRIVER", "/nonexistent/not-the-real-safaridriver"
    )
