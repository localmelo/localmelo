"""Fixtures for the session tests: a temporary HOME, and no store left open.

With HOME in a temporary directory, the default sessions files and the CLI
host id resolve there, so no test opens ``~/.cache/localmelo``. Every
store a test builds is closed after it, also when the test fails: an open
aiosqlite connection keeps a thread that would hold up the process.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.sessions.store import SessionStore


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


@pytest.fixture(autouse=True)
def _close_stores(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    stores: list[SessionStore] = []
    original = SessionStore.__init__

    def tracked(self: SessionStore, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        stores.append(self)

    monkeypatch.setattr(SessionStore, "__init__", tracked)
    yield
    for store in stores:
        conn, store._conn = store._conn, None
        if conn is not None:
            conn.stop()
        store._release_owner()
