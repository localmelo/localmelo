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

from localmelo.support.sessions.reader import SessionReader
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
    readers: list[SessionReader] = []
    store_init = SessionStore.__init__
    reader_init = SessionReader.__init__

    def tracked_store(self: SessionStore, *args: Any, **kwargs: Any) -> None:
        store_init(self, *args, **kwargs)
        stores.append(self)

    def tracked_reader(self: SessionReader, *args: Any, **kwargs: Any) -> None:
        reader_init(self, *args, **kwargs)
        readers.append(self)

    monkeypatch.setattr(SessionStore, "__init__", tracked_store)
    monkeypatch.setattr(SessionReader, "__init__", tracked_reader)
    yield
    for store in stores:
        conn, store._conn = store._conn, None
        if conn is not None:
            conn.stop()
        store._release_owner()
    for reader in readers:
        conn, reader._conn = reader._conn, None
        if conn is not None:
            conn.stop()
