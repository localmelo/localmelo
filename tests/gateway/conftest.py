"""Fixtures for the gateway tests: a temporary HOME.

The console opens ``~/.cache/localmelo/gateway/sessions.db`` when the
gateway starts, and the web workspace's call log and history also live
under ``~/.cache/localmelo``. A temporary HOME keeps every gateway test
away from the user's cache, and the tracker closes any store a test left
open so an aiosqlite thread cannot hold up the process.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.gateway.console.front import Front
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
    store_init = SessionStore.__init__

    def tracked_store(self: SessionStore, *args: Any, **kwargs: Any) -> None:
        store_init(self, *args, **kwargs)
        stores.append(self)

    monkeypatch.setattr(SessionStore, "__init__", tracked_store)
    yield
    for store in stores:
        conn, store._conn = store._conn, None
        if conn is not None:
            conn.stop()
        store._release_owner()


# The real request for a title, kept for the one test that checks it.
REAL_ASK_TITLE = Front._ask_title


@pytest.fixture(autouse=True)
def _no_title_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """A task's title is asked of the model only in production.

    Without this, a test that starts a task from a long request would build a
    provider from its config and send a request; a test that wants a model's
    title passes ``ConsoleOptions(titler=…)``.
    """

    async def no_model(self: Front, request: str) -> str:
        raise RuntimeError("a test sends no title request")

    monkeypatch.setattr(Front, "_ask_title", no_model)
