"""Fixtures for every test."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point a test at a real memory dir.

    With ``LOCALMELO_PERSIST_MEMORY`` set, as for the benchmark, every
    Agent built without ``AgentOptions.memory_dir`` keeps its history in
    SQLite, in ``LOCALMELO_MEMORY_DIR`` or ``~/.cache/localmelo/memory``.
    A test that wants persistence sets the variables itself, or passes
    ``memory_dir``. Some test modules have a copy of this fixture, which
    takes its place there.
    """
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)
