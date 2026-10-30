"""Fixtures and run-wide guards for every test."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

from localmelo.tests._cache_guard import RealCacheGuard, real_cache_dir

# The home the test process started with, before any fixture redirects it.
_REAL_HOME = os.environ.get("HOME")
# Refuses writes to, and resolutions of, the user's real cache.
_REAL_CACHE_GUARD = RealCacheGuard(real_cache_dir())


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


@pytest.fixture(autouse=True)
def _no_env_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment set how long history keeps reasoning.

    A test that wants ``LOCALMELO_THINKING_RETENTION_DAYS`` sets it itself.
    A fixture of its own, since some test modules replace
    ``_no_env_persistence`` with a copy of the same name.
    """
    monkeypatch.delenv("LOCALMELO_THINKING_RETENTION_DAYS", raising=False)


@pytest.fixture(autouse=True)
def _forget_servers() -> None:
    """Start every test knowing nothing about which server is oMLX.

    A model listing remembers, per URL, whether the server named oMLX as its
    owner; the label of a later banner reads it. Without this a test that
    lists an oMLX server would rename the same URL in another test.
    """
    from localmelo.support.backends import discovery

    discovery.forget_servers()


@pytest.fixture(autouse=True)
def _isolated_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never let a test resolve ``~`` to the user's own home.

    ``~/.cache/localmelo`` then resolves under a temporary directory: the
    config, the memory stores, the sessions files, the CLI history and log,
    and the gateway's call log. The home is its own directory, not inside
    the test's ``tmp_path``, whose contents tests check. The redirect
    happens once, so a directory's own fixture or a test that sets HOME
    keeps its choice.
    """
    if os.environ.get("HOME") != _REAL_HOME:
        return
    home = tmp_path_factory.mktemp("user-home")
    monkeypatch.setenv("HOME", str(home))


@pytest.fixture(autouse=True)
def _guard_real_cache(request: pytest.FixtureRequest) -> Iterator[None]:
    """Refuse the user's real home and cache while a test runs.

    Writing under the real cache raises, and so does a ``~`` path that
    resolves to the real home, which a test should not use. A test that
    needs the real home on purpose is marked ``allow_real_cache``; writing
    the real cache is never allowed.
    """
    _REAL_CACHE_GUARD.allow_real_home = (
        request.node.get_closest_marker("allow_real_cache") is not None
    )
    _REAL_CACHE_GUARD.watch_home = True
    try:
        yield
    finally:
        _REAL_CACHE_GUARD.watch_home = False
        _REAL_CACHE_GUARD.allow_real_home = False


def pytest_sessionstart(session: Any) -> None:
    _REAL_CACHE_GUARD.install()


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    _REAL_CACHE_GUARD.uninstall()
