"""Fixtures and run-wide guards for every test."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from localmelo.tests._cache_guard import RealCacheGuard, real_cache_dir

if TYPE_CHECKING:
    from localmelo.tests.fixtures.melo_serve import FakeServe
    from localmelo.tests.fixtures.setup_tools import Fakes

# The home the test process started with, before any fixture redirects it.
_REAL_HOME = os.environ.get("HOME")
# Refuses writes to, and resolutions of, the user's real cache.
_REAL_CACHE_GUARD = RealCacheGuard(real_cache_dir())
# The setup plan's own functions, kept before ``_no_setup_plan`` replaces them.
_REAL_SETUP: dict[str, Any] = {}


@pytest.fixture(autouse=True)
def _no_setting_overrides() -> Iterator[None]:
    """``-X`` settings belong to the process: no test inherits another's."""
    yield
    from localmelo.support import config

    config.set_overrides({})


@pytest.fixture(autouse=True)
def fake_keychain(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """No test reaches the real keychain: ``security`` is an in-memory stand-in.

    A test that reads or changes a saved key takes this fixture and works with
    ``fake_keychain.items``.
    """
    from localmelo.support import keychain
    from localmelo.tests.fake_keychain import FakeSecurity

    fake = FakeSecurity()
    monkeypatch.setattr(keychain, "_security", fake)
    monkeypatch.setattr(keychain, "available", lambda: True)
    keychain._cache.clear()
    yield fake
    keychain._cache.clear()


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
def _no_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches the desktop control helper or asks which app is in front.

    The helper would read the person's real screen. A test of other apps'
    windows builds a ``DesktopWatch`` over the fake helper and a scripted front
    app itself.
    """
    from localmelo.support.browser import desktop

    async def nobody_in_front() -> None:
        return None

    monkeypatch.setenv(desktop.HELPER_ENV, "")
    monkeypatch.setattr(desktop, "front_app", nobody_in_front)


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No test opens a real browser, or takes the run for a person at a terminal.

    A test of opening fakes ``console_link.can_open`` and ``webbrowser.open`` itself.
    """
    import webbrowser

    from localmelo.support import console_link

    opened: list[str] = []

    def refuse(url: str, *args: object, **kwargs: object) -> bool:
        opened.append(url)
        return False

    monkeypatch.setattr(console_link, "can_open", lambda: False)
    monkeypatch.setattr(webbrowser, "open", refuse)
    yield
    assert not opened, f"a test opened a browser: {opened}"


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
def melo_serve_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """melo_serve's home is an empty temporary folder: no test reads this Mac's own.

    The state file and the control socket of the melo_serve that may run here
    are never read or connected to, and no test finds this Mac's own
    ``melo-serve`` command either: ``melo_serve.command()`` and
    ``melo_serve.refusal()`` (of ``melo-serve``) are None. A test
    writes a fake ``state.json`` in the folder (``tests/fixtures/melo_serve.py``)
    and asks for ``melo_serve_socket`` to make it running.
    """
    from localmelo.support import melo_serve

    home = tmp_path_factory.mktemp("melo-serve")
    monkeypatch.setenv("MELO_SERVE_HOME", str(home))
    monkeypatch.delenv("MELO_SERVE_SOCKET", raising=False)
    monkeypatch.setattr(melo_serve, "command", lambda: None)
    refused = melo_serve.refusal
    monkeypatch.setattr(
        melo_serve,
        "refusal",
        lambda name=melo_serve.COMMAND, *args, **kwargs: (
            None if name == melo_serve.COMMAND else refused(name, *args, **kwargs)
        ),
    )
    return home


@pytest.fixture(autouse=True)
def _no_setup_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first run's setup plan is unavailable: no test runs ``uv`` or fetches melo_serve.

    A test of the setup screen fakes ``run.plan`` itself.
    """
    from localmelo.support.server_setup import run

    _REAL_SETUP.setdefault("plan", run.plan)
    _REAL_SETUP.setdefault("installed_version", run.installed_version)
    monkeypatch.setattr(
        run, "plan", lambda timeout=0.0, pick="": run.Plan(False, "no plan in tests")
    )
    monkeypatch.setattr(run, "installed_version", lambda: "")


@pytest.fixture
def real_setup_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """This test runs the real setup plan, over the fake ``uv`` and ``melo-serve`` of ``setup_fakes``."""
    from localmelo.support.server_setup import run

    monkeypatch.setattr(run, "plan", _REAL_SETUP["plan"])
    monkeypatch.setattr(run, "installed_version", _REAL_SETUP["installed_version"])


@pytest.fixture
def setup_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    melo_serve_home: Path,
    melo_serve_socket: Path,
    real_setup_plan: None,
) -> Fakes:
    """Fake ``uv`` and ``melo-serve`` on PATH, and a melo_serve that runs, as ``use`` leaves it.

    The first-run setup job and plan run for real over them; nothing is installed,
    downloaded or started.
    """
    from localmelo.tests.fixtures.melo_serve import role, write_state
    from localmelo.tests.fixtures.setup_tools import Fakes

    fake = Fakes(tmp_path / "fake", monkeypatch)
    write_state(
        melo_serve_home,
        chat=role("Qwen3 8B", "omlx", "oMLX"),
        embedding=role("Qwen3-Embedding-0.6B", "omlx", "oMLX"),
    )
    return fake


@pytest.fixture
def melo_serve_socket(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A control socket that accepts connections, so a fake ``state.json`` is running.

    Without it nothing answers on the socket and no state is running.
    """
    from localmelo.tests.fixtures.melo_serve import listening

    with listening() as path:
        monkeypatch.setenv("MELO_SERVE_SOCKET", str(path))
        yield path


@pytest.fixture
def serve_fake(
    melo_serve_home: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[FakeServe]:
    """A running fake melo_serve that serves the 27B and answers ``/picks`` and ``/switch``."""
    from localmelo.tests.fixtures.melo_serve import FakeServe, listening

    fake = FakeServe(melo_serve_home)
    fake.write()
    with listening(fake) as path:
        monkeypatch.setenv("MELO_SERVE_SOCKET", str(path))
        yield fake


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
