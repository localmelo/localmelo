"""The gateway's login service: what ``melo serve install`` writes and says.

``launchctl`` is a fake and the plist folder is a temporary one, so nothing
here touches the real service.
"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from localmelo.support import gateway_endpoint
from localmelo.support.gateway import daemon


class FakeLaunchctl:
    """A launchd with one service: loaded or not, and a record of what was asked."""

    def __init__(self) -> None:
        self.loaded = False
        self.calls: list[str] = []

    def __call__(self, *args: str) -> subprocess.CompletedProcess[str]:
        self.calls.append(args[0])
        if args[0] == "bootstrap":
            self.loaded = True
        elif args[0] == "bootout":
            self.loaded = False
        asks_the_service = args[0] in ("print", "kickstart")
        code = 113 if asks_the_service and not self.loaded else 0
        return subprocess.CompletedProcess(args, code, "state = running\npid = 7\n", "")


@pytest.fixture
def launchd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeLaunchctl:
    fake = FakeLaunchctl()
    monkeypatch.setattr(daemon, "_launchctl", fake)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(daemon.AGENTS_ENV, str(tmp_path / "agents"))
    monkeypatch.setattr(gateway_endpoint, "live", lambda *a, **k: None)
    return fake


def test_the_plist_restarts_a_crash_but_not_a_clean_stop(
    launchd: FakeLaunchctl,
) -> None:
    plist = plistlib.loads(daemon.render(9000))
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["RunAtLoad"] is True
    assert plist["ProgramArguments"][-2:] == ["--port", "9000"]


def test_install_says_what_it_did_and_a_second_run_changes_nothing(
    launchd: FakeLaunchctl, tmp_path: Path
) -> None:
    first = daemon.install()
    assert first.line == "installed ai.localmelo.gateway (starts at login)"
    assert Path(first.path) == tmp_path / "agents" / "ai.localmelo.gateway.plist"
    assert launchd.calls == ["print", "bootstrap"]
    launchd.calls.clear()
    again = daemon.install()
    assert again.line == "already installed, skipped"
    assert launchd.calls == ["print"]  # no bootout: the gateway keeps running


def test_install_with_another_port_restarts_the_gateway(launchd: FakeLaunchctl) -> None:
    daemon.install()
    launchd.calls.clear()
    changed = daemon.install(9000)
    assert changed.line == "updated ai.localmelo.gateway; the gateway restarted"
    assert launchd.calls == ["print", "bootout", "bootstrap"]


def test_install_next_to_a_gateway_that_is_not_the_services_waits_for_login(
    launchd: FakeLaunchctl, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = gateway_endpoint.Endpoint(1, "127.0.0.1", 8401, "", "id", 0.0)
    monkeypatch.setattr(gateway_endpoint, "live", lambda *a, **k: live)
    done = daemon.install()
    assert done.line == "installed ai.localmelo.gateway; it takes over at next login"
    assert Path(done.path).exists()
    assert "bootstrap" not in launchd.calls  # two gateways never run


def test_a_launchctl_that_hangs_is_a_failed_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def hang(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("launchctl", 1)

    monkeypatch.setattr(subprocess, "run", hang)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(daemon.AGENTS_ENV, str(tmp_path / "agents"))
    with pytest.raises(RuntimeError, match="timed out"):
        daemon.install()


def test_start_kicks_a_loaded_service_and_loads_one_that_is_not(
    launchd: FakeLaunchctl,
) -> None:
    daemon.install()
    launchd.calls.clear()
    daemon.start()
    assert launchd.calls == ["print", "kickstart"]
    launchd.loaded = False
    launchd.calls.clear()
    daemon.start()
    assert launchd.calls == ["print", "bootstrap"]


def test_uninstall_removes_the_plist_and_unloads_the_service(
    launchd: FakeLaunchctl,
) -> None:
    path = Path(daemon.install().path)
    assert daemon.uninstall() is True
    assert not path.exists() and launchd.loaded is False
    assert daemon.uninstall() is False
