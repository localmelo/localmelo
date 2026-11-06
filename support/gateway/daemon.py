"""The gateway's login service: a per-user LaunchAgent (``melo serve install``).

It starts at login and after a crash, not after a clean stop
(``KeepAlive.SuccessfulExit = false``), so ``melo serve stop`` keeps it stopped
until the next login or ``melo serve``. Every ``launchctl`` call goes through
:func:`_launchctl`, and ``LOCALMELO_LAUNCH_AGENTS`` moves the plist folder
(tests), so nothing else touches the real service.
"""

from __future__ import annotations

import os
import plistlib
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Final, Literal

from localmelo.support import gateway_endpoint

LABEL: Final = "ai.localmelo.gateway"
AGENTS_ENV: Final = "LOCALMELO_LAUNCH_AGENTS"  # moves the plist folder (tests)
LAUNCHCTL_TIMEOUT_S: Final = 15.0
# How long ``melo serve`` waits for a gateway it asked launchd to start.
START_WAIT_S: Final = 20.0
LOG_DIR: Final = "~/.localmelo/logs"
# The line that says where to look when the service does not come up.
LOG_HINT: Final = f"{LOG_DIR}/gateway.stderr.log"

Outcome = Literal["installed", "updated", "unchanged", "deferred"]


@dataclass(frozen=True)
class Installed:
    """What :func:`install` did: the outcome, the plist, and the line that says so."""

    outcome: Outcome
    path: str
    line: str


def plist_path() -> str:
    folder = os.environ.get(AGENTS_ENV) or "~/Library/LaunchAgents"
    return os.path.join(os.path.expanduser(folder), f"{LABEL}.plist")


def render(port: int = 8401) -> bytes:
    """The plist that runs ``python -m localmelo --serve --port PORT`` for this interpreter."""
    log_dir = os.path.expanduser(LOG_DIR)
    plist: dict[str, Any] = {
        "Label": LABEL,
        "ProgramArguments": [
            sys.executable,
            "-m",
            "localmelo",
            "--serve",
            "--port",
            str(port),
        ],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 3,
        "StandardOutPath": f"{log_dir}/gateway.stdout.log",
        "StandardErrorPath": f"{log_dir}/gateway.stderr.log",
    }
    return plistlib.dumps(plist)


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``launchctl``; a call that hangs or cannot run is a failed one (code 124 / 127)."""
    try:
        return subprocess.run(
            ["launchctl", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=LAUNCHCTL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", "launchctl timed out")
    except OSError as exc:
        return subprocess.CompletedProcess(args, 127, "", str(exc))


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _print() -> subprocess.CompletedProcess[str]:
    return _launchctl("print", f"{_domain()}/{LABEL}")


def _bootstrap(path: str) -> None:
    done = _launchctl("bootstrap", _domain(), path)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or "launchctl bootstrap failed")


def is_installed() -> bool:
    """Whether the plist is there (the service runs at login, loaded or not)."""
    return sys.platform == "darwin" and os.path.exists(plist_path())


def install(port: int = 8401) -> Installed:
    """Make the service match what :func:`render` writes now.

    ``unchanged``: the same plist is loaded, nothing was touched (the gateway
    keeps running). ``installed``: new, or only needed loading. ``updated``: the
    plist differed, so the gateway restarted. ``deferred``: a gateway that is
    not the service's own is live (the app's, or a foreground ``melo serve``),
    so the plist is written and not loaded: two gateways never run, and
    launchd loads it at the next login.
    """
    if sys.platform != "darwin":
        raise RuntimeError("daemon install only supported on macOS (launchd)")
    path = plist_path()
    wanted = render(port)
    try:
        with open(path, "rb") as file:
            current: bytes | None = file.read()
    except OSError:
        current = None
    loaded = _print().returncode == 0
    if current == wanted and loaded:
        return Installed("unchanged", path, "already installed, skipped")
    if current != wanted:
        os.makedirs(os.path.expanduser(LOG_DIR), exist_ok=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            file.write(wanted)
    if not loaded and gateway_endpoint.live() is not None:
        return Installed(
            "deferred", path, f"installed {LABEL}; it takes over at next login"
        )
    if loaded:
        _launchctl("bootout", f"{_domain()}/{LABEL}")
    _bootstrap(path)
    if loaded:
        return Installed("updated", path, f"updated {LABEL}; the gateway restarted")
    return Installed("installed", path, f"installed {LABEL} (starts at login)")


def start() -> None:
    """Run the installed service now: kick it if it is loaded, else load it."""
    if _print().returncode == 0:
        done = _launchctl("kickstart", f"{_domain()}/{LABEL}")
        if done.returncode != 0:
            raise RuntimeError(done.stderr.strip() or "launchctl kickstart failed")
    else:
        _bootstrap(plist_path())


def uninstall() -> bool:
    path = plist_path()
    existed = os.path.exists(path)
    _launchctl("bootout", f"{_domain()}/{LABEL}")
    if existed:
        os.remove(path)
    return existed


def status() -> dict[str, object]:
    installed = os.path.exists(plist_path())
    result = _print()
    if result.returncode != 0:
        return {"installed": installed, "running": False}
    found = re.search(r"\bpid = (\d+)", result.stdout)
    return {
        "installed": True,
        "running": "state = running" in result.stdout,
        "pid": int(found.group(1)) if found else None,
    }
