"""melo's own gateway, which no call auto mode decides may reach (#339).

The gateway registers its port and its private files when it starts
(:func:`register`). Auto mode's sandbox
(:class:`~localmelo.melo.executor.sandbox.AutoConfinement`) then denies
reading or writing those files and connecting to that port, and the action
review counts a request to melo's own API as a change to localmelo itself.
Without a gateway in this process the default files under the home folder
are guarded all the same, and the port is the one the settings give the
gateway (:func:`expect`, set by the terminal), when known.
"""

from __future__ import annotations

import contextlib
import os
import pwd
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

# The gateway's folder under a home folder; its sign-in token is in auth/.
GATEWAY_DIR: Final = (".cache", "localmelo", "gateway")
# A SQLite file's companions, which hold its pages too.
_SQLITE_PARTS: Final = ("-wal", "-shm", "-journal")


@dataclass(frozen=True)
class OwnGateway:
    """The ports melo's gateway may listen on (none known: empty) and its files."""

    ports: tuple[int, ...]
    files: tuple[str, ...]

    @property
    def port(self) -> int | None:
        """The first of :attr:`ports`, or ``None``."""
        return self.ports[0] if self.ports else None


_registered: list[tuple[int | None, tuple[str, ...]]] = []
# The ports a process that runs no gateway knows of: the settings', the live one's.
_expected: list[int] = []


def _homes() -> list[str]:
    """The home folder of this process, and the account's own when they differ."""
    homes = [os.path.expanduser("~")]
    with contextlib.suppress(KeyError, OSError):
        homes.append(pwd.getpwuid(os.getuid()).pw_dir)
    return homes


def _with_parts(path: str) -> list[str]:
    return (
        [path, *(path + part for part in _SQLITE_PARTS)]
        if path.endswith(".db")
        else [path]
    )


def default_files(home: str) -> tuple[str, ...]:
    """The gateway's sign-in folder and session store under *home*."""
    base = os.path.join(home, *GATEWAY_DIR)
    return (os.path.join(base, "auth"), *_with_parts(os.path.join(base, "sessions.db")))


def register(port: int | None, files: Sequence[str] = ()) -> None:
    """Record the gateway this process serves: its *port* and private *files*.

    A ``.db`` file brings its companions (``-wal``, ``-shm``, ``-journal``).
    """
    if port is not None and (isinstance(port, bool) or not 0 < port < 65536):
        raise ValueError(f"port must be from 1 to 65535, got {port!r}")
    _registered.append((port, tuple(str(f) for f in files)))


def expect(*ports: int | None) -> None:
    """The ports a gateway of this setup listens on, for a process that runs none.

    The terminal passes ``[gateway] port`` and the running gateway's (its
    endpoint file); no port forgets them. Any port out of range is left out.
    """
    _expected.clear()
    for port in ports:
        if isinstance(port, bool) or not isinstance(port, int):
            continue
        if 0 < port < 65536 and port not in _expected:
            _expected.append(port)


def unregister(port: int | None) -> None:
    """Forget the newest gateway registered with *port* (it stopped)."""
    for i in range(len(_registered) - 1, -1, -1):
        if _registered[i][0] == port:
            del _registered[i]
            return


def current() -> OwnGateway:
    """The ports to close and every file to guard, as real paths.

    The ports are the registered gateways' (newest first) and the expected ones.
    """
    found: list[str] = []
    named = [f for home in _homes() for f in default_files(home)]
    named += [
        part for _, files in _registered for f in files for part in _with_parts(f)
    ]
    for path in named:
        real = os.path.realpath(path)
        if real not in found:
            found.append(real)
    ports = [port for port, _ in reversed(_registered) if port is not None]
    ports += [port for port in _expected if port not in ports]
    return OwnGateway(ports=tuple(ports), files=tuple(found))
