"""The cached model-server reachability check of ``GET /api/health``.

The endpoint never waits for a probe. It returns what the last probe found
and starts one in the background when that result is older than
:data:`PROBE_INTERVAL_S`, or :data:`PROBE_UNREACHABLE_INTERVAL_S` while
the server does not answer. At most one probe runs at a time.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Final, Protocol

import httpx

from localmelo.support.backends.discovery import DISCOVERY_TIMEOUT_S, DiscoveryError
from localmelo.support.backends.discovery import list_models as discovery_list_models

if TYPE_CHECKING:
    from localmelo.support.config import Config

# Seconds a reachable result is used before a fresh probe starts.
PROBE_INTERVAL_S: Final = 30.0
# Seconds an unreachable result is used before a fresh probe starts.
PROBE_UNREACHABLE_INTERVAL_S: Final = 5.0
# The backends that run a local model server, so health can probe one.
LOCAL_BACKENDS: Final = frozenset({"mlc", "ollama", "vllm", "sglang"})

ProbeFunction = Callable[[str, str], Awaitable[tuple[bool, str]]]


class Probe(Protocol):
    """What ``GET /api/health`` needs from a model-server probe."""

    def snapshot(self) -> dict[str, Any] | None: ...

    async def aclose(self) -> None: ...


def url_label(url: str) -> str:
    """An absolute http(s) *url* without user info, or ``""``.

    A malformed URL never reaches the health response as a 500, and a
    relative value such as ``user:secret@host:9/v1`` is not echoed back: an
    unparseable or non-http label is empty, so a credential in a broken
    ``chat_url`` cannot leak.
    """
    try:
        parsed = httpx.URL(url)
    except (TypeError, ValueError, httpx.InvalidURL):
        return ""
    if parsed.scheme not in ("http", "https") or not parsed.host:
        return ""
    try:
        return str(parsed.copy_with(username="", password=""))
    except (TypeError, ValueError):
        return ""


def _first_line(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line


def _exception_text(exc: BaseException) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


async def default_probe(backend: str, url: str) -> tuple[bool, str]:
    """Whether the server at *url* answers, and the first line of its error.

    A server that answers at all is reachable, also when it answers with an
    error: ``DiscoveryError.reachable`` says which it is. The error is
    ``""`` when the server is reachable, as the health shape promises.
    """
    try:
        await discovery_list_models(backend, url, timeout_s=DISCOVERY_TIMEOUT_S)
    except DiscoveryError as exc:
        return exc.reachable, "" if exc.reachable else _first_line(str(exc))
    except Exception as exc:  # anything else the connection raised
        return False, _first_line(_exception_text(exc))
    return True, ""


class ModelServerProbe:
    """A cached reachability check of the settings' local model server.

    *probe* replaces the network in tests. The object starts a probe from
    :meth:`snapshot`, which runs inside the gateway's loop, and keeps one
    task at a time.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        probe: ProbeFunction | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cfg = cfg
        self._probe = probe or default_probe
        self._clock = clock
        self._result: tuple[bool, str, float] | None = None
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    def model_server(self) -> dict[str, Any] | None:
        """The object ``GET /api/health`` reports, or ``None`` for a cloud backend.

        ``reachable`` is ``None`` until the first probe ends; the call also
        starts a probe when the cached one is old enough.
        """
        backend = self._cfg.chat_backend
        if backend not in LOCAL_BACKENDS:
            return None
        section = getattr(self._cfg, backend, None)
        url = getattr(section, "chat_url", "") if section is not None else ""
        label = url_label(url)
        if not url:
            return {
                "url_label": "",
                "reachable": False,
                "error": "No chat URL is configured.",
                "checked_s_ago": 0,
            }
        now = self._clock()
        if self._should_probe(now):
            self._start(backend, url)
        if self._result is None:
            return {
                "url_label": label,
                "reachable": None,
                "error": "",
                "checked_s_ago": 0,
            }
        reachable, error, checked_at = self._result
        return {
            "url_label": label,
            "reachable": reachable,
            "error": error,
            "checked_s_ago": int(max(0.0, now - checked_at)),
        }

    def snapshot(self) -> dict[str, Any] | None:
        """:meth:`model_server`'s answer, the name the console calls it by."""
        return self.model_server()

    def _should_probe(self, now: float) -> bool:
        if self._closed or self._task is not None:
            return False
        if self._result is None:
            return True
        reachable, _, checked_at = self._result
        interval = PROBE_INTERVAL_S if reachable else PROBE_UNREACHABLE_INTERVAL_S
        return now - checked_at >= interval

    def _start(self, backend: str, url: str) -> None:
        try:
            self._task = asyncio.create_task(self._run(backend, url))
        except RuntimeError:  # no running loop: tests that only read the cache
            self._task = None

    async def _run(self, backend: str, url: str) -> None:
        try:
            reachable, error = await self._probe(backend, url)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a probe must never end the health route
            reachable, error = False, _first_line(_exception_text(exc))
        self._result = (reachable, error, self._clock())
        self._task = None

    async def aclose(self) -> None:
        """Cancel a probe in flight; idempotent."""
        self._closed = True
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
