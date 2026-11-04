"""Whether the chat model reads pictures, for the composer and for a send.

The answer is the model provider's own ``takes_images()``: a hosted preset says
it, oMLX and Ollama are asked, and any other server is unknown. ``True``,
``False`` or ``None`` (unknown) is kept for :data:`TTL_S` seconds per model
setting. Nothing asks the server until a picture is added: the upload's answer
and a send that carries pictures wait for :meth:`Vision.reads`.

A model that refuses a turn because of its pictures (the turn ends with
:data:`~localmelo.melo.schema.PICTURES_REFUSED_ERROR`) is text only for the time being:
:meth:`Vision.listen` hears that end and :meth:`Vision.refused` caches ``False`` for
the model setting like any other verdict. It lasts :data:`TTL_S` seconds or until the
model setting changes; then the server is asked again, and what it answers stands. What a
refusal taught is a guess from one error, so it never outlives the cache.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any, Final

from localmelo.melo.schema import PICTURES_REFUSED_ERROR

if TYPE_CHECKING:
    from localmelo.support.sessions.store import StoredEvent

    from . import Console

TTL_S: Final = 60.0
# A model server that does not answer in this time is "unknown", not "text only".
CHECK_TIMEOUT_S: Final = 4.0


class Vision:
    """The cached verdict on the chat model of the settings."""

    _listening = False

    def __init__(self, console: Console) -> None:
        self._console = console
        self._result: tuple[tuple[Any, ...], bool | None, float] | None = None
        self._task: asyncio.Task[None] | None = None

    def _key(self) -> tuple[Any, ...]:
        cfg = self._console.cfg
        section = getattr(cfg, cfg.chat_backend, None) if cfg.chat_backend else None
        return (
            cfg.chat_backend,
            getattr(section, "chat_model", ""),
            getattr(section, "chat_url", ""),
            getattr(section, "base_url", ""),
        )

    def _fresh(self) -> bool:
        return (
            self._result is not None
            and self._result[0] == self._key()
            and time.monotonic() - self._result[2] < TTL_S
        )

    async def _ask(self) -> bool | None:
        from localmelo.support.backends.registry import get_backend

        cfg = self._console.cfg
        if not cfg.is_configured:
            return None
        provider = None
        try:
            provider = get_backend(cfg.chat_backend).build_chat_provider(cfg)
            ask = getattr(provider, "takes_images", None)
            if not callable(ask):
                return None
            verdict = await asyncio.wait_for(ask(), CHECK_TIMEOUT_S)
        except Exception:
            return None
        finally:
            close = getattr(provider, "close", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    await close()
        return verdict if isinstance(verdict, bool) else None

    async def _refresh(self) -> None:
        key = self._key()
        verdict = await self._ask()
        self._result = (key, verdict, time.monotonic())

    async def reads(self) -> bool | None:
        """``True`` if the model reads pictures, ``False`` if it is text only, ``None`` if unknown."""
        if not self._fresh():
            if self._task is None or self._task.done():
                self._task = asyncio.ensure_future(self._refresh())
            await asyncio.shield(self._task)
        assert self._result is not None
        return self._result[1] if self._result[0] == self._key() else None

    def known(self) -> bool | None:
        """The last verdict on the model of the settings, without asking the server again.

        ``None`` before any verdict, and after the model setting changed. A run
        that reads words added mid-turn asks this: it cannot wait for a server.
        """
        result = self._result
        if result is None or result[0] != self._key():
            return None
        return result[1]

    def refused(self) -> None:
        """A turn was refused for its pictures: the model of the settings is text only, for now."""
        self._result = (self._key(), False, time.monotonic())

    def listen(self, store: Any) -> None:
        """Hear the end of every turn in *store*, once, to learn of a refusal for pictures."""
        if not self._listening:
            self._listening = True
            store.add_listener(self._heard)

    def _heard(self, stored: StoredEvent) -> None:
        event = stored.event
        if event.get("type") != "turn.finished":
            return
        data = event.get("data") or {}
        if (
            data.get("status") == "provider_error"
            and data.get("error") == PICTURES_REFUSED_ERROR
        ):
            self.refused()

    async def aclose(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
