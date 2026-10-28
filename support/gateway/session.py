from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from localmelo.melo.agent import Agent, AgentOptions

if TYPE_CHECKING:
    from localmelo.support.config import Config

logger = logging.getLogger("localmelo.gateway.session")


class SessionBusyError(RuntimeError):
    """The requested action would interrupt an active session."""


class AgentLike(Protocol):
    async def run(self, query: str) -> str: ...

    async def close(self) -> None: ...


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    status: str = "idle"  # idle, running, closed
    agent: AgentLike = field(default=None)  # type: ignore[assignment]
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def touch(self) -> None:
        self.last_active = time.time()

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_active

    @property
    def is_busy(self) -> bool:
        return self._lock.locked()


class SessionManager:
    def __init__(
        self,
        max_sessions: int = 50,
        idle_ttl: float = 3600.0,
        config: Config | None = None,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._max = max_sessions
        self._idle_ttl = idle_ttl
        self._config = config

    def _create_agent(self) -> AgentLike:
        if self._config is None:
            raise RuntimeError("SessionManager requires a config to create agents")
        # A session keeps its answers, so a follow-up sees them (#37's option).
        return Agent(
            config=self._config,
            options=AgentOptions(record_answers=True, mode="gateway"),
        )

    def get(self, session_id: str) -> Session | None:
        s = self._sessions.get(session_id)
        if s and s.status != "closed":
            s.touch()
            return s
        return None

    async def create(self, session_id: str | None = None) -> Session:
        await self._evict_idle()
        if len(self._sessions) >= self._max:
            await self._evict_oldest()

        agent = self._create_agent()
        s = Session(agent=agent)
        if session_id:
            s.session_id = session_id
        self._sessions[s.session_id] = s
        return s

    async def get_or_create(self, session_id: str | None = None) -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing:
                return existing
        return await self.create(session_id)

    async def close(self, session_id: str) -> bool:
        s = self._sessions.get(session_id)
        if not s:
            return False
        if s.is_busy:
            raise SessionBusyError(
                f"Session {session_id} is running a task; wait for it to finish "
                "before closing it."
            )
        s.status = "closed"
        try:
            await s.agent.close()
        except Exception:
            logger.warning(
                "error closing agent for session %s", session_id, exc_info=True
            )
        del self._sessions[session_id]
        return True

    def list_sessions(self) -> list[dict[str, object]]:
        return [
            {
                "session_id": s.session_id,
                "status": s.status,
                "idle_seconds": round(s.idle_seconds, 1),
                "created_at": s.created_at,
                "is_busy": s.is_busy,
            }
            for s in self._sessions.values()
            if s.status != "closed"
        ]

    async def _evict_idle(self) -> None:
        expired = [
            sid
            for sid, s in self._sessions.items()
            if s.idle_seconds > self._idle_ttl and s.status == "idle" and not s.is_busy
        ]
        for sid in expired:
            s = self._sessions.pop(sid, None)
            if s:
                try:
                    await s.agent.close()
                except Exception:
                    logger.warning(
                        "error closing evicted session %s", sid, exc_info=True
                    )

    async def _evict_oldest(self) -> None:
        if not self._sessions:
            return
        # Only evict idle, non-busy sessions
        candidates = [
            s for s in self._sessions.values() if s.status == "idle" and not s.is_busy
        ]
        if not candidates:
            return
        oldest = min(candidates, key=lambda s: s.last_active)
        s = self._sessions.pop(oldest.session_id, None)
        if s:
            try:
                await s.agent.close()
            except Exception:
                logger.warning(
                    "error closing evicted session %s", oldest.session_id, exc_info=True
                )

    async def close_all(self) -> None:
        for s in list(self._sessions.values()):
            try:
                await s.agent.close()
            except Exception:
                logger.warning("error closing session %s", s.session_id, exc_info=True)
        self._sessions.clear()
