from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.support.backends.openai_compat import normalize_url
from localmelo.support.providers.embedding.openai_compat import OpenAICompatEmbedding
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_SCENARIOS_PATH = (
    Path(__file__).resolve().parents[2] / "tests" / "smoke" / "data" / "scenarios.json"
)


def _normalize_input_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


def _strip_v1(url: str) -> str:
    url = _normalize_input_url(url)
    if url.endswith("/v1"):
        return url[:-3]
    return url


def _detect_adapter(chat_url: str, hint: str | None) -> str:
    if hint and hint not in {"", "auto"}:
        return hint
    lowered = _normalize_input_url(chat_url).lower()
    if "11434" in lowered or "ollama" in lowered:
        return "ollama"
    return "mlc"


def _parse_model_ids(payload: Any) -> list[str]:
    items: list[Any] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            items = payload["data"]
        elif isinstance(payload.get("models"), list):
            items = payload["models"]

    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        model_id = ""
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict):
            for key in ("id", "name", "model"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    model_id = value.strip()
                    break
        if model_id and model_id not in seen:
            seen.add(model_id)
            names.append(model_id)
    return names


def _load_scenarios() -> list[dict[str, Any]]:
    with _SCENARIOS_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    scenarios: list[dict[str, Any]] = []
    for sc in raw:
        scenarios.append(
            {
                "id": sc["id"],
                "name": sc["name"],
                "description": sc.get("description", ""),
                "seed_memories": list(sc.get("seed_memories", [])),
                "queries": list(sc.get("queries", [])),
            }
        )
    return scenarios


def split_thinking_and_answer(text: str) -> tuple[str, str]:
    content = (text or "").strip()
    if not content:
        return "", ""
    match = _THINK_RE.search(content)
    if not match:
        return "", content
    thinking = match.group(1).strip()
    answer = (content[: match.start()] + content[match.end() :]).strip()
    return thinking, answer


@dataclass
class PlaygroundSession:
    agent: Agent
    adapter: str
    chat_url: str
    chat_model: str
    embedding_url: str = ""
    embedding_model: str = ""
    scenario_id: str = ""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    messages: list[dict[str, str]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def touch(self) -> None:
        self.last_active = time.time()


class SmokePlayground:
    def __init__(self, max_sessions: int = 12, idle_ttl: float = 7200.0) -> None:
        self._sessions: dict[str, PlaygroundSession] = {}
        self._max = max_sessions
        self._idle_ttl = idle_ttl
        self._scenarios = _load_scenarios()
        self._scenario_map = {sc["id"]: sc for sc in self._scenarios}

    def scenarios_payload(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for sc in self._scenarios:
            payload.append(
                {
                    "id": sc["id"],
                    "name": sc["name"],
                    "description": sc["description"],
                    "seed_count": len(sc["seed_memories"]),
                    "seed_memories": list(sc["seed_memories"]),
                    "queries": list(sc["queries"]),
                }
            )
        return payload

    async def discover(
        self,
        *,
        chat_url: str,
        embedding_url: str = "",
        adapter: str | None = None,
    ) -> dict[str, Any]:
        chat_url = _normalize_input_url(chat_url)
        embedding_url = _normalize_input_url(embedding_url or chat_url)
        if not chat_url:
            raise ValueError("chat_url is required")

        detected = _detect_adapter(chat_url, adapter)
        chat_models = await self._fetch_models(chat_url)
        embedding_models = (
            await self._fetch_models(embedding_url)
            if embedding_url
            else list(chat_models)
        )

        return {
            "adapter": detected,
            "chat_url": chat_url,
            "embedding_url": embedding_url,
            "chat_models": chat_models,
            "embedding_models": embedding_models,
        }

    async def create_session(
        self,
        *,
        chat_url: str,
        chat_model: str,
        embedding_url: str = "",
        embedding_model: str = "",
        adapter: str | None = None,
        scenario_id: str = "",
    ) -> dict[str, Any]:
        await self._evict_idle()
        if len(self._sessions) >= self._max:
            await self._evict_oldest()

        chat_url = _normalize_input_url(chat_url)
        embedding_url = _normalize_input_url(embedding_url or "")
        if not chat_url:
            raise ValueError("chat_url is required")
        if not chat_model:
            raise ValueError("chat_model is required")

        detected = _detect_adapter(chat_url, adapter)
        llm = self._build_llm_provider(detected, chat_url, chat_model)
        embedding = self._build_embedding_provider(embedding_url, embedding_model)
        agent = Agent(llm=llm, embedding=embedding)

        scenario_summary = None
        if scenario_id == "__all__":
            total = 0
            for sc in self._scenarios:
                for fact in sc["seed_memories"]:
                    await agent.hippo.memorize(fact)
                    total += 1
            scenario_summary = {
                "id": "__all__",
                "name": "All Memories",
                "description": f"Loaded {total} seed memories from all sets",
                "seed_count": total,
                "queries": [q for sc in self._scenarios for q in sc.get("queries", [])],
            }
        else:
            scenario = self._scenario_map.get(scenario_id) if scenario_id else None
            if scenario:
                for fact in scenario["seed_memories"]:
                    await agent.hippo.memorize(fact)
                scenario_summary = self._scenario_summary(scenario)

        session = PlaygroundSession(
            agent=agent,
            adapter=detected,
            chat_url=chat_url,
            chat_model=chat_model,
            embedding_url=embedding_url,
            embedding_model=embedding_model,
            scenario_id=scenario_id,
        )
        self._sessions[session.session_id] = session

        return {
            "session_id": session.session_id,
            "adapter": detected,
            "chat_url": chat_url,
            "chat_model": chat_model,
            "embedding_url": embedding_url,
            "embedding_model": embedding_model,
            "scenario": scenario_summary,
        }

    async def run(
        self, session_id: str, query: str, *, timeout: float = 600.0
    ) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("session not found")
        if not query.strip():
            raise ValueError("query is required")

        async with session.lock:
            session.touch()
            result = await asyncio.wait_for(
                session.agent.run(query.strip()), timeout=timeout
            )
            thinking, answer = split_thinking_and_answer(result)
            session.messages.append({"role": "user", "content": query.strip()})
            session.messages.append(
                {
                    "role": "assistant",
                    "content": result,
                    "thinking": thinking,
                    "answer": answer,
                }
            )
            return {
                "session_id": session.session_id,
                "result": result,
                "thinking": thinking,
                "answer": answer or result,
            }

    async def close(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        await session.agent.close()
        return True

    async def close_all(self) -> None:
        for session in list(self._sessions.values()):
            with contextlib.suppress(Exception):
                await session.agent.close()
        self._sessions.clear()

    async def _fetch_models(self, url: str) -> list[str]:
        async with httpx.AsyncClient(
            base_url=normalize_url(_normalize_input_url(url)),
            timeout=20.0,
        ) as client:
            resp = await client.get("/models")
            resp.raise_for_status()
            models = _parse_model_ids(resp.json())
        if not models:
            raise ValueError(f"No models found at {url}")
        return models

    def _build_llm_provider(
        self, adapter: str, chat_url: str, chat_model: str
    ) -> BaseLLMProvider:
        if adapter == "ollama":
            return OllamaNativeChat(
                base_url=_strip_v1(chat_url),
                model=chat_model,
                timeout=300.0,
            )
        return OpenAICompatLLM(
            base_url=normalize_url(chat_url),
            model=chat_model,
            timeout=300.0,
        )

    def _build_embedding_provider(
        self, embedding_url: str, embedding_model: str
    ) -> BaseEmbeddingProvider | None:
        if not embedding_url or not embedding_model:
            return None
        return OpenAICompatEmbedding(
            base_url=normalize_url(embedding_url),
            model=embedding_model,
            timeout=180.0,
        )

    def _scenario_summary(
        self, scenario: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if scenario is None:
            return None
        return {
            "id": scenario["id"],
            "name": scenario["name"],
            "description": scenario["description"],
            "seed_count": len(scenario["seed_memories"]),
            "queries": list(scenario["queries"]),
        }

    async def _evict_idle(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, session in self._sessions.items()
            if (now - session.last_active) > self._idle_ttl
            and not session.lock.locked()
        ]
        for sid in expired:
            session = self._sessions.pop(sid, None)
            if session is not None:
                with contextlib.suppress(Exception):
                    await session.agent.close()

    async def _evict_oldest(self) -> None:
        if not self._sessions:
            return
        idle_sessions = [s for s in self._sessions.values() if not s.lock.locked()]
        if not idle_sessions:
            return
        oldest = min(idle_sessions, key=lambda s: s.last_active)
        self._sessions.pop(oldest.session_id, None)
        with contextlib.suppress(Exception):
            await oldest.agent.close()
