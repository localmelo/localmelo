from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.memory._sqlite import thinking_retention_days
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.memory.procedures.store import SqliteProcedures
from localmelo.melo.schema import RunResult, answer_text, split_thinking
from localmelo.support.backends.discovery import parse_model_ids
from localmelo.support.backends.openai_compat import normalize_url
from localmelo.support.gateway.chat_gate import (
    ChatGate,
    GatedEmbedding,
    GatedLLM,
)
from localmelo.support.gateway.machine_monitor import (
    BATTERY_POLL_S,
    LOCK_GUARD_POLL_S,
    LOCK_POLL_S,
    LOCK_SETTLE_S,
    LockRead,
    MachineMonitor,
    MachineSnapshot,
)
from localmelo.support.gateway.session import SessionBusyError
from localmelo.support.providers.embedding.openai_compat import OpenAICompatEmbedding
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

# The default store of a playground's own memories, as a ``~`` string so it
# resolves when a playground opens one: a test's temporary HOME then applies.
_DEFAULT_MEMORY_DIR = "~/.cache/localmelo/playground"

_SCENARIOS_PATH = (
    Path(__file__).resolve().parents[2] / "tests" / "smoke" / "data" / "scenarios.json"
)


class LockHeldError(Exception):
    """A create that would embed its seeds was refused: a foreign lock is held."""


async def _answer(agent: Agent, query: str) -> RunResult:
    """Run *query* and return how the run ended.

    The run has an event sink, so a Stop (a cancellation, see _stop) ends
    it as ``cancelled`` and closes its turn in working memory, with the
    label of an interrupted turn as the kept answer, instead of leaving
    the query unanswered there. The workspace shows no live events yet:
    the sink keeps only the outcome that ``run.finished`` reports, which
    this coroutine returns when the cancellation reaches it. That is also
    the run's own outcome when the Stop came after the run had ended,
    during its final save, so its answer, which working memory keeps
    already, is not lost.
    """
    finished: list[RunResult] = []

    def keep_outcome(event: AgentEvent) -> None:
        if event.type == "run.finished":
            finished.append(RunResult(**event.data["result"]))

    try:
        return await agent.run_task(query, on_event=keep_outcome)
    except asyncio.CancelledError:
        if not finished:
            raise  # stopped before the run began
        return finished[-1]


def _stop(task: asyncio.Task[RunResult], message: str | None = None) -> None:
    """Cancel *task*, a run of _answer, unless it is done or cancelled already.

    Once is enough: run_task then returns once the run is saved. A second
    cancellation would stop only its wait for that save (see
    Agent._finalize_in_task), so the next run could start while the save
    of this one is still under way.
    """
    if not task.done() and not task.cancelling():
        task.cancel(message)


async def _stopped(
    task: asyncio.Task[RunResult], message: str | None = None
) -> RunResult | None:
    """Stop *task* (see _stop) and wait for it to end.

    Returns how its run ended, or None if the run never began.
    """
    _stop(task, message)
    await asyncio.wait({task})
    return None if task.cancelled() else task.result()


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
        if hint not in {"mlc", "omlx", "ollama", "vllm", "sglang"}:
            raise ValueError(f"Unsupported adapter: {hint}")
        return hint
    lowered = _normalize_input_url(chat_url).lower()
    if "11434" in lowered or "ollama" in lowered:
        return "ollama"
    return "mlc"


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


@dataclass
class PlaygroundSession:
    agent: Agent
    adapter: str
    chat_url: str
    chat_model: str
    embedding_url: str = ""
    embedding_model: str = ""
    scenario_id: str = ""
    scenario: dict[str, Any] | None = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    messages: list[dict[str, str]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    task: asyncio.Task[RunResult] | None = field(default=None, repr=False)
    cancel_requested: bool = False
    # The chat's scheduling level, P1-P4; set by the page's chip.
    level: int = 2
    level_source: str = "computed"

    def touch(self) -> None:
        self.last_active = time.time()


class SmokePlayground:
    def __init__(
        self,
        max_sessions: int = 12,
        idle_ttl: float = 7200.0,
        memory_dir: Path | str | None = None,
        policy: str = "off",
        n_srv: int = 8,
        r_f: int = 1,
        n_fg: int = 1,
        call_log_dir: Path | str | None = None,
        lock_path: Path | str | None = None,
        lock_self: str = "",
        lock_gate: str = "record",
        lock_read: LockRead | None = None,
        battery_runner: Callable[[], str] | None = None,
        lock_poll_s: float = LOCK_POLL_S,
        battery_poll_s: float = BATTERY_POLL_S,
        settle_s: float = LOCK_SETTLE_S,
        guard_poll_s: float = LOCK_GUARD_POLL_S,
    ) -> None:
        self._sessions: dict[str, PlaygroundSession] = {}
        self._max = max_sessions
        self._idle_ttl = idle_ttl
        self._scenarios = _load_scenarios()
        self._scenario_map = {sc["id"]: sc for sc in self._scenarios}
        self._creation_lock = asyncio.Lock()

        # Share persisted memories, but let each agent own its connections.
        # Closing one session must not close resources used by another run.
        mem_dir = Path(memory_dir or _DEFAULT_MEMORY_DIR).expanduser()
        mem_dir.mkdir(parents=True, exist_ok=True)
        self._history_path = mem_dir / "history.db"
        # Read once, so every chat's history keeps reasoning as long.
        self._thinking_retention_days = thinking_retention_days()
        self._long_term_path = mem_dir / "long_term.db"
        self._procedures_path = mem_dir / "procedures.db"
        # One gate for every chat; call_log_dir=None writes no call log.
        self._gate = ChatGate(
            policy=policy,
            n_srv=n_srv,
            r_f=r_f,
            n_fg=n_fg,
            call_log_dir=call_log_dir,
            retention_days=self._thinking_retention_days,
            lock_gate=lock_gate,
        )
        # While a call is held the run wakes at least this often (the focus
        # window, from the policy's Params): a hold caused only by the typing
        # heartbeat has no event to end it once the user stops typing, so the
        # bounded wake keeps it from waiting forever (R10). It never raises
        # the run's TimeoutError; it runs a no-event transition and waits.
        self._hold_wake_s = self._gate.hold_wake_s
        # A monitor exists when there is a lock path to watch or a battery to
        # read. The gateway always gives a battery runner, so it always has
        # one; a test that passes neither has none (R3).
        self._monitor: MachineMonitor | None = None
        if lock_path is not None or battery_runner is not None:
            self._monitor = MachineMonitor(
                lock_path=lock_path,
                lock_self=lock_self,
                lock_gate=lock_gate,
                write=self._gate.write_machine,
                on_change=self._on_machine,
                lock_read=lock_read,
                battery_runner=battery_runner,
                lock_poll_s=lock_poll_s,
                battery_poll_s=battery_poll_s,
                settle_s=settle_s,
                guard_poll_s=guard_poll_s,
            )

    def _on_machine(self, snapshot: MachineSnapshot) -> None:
        """Feed one observed snapshot to the gate; no await between the two."""
        self._gate.set_machine(snapshot.machine, payload=snapshot.payload())

    async def start(self) -> None:
        """Start the machine monitor; idempotent."""
        if self._monitor is not None:
            await self._monitor.start()

    async def stop(self) -> None:
        """Stop the machine monitor; idempotent."""
        if self._monitor is not None:
            await self._monitor.stop()

    def machine_payload(self) -> dict[str, Any] | None:
        """The machine snapshot for the page, or ``None`` without a monitor."""
        return None if self._monitor is None else self._monitor.payload()

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
            if embedding_url != chat_url
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
        chat_url = _normalize_input_url(chat_url)
        embedding_url = _normalize_input_url(
            embedding_url or (chat_url if embedding_model else "")
        )
        chat_model = chat_model.strip()
        if not chat_url:
            raise ValueError("chat_url is required")
        if not chat_model:
            raise ValueError("chat_model is required")
        if (
            scenario_id
            and scenario_id != "__all__"
            and scenario_id not in self._scenario_map
        ):
            raise ValueError("Unknown seed memory set")
        if (
            self._gate.lock_gate == "wait"
            and self._gate.machine.lock == "foreign"
            and scenario_id
            and embedding_model
        ):
            # Seeding embeds before the chat exists; parking that call would
            # hide a model request behind a create with no chat and no timer
            # (D14). Answer 409 with the holder and send nothing.
            holder = str(self._gate.machine_payload().get("holder") or "a session")
            raise LockHeldError(
                f"GPU lock held by {holder}; try again when it is released"
            )

        detected = _detect_adapter(chat_url, adapter)
        # Seeding may await embedding requests. Reserve capacity across the
        # complete creation operation instead of racing other creations.
        async with self._creation_lock:
            if len(self._sessions) >= self._max and all(
                session.lock.locked() for session in self._sessions.values()
            ):
                raise SessionBusyError(
                    "All sessions are busy. Stop a response before creating a chat."
                )
            session = await self._create_session(
                detected,
                chat_url,
                chat_model,
                embedding_url,
                embedding_model,
                scenario_id,
            )
            try:
                # Do not discard a conversation until its replacement is ready.
                # A previously idle chat may also have started a run while the
                # new agent was waiting for seed embeddings.
                await self._evict_idle()
                if len(self._sessions) >= self._max:
                    await self._evict_oldest()
                if len(self._sessions) >= self._max:
                    raise SessionBusyError(
                        "All sessions are busy. Stop a response before creating a chat."
                    )
            except BaseException:
                await session.agent.close()
                raise
            self._sessions[session.session_id] = session
            return self.get_session(session.session_id)

    async def _create_session(
        self,
        detected: str,
        chat_url: str,
        chat_model: str,
        embedding_url: str,
        embedding_model: str,
        scenario_id: str,
    ) -> PlaygroundSession:
        # The session id exists before its providers do: a gated provider
        # tags every call with it, and the gate keys lanes, levels and the
        # log by it.
        session_id = uuid.uuid4().hex[:12]
        client = self._gate.client(session_id)
        llm = GatedLLM(self._build_llm_provider(detected, chat_url, chat_model), client)
        inner_embedding = self._build_embedding_provider(embedding_url, embedding_model)
        embedding = (
            GatedEmbedding(inner_embedding, client)
            if inner_embedding is not None
            else None
        )
        # Each chat keeps its answers, so a follow-up sees them (#37's option).
        agent = Agent(
            llm=llm, embedding=embedding, options=AgentOptions(record_answers=True)
        )

        await agent.hippo.aclose()
        agent.hippo.history = SqliteHistory(
            self._history_path, thinking_retention_days=self._thinking_retention_days
        )
        if embedding:
            agent.hippo.long = SqliteLongTerm(self._long_term_path)
        agent.hippo.procedures = SqliteProcedures(self._procedures_path)

        try:
            scenario_summary = await self._seed_memories(agent, scenario_id)
        except BaseException:
            await agent.close()
            raise

        return PlaygroundSession(
            agent=agent,
            session_id=session_id,
            adapter=detected,
            chat_url=chat_url,
            chat_model=chat_model,
            embedding_url=embedding_url,
            embedding_model=embedding_model,
            scenario_id=scenario_id,
            scenario=scenario_summary,
        )

    async def _seed_memories(
        self, agent: Agent, scenario_id: str
    ) -> dict[str, Any] | None:
        if scenario_id == "__all__":
            total = 0
            for sc in self._scenarios:
                for fact in sc["seed_memories"]:
                    await agent.hippo.memorize(fact)
                    total += 1
            return {
                "id": "__all__",
                "name": "All Memories",
                "description": f"Loaded {total} seed memories from all sets",
                "seed_count": total,
                "queries": [q for sc in self._scenarios for q in sc.get("queries", [])],
            }
        scenario = self._scenario_map.get(scenario_id)
        if scenario:
            for fact in scenario["seed_memories"]:
                await agent.hippo.memorize(fact)
        return self._scenario_summary(scenario)

    def list_sessions(self) -> list[dict[str, Any]]:
        return [self._session_payload(s) for s in self._sessions.values()]

    def get_session(self, session_id: str) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("session not found")
        return {
            **self._session_payload(session),
            "messages": [dict(m) for m in session.messages],
        }

    def _session_payload(self, session: PlaygroundSession) -> dict[str, Any]:
        title = next(
            (m["content"] for m in session.messages if m["role"] == "user"), "New chat"
        )
        return {
            "session_id": session.session_id,
            "adapter": session.adapter,
            "chat_url": session.chat_url,
            "chat_model": session.chat_model,
            "embedding_url": session.embedding_url,
            "embedding_model": session.embedding_model,
            "scenario_id": session.scenario_id,
            "scenario": session.scenario,
            "created_at": session.created_at,
            "last_active": session.last_active,
            "title": " ".join(title.split())[:80],
            "is_busy": session.lock.locked(),
            "level": session.level,
            "held": self._gate.held(session.session_id),
        }

    async def run(
        self, session_id: str, query: str, *, timeout: float = 600.0
    ) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("session not found")
        if not query.strip():
            raise ValueError("query is required")
        if session.lock.locked():
            raise SessionBusyError(
                "This chat is already responding. Stop it or wait for it to finish."
            )

        async with session.lock:
            session.touch()
            session.cancel_requested = False
            # The run sets the lane itself, so its first call is a lane call
            # whatever order the page's presence heartbeat arrives in (R29).
            started = self._gate.turn_start(session_id)
            self._gate.focus(session_id)
            session.messages.append({"role": "user", "content": query.strip()})
            task = asyncio.create_task(_answer(session.agent, query.strip()))
            session.task = task
            outcome: RunResult | None
            timed_out = False
            try:
                # Shielded, so that only _stop cancels the run: the chat
                # stays busy until the run is saved. Under a lock hold the
                # wait is the gate's even when the policy is off (R12).
                if not self._gate.enforces_holds:
                    outcome = await asyncio.wait_for(asyncio.shield(task), timeout)
                else:
                    outcome = await self._wait_for_turn(
                        session_id, task, timeout, started
                    )
            except TimeoutError:
                if task.done():  # raised by the run, not this timeout
                    session.messages.append(
                        {"role": "note", "content": "Response timed out."}
                    )
                    raise
                outcome = await _stopped(task, f"Timed out after {timeout:g} s")
                timed_out = not session.cancel_requested
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    # This request was cancelled: stop its run too, then let
                    # the cancellation go on.
                    self._close_turn(session, await _stopped(task), "Response stopped.")
                    raise
                outcome = None  # stopped before the run began
            except Exception as exc:
                session.messages.append({"role": "note", "content": f"Error: {exc}"})
                raise
            finally:
                session.task = None
                session.touch()
            note = "Response timed out." if timed_out else "Response stopped."
            response = self._close_turn(session, outcome, note)
            if timed_out and response["cancelled"]:
                raise TimeoutError(f"Timed out after {timeout:g} s")
            return response

    async def _wait_for_turn(
        self,
        session_id: str,
        task: asyncio.Task[RunResult],
        timeout: float,
        started: int,
    ) -> RunResult:
        """The run's wait while the gate may hold: *timeout* counts active time.

        While the chat is held the wait is bounded by a wake of at most the
        focus window (``ChatGate.hold_wake_s``, section 4.6 as amended): the hold
        itself carries no timeout, and the wake only re-runs a gate
        transition and waits again, so it never fires the run's TimeoutError.
        The event the gate sets on every hold change wakes the loop earlier
        to recompute the remaining time. Both the clock read and the event
        clear happen in one synchronous segment, so a hold cannot start
        between them unnoticed.
        """
        gate = self._gate
        while True:
            gate.wait_event.clear()
            now = gate.now()
            if gate.held_since(session_id) is not None:
                # A wake, never the run's timeout: the hold is re-evaluated
                # and the wait continues if the policy still holds the call.
                remaining: float | None = self._hold_wake_s
            else:
                spent = gate.held_seconds(session_id)
                remaining = timeout - ((now - started) / 1_000_000 - spent)
                if remaining <= 0:
                    raise TimeoutError
            waiter = asyncio.ensure_future(gate.wait_event.wait())
            try:
                await asyncio.wait(
                    {task, waiter},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                waiter.cancel()
            if task.done():
                return task.result()
            if gate.held_since(session_id) is not None:
                gate.wake()

    def _close_turn(
        self, session: PlaygroundSession, outcome: RunResult | None, note: str
    ) -> dict[str, Any]:
        """End the turn in the transcript; return the response of run().

        A run that ended otherwise than ``cancelled`` adds its answer, also
        when a Stop or the timeout came during its final save, since working
        memory keeps that answer already. A stopped run adds *note*.
        """
        if outcome is None or outcome.status == "cancelled":
            session.messages.append({"role": "note", "content": note})
            return {
                "session_id": session.session_id,
                "cancelled": True,
                "result": "",
                "thinking": "",
                "answer": "",
            }
        result = outcome.result or f"[{outcome.status}] {outcome.error}"
        # The split and the label that working memory keeps (see
        # Agent._record_answer), so the transcript shows the same answer.
        thinking, answer = split_thinking(result)
        answer = answer_text(outcome.status, answer)
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
            "cancelled": False,
            "result": result,
            "thinking": thinking,
            "answer": answer,
        }

    def presence(self, session_id: str) -> None:
        """A presence heartbeat: the chat is focused and being typed in."""
        if session_id not in self._sessions:
            raise KeyError("session not found")
        self._gate.presence(session_id)

    def set_level(self, session_id: str, level: int) -> dict[str, Any]:
        """Set and return a chat's level, P1-P4."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("session not found")
        value = self._gate.set_level(session_id, level)
        session.level = value
        session.level_source = "user"
        return {"session_id": session_id, "level": value}

    async def cancel(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("session not found")
        if session.task is None or session.task.done():
            return False
        session.cancel_requested = True
        _stop(session.task)
        # Wait until run() has closed the turn and released its session lock.
        async with session.lock:
            return True

    async def close(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            return False
        if session.lock.locked():
            raise SessionBusyError(
                "This chat is still responding. Stop it before closing it."
            )
        self._sessions.pop(session_id, None)
        await session.agent.close()
        self._gate.forget(session_id)
        return True

    async def close_all(self) -> None:
        # The monitor writes through the gate's log, so stop it before the
        # gate closes the files (R21).
        await self.stop()
        for session in list(self._sessions.values()):
            with contextlib.suppress(Exception):
                await self.cancel(session.session_id)
                await session.agent.close()
        self._sessions.clear()
        # After the agents: the wrappers' inner providers are closed already.
        await self._gate.close()

    async def _fetch_models(self, url: str) -> list[str]:
        async with httpx.AsyncClient(
            base_url=normalize_url(_normalize_input_url(url)),
            timeout=20.0,
        ) as client:
            resp = await client.get("/models")
            resp.raise_for_status()
            models = parse_model_ids(resp.json())
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
                self._gate.forget(sid)

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
        self._gate.forget(oldest.session_id)
