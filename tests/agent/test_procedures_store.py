"""Where an agent keeps its procedural store: it follows memory persistence.

With persistence on (``AgentOptions.memory_dir`` or
``LOCALMELO_PERSIST_MEMORY``) the store is ``procedures.db`` in the memory
directory; otherwise it lives in RAM for the agent's life (decision L-16).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.procedures.store import (
    MemoryProcedures,
    Procedures,
    SqliteProcedures,
)
from localmelo.melo.schema import Message, ToolDef


class QuietLLM(BaseLLMProvider):
    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        return Message(role="assistant", content="ok")

    async def close(self) -> None:
        pass


class ClosingProcedures(MemoryProcedures):
    def __init__(self) -> None:
        super().__init__()
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


def test_hippo_keeps_a_store_in_ram_by_default() -> None:
    assert type(Hippo().procedures) is MemoryProcedures


def test_hippo_takes_an_injected_store() -> None:
    store = MemoryProcedures()
    assert Hippo(procedures=store).procedures is store


async def test_hippo_aclose_closes_the_store() -> None:
    store = ClosingProcedures()
    await Hippo(procedures=store).aclose()
    assert store.closed == 1


async def test_an_agent_without_persistence_keeps_the_store_in_ram() -> None:
    agent = Agent(llm=QuietLLM())
    try:
        assert type(agent.hippo.procedures) is MemoryProcedures
    finally:
        await agent.close()


@pytest.mark.parametrize("memory", [True, False])
async def test_a_memory_dir_gives_procedures_db(tmp_path: Path, memory: bool) -> None:
    # memory=False keeps history in SQLite too: the store follows persistence.
    agent = Agent(
        llm=QuietLLM(),
        options=AgentOptions(memory_dir=str(tmp_path), memory=memory),
    )
    try:
        store = agent.hippo.procedures
        assert isinstance(store, SqliteProcedures)
        assert store.path == str(tmp_path / "procedures.db")
    finally:
        await agent.close()


async def test_the_environment_opt_in_gives_procedures_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALMELO_PERSIST_MEMORY", "1")
    monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(tmp_path / "env"))
    agent = Agent(llm=QuietLLM())
    try:
        store = agent.hippo.procedures
        assert isinstance(store, SqliteProcedures)
        assert store.path == str(tmp_path / "env" / "procedures.db")
    finally:
        await agent.close()


async def test_closing_the_agent_closes_the_store() -> None:
    agent = Agent(llm=QuietLLM())
    store = ClosingProcedures()
    agent.hippo.procedures = store
    await agent.close()
    assert store.closed == 1


def test_the_store_is_a_procedures_store() -> None:
    assert isinstance(Hippo().procedures, Procedures)


async def test_hippo_close_drops_the_store_s_connection(tmp_path: Path) -> None:
    # The sync shim, as for SqliteHistory and SqliteLongTerm: no await, the
    # connection reference dropped.
    store = SqliteProcedures(tmp_path / "procedures.db")
    await store.export_rows()  # opens the connection
    assert store._conn is not None
    Hippo(procedures=store).close()  # should not raise
    assert store._conn is None
