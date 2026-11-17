"""Where an agent keeps its procedural store: it follows memory persistence.

With persistence on (``AgentOptions.memory_dir`` or
``LOCALMELO_PERSIST_MEMORY``) the store is ``procedures.db`` in the memory
directory; otherwise it lives in RAM for the agent's life (decision L-16).
The builders of chats and tasks pass ``persist_procedures``: the record then
goes to the default directory, alone and without task text.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.procedures.normalize import key_for_routine
from localmelo.melo.memory.procedures.store import (
    MemoryProcedures,
    Procedures,
    SqliteProcedures,
)
from localmelo.melo.schema import Message, ToolDef
from localmelo.tests.agent.procedural_helpers import (
    QUERY,
    ScriptedLLM,
    fix_run,
    make_agent,
    passing_tools,
    query_key,
)


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
    connection = store._conn
    assert connection is not None
    try:
        Hippo(procedures=store).close()  # should not raise
        assert store._conn is None
    finally:
        # The shim only drops its reference; close what this test opened
        # before its event loop exits.
        await connection.close()


# ── persist_procedures: the record alone, with no task text ─────────────────


def default_memory_dir() -> Path:
    """The default memory directory, under the test's temporary HOME."""
    return Path(os.path.expanduser("~/.cache/localmelo/memory"))


async def routines(agent: Agent) -> list[dict[str, Any]]:
    return [
        row
        for row in await agent.hippo.procedures.export_rows()
        if row["table"] == "routines"
    ]


@pytest.mark.parametrize("where", ["home", "env"])
async def test_persist_procedures_puts_only_the_record_in_the_default_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    directory = default_memory_dir()
    if where == "env":
        directory = tmp_path / "elsewhere"
        monkeypatch.setenv("LOCALMELO_MEMORY_DIR", str(directory))
    agent = make_agent(
        ScriptedLLM(fix_run()), tools=passing_tools(), persist_procedures=True
    )
    store = agent.hippo.procedures
    assert isinstance(store, SqliteProcedures)
    assert store.path == str(directory / "procedures.db")
    assert agent._memory_dir is None  # the memory opt-in is not on
    await agent.run_task(QUERY, own_chars=len(QUERY))
    await agent.close()
    assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
    files = {
        p.name for p in directory.iterdir() if not p.name.endswith(("-wal", "-shm"))
    }
    assert files == {"procedures.db"}  # no history.db, no long_term.db


async def test_an_agent_without_the_argument_creates_no_file() -> None:
    agent = make_agent(ScriptedLLM(fix_run()), tools=passing_tools())
    await agent.run_task(QUERY, own_chars=len(QUERY))
    await agent.close()
    assert not default_memory_dir().exists()


async def test_the_memory_opt_in_wins_over_persist_procedures(tmp_path: Path) -> None:
    # As for any agent with the opt-in: all stores, the text, the key of the query.
    agent = make_agent(
        ScriptedLLM(fix_run()),
        tools=passing_tools(),
        persist_procedures=True,
        memory_dir=str(tmp_path),
    )
    query = f"{QUERY}\n\nnews: a task ended"
    await agent.run_task(query, own_chars=len(QUERY))
    (routine,) = await routines(agent)
    await agent.close()
    assert routine["key"] == query_key(agent, query).key
    assert routine["query_text"] == query_key(agent, query).query_text != ""
    assert (tmp_path / "history.db").exists()
    assert not default_memory_dir().exists()


@pytest.mark.parametrize(
    ("query", "own_chars"),
    [(QUERY, None), (QUERY, 0), (f"  \n{QUERY}", 3)],
    ids=["not given", "empty", "blank"],
)
async def test_a_run_without_own_words_gets_no_routine(
    query: str, own_chars: int | None
) -> None:
    agent = make_agent(
        ScriptedLLM(fix_run()), tools=passing_tools(), persist_procedures=True
    )
    result = await agent.run_task(query, own_chars=own_chars)
    assert result.status == "completed"
    assert await agent.hippo.procedures.export_rows() == []
    await agent.close()


async def test_a_named_routine_keeps_its_name_and_no_text() -> None:
    agent = make_agent(
        ScriptedLLM(fix_run()), tools=passing_tools(), persist_procedures=True
    )
    await agent.run_task(QUERY, routine="daily report", own_chars=len(QUERY))
    (routine,) = await routines(agent)
    await agent.close()
    assert routine["name"] == "daily report"
    assert routine["key"] == key_for_routine("daily report").key
    assert routine["query_text"] == ""
