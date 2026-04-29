"""Tests for memory persistence (SQLite) and registry/index split."""

import asyncio
from pathlib import Path

import pytest

from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.memory.tools import ToolIndex, ToolRegistry
from localmelo.melo.schema import StepRecord, TaskRecord, ToolCall, ToolDef, ToolResult

# ── ToolRegistry vs ToolIndex split ──


class TestToolIndex:
    """ToolIndex is the search-only component."""

    def test_index_and_search(self) -> None:
        idx = ToolIndex()
        idx.index(
            ToolDef(
                name="shell_exec",
                description="execute shell commands",
                parameters={},
                semantic_tags=["shell", "bash"],
            )
        )
        idx.index(
            ToolDef(
                name="file_read",
                description="read file contents",
                parameters={},
                semantic_tags=["file", "read"],
            )
        )
        results = idx.search("run a shell command")
        assert len(results) > 0
        assert results[0].name == "shell_exec"

    def test_search_empty(self) -> None:
        idx = ToolIndex()
        assert idx.search("anything") == []

    def test_remove(self) -> None:
        idx = ToolIndex()
        td = ToolDef(name="a", description="alpha tool", parameters={})
        idx.index(td)
        assert len(idx.search("alpha")) == 1
        idx.remove("a")
        assert idx.search("alpha") == []


class TestRegistryIndexSplit:
    """ToolRegistry combines authoritative lookup + delegated search."""

    def test_registry_get_not_in_index(self) -> None:
        """Registry is the authoritative source; index is for search only."""
        reg = ToolRegistry()
        td = ToolDef(name="my_tool", description="does things", parameters={})
        reg.register(td)
        # Authoritative lookup works
        assert reg.get("my_tool") is td
        # Search via index also works
        assert len(reg.search("does things")) == 1

    def test_index_accessible(self) -> None:
        """The underlying ToolIndex is exposed for direct use."""
        reg = ToolRegistry()
        td = ToolDef(name="x", description="x tool", parameters={})
        reg.register(td)
        assert isinstance(reg.index, ToolIndex)
        assert len(reg.index.search("x tool")) == 1

    def test_registry_public_api(self) -> None:
        """register/get/list_all/search work as expected."""
        reg = ToolRegistry()
        reg.register(
            ToolDef(
                name="shell_exec",
                description="execute shell commands",
                parameters={},
                semantic_tags=["shell", "bash", "command"],
            )
        )
        reg.register(
            ToolDef(
                name="file_read",
                description="read file contents",
                parameters={},
                semantic_tags=["file", "read"],
            )
        )
        # get
        assert reg.get("shell_exec") is not None
        assert reg.get("nonexistent") is None
        # list_all
        assert len(reg.list_all()) == 2
        # search
        results = reg.search("run a shell command")
        assert len(results) > 0
        assert results[0].name == "shell_exec"


# ── SqliteHistory persistence ──


class TestSqliteHistory:
    @pytest.mark.asyncio
    async def test_save_and_get_task(self, tmp_path: Path) -> None:
        db = tmp_path / "history.db"
        h = SqliteHistory(db)
        task = TaskRecord(query="hello", task_id="t1")
        await h.save_task(task)
        got = await h.get_task("t1")
        assert got is not None
        assert got.query == "hello"
        await h.aclose()

    @pytest.mark.asyncio
    async def test_add_step_with_tool_call(self, tmp_path: Path) -> None:
        db = tmp_path / "history.db"
        h = SqliteHistory(db)
        task = TaskRecord(query="test", task_id="t2")
        await h.save_task(task)

        step = StepRecord(
            thought="planning",
            tool_call=ToolCall(tool_name="shell_exec", arguments={"cmd": "ls"}),
            tool_result=ToolResult(
                tool_name="shell_exec", output="file.txt", duration_ms=10.0
            ),
        )
        await h.add_step("t2", step)
        steps = await h.get_steps("t2")
        assert len(steps) == 1
        assert steps[0].thought == "planning"
        assert steps[0].tool_call is not None
        assert steps[0].tool_call.tool_name == "shell_exec"
        assert steps[0].tool_call.arguments == {"cmd": "ls"}
        assert steps[0].tool_result is not None
        assert steps[0].tool_result.output == "file.txt"
        await h.aclose()

    @pytest.mark.asyncio
    async def test_persistence_across_reopen(self, tmp_path: Path) -> None:
        """Data survives closing and reopening the database."""
        db = tmp_path / "history.db"

        # Session 1: write data
        h1 = SqliteHistory(db)
        task = TaskRecord(query="persist me", task_id="tp")
        await h1.save_task(task)
        await h1.add_step("tp", StepRecord(thought="step-one"))
        await h1.add_step("tp", StepRecord(thought="step-two"))
        await h1.aclose()

        # Session 2: read data back
        h2 = SqliteHistory(db)
        got = await h2.get_task("tp")
        assert got is not None
        assert got.query == "persist me"
        assert len(got.steps) == 2
        assert got.steps[0].thought == "step-one"
        assert got.steps[1].thought == "step-two"
        await h2.aclose()

    @pytest.mark.asyncio
    async def test_save_task_updates_status(self, tmp_path: Path) -> None:
        db = tmp_path / "history.db"
        h = SqliteHistory(db)
        task = TaskRecord(query="q", task_id="tu", status="running")
        await h.save_task(task)

        task.status = "completed"
        task.result = "done"
        await h.save_task(task)

        got = await h.get_task("tu")
        assert got is not None
        assert got.status == "completed"
        assert got.result == "done"
        await h.aclose()

    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self, tmp_path: Path) -> None:
        db = tmp_path / "history.db"
        h = SqliteHistory(db)
        assert await h.get_task("nope") is None
        await h.aclose()

    @pytest.mark.asyncio
    async def test_step_ordering(self, tmp_path: Path) -> None:
        """Steps maintain insertion order across reopen."""
        db = tmp_path / "history.db"
        h = SqliteHistory(db)
        await h.save_task(TaskRecord(query="q", task_id="so"))
        for i in range(5):
            await h.add_step("so", StepRecord(thought=f"s{i}"))
        await h.aclose()

        h2 = SqliteHistory(db)
        steps = await h2.get_steps("so")
        assert [s.thought for s in steps] == [f"s{i}" for i in range(5)]
        await h2.aclose()

    @pytest.mark.asyncio
    async def test_concurrent_add_step_preserves_order(self, tmp_path: Path) -> None:
        """Concurrent add_step calls produce dense, unique seq values.

        The SELECT MAX(seq) → INSERT pair must be atomic. Without the
        write lock, racing tasks would collide on the same ``seq`` and
        the resulting steps would either duplicate or be reordered on
        reload.
        """
        db = tmp_path / "history.db"
        h = SqliteHistory(db)
        await h.save_task(TaskRecord(query="q", task_id="cc"))

        n = 20
        await asyncio.gather(
            *(h.add_step("cc", StepRecord(thought=f"s{i}")) for i in range(n))
        )

        # Steps are returned ordered by seq. Each thought must appear
        # exactly once and the count must equal n.
        steps = await h.get_steps("cc")
        thoughts = sorted(s.thought for s in steps)
        assert thoughts == sorted(f"s{i}" for i in range(n))
        assert len(steps) == n
        await h.aclose()


# ── SqliteLongTerm persistence ──


class TestSqliteLongTerm:
    @pytest.mark.asyncio
    async def test_add_and_search(self, tmp_path: Path) -> None:
        db = tmp_path / "long.db"
        lt = SqliteLongTerm(db)
        await lt.add("hello world", [1.0, 0.0, 0.0])
        await lt.add("goodbye world", [0.0, 1.0, 0.0])
        results = await lt.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0] == "hello world"
        await lt.aclose()

    @pytest.mark.asyncio
    async def test_empty_search(self, tmp_path: Path) -> None:
        db = tmp_path / "long.db"
        lt = SqliteLongTerm(db)
        results = await lt.search([1.0, 0.0], top_k=5)
        assert results == []
        await lt.aclose()

    @pytest.mark.asyncio
    async def test_persistence_across_reopen(self, tmp_path: Path) -> None:
        """Vectors survive closing and reopening the database."""
        db = tmp_path / "long.db"

        # Session 1: write
        lt1 = SqliteLongTerm(db)
        await lt1.add("alpha", [1.0, 0.0, 0.0], metadata={"src": "test"})
        await lt1.add("beta", [0.0, 1.0, 0.0])
        await lt1.aclose()

        # Session 2: search
        lt2 = SqliteLongTerm(db)
        results = await lt2.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0] == "alpha"
        assert results[0][2] == {"src": "test"}
        await lt2.aclose()

    @pytest.mark.asyncio
    async def test_search_after_reload(self, tmp_path: Path) -> None:
        """Multiple entries survive reload and rank correctly."""
        db = tmp_path / "long.db"

        lt = SqliteLongTerm(db)
        await lt.add("doc about cats", [0.9, 0.1, 0.0])
        await lt.add("doc about dogs", [0.1, 0.9, 0.0])
        await lt.add("doc about fish", [0.0, 0.1, 0.9])
        await lt.aclose()

        lt2 = SqliteLongTerm(db)
        results = await lt2.search([0.9, 0.1, 0.0], top_k=2)
        assert len(results) == 2
        # "cats" should rank first (closest vector)
        assert results[0][0] == "doc about cats"
        await lt2.aclose()

    @pytest.mark.asyncio
    async def test_metadata_preserved(self, tmp_path: Path) -> None:
        db = tmp_path / "long.db"
        lt = SqliteLongTerm(db)
        await lt.add("x", [1.0, 0.0], metadata={"step_id": "abc", "num": 42})
        results = await lt.search([1.0, 0.0], top_k=1)
        assert results[0][2] == {"step_id": "abc", "num": 42}
        await lt.aclose()
