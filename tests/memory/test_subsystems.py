import pytest

from localmelo.melo.memory.history import History
from localmelo.melo.memory.long import LongTerm
from localmelo.melo.memory.short import ShortTerm, WorkingMemory
from localmelo.melo.memory.tools import ToolRegistry
from localmelo.melo.schema import (
    Message,
    ReflectionEntry,
    StepRecord,
    TaskRecord,
    ToolDef,
)


class TestShortTerm:
    def test_append_and_window(self) -> None:
        st = ShortTerm(max_len=3)
        st.append(Message(role="user", content="a"))
        st.append(Message(role="user", content="b"))
        assert len(st.get_window()) == 2

    def test_max_len(self) -> None:
        st = ShortTerm(max_len=2)
        st.append(Message(role="user", content="a"))
        st.append(Message(role="user", content="b"))
        st.append(Message(role="user", content="c"))
        window = st.get_window()
        assert len(window) == 2
        assert window[0].content == "b"
        assert window[1].content == "c"

    def test_clear(self) -> None:
        st = ShortTerm(max_len=5)
        st.append(Message(role="user", content="a"))
        st.clear()
        assert len(st.get_window()) == 0


class TestWorkingMemory:
    def test_add_and_get_reflections(self) -> None:
        wm = WorkingMemory(max_len=5)
        entry = ReflectionEntry(attempt_id=0, summary="tried X")
        wm.add_reflection(entry)
        assert len(wm.get_reflections()) == 1
        assert wm.get_reflections()[0].summary == "tried X"

    def test_clear_preserves_reflections(self) -> None:
        wm = WorkingMemory(max_len=5)
        wm.append(Message(role="user", content="hello"))
        wm.add_reflection(ReflectionEntry(attempt_id=0, summary="r1"))
        wm.clear()
        assert len(wm.get_window()) == 0
        assert len(wm.get_reflections()) == 1

    def test_clear_all(self) -> None:
        wm = WorkingMemory(max_len=5)
        wm.append(Message(role="user", content="hello"))
        wm.add_reflection(ReflectionEntry(attempt_id=0, summary="r1"))
        wm.clear_all()
        assert len(wm.get_window()) == 0
        assert len(wm.get_reflections()) == 0

    def test_backward_compat_alias(self) -> None:
        st = ShortTerm(max_len=3)
        assert isinstance(st, WorkingMemory)


class TestLongTerm:
    @pytest.mark.asyncio
    async def test_add_and_search(self) -> None:
        lt = LongTerm()
        await lt.add("hello world", [1.0, 0.0, 0.0])
        await lt.add("goodbye world", [0.0, 1.0, 0.0])
        results = await lt.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0] == "hello world"

    @pytest.mark.asyncio
    async def test_empty_search(self) -> None:
        lt = LongTerm()
        results = await lt.search([1.0, 0.0], top_k=5)
        assert results == []


class TestHistory:
    @pytest.mark.asyncio
    async def test_save_and_get_task(self) -> None:
        h = History()
        task = TaskRecord(query="test")
        await h.save_task(task)
        got = await h.get_task(task.task_id)
        assert got is not None
        assert got.query == "test"

    @pytest.mark.asyncio
    async def test_add_step(self) -> None:
        h = History()
        task = TaskRecord(query="test")
        await h.save_task(task)
        step = StepRecord(thought="thinking")
        await h.add_step(task.task_id, step)
        steps = await h.get_steps(task.task_id)
        assert len(steps) == 1
        assert steps[0].thought == "thinking"


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        reg = ToolRegistry()
        td = ToolDef(name="shell_exec", description="run shell", parameters={})
        reg.register(td)
        assert reg.get("shell_exec") is not None
        assert reg.get("nonexistent") is None

    def test_search(self) -> None:
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
        results = reg.search("run a shell command")
        assert len(results) > 0
        assert results[0].name == "shell_exec"

    def test_list_all(self) -> None:
        reg = ToolRegistry()
        reg.register(ToolDef(name="a", description="a", parameters={}))
        reg.register(ToolDef(name="b", description="b", parameters={}))
        assert len(reg.list_all()) == 2
