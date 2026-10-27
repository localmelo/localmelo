"""The web workspace keeps its procedural store beside its history."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from localmelo.melo.contracts.providers import BaseLLMProvider  # noqa: E402
from localmelo.melo.memory.procedures.store import SqliteProcedures  # noqa: E402
from localmelo.melo.schema import Message, ToolDef  # noqa: E402
from localmelo.support.gateway.playground import SmokePlayground  # noqa: E402


class FakeLLM(BaseLLMProvider):
    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        return Message(role="assistant", content="Hello")

    async def close(self) -> None:
        pass


@pytest.fixture()
async def manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[SmokePlayground]:
    manager = SmokePlayground(memory_dir=tmp_path)
    monkeypatch.setattr(manager, "_build_llm_provider", lambda *args: FakeLLM())
    yield manager
    await manager.close_all()


async def create(manager: SmokePlayground) -> str:
    payload = await manager.create_session(
        chat_url="http://127.0.0.1:8000/v1", chat_model="Qwen-27B", adapter="omlx"
    )
    return str(payload["session_id"])


async def test_each_chat_keeps_its_store_in_the_workspace_file(
    manager: SmokePlayground, tmp_path: Path
) -> None:
    first = await create(manager)
    second = await create(manager)
    stores = [manager._sessions[sid].agent.hippo.procedures for sid in (first, second)]
    for store in stores:
        assert isinstance(store, SqliteProcedures)
        assert store.path == str(tmp_path / "procedures.db")
    # Each chat owns its connection: closing one leaves the other's open.
    assert stores[0] is not stores[1]
    await stores[1].export_rows()
    await manager.close(first)
    assert await stores[1].export_rows() == []
