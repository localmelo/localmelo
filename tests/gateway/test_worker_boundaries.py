"""Worker provider calls wait at the pause boundary."""

from __future__ import annotations

import asyncio

from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.workers import _WorkerEmbedding, _WorkerLLM


class RecordingLLM(BaseLLMProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[list[Message], list[ToolDef] | None]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append((messages, tools))
        return Message(role="assistant", content="reply")

    async def close(self) -> None:
        pass


class RecordingEmbedding(BaseEmbeddingProvider):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0] for _ in texts]

    async def close(self) -> None:
        pass


async def test_worker_providers_wait_and_release() -> None:
    gate = ChatGate(policy="off")
    ready = asyncio.Event()
    llm = RecordingLLM()
    embedding = RecordingEmbedding()
    client = gate.client("worker")
    worker_llm = _WorkerLLM(llm, client, ready)
    worker_embedding = _WorkerEmbedding(embedding, client, ready)
    messages = [Message(role="user", content="hello")]
    chat = asyncio.create_task(worker_llm.chat(messages))
    embed = asyncio.create_task(worker_embedding.embed(["hello"]))
    try:
        await asyncio.sleep(0)
        assert not chat.done() and not embed.done()
        assert not llm.calls and not embedding.calls and not gate._calls
        ready.set()
        async with asyncio.timeout(3):
            reply, vectors = await asyncio.gather(chat, embed)
        assert reply.content == "reply"
        assert vectors == [[1.0]]
        assert llm.calls == [(messages, None)]
        assert embedding.calls == [["hello"]]

        ready.clear()
        chat_again = asyncio.create_task(worker_llm.chat(messages))
        try:
            await asyncio.sleep(0)
            assert not chat_again.done() and len(llm.calls) == 1
            ready.set()
            async with asyncio.timeout(3):
                await chat_again
            assert len(llm.calls) == 2
        finally:
            chat_again.cancel()
            await asyncio.gather(chat_again, return_exceptions=True)
    finally:
        chat.cancel()
        embed.cancel()
        await asyncio.gather(chat, embed, return_exceptions=True)
        await gate.close()
