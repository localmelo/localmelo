"""A fake embedding model and a fake chat model for the memory-across-chats tests.

The vectors are concepts, not a model: one dimension per concept below, set when
the text holds one of its words (English or Chinese), plus a few small hashed
dimensions for the other words. So "weekly sync" lands near "group meeting" and
"我的组会" near "my group meeting", which BM25 alone cannot match.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from localmelo.melo.contracts.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderError,
)
from localmelo.melo.schema import Message, ToolDef

CONCEPTS: dict[str, tuple[str, ...]] = {
    "meeting": ("meeting", "sync", "standup", "组会", "会议"),
    "drink": ("drink", "tea", "coffee", "喝", "茶"),
    "home": ("live", "home", "city", "住"),
    "dentist": ("dentist", "teeth", "牙医"),
    "lab": ("lab", "robotics", "实验室"),
    "ci": ("ci", "test", "fixture", "测试"),
    "answers": ("answers", "short", "brief", "简短", "回答"),
    "landlord": ("landlord", "rent", "房东"),
    "cat": ("cat", "猫"),
}
_RESIDUAL = 32
_QUERY = re.compile(r"^Instruct: .*?\nQuery: ", re.DOTALL)


def vector(text: str) -> list[float]:
    folded = _QUERY.sub("", text).lower()
    out = [0.0] * (len(CONCEPTS) + _RESIDUAL)
    for index, words in enumerate(CONCEPTS.values()):
        if any(word in folded for word in words):
            out[index] = 1.0
    for word in re.findall(r"[a-z]+", folded):
        if not any(word in words for words in CONCEPTS.values()):
            slot = int(hashlib.sha256(word.encode()).hexdigest(), 16) % _RESIDUAL
            out[len(CONCEPTS) + slot] += 0.15
    if not any(out):
        out[-1] = 1.0
    norm = math.sqrt(sum(v * v for v in out))
    return [v / norm for v in out]


class FakeEmbedding(BaseEmbeddingProvider):
    """In process; *delay_s* makes every call that slow, *fail* makes it raise."""

    def __init__(self, model: str = "fake-embed", delay_s: float = 0.0) -> None:
        self.model = model
        self.delay_s = delay_s
        self.fail = False
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("the embedding server is down")
        return [vector(text) for text in texts]

    async def close(self) -> None:
        return None


class EmbeddingServer:
    """``POST /v1/embeddings`` on 127.0.0.1 and a port the system picks."""

    def __init__(self) -> None:
        self.inputs: list[list[str]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                texts = list(body["input"])
                server.inputs.append(texts)
                data = [
                    {"index": i, "embedding": vector(text)}
                    for i, text in enumerate(texts)
                ]
                raw = json.dumps({"data": data, "model": body["model"]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}/v1"

    def start(self) -> EmbeddingServer:
        self._thread.start()
        return self

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


class FakeChat(BaseLLMProvider):
    """A chat model in process that answers the retrieval request.

    It picks the catalog lines (``[alias] label · text``) that hold *needle*;
    *reply* sends that text instead, *delay_s* makes it slow, *fail* makes it
    raise a provider error. ``usage`` is what the server would report.
    """

    def __init__(
        self,
        needle: str = "",
        *,
        reply: str | None = None,
        delay_s: float = 0.0,
        fail: bool = False,
    ) -> None:
        self.model = "fake-chat"
        self.needle = needle
        self.reply = reply
        self.delay_s = delay_s
        self.fail = fail
        self.calls: list[str] = []
        self.last_usage: dict[str, int] | None = None

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        assert tools is None
        user = str(messages[-1].content)
        self.calls.append(user)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail:
            raise ProviderError("the chat server is down")
        picked = [
            alias
            for alias, line in re.findall(r"^\[(\w+)\] (.*)$", user, re.MULTILINE)
            if self.needle.lower() in line.lower()
        ]
        content = self.reply if self.reply is not None else json.dumps({"ids": picked})
        self.last_usage = {"prompt_tokens": 321, "completion_tokens": 9}
        return Message("assistant", content, usage=dict(self.last_usage))

    async def close(self) -> None:
        return None
