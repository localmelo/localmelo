"""The ``llm`` retriever: its request, the catalog, the reply, the fallbacks.

Requests go to scripted ``send`` functions, or through
:func:`send_from_provider` to ``OpenAICompatLLM`` on an
``httpx.MockTransport``: no network. Every expected text was written by
hand from the rules of the R3 spec (Requirement 10).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from localmelo.melo.contracts.providers import (
    LLM_REQUEST_KIND,
    BaseLLMProvider,
    ProviderError,
)
from localmelo.melo.memory.retrieval import Doc, Hit, send_from_provider
from localmelo.melo.memory.retrieval.bm25 import BM25Retriever
from localmelo.melo.memory.retrieval.llm import (
    CATALOG_BLOCK_CHARS,
    CATALOG_CHARS,
    CATALOG_ENTRY_CHARS,
    RETRIEVE_SYSTEM_PROMPT,
    RETRIEVE_TIMEOUT_S,
    LLMRetriever,
)
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


class ScriptedSend:
    """A ``send`` that records each request and answers from a script.

    Each item is the content of a reply, a whole reply, or an exception to
    raise; the last item answers every later request.
    """

    def __init__(self, *script: str | Message | BaseException) -> None:
        self.script = script
        self.calls: list[list[Message]] = []

    async def __call__(self, messages: list[Message]) -> Message:
        self.calls.append(messages)
        item = self.script[min(len(self.calls), len(self.script)) - 1]
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Message):
            return item
        return Message("assistant", item)

    @property
    def user_text(self) -> str:
        return self.calls[-1][1].content


def _ids(hits: list[Hit]) -> list[str]:
    return [hit.doc_id for hit in hits]


QUERY = "foldr argument order"
DOCS = [
    Doc("t:n1", "note", "the foldr argument order is reversed"),
    Doc("t:n2", "note", "foldr argument order fixed in the draft"),
    Doc("t:d1", "draft", "def foldr(function, list, initial): pass"),
    Doc("t:n3", "note", "the foldr argument order needs a test"),
    Doc("t:t1", "thought", "foldr folds from the right; argument order matters"),
]


# ── The request ─────────────────────────────────────────────────────────────


def test_the_system_prompt() -> None:
    assert RETRIEVE_SYSTEM_PROMPT == (
        "You select memory entries that help an agent with its next step. Reply "
        'with JSON only: {"ids": [...]}, most useful first, at most the number '
        'asked for, or {"ids": []}.'
    )
    assert (CATALOG_ENTRY_CHARS, CATALOG_CHARS, CATALOG_BLOCK_CHARS) == (
        300,
        12_000,
        3_000,
    )
    assert RETRIEVE_TIMEOUT_S == 30.0


async def test_the_exact_request_for_a_small_catalog() -> None:
    docs = [
        Doc(
            "t-en:n1",
            "note",
            "plan: read list_ops.py for the signatures\nfound: two tests",
            title="plan: read list_ops.py for the signatures",
            meta={"task_id": "t-en", "step": 2},
        ),
        Doc(
            "t-en:d2",
            "draft",
            "def append(a, b):\n    return a + b",
            title="list_ops.py, 2 lines: def append(a, b):",
            meta={"task_id": "t-en", "step": 4},
        ),
        # No title: the text's first non-blank line, whitespace collapsed.
        Doc(
            "t-en:t2.1",
            "thought",
            "foldr applies the function from the right.\n\nThe accumulator is last.",
            meta={"task_id": "t-en", "step": 4},
        ),
        Doc(
            "t-en:n3",
            "note",
            "  \n  observed:   read   list_ops.py  \nmore",
            meta={"task_id": "t-en", "step": 0},
        ),
        Doc("t-en:r4", "reflection", "Next attempt: start from the tests."),
        # A step that is not a positive int is left out.
        Doc(
            "s1:g1",
            "digest",
            "The user works on list ops.",
            title="  Session \t digest ",
            meta={"step": True},
        ),
        Doc("m:7", "long", "The user prefers tabs.", meta={"step": "3"}),
    ]
    send = ScriptedSend('{"ids": []}')
    result = await LLMRetriever(send).search(
        "  write list_ops.py; open: foldr argument order?  ",
        docs,
        k=3,
        exclude={"t-en:n3"},
    )
    assert (result.hits, result.retriever, result.fallback) == ([], "llm", "")

    [messages] = send.calls
    assert [message.role for message in messages] == ["system", "user"]
    assert messages[0].content == RETRIEVE_SYSTEM_PROMPT
    assert messages[1].content == (
        "<memory>\n"
        "[n1] note · step 2 · plan: read list_ops.py for the signatures\n"
        "[d1] draft · step 4 · list_ops.py, 2 lines: def append(a, b):\n"
        "[t1] thinking · step 4 · foldr applies the function from the right.\n"
        "[n2] note · observed: read list_ops.py\n"
        "[r1] reflection · Next attempt: start from the tests.\n"
        "[g1] digest · Session digest\n"
        "[m1] memory · The user prefers tabs.\n"
        "</memory>\n"
        "Do not return: n2\n"
        "Query: write list_ops.py; open: foldr argument order?\n"
        "Return at most 3 ids."
    )
    assert all(not message.tool_calls for message in messages)


async def test_an_appended_doc_keeps_the_request_prefix() -> None:
    send = ScriptedSend('{"ids": []}')
    retriever = LLMRetriever(send)
    await retriever.search(QUERY, DOCS, k=3)
    appended = [*DOCS, Doc("t:n4", "note", "foldr argument order: accumulator last")]
    await retriever.search(QUERY, appended, k=3)

    old, new = send.calls
    assert new[0].content == old[0].content
    old_entries = old[1].content[: old[1].content.index("</memory>\n")]
    assert old_entries.endswith(
        "[t1] thinking · foldr folds from the right; argument order matters\n"
    )
    assert new[1].content.startswith(old_entries)
    assert (
        new[1]
        .content[len(old_entries) :]
        .startswith("[n4] note · foldr argument order: accumulator last\n</memory>\n")
    )


async def test_aliases_count_per_kind_and_excluded_docs_keep_their_place() -> None:
    send = ScriptedSend('{"ids": []}')
    retriever = LLMRetriever(send)
    await retriever.search(QUERY, DOCS, k=3)
    plain = send.user_text
    await retriever.search(QUERY, DOCS, k=3, exclude={"t:n2", "t:d1", "t:n9"})
    excluded = send.user_text

    lines = plain.splitlines()
    assert [line.split("]")[0] + "]" for line in lines[1:6]] == [
        "[n1]",
        "[n2]",
        "[d1]",
        "[n3]",
        "[t1]",
    ]
    # The same catalog; the excluded docs among DOCS are named, in catalog
    # order, and t:n9, which is not among them, is ignored.
    assert excluded.splitlines()[:7] == lines[:7]
    assert excluded.splitlines()[7] == "Do not return: n2, d1"
    assert "Do not return" not in plain


async def test_a_long_entry_line_is_shortened() -> None:
    docs = [Doc("t:n1", "note", "x", title="foldr " * 80, meta={"step": 12})]
    send = ScriptedSend('{"ids": []}')
    await LLMRetriever(send).search(QUERY, docs, k=1)
    line = send.user_text.splitlines()[1]
    assert len(line) <= CATALOG_ENTRY_CHARS
    assert line.startswith("[n1] note · step 12 · foldr foldr")
    assert line.endswith("foldr…")


# ── The catalog cap ─────────────────────────────────────────────────────────


def _sized_docs(count: int) -> list[Doc]:
    """*count* notes whose entry lines are 100 characters, 101 with "\\n"."""
    docs = []
    for i in range(1, count + 1):
        head = f"[n{i}] note · "
        docs.append(Doc(f"t:n{i}", "note", "foldr", title="x" * (100 - len(head))))
    return docs


async def _catalog_lines(count: int, **search: Any) -> tuple[list[str], str]:
    send = ScriptedSend(search.pop("reply", '{"ids": []}'))
    await LLMRetriever(send).search(QUERY, _sized_docs(count), k=3, **search)
    text = send.user_text
    body = text[len("<memory>\n") : text.index("</memory>\n")]
    lines = body.splitlines()
    assert all(len(line) == 100 for line in lines)
    return lines, text


class TestCatalogCap:
    async def test_a_catalog_that_fits_is_whole(self) -> None:
        # 118 lines of 101 characters: 11,918, within 12,000.
        lines, _ = await _catalog_lines(118)
        assert len(lines) == 118
        assert lines[0].startswith("[n1] ")

    async def test_the_oldest_lines_go_in_whole_blocks(self) -> None:
        # 119 lines: 12,019, 19 over. m = 3,000, and the shortest run of
        # oldest lines of at least 3,000 characters is 30 lines (3,030).
        lines, _ = await _catalog_lines(119)
        assert len(lines) == 89
        assert lines[0].startswith("[n31] ")
        assert lines[-1].startswith("[n119] ")

    async def test_the_cut_moves_only_when_m_does(self) -> None:
        _, first = await _catalog_lines(119)
        # 148 lines: 14,948, 2,948 over: m is still 3,000, the cut stays.
        lines, later = await _catalog_lines(148)
        assert lines[0].startswith("[n31] ")
        entries = first[: first.index("</memory>\n")]
        assert later.startswith(entries)
        # 149 lines: 15,049, 3,049 over: m = 6,000, 60 lines (6,060) go.
        lines, _ = await _catalog_lines(149)
        assert lines[0].startswith("[n61] ")
        assert sum(len(line) + 1 for line in lines) == 89 * 101 <= CATALOG_CHARS

    async def test_a_dropped_doc_cannot_be_returned_or_excluded(self) -> None:
        send = ScriptedSend('{"ids": ["n1", "n30", "n31", "n119"]}')
        result = await LLMRetriever(send).search(
            QUERY, _sized_docs(119), k=5, exclude={"t:n5", "t:n40"}
        )
        assert _ids(result.hits) == ["t:n31", "t:n119"]
        assert "Do not return: n40\n" in send.user_text


# ── The reply ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"ids": ["n2", "d1"]}', ["t:n2", "t:d1"]),
        ('```json\n{"ids": ["n2", "d1"]}\n```', ["t:n2", "t:d1"]),
        ('Here you go: {"ids": ["d1"]} I hope it helps.', ["t:d1"]),
        ('I read {"note": "x"} first, then chose {"ids": ["n1"]}.', ["t:n1"]),
        ('Mind {braces}. {"ids": ["t1"]}', ["t:t1"]),
        ('["n1", "t1"]', ["t:n1", "t:t1"]),
        ('{"ids": ["n1"], "why": "it names foldr"}', ["t:n1"]),
        ('{"ids": ["[n2]", " N1 ", "[D1]"]}', ["t:n2", "t:n1", "t:d1"]),
        ('{"ids": ["n1", "n1", "N1", "n2"]}', ["t:n1", "t:n2"]),
        ('{"ids": ["x9", "n7", 3, null, "t:n1", "n2"]}', ["t:n2"]),
        ('{"ids": ["n1", "n2", "d1", "n3"]}', ["t:n1", "t:n2", "t:d1"]),
    ],
)
async def test_ids_are_read_tolerantly(content: str, expected: list[str]) -> None:
    result = await LLMRetriever(ScriptedSend(content)).search(QUERY, DOCS, k=3)
    assert (result.retriever, result.fallback) == ("llm", "")
    assert _ids(result.hits) == expected
    assert [hit.rank for hit in result.hits] == list(range(1, len(expected) + 1))
    assert [hit.score for hit in result.hits] == [
        1 / rank for rank in range(1, len(expected) + 1)
    ]


async def test_excluded_aliases_are_dropped() -> None:
    send = ScriptedSend('{"ids": ["n3", "n2", "n1"]}')
    result = await LLMRetriever(send).search(QUERY, DOCS, k=3, exclude={"t:n3"})
    assert _ids(result.hits) == ["t:n2", "t:n1"]


async def test_an_empty_list_is_no_hits_not_a_failure() -> None:
    reply = Message("assistant", '{"ids": []}', usage={"prompt_tokens": 90})
    result = await LLMRetriever(ScriptedSend(reply)).search(QUERY, DOCS, k=3)
    assert (result.hits, result.retriever, result.fallback) == ([], "llm", "")
    assert result.usage == {"prompt_tokens": 90}


async def test_only_content_is_read() -> None:
    reply = Message("assistant", "", thinking='{"ids": ["n1"]}')
    result = await LLMRetriever(ScriptedSend(reply)).search(QUERY, DOCS, k=3)
    assert result.fallback == "parse"


# ── Fallbacks ───────────────────────────────────────────────────────────────


async def _bm25_hits(**search: Any) -> list[Hit]:
    return (await BM25Retriever().search(QUERY, DOCS, k=3, **search)).hits


@pytest.mark.parametrize(
    "content",
    [
        "no JSON here",
        "",
        '"n1"',
        '{"answer": ["n1"]}',
        '{"ids": "n1"}',
        '{"ids": {"n1": 1}}',
        'first {"ids": "n1"}, then {"ids": ["n2"]}',
        '["n1", ',
    ],
)
async def test_a_reply_without_a_list_of_ids_falls_back(content: str) -> None:
    usage = {"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127}
    send = ScriptedSend(Message("assistant", content, usage=usage))
    result = await LLMRetriever(send).search(QUERY, DOCS, k=3, exclude={"t:n1"})
    assert (result.retriever, result.fallback) == ("bm25", "parse")
    assert result.hits == await _bm25_hits(exclude={"t:n1"})
    assert result.hits
    # The response came back: its usage stays.
    assert result.usage == usage


async def test_a_timeout_falls_back() -> None:
    cancelled = asyncio.Event()

    async def slow(messages: list[Message]) -> Message:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return Message("assistant", '{"ids": ["n1"]}')

    result = await LLMRetriever(slow, timeout_s=0.05).search(QUERY, DOCS, k=3)
    assert (result.retriever, result.fallback, result.usage) == ("bm25", "timeout", {})
    assert result.hits == await _bm25_hits()
    assert cancelled.is_set()


async def test_a_provider_error_falls_back() -> None:
    send = ScriptedSend(ProviderError("HTTP 503", status_code=503, retryable=True))
    result = await LLMRetriever(send).search(QUERY, DOCS, k=3)
    assert (result.retriever, result.fallback, result.usage) == (
        "bm25",
        "provider_error",
        {},
    )
    assert result.hits == await _bm25_hits()
    assert len(send.calls) == 1  # no retry


async def test_the_fallback_takes_the_floor_options() -> None:
    send = ScriptedSend(ProviderError("down"))
    docs = [Doc("t:n1", "note", "read list_ops.py: 120 chars")]
    floored = await LLMRetriever(send).search("read list_ops.py", docs, k=3)
    assert floored.hits == []
    unfloored = await LLMRetriever(send, floor_terms=0).search(
        "read list_ops.py", docs, k=3
    )
    assert _ids(unfloored.hits) == ["t:n1"]


async def test_another_exception_propagates() -> None:
    send = ScriptedSend(RuntimeError("a bug"))
    with pytest.raises(RuntimeError, match="a bug"):
        await LLMRetriever(send).search(QUERY, DOCS, k=3)


async def test_a_cancellation_propagates() -> None:
    started = asyncio.Event()

    async def hang(messages: list[Message]) -> Message:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("not reached")

    task = asyncio.create_task(LLMRetriever(hang).search(QUERY, DOCS, k=3))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_cancelled_error_from_send_propagates() -> None:
    send = ScriptedSend(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await LLMRetriever(send).search(QUERY, DOCS, k=3)


async def test_usage_keeps_the_integer_keys() -> None:
    usage = {"prompt_tokens": 50, "completion_tokens": 4, "flag": True}
    reply = Message("assistant", '{"ids": ["n1"]}', usage=usage)
    result = await LLMRetriever(ScriptedSend(reply)).search(QUERY, DOCS, k=3)
    assert result.usage == {"prompt_tokens": 50, "completion_tokens": 4}


# ── send_from_provider ──────────────────────────────────────────────────────


class Server:
    """A fake OpenAI-compatible server on an ``httpx.MockTransport``."""

    def __init__(self, status: int = 200, content: str = '{"ids": ["n2"]}') -> None:
        self.status = status
        self.content = content
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, text="busy")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": self.content}}
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 4},
            },
        )

    def provider(self) -> OpenAICompatLLM:
        return OpenAICompatLLM(
            "http://retrieval.test/v1",
            "qwen-test",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            transport=httpx.MockTransport(self.handler),
        )


async def test_send_from_provider_sends_one_plain_chat_request() -> None:
    server = Server()
    llm = server.provider()
    try:
        result = await LLMRetriever(send_from_provider(llm)).search(QUERY, DOCS, k=3)
    finally:
        await llm.close()

    assert (_ids(result.hits), result.retriever, result.fallback) == (
        ["t:n2"],
        "llm",
        "",
    )
    assert result.usage == {
        "prompt_tokens": 50,
        "completion_tokens": 4,
        "total_tokens": 54,
    }
    [request] = server.requests
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["X-Localmelo-Attempt"] == "1"
    body = json.loads(request.content)
    assert "tools" not in body
    assert "tool_choice" not in body
    assert body["model"] == "qwen-test"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == RETRIEVE_SYSTEM_PROMPT
    assert body["messages"][1]["content"].startswith("<memory>\n[n1] note · ")


async def test_send_from_provider_does_not_retry() -> None:
    server = Server(status=503)
    llm = server.provider()
    try:
        result = await LLMRetriever(send_from_provider(llm)).search(QUERY, DOCS, k=3)
    finally:
        await llm.close()
    assert (result.retriever, result.fallback) == ("bm25", "provider_error")
    assert len(server.requests) == 1


_MISSING = object()


class UsageLLM(BaseLLMProvider):
    """A provider that sets ``last_usage`` (unless it is _MISSING) in chat()."""

    def __init__(self, last_usage: object = _MISSING) -> None:
        self.next_usage = last_usage
        self.tools: list[list[ToolDef] | None] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.tools.append(tools)
        await asyncio.sleep(0)
        if self.next_usage is not _MISSING:
            self.last_usage = self.next_usage  # type: ignore[assignment]
        return Message(
            "assistant",
            '{"ids": ["n1"]}',
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )

    async def close(self) -> None:
        pass


async def test_send_from_provider_takes_last_usage() -> None:
    llm = UsageLLM(
        {
            "prompt_tokens": 10,
            "cached_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 12,
            "reasoning_tokens": 0,
            "flag": True,
            "ratio": 0.5,
            "name": "x",
        }
    )
    reply = await send_from_provider(llm)([Message("user", "hi")])
    assert reply.usage == {
        "prompt_tokens": 10,
        "cached_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 12,
        "reasoning_tokens": 0,
    }
    assert llm.tools == [None]

    result = await LLMRetriever(send_from_provider(llm)).search(QUERY, DOCS, k=3)
    assert result.usage["cached_tokens"] == 8


@pytest.mark.parametrize("last_usage", [_MISSING, None, [("prompt_tokens", 99)], 7])
async def test_send_from_provider_keeps_the_reply_usage_otherwise(
    last_usage: object,
) -> None:
    llm = UsageLLM(last_usage)
    reply = await send_from_provider(llm)([Message("user", "hi")])
    assert reply.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }


async def test_cached_tokens_reach_the_result_through_the_real_provider() -> None:
    # R1: OpenAICompatLLM keeps the whole usage in last_usage, and every
    # retrieval request names its kind.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": '{"ids": ["n2"]}'}}
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 4,
                    "total_tokens": 54,
                    "prompt_tokens_details": {"cached_tokens": 32},
                    "completion_tokens_details": {"reasoning_tokens": 3},
                },
            },
        )

    llm = OpenAICompatLLM(
        "http://retrieval.test/v1",
        "qwen-test",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await LLMRetriever(send_from_provider(llm)).search(QUERY, DOCS, k=3)
    finally:
        await llm.close()
    assert result.usage == {
        "prompt_tokens": 50,
        "completion_tokens": 4,
        "total_tokens": 54,
        "cached_tokens": 32,
        "reasoning_tokens": 3,
    }
    assert [r.headers["X-Localmelo-Call-Kind"] for r in requests] == ["retrieve"]
    # The kind is set only while the request is sent.
    assert LLM_REQUEST_KIND.get() == ""
