"""``llm``: the chat model picks the documents from a catalog of them.

One chat request without tools: a fixed system prompt
(:data:`RETRIEVE_SYSTEM_PROMPT`), and a user message that lists every doc
as one catalog line under a short alias (``n3`` is the third note of
``docs``), names the excluded aliases and asks for at most *k* ids. The
catalog keeps excluded docs in their place, and a caller that only appends
to ``docs`` keeps every alias, so the request's prefix stays the same from
call to call and a server can reuse its cache.

On a timeout, a :class:`~localmelo.melo.contracts.providers.ProviderError`
or a reply without readable ids, an internal ``bm25`` retriever answers
instead, and the result names the reason in ``fallback``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import Any

from localmelo.melo.contracts.providers import BaseLLMProvider, ProviderError
from localmelo.melo.memory.retrieval.base import (
    Doc,
    RetrievalResult,
    Send,
    elapsed_ms,
    int_step,
    is_number,
    ranked,
    search_candidates,
)
from localmelo.melo.memory.retrieval.bm25 import (
    DEFAULT_FLOOR_TERMS,
    DEFAULT_RELATIVE_CUT,
    BM25Retriever,
)
from localmelo.melo.schema import Message

# Character caps, with their token equivalents at about 3.45 characters
# per token.
CATALOG_ENTRY_CHARS = 300  # ≈ 87 tokens: one catalog line
CATALOG_CHARS = 12_000  # ≈ 3,478 tokens: every catalog line, newlines included
CATALOG_BLOCK_CHARS = 3_000  # ≈ 870 tokens: old lines are dropped in steps of this

RETRIEVE_TIMEOUT_S = 30.0

RETRIEVE_SYSTEM_PROMPT = (
    "You select memory entries that help an agent with its next step. "
    'Reply with JSON only: {"ids": [...]}, most useful first, at most the '
    'number asked for, or {"ids": []}.'
)

# The alias letter and the catalog label of each doc kind.
_KIND_LETTERS = {
    "note": "n",
    "draft": "d",
    "thought": "t",
    "digest": "g",
    "reflection": "r",
    "long": "m",
}
_KIND_LABELS = {
    "note": "note",
    "draft": "draft",
    "thought": "thinking",
    "digest": "digest",
    "reflection": "reflection",
    "long": "memory",
}

# What an id in the reply is stripped of before it is read as an alias.
_ALIAS_STRIP = " \t\r\n\f\v[]"


class _ParseError(Exception):
    """The reply holds no readable list of ids."""


def _check_timeout(timeout_s: object) -> None:
    """Raise :class:`ValueError` unless *timeout_s* is a number above 0."""
    if not is_number(timeout_s) or not timeout_s > 0:
        raise ValueError(f"timeout_s must be a number > 0, got {timeout_s!r}")


class LLMRetriever:
    """Asks the chat model, through *send*, which docs help with the query.

    ``Hit.score`` is ``1 / rank``. ``usage`` is the response's, a parse
    failure included; ``{}`` when no response came back.
    """

    name: str

    def __init__(
        self,
        send: Send,
        *,
        timeout_s: float = RETRIEVE_TIMEOUT_S,
        floor_terms: int = DEFAULT_FLOOR_TERMS,
        relative_cut: float = DEFAULT_RELATIVE_CUT,
    ) -> None:
        _check_timeout(timeout_s)
        self.name = "llm"
        self.timeout_s = timeout_s
        self._send = send
        self._fallback = BM25Retriever(
            floor_terms=floor_terms, relative_cut=relative_cut
        )

    async def search(
        self,
        query: str,
        docs: Sequence[Doc],
        *,
        k: int,
        exclude: AbstractSet[str] = frozenset(),
    ) -> RetrievalResult:
        start = time.perf_counter()
        if not search_candidates(query, docs, k=k, exclude=exclude):
            return RetrievalResult(hits=[], retriever=self.name, ms=elapsed_ms(start))

        catalog = _catalog(docs)
        messages = [
            Message("system", RETRIEVE_SYSTEM_PROMPT),
            Message("user", _user_text(catalog, exclude, query, k)),
        ]
        usage: dict[str, int] = {}
        try:
            # No retries: one request, cut at timeout_s.
            reply = await asyncio.wait_for(self._send(messages), self.timeout_s)
            usage = _integer_items(reply.usage)
            ids = _reply_ids(reply.content or "")
        except TimeoutError:
            reason = "timeout"
        except ProviderError:
            reason = "provider_error"
        except _ParseError:
            reason = "parse"
        else:
            aliases = {alias: doc.doc_id for alias, doc, _line in catalog}
            chosen: list[str] = []
            for item in ids:
                if not isinstance(item, str):
                    continue
                doc_id = aliases.get(item.strip(_ALIAS_STRIP).lower())
                if doc_id is None or doc_id in exclude or doc_id in chosen:
                    continue
                chosen.append(doc_id)
                if len(chosen) == k:
                    break
            return RetrievalResult(
                hits=ranked(
                    (doc_id, 1 / rank) for rank, doc_id in enumerate(chosen, 1)
                ),
                retriever=self.name,
                ms=elapsed_ms(start),
                usage=usage,
            )

        answer = await self._fallback.search(query, docs, k=k, exclude=exclude)
        return RetrievalResult(
            hits=answer.hits,
            retriever=answer.retriever,
            ms=elapsed_ms(start),
            fallback=reason,
            usage=usage,
        )


def send_from_provider(llm: BaseLLMProvider) -> Send:
    """A :data:`~localmelo.melo.memory.retrieval.base.Send` on *llm*.

    Each call sends one ``llm.chat(messages, tools=None)``, with no retry.
    When the provider has a ``last_usage`` dict once ``chat()`` returns,
    its int items replace the reply's ``usage``, so usage keys that the
    reply's own three leave out (``cached_tokens``) reach the result.
    """

    async def send(messages: list[Message]) -> Message:
        reply = await llm.chat(messages, tools=None)
        # Read right after chat() returns, with no await in between, so the
        # value belongs to this call even while other calls run.
        last_usage = getattr(llm, "last_usage", None)
        if isinstance(last_usage, dict):
            reply.usage = _integer_items(last_usage)
        return reply

    return send


def _integer_items(usage: object) -> dict[str, int]:
    """The items of *usage*, a dict, whose values are ints (not bools)."""
    if not isinstance(usage, dict):
        return {}
    return {
        key: value
        for key, value in usage.items()
        if isinstance(key, str)
        and isinstance(value, int)
        and not isinstance(value, bool)
    }


# ── The request ──


def _catalog(docs: Sequence[Doc]) -> list[tuple[str, Doc, str]]:
    """The catalog of *docs*: ``(alias, doc, line)`` in ``docs`` order.

    Each line ends in ``\\n``. When the lines pass :data:`CATALOG_CHARS`
    in all, the shortest run of oldest lines whose length is at least m is
    dropped, m being the excess rounded up to a multiple of
    :data:`CATALOG_BLOCK_CHARS`. The rest then fits, and while only lines
    are appended the cut point moves only when m does.
    """
    counts: dict[str, int] = {}
    entries: list[tuple[str, Doc, str]] = []
    for doc in docs:
        counts[doc.kind] = counts.get(doc.kind, 0) + 1
        alias = f"{_KIND_LETTERS[doc.kind]}{counts[doc.kind]}"
        entries.append((alias, doc, _entry_line(alias, doc) + "\n"))
    total = sum(len(line) for _alias, _doc, line in entries)
    if total <= CATALOG_CHARS:
        return entries
    excess = total - CATALOG_CHARS
    drop = -(-excess // CATALOG_BLOCK_CHARS) * CATALOG_BLOCK_CHARS
    dropped = 0
    first = 0
    while first < len(entries) and dropped < drop:
        dropped += len(entries[first][2])
        first += 1
    return entries[first:]


def _entry_line(alias: str, doc: Doc) -> str:
    """``[<alias>] <label> · step <n> · <title>``, shortened to fit.

    The step part appears when ``meta["step"]`` is a positive int. The
    title is ``doc.title``, else the text's first non-blank line, with
    whitespace collapsed.
    """
    parts = [f"[{alias}] {_KIND_LABELS[doc.kind]}"]
    step = int_step(doc.meta)
    if step is not None and step > 0:
        parts.append(f"step {step}")
    title = _collapse(doc.title) or _collapse(_first_nonblank_line(doc.text))
    if title:
        parts.append(title)
    return _shorten(" · ".join(parts), CATALOG_ENTRY_CHARS)


def _user_text(
    catalog: Sequence[tuple[str, Doc, str]],
    exclude: AbstractSet[str],
    query: str,
    k: int,
) -> str:
    parts = ["<memory>\n"]
    parts.extend(line for _alias, _doc, line in catalog)
    parts.append("</memory>\n")
    hidden = [alias for alias, doc, _line in catalog if doc.doc_id in exclude]
    if hidden:
        parts.append(f"Do not return: {', '.join(hidden)}\n")
    parts.append(f"Query: {query.strip()}\n")
    parts.append(f"Return at most {k} ids.")
    return "".join(parts)


# ── The reply ──


def _reply_ids(content: str) -> list[Any]:
    """The list of ids in *content*; :class:`_ParseError` without one.

    A Markdown fence around the reply is stripped. When the rest is JSON as
    a whole, a list is the ids and an object gives its ``"ids"``.
    Otherwise the first balanced ``{...}`` that decodes to an object with
    ``"ids"`` gives them. The ids must be a list.

    The scan follows ``_json_values`` in ``melo/agent/agent.py``, which is
    private to the agent loop: after a ``{`` that starts no object, the
    search goes on from where the decoder gave up, and after an object,
    from its end.
    """
    body = _strip_code_fences(content)
    try:
        value = json.loads(body)
    except (ValueError, RecursionError):
        pass
    else:
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and "ids" in value:
            return _ids_list(value["ids"])
        raise _ParseError("the reply is JSON without ids")
    decoder = json.JSONDecoder()
    start = body.find("{")
    while start != -1:
        try:
            value, end = decoder.raw_decode(body, start)
        except json.JSONDecodeError as exc:
            start = body.find("{", max(exc.pos, start + 1))
            continue
        except RecursionError:
            break
        if isinstance(value, dict) and "ids" in value:
            return _ids_list(value["ids"])
        start = body.find("{", end)
    raise _ParseError("the reply holds no JSON object with ids")


def _ids_list(ids: object) -> list[Any]:
    if not isinstance(ids, list):
        raise _ParseError("the ids are not a list")
    return ids


def _strip_code_fences(text: str) -> str:
    """*text* without a Markdown code fence around it, such as ```json ... ```.

    A copy of ``_strip_code_fences`` in ``melo/agent/agent.py``, which is
    private to the agent loop.
    """
    body = text.strip()
    if body.startswith("```"):
        # The opening fence may name a language; drop its whole line.
        newline = body.find("\n")
        body = body[newline + 1 :] if newline != -1 else body[3:]
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


# ── Text helpers ──


def _shorten(text: str, width: int) -> str:
    """*text* in at most *width* characters, cut at a word boundary, ending in "…".

    Text that fits is returned as it is. When the cut falls inside the
    first word, there is no boundary to go back to, and the text is cut
    where *width* falls.

    A copy of ``_shorten`` in ``melo/agent/agent.py``, which is private to
    the agent loop.
    """
    if len(text) <= width:
        return text
    if width < 1:
        return ""
    head = text[: width - 1]
    if not text[len(head)].isspace() and not head[-1:].isspace():
        # The cut falls inside a word: leave that word out.
        words = head.rsplit(None, 1)
        if len(words) == 2:
            head = words[0]
    return head.rstrip() + "…"


def _collapse(text: str) -> str:
    """*text* with each run of whitespace made one space, and stripped."""
    return " ".join(text.split())


def _first_nonblank_line(text: str) -> str:
    """The first line of *text* that is not blank; "" if none."""
    for line in text.splitlines():
        if line.strip():
            return line
    return ""
