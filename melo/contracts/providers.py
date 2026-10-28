from __future__ import annotations

import contextvars
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal

from localmelo.melo.schema import Message, ToolDef

# 1-based attempt number of the LLM request being sent. The agent sets it
# around each request, retries included, so a provider can pass it on
# without a signature change: the HTTP chat providers send it as the
# ``X-Localmelo-Attempt`` header, which lets a proxy match its request log
# against a run's ``n_llm`` and ``llm_errors``.
LLM_REQUEST_ATTEMPT: contextvars.ContextVar[int] = contextvars.ContextVar(
    "LLM_REQUEST_ATTEMPT", default=1
)

# Call kind of the LLM request being sent, one of LLM_REQUEST_KINDS, or ""
# when no caller set one. The agent sets it around each request, retries
# included, as it sets LLM_REQUEST_ATTEMPT, so a provider can pass it on
# without a signature change: the HTTP chat providers send a non-empty
# kind as the ``X-Localmelo-Call-Kind`` header, which tells a proxy a
# planning request from a reflection.
LLM_REQUEST_KIND: contextvars.ContextVar[str] = contextvars.ContextVar(
    "LLM_REQUEST_KIND", default=""
)

# The call kinds: runs send "plan" and "reflect", Chat.estimate_steps sends
# "estimate", and melo-retrieve's retrieval requests "retrieve". History's
# llm_calls rows and melo-retrieve's reader of them share this vocabulary.
LLM_REQUEST_KINDS: Final = ("plan", "reflect", "retrieve", "estimate")


class ProviderError(Exception):
    """A request to a model server failed.

    Providers raise it in place of their transport's exceptions, so the
    agent can retry the request or end the run with ``provider_error``
    without knowing how the provider reaches its server.

    ``status_code`` is the HTTP status of the failed response, or ``None``
    when no response came back (a timeout, a connection failure).
    ``retryable`` says whether sending the same request again may succeed:
    true for timeouts, transport errors and HTTP 429, 502, 503 and 504;
    false for every other error status and for a malformed response body.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        context_overflow: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.context_overflow = context_overflow


@dataclass(frozen=True)
class StreamFragment:
    """A provisional delta or completed reply from an opted-in provider stream.

    Deltas are untrusted and must be filtered before display; they must not be
    persisted. Tool fragments contain only an index, never partial arguments.
    Only ``reply`` carries complete tool calls. Failures raise ProviderError.
    """

    kind: Literal["content", "reasoning", "tool", "usage", "reply"]
    text: str = ""
    tool_index: int | None = None
    usage: dict[str, int] | None = None
    finish_reason: str = ""
    reply: Message | None = None


class BaseLLMProvider(ABC):
    """Abstract interface for chat/completion model backends.

    Any backend that can handle chat messages and (optionally) tool calls
    should subclass this.  One implementation (OpenAICompatLLM) covers all
    backends that expose the OpenAI ``/chat/completions`` format: MLC LLM,
    Ollama, vLLM, LMStudio, OpenAI, Groq, Together AI, etc.

    After a successful :meth:`chat`, :attr:`last_reasoning_key` names the
    response key the reasoning came from and :attr:`last_usage` holds the
    whole usage the server reported; the agent records both with the
    reply. A provider that sets neither keeps the class defaults, ``""``
    and ``None``. Both are class attributes, so a wrapper that forwards
    attributes reads the defaults unless it copies them from the provider
    it wraps right after that provider's ``chat()`` returns.

    :attr:`model` names the model the provider's requests go to; the
    procedural store keeps step templates per model. The class default
    ``""`` means unknown, and so does ``"default"``: a run on such a
    provider shows past runs' facts but never a bar. A wrapper copies it
    from the provider it wraps. A subclass keeps it a plain attribute or a
    property with a setter: mypy refuses a read-only property in place of
    the base class's writable attribute.
    """

    # The id of the model this provider's requests go to, such as
    # "qwen3-8b"; "" when unknown. Built-in providers set it on the
    # instance from their configuration.
    model: str = ""

    # The key of the response that the last successful chat() read the
    # reasoning (``Message.thinking``) from, such as "reasoning_content", or
    # "" for none. A provider sets it before chat() returns, with no await
    # in between; one that never sets it keeps this default.
    last_reasoning_key: str = ""
    # The whole usage the server reported for the last successful chat(),
    # token counts only: Message.usage's keys and any the server adds, such
    # as "cached_tokens". Set as last_reasoning_key is; None when unknown.
    last_usage: dict[str, int] | None = None

    # The request key this provider can send Message.thinking back under:
    # None (never sends it), "auto" (the key the last reply came from),
    # "reasoning_content" or "reasoning" (an OpenAI-compatible server), or
    # "thinking" (Ollama native). A provider sets it on the instance; the
    # class default means the agent has no field to replay.
    reasoning_field: str | None = None
    # Whether the server rejects a request whose assistant turns omit the
    # reasoning it produced. Such a provider gets whole-unit eviction and
    # no cap; ``drop`` is refused for it.
    requires_reasoning_replay: bool = False
    # Extra body fields for one call kind, merged over the provider's
    # ``extra_body`` for that request (R5's --retrieval-extra-body). Empty
    # and immutable by default, so a provider without one is unaffected.
    extra_body_by_kind: Mapping[str, Mapping[str, Any]] = MappingProxyType({})

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> Message:
        """Send messages and return the assistant response.

        A failed request should raise :class:`ProviderError`; the agent
        retries it when ``retryable`` is true.
        """

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> AsyncGenerator[StreamFragment, None]:
        """Opt in to streaming; providers without it return one whole reply.

        Close the iterator on early exit. Cancellation propagates and releases
        the provider's in-flight response; errors raise ``ProviderError``.
        """
        yield StreamFragment(kind="reply", reply=await self.chat(messages, tools))

    @abstractmethod
    async def close(self) -> None:
        """Release underlying resources (HTTP connections, etc.)."""


class BaseEmbeddingProvider(ABC):
    """Abstract interface for embedding model backends.

    Any backend that can turn text into dense vectors should subclass this.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text.

        A failed request should raise :class:`ProviderError`.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release underlying resources."""
