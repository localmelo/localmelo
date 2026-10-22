from __future__ import annotations

import contextvars
from abc import ABC, abstractmethod

from localmelo.melo.schema import Message, ToolDef

# 1-based attempt number of the LLM request being sent. The agent sets it
# around each request, retries included, so a provider can pass it on
# without a signature change: the HTTP chat providers send it as the
# ``X-Localmelo-Attempt`` header, which lets a proxy match its request log
# against a run's ``n_llm`` and ``llm_errors``.
LLM_REQUEST_ATTEMPT: contextvars.ContextVar[int] = contextvars.ContextVar(
    "LLM_REQUEST_ATTEMPT", default=1
)


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
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class BaseLLMProvider(ABC):
    """Abstract interface for chat/completion model backends.

    Any backend that can handle chat messages and (optionally) tool calls
    should subclass this.  One implementation (OpenAICompatLLM) covers all
    backends that expose the OpenAI ``/chat/completions`` format: MLC LLM,
    Ollama, vLLM, LMStudio, OpenAI, Groq, Together AI, etc.
    """

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
