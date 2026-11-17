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

# Request fields to add to the chat requests sent in this context, empty when
# none. The agent sets it around the one planning call it retries with
# thinking turned off (a thinking model that ended a turn with its answer only
# in its thinking), so a provider can merge it in without a signature change.
LLM_REQUEST_EXTRA_BODY: contextvars.ContextVar[Mapping[str, Any]] = (
    contextvars.ContextVar("LLM_REQUEST_EXTRA_BODY", default=MappingProxyType({}))
)

# The call kinds: runs send "plan" and "reflect", melo-retrieve's retrieval
# requests "retrieve", and the review of a step the rules left open "review"
# (#339). History's llm_calls rows and melo-retrieve's reader of them share
# this vocabulary.
LLM_REQUEST_KINDS: Final = ("plan", "reflect", "retrieve", "review")


def thinking_disabled(body: Mapping[str, Any]) -> bool:
    """Whether a request body tells the model not to think (#357).

    That is ``chat_template_kwargs.enable_thinking`` set to false (vLLM,
    oMLX and other servers that render the model's chat template),
    ``think`` set to false (Ollama) or ``thinking.type`` set to
    ``"disabled"`` (hosted services such as DeepSeek). Anything else,
    including no setting, is a model that may think.
    """
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, Mapping) and kwargs.get("enable_thinking") is False:
        return True
    switch = body.get("thinking")
    if isinstance(switch, Mapping) and switch.get("type") == "disabled":
        return True
    return body.get("think") is False


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
    ``detail`` is what the server said when the message does not carry it (a
    streamed error names only the status); it is for reading, not for showing.
    ``served`` names the model the server serves now when it refused the one the
    request named because it was switched meanwhile (a 409 ``model_switched``);
    a provider asks again, once, for that one.
    ``switching_s`` is set while the server is switching its model (a 503
    ``switching``): the seconds it says to wait, 0.0 when it names none, and
    ``None`` for any other error. A provider waits that out rather than fail.
    ``notice`` is a sentence for the person when the provider has already put
    the cause right and the request is worth sending again (a model the server
    no longer has, swapped for the one it offers): the retry notice shows it
    alone, in place of "Model request failed".
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        context_overflow: bool = False,
        detail: str = "",
        served: str = "",
        switching_s: float | None = None,
        notice: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.context_overflow = context_overflow
        self.detail = detail
        self.served = served
        self.switching_s = switching_s
        self.notice = notice


@dataclass(frozen=True)
class StreamFragment:
    """A provisional delta or completed reply from an opted-in provider stream.

    Deltas are untrusted and must be filtered before display; they must not be
    persisted. Tool fragments contain only an index, never partial arguments.
    Only ``reply`` carries complete tool calls. Failures raise ProviderError.

    ``whole`` is set on a ``reply`` from a server that was asked to stream and
    sent the reply whole instead: it answered with one body, or ended the stream
    before it delivered anything and answered a second request. A provider that
    sets it has stopped asking that server to stream.
    """

    kind: Literal["content", "reasoning", "tool", "usage", "reply"]
    text: str = ""
    tool_index: int | None = None
    usage: dict[str, int] | None = None
    finish_reason: str = ""
    reply: Message | None = None
    whole: bool = False


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

    # Whether this provider's planning requests tell the model not to think
    # (see thinking_disabled). A provider sets it on the instance from its
    # request fields; a wrapper copies it. The agent then takes a reply's
    # words, wherever the server put them, as the model's answer (#357).
    thinking_off: bool = False
    # The request fields that turn the model's thinking off, for the one retry
    # of a reply that was only thinking (#469); empty when the provider cannot
    # or already asks for no thinking. A provider sets it on the instance; a
    # wrapper copies it.
    thinking_off_body: Mapping[str, Any] = MappingProxyType({})

    # The request key this provider can send Message.thinking back under:
    # None (never sends it), "auto" (the key the last reply came from),
    # "reasoning_content" or "reasoning" (an OpenAI-compatible server), or
    # "thinking" (Ollama native). A provider sets it on the instance; the
    # class default means the agent has no field to replay.
    reasoning_field: str | None = None
    # Whether the server rejects a request whose assistant turns omit the
    # reasoning it produced. Such a provider gets whole-unit eviction and
    # no cap.
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

    async def takes_images(self) -> bool | None:
        """Whether the server says :attr:`model` reads images; ``None`` when unknown.

        The agent asks once, when a tool first returns an image and
        ``AgentOptions.tool_images`` is ``"auto"``. The default does not know.
        """
        return None

    async def context_length(self) -> int | None:
        """The context length, in tokens, the server runs :attr:`model` with; ``None`` when unknown.

        Never the model's own maximum where the server may run it with less. The
        agent asks once, before its first run, when nothing sets its context
        budget (``AgentOptions.context_window``). The default does not know.
        """
        return None

    def output_limit(self) -> int | None:
        """The most output tokens a planning request asks for, when the provider sends a limit; else ``None``."""
        return None

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
