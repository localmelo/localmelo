"""Build the Agents that sessions run on.

A session's Agent works in a workspace, keeps each turn's answer for the
next one, offers every tool (no retrieval of tools per step) and lets a
shell or Python command run for :data:`SESSION_TOOL_TIMEOUT_S`. The model
comes from a URL given on the command line or from the config file, and a
surface may wrap the providers it builds (:func:`build_session_agent`), so
that a gate in front of every model call is the one the Agent's ``Chat``
was made with.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import logging
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.contracts.tools import BaseToolProvider
from localmelo.support.sessions.identity import (
    config_identity,
    kept_across_chats,
    url_identity,
)

if TYPE_CHECKING:
    from localmelo.support.config import Config
    from localmelo.support.memory import ChatMemory

# Shell and Python commands of a session stop after this many seconds: test
# suites, installs and builds take longer than a task's 30 s and 60 s.
SESSION_TOOL_TIMEOUT_S: Final = 600.0

# What a surface may do to a provider the builder built, before the Agent
# is made from it: the console puts its scheduler gate in front of every
# call this way.
LLMWrapper = Callable[[BaseLLMProvider], BaseLLMProvider]
EmbeddingWrapper = Callable[[BaseEmbeddingProvider], BaseEmbeddingProvider]

# The settings that keep each request an extension of the previous one. A config
# file that lacks a key has its default; one that sets a key keeps it.
CACHE_SETTINGS: Final = (
    "fixed_prefix",
    "window_eviction",
    "reflect_tool_guard",
    "cache_layout",
    "window_messages",
)


def configured_cache_options(
    cfg: Config | None,
    options: AgentOptions,
    keep: Collection[str] = (),
    *,
    backend: str | None = None,
) -> AgentOptions:
    """*options* with the config's cache settings, except the names in *keep*.

    A window that the config leaves unset is the default of *backend*, the key
    of the server the run talks to (``openai-compatible`` for a URL); it is
    ``cfg.chat_backend`` when None.
    """
    if cfg is None:
        return options
    values = {name: getattr(cfg.agent, name) for name in CACHE_SETTINGS}
    values["window_messages"] = cfg.agent.window_for(backend or cfg.chat_backend)
    changes = {
        name: value
        for name, value in values.items()
        if name not in keep and getattr(options, name) != value
    }
    return dataclasses.replace(options, **changes) if changes else options


# The settings of the pictures that tools return (AgentOptions.tool_images).
IMAGE_SETTINGS: Final = ("tool_images", "tool_images_keep", "tool_images_retire")


def configured_image_options(cfg: Config, options: AgentOptions) -> AgentOptions:
    """*options* with the config's ``tool_images`` keys; ValueError for a bad value."""
    values = {name: getattr(cfg.agent, name) for name in IMAGE_SETTINGS}
    if all(getattr(options, name) == value for name, value in values.items()):
        return options
    try:
        return dataclasses.replace(options, **values)
    except ValueError as exc:
        raise ValueError(f"[agent] {exc}") from None


def configured_context_options(
    cfg: Config | None, options: AgentOptions, overrides: Mapping[str, int | None]
) -> AgentOptions:
    """Resolve context settings together so flags may override individual keys."""
    names = ("context_window", "output_reserve", "safety_margin")
    fields: dict[str, Any] = {
        name: (
            overrides.get(name)
            if overrides.get(name) is not None
            else (
                getattr(cfg.agent, name, None)
                if cfg is not None and getattr(cfg.agent, name, None) is not None
                else getattr(options, name)
            )
        )
        for name in names
    }
    if all(getattr(options, name) == value for name, value in fields.items()):
        return options
    return dataclasses.replace(options, **fields)


def configured_session_options(
    cfg: Config | None,
    options: AgentOptions | None = None,
    *,
    explicit: Collection[str] = (),
    budget_overrides: Mapping[str, int | None] | None = None,
) -> AgentOptions:
    """Config defaults for a session; names in *explicit* retain the caller's values."""
    options = options or AgentOptions()
    overrides = {
        name: getattr(options, name)
        for name in ("context_window", "output_reserve", "safety_margin")
        if name in explicit
    }
    overrides.update(budget_overrides or {})
    options = configured_context_options(cfg, options, overrides)
    if cfg is None:
        return options
    if overrides:
        cfg = copy.deepcopy(cfg)
        for name in ("context_window", "output_reserve", "safety_margin"):
            setattr(cfg.agent, name, getattr(options, name))
    fields: dict[str, Any] = {
        name: getattr(cfg.agent, name)
        for name in (
            "reasoning",
            "replay_turns",
            "note_writer",
            *CACHE_SETTINGS,
            "captured_history",
            *IMAGE_SETTINGS,
        )
        if name not in explicit
    }
    if "window_messages" in fields:
        fields["window_messages"] = cfg.agent.window_for(cfg.chat_backend)
    if fields.get("reasoning", options.reasoning) != "legacy":
        for name, key in (
            ("retriever", "retriever"),
            ("recall_at", "recall_at"),
            ("recall_k", "k"),
            ("recall_kinds", "kinds"),
            ("recall_placement", "placement"),
            ("recall_min_score", "min_score"),
            ("cross_task_ratio", "cross_task_ratio"),
            ("cross_task_cap", "cross_task_cap"),
            ("cross_task_cold", "cross_task_cold"),
        ):
            if name not in explicit:
                value = getattr(cfg.retrieval, key)
                if name in ("recall_at", "recall_kinds"):
                    value = frozenset(value) if value else None
                fields[name] = value
    env = cfg.tool_environment()
    env.update(options.tool_env or {})
    fields["tool_env"] = env or None
    return dataclasses.replace(options, **fields)


def session_options(options: AgentOptions, workspace: str) -> AgentOptions:
    """*options* for a session's Agent that works in *workspace*.

    It replaces the session's workspace, answer recording, exposure, tool
    timeout and replay scope. The caller's ``mode`` survives, so the UI's
    ``interactive`` and the console's ``gateway`` reach the Agent.
    """
    return dataclasses.replace(
        options,
        workspace_root=workspace,
        record_answers=True,
        tool_exposure="all",
        tool_timeout_s=SESSION_TOOL_TIMEOUT_S,
        replay_scope="window",
    )


def load_mcp_providers(path: str) -> list[BaseToolProvider]:
    """Fresh MCP tool providers for the config file at *path*.

    ``~`` is expanded; a file that cannot be read or is not a valid MCP
    config raises :class:`~localmelo.support.mcp.McpConfigError`.
    """
    from localmelo.support.mcp import McpToolProvider, load_mcp_config

    return [McpToolProvider(load_mcp_config(os.path.expanduser(path)))]


# A step review (request kind "review", #339) is a short JSON answer: no
# thinking, no sampling spread, and a small reply, so a thinking local model
# answers in time instead of timing out into a block.
REVIEW_MAX_TOKENS: Final = 200
# The backends whose OpenAI-compatible servers take chat_template_kwargs
# (None: a --base-url server, a local one).
_TEMPLATE_KWARGS_BACKENDS: Final = frozenset({"vllm", "sglang", None})
_LOCAL_BACKENDS: Final = frozenset({"vllm", "sglang", "mlc", "ollama", None})


def review_body(llm: BaseLLMProvider, backend: str | None) -> dict[str, Any]:
    """The extra request body of a step review for *llm* on *backend*.

    Ollama's native API: ``think`` off and ``options`` with temperature 0 and
    :data:`REVIEW_MAX_TOKENS` (``num_predict``), over the provider's own
    options. An OpenAI-compatible server: temperature 0 and ``max_tokens``,
    and, where the server takes it, ``chat_template_kwargs.enable_thinking``
    off over the provider's own kwargs. A cloud backend gets nothing: its
    models answer fast, and some refuse these fields.
    """
    if backend not in _LOCAL_BACKENDS:
        return {}
    own = getattr(llm, "extra_body", None) or {}
    from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat

    if isinstance(llm, OllamaNativeChat):
        found = own.get("options")
        options: Mapping[str, Any] = found if isinstance(found, Mapping) else {}
        return {
            "think": False,
            "options": {
                **options,
                "temperature": 0,
                "num_predict": REVIEW_MAX_TOKENS,
            },
        }
    body: dict[str, Any] = {"temperature": 0, "max_tokens": REVIEW_MAX_TOKENS}
    if backend in _TEMPLATE_KWARGS_BACKENDS:
        kwargs = own.get("chat_template_kwargs")
        kwargs = dict(kwargs) if isinstance(kwargs, Mapping) else {}
        body["chat_template_kwargs"] = {**kwargs, "enable_thinking": False}
    return body


def with_review_body(llm: BaseLLMProvider, backend: str | None) -> None:
    """Give *llm*'s step reviews :func:`review_body`, keeping its other kinds."""
    body = review_body(llm, backend)
    if body:
        llm.extra_body_by_kind = {**llm.extra_body_by_kind, "review": body}


async def build_session_agent(
    *,
    cfg: Config | None,
    base_url: str | None,
    model: str | None,
    workspace: str,
    options: AgentOptions | None = None,
    embedding_base_url: str | None = None,
    embedding_model: str | None = None,
    request_timeout_s: float | None = None,
    reasoning_field: str | None = None,
    retrieval_extra_body: Mapping[str, Any] | None = None,
    mcp_config: str | None = None,
    browser_driver: str = "off",
    web_fetch: bool = False,
    wrap_llm: LLMWrapper | None = None,
    wrap_embedding: EmbeddingWrapper | None = None,
    identify: bool = False,
    memory: ChatMemory | None = None,
    tools: Sequence[BaseToolProvider] = (),
    agent_class: type[Agent] | None = None,
) -> Agent:
    """A started Agent for a session in *workspace*.

    With *base_url*, an OpenAI-compatible endpoint used as given, with no
    API key, *model* (``default`` when None) and an embedding provider only
    when *embedding_base_url* names one. Otherwise the backends of *cfg*, on
    a copy whose chat model is *model* when one is given and whose
    ``request_timeout_s`` is *request_timeout_s* when that is set; *cfg*
    itself is not changed. ``agent.start()`` is awaited; when it raises, the
    Agent is closed and the exception propagates.

    If *options* is omitted, the config's agent/retrieval defaults are used;
    an explicit options object takes precedence. *reasoning_field* overrides
    the config's replay key, except Ollama's fixed native key. The retrieval
    extra body applies only to retrieve requests and is never saved.
    *browser_driver* names the browser the Agent may read web pages in (see
    :mod:`localmelo.support.browser`); ``off`` gives it no browser tools. Only
    the surfaces that ask a person (the terminal chat and the console) pass it.
    *web_fetch* gives it the ``web_fetch`` tool (see
    :mod:`localmelo.support.web`): the chats and the console's tasks pass
    ``[tools] web_fetch``; direct runs pass nothing, so they never have it.

    *wrap_llm* and *wrap_embedding* are called with the provider the builder
    made, and the Agent is made with what they return, so the provider its
    ``Chat`` holds is the wrapped one. None, the default, passes the provider
    through as this builder always did. An embedding that was not built is
    not wrapped.

    *identify* gives the Agent the identity of a chat (see
    :mod:`~localmelo.support.sessions.identity`): melo, and the model, server
    and place of the backend the Agent is built on, unless *options* set an
    identity already. The terminal chat and the console pass it; workers and
    the other callers do not, and their requests carry no identity.

    *memory*, when it is on, gives a chat memory across chats: its identity
    says so and pins the user's About-you and Preferences notes, and the
    Agent gets ``search_memory`` and ``remember``. The runner does the rest per turn.

    *tools* are more tool providers for this Agent (the main chat's task
    tools), and *agent_class* the class it is made of (the main chat's
    :class:`~localmelo.support.gateway.console.front.FrontAgent`); every other
    caller passes neither.
    """
    agent_options = session_options(
        configured_session_options(cfg) if options is None else options, workspace
    )
    if retrieval_extra_body is not None and (
        agent_options.reasoning == "legacy"
        or agent_options.retriever in ("off", "bm25", "embedding")
    ):
        raise ValueError(
            "retrieval_extra_body needs an llm/auto retriever outside legacy"
        )
    providers = load_mcp_providers(mcp_config) if mcp_config else []
    providers.extend(tools)
    notes: str | None = None
    if memory is not None and memory.enabled:
        try:
            notes = await memory.pinned_quotes()
        except Exception:  # a chat still opens without its pinned notes
            logging.getLogger(__name__).warning("Could not read notes", exc_info=True)
            notes = ""
        providers.append(memory.tools())
    if browser_driver != "off":
        from localmelo.support.browser import provider_for

        browser = provider_for(browser_driver)
        if browser is not None:
            providers.append(browser)
    if web_fetch:
        from localmelo.support.web import WebFetchProvider

        providers.append(WebFetchProvider(browser=browser_driver != "off"))
    llm: BaseLLMProvider | None = None
    embedding: BaseEmbeddingProvider | None = None
    from_config = False
    if base_url:
        from localmelo.support.providers.embedding.openai_compat import (
            OpenAICompatEmbedding,
        )
        from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

        # Passed only when set: httpx reads timeout=None as no timeout at all.
        timeout: dict[str, Any] = (
            {} if request_timeout_s is None else {"timeout": request_timeout_s}
        )
        if (
            agent_options.reasoning in ("replay", "compress+replay")
            and reasoning_field is not None
        ):
            timeout["reasoning_field"] = reasoning_field
        llm = OpenAICompatLLM(base_url=base_url, model=model or "default", **timeout)
        embedding = (
            OpenAICompatEmbedding(
                base_url=embedding_base_url, model=embedding_model or "default"
            )
            if embedding_base_url
            else None
        )
    else:
        if cfg is None:
            raise ValueError("build_session_agent needs a cfg or a base_url")
        cfg = copy.deepcopy(cfg)
        section = getattr(cfg, cfg.chat_backend, None)
        if model and section is not None:
            section.chat_model = model
        if request_timeout_s is not None:
            cfg.request_timeout_s = request_timeout_s
        if reasoning_field is not None:
            if cfg.chat_backend == "ollama":
                raise ValueError(
                    "--reasoning-field is not available with Ollama's fixed thinking field"
                )
            if section is not None:
                section.reasoning_field = reasoning_field
        if (
            wrap_llm is None
            and wrap_embedding is None
            and retrieval_extra_body is None
            and agent_options.reasoning == "legacy"
        ):
            # Keep the original legacy path. Other modes may be refused by
            # Agent.__init__, so own their providers to close them on refusal.
            from_config = True
        else:
            # The providers must exist before the Agent's Chat is made, so
            # build them here exactly as Agent(config=...) would.
            from localmelo.melo.agent.agent import _providers_from_config

            llm, embedding = _providers_from_config(cfg)
    if identify and agent_options.identity is None:
        identity = None
        if base_url:
            kept = kept_across_chats(agent_options, has_embedding=embedding is not None)
            identity = url_identity(base_url, model, kept, notes)
        elif cfg is not None:
            kept = kept_across_chats(agent_options, has_embedding=cfg.has_embedding)
            identity = config_identity(cfg, kept, notes)
        agent_options = dataclasses.replace(agent_options, identity=identity)
    try:
        if not from_config:
            assert llm is not None
            replay = agent_options.reasoning in ("replay", "compress+replay") or (
                agent_options.reasoning == "compress" and llm.requires_reasoning_replay
            )
            if cfg is not None and cfg.chat_backend != "ollama" and not replay:
                llm.reasoning_field = None
            if retrieval_extra_body is not None:
                from localmelo.support.providers.llm._protocol import prepare_extra_body

                llm.extra_body_by_kind = {
                    "retrieve": prepare_extra_body(
                        copy.deepcopy(dict(retrieval_extra_body))
                    )
                }
            with_review_body(llm, None if cfg is None else cfg.chat_backend)
            if wrap_llm is not None:
                llm = wrap_llm(llm)
            if wrap_embedding is not None and embedding is not None:
                embedding = wrap_embedding(embedding)
            agent = (agent_class or Agent)(
                llm=llm,
                embedding=embedding,
                options=agent_options,
                tool_providers=providers,
            )
        else:
            assert cfg is not None
            agent = (agent_class or Agent)(
                config=cfg, options=agent_options, tool_providers=providers
            )
            built = getattr(agent, "_llm", None)
            if isinstance(built, BaseLLMProvider):
                with_review_body(built, cfg.chat_backend)
    except BaseException:
        # No Agent exists to own these if a wrapper or constructor refused.
        for provider in (llm, embedding, *providers):
            if provider is not None:
                with contextlib.suppress(Exception):
                    await provider.close()
        raise
    try:
        await agent.start()
    except BaseException:
        await agent.close()
        raise
    return agent
