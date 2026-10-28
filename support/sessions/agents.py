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
import os
from collections.abc import Callable, Collection, Mapping
from typing import TYPE_CHECKING, Any, Final

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.contracts.tools import BaseToolProvider

if TYPE_CHECKING:
    from localmelo.support.config import Config

# Shell and Python commands of a session stop after this many seconds: test
# suites, installs and builds take longer than a task's 30 s and 60 s.
SESSION_TOOL_TIMEOUT_S: Final = 600.0

# What a surface may do to a provider the builder built, before the Agent
# is made from it: the console puts its scheduler gate in front of every
# call this way.
LLMWrapper = Callable[[BaseLLMProvider], BaseLLMProvider]
EmbeddingWrapper = Callable[[BaseEmbeddingProvider], BaseEmbeddingProvider]


def configured_session_options(
    cfg: Config | None,
    options: AgentOptions | None = None,
    *,
    explicit: Collection[str] = (),
) -> AgentOptions:
    """Config defaults for a session; names in *explicit* retain the caller's values."""
    options = options or AgentOptions()
    if cfg is None:
        return options
    fields: dict[str, Any] = {
        name: getattr(cfg.agent, name)
        for name in ("reasoning", "replay_turns", "note_writer")
        if name not in explicit
    }
    if fields.get("reasoning", options.reasoning) != "legacy":
        for name, key in (
            ("retriever", "retriever"),
            ("recall_at", "recall_at"),
            ("recall_k", "k"),
            ("recall_kinds", "kinds"),
            ("recall_placement", "placement"),
            ("recall_min_score", "min_score"),
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
    wrap_llm: LLMWrapper | None = None,
    wrap_embedding: EmbeddingWrapper | None = None,
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

    *wrap_llm* and *wrap_embedding* are called with the provider the builder
    made, and the Agent is made with what they return, so the provider its
    ``Chat`` holds is the wrapped one. None, the default, passes the provider
    through as this builder always did. An embedding that was not built is
    not wrapped.
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
            if wrap_llm is not None:
                llm = wrap_llm(llm)
            if wrap_embedding is not None and embedding is not None:
                embedding = wrap_embedding(embedding)
            agent = Agent(
                llm=llm,
                embedding=embedding,
                options=agent_options,
                tool_providers=providers,
            )
        else:
            assert cfg is not None
            agent = Agent(config=cfg, options=agent_options, tool_providers=providers)
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
