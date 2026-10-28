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

import copy
import dataclasses
import os
from collections.abc import Callable
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


def session_options(options: AgentOptions, workspace: str) -> AgentOptions:
    """*options* for a session's Agent that works in *workspace*.

    It replaces the session's workspace, answer recording, exposure and
    tool timeout only. The caller's ``mode`` survives, so the UI's
    ``interactive`` and the console's ``gateway`` reach the Agent.
    """
    return dataclasses.replace(
        options,
        workspace_root=workspace,
        record_answers=True,
        tool_exposure="all",
        tool_timeout_s=SESSION_TOOL_TIMEOUT_S,
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
    options: AgentOptions,
    embedding_base_url: str | None = None,
    embedding_model: str | None = None,
    request_timeout_s: float | None = None,
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

    *wrap_llm* and *wrap_embedding* are called with the provider the builder
    made, and the Agent is made with what they return, so the provider its
    ``Chat`` holds is the wrapped one. None, the default, passes the provider
    through as this builder always did. An embedding that was not built is
    not wrapped.
    """
    agent_options = session_options(options, workspace)
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
        if wrap_llm is None and wrap_embedding is None:
            # No hook: the Agent builds the config's providers itself, as it
            # did before the hook existed.
            from_config = True
        else:
            # The providers must exist before the Agent's Chat is made, so
            # build them here exactly as Agent(config=...) would.
            from localmelo.melo.agent.agent import _providers_from_config

            llm, embedding = _providers_from_config(cfg)
    if not from_config:
        assert llm is not None
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
    try:
        await agent.start()
    except BaseException:
        await agent.close()
        raise
    return agent
