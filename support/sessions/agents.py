"""Build the Agents that sessions run on.

A session's Agent works in a workspace, keeps each turn's answer for the
next one, offers every tool (no retrieval of tools per step) and lets a
shell or Python command run for :data:`SESSION_TOOL_TIMEOUT_S`. The model
comes from a URL given on the command line or from the config file.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING, Any, Final

from localmelo.melo.agent import Agent, AgentOptions

if TYPE_CHECKING:
    from localmelo.support.config import Config

# Shell and Python commands of a session stop after this many seconds: test
# suites, installs and builds take longer than a task's 30 s and 60 s.
SESSION_TOOL_TIMEOUT_S: Final = 600.0


def session_options(options: AgentOptions, workspace: str) -> AgentOptions:
    """*options* for a session's Agent that works in *workspace*."""
    return dataclasses.replace(
        options,
        workspace_root=workspace,
        record_answers=True,
        tool_exposure="all",
        tool_timeout_s=SESSION_TOOL_TIMEOUT_S,
    )


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
) -> Agent:
    """A started Agent for a session in *workspace*.

    With *base_url*, an OpenAI-compatible endpoint used as given, with no
    API key, *model* (``default`` when None) and an embedding provider only
    when *embedding_base_url* names one. Otherwise the backends of *cfg*, on
    a copy whose chat model is *model* when one is given and whose
    ``request_timeout_s`` is *request_timeout_s* when that is set; *cfg*
    itself is not changed. ``agent.start()`` is awaited; when it raises, the
    Agent is closed and the exception propagates.
    """
    agent_options = session_options(options, workspace)
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
        agent = Agent(llm=llm, embedding=embedding, options=agent_options)
    else:
        if cfg is None:
            raise ValueError("build_session_agent needs a cfg or a base_url")
        cfg = copy.deepcopy(cfg)
        section = getattr(cfg, cfg.chat_backend, None)
        if model and section is not None:
            section.chat_model = model
        if request_timeout_s is not None:
            cfg.request_timeout_s = request_timeout_s
        agent = Agent(config=cfg, options=agent_options)
    try:
        await agent.start()
    except BaseException:
        await agent.close()
        raise
    return agent
