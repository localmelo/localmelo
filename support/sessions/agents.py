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
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts import work_folder
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.contracts.tools import BaseToolProvider
from localmelo.melo.schema import EXTEND_STEPS
from localmelo.support.backends import providers_for
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.thinking import is_local, thinking_off_fields
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
    "window_eviction",
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


# The three that make up the context budget: all set or none (AgentOptions).
CONTEXT_SETTINGS: Final = ("context_window", "output_reserve", "safety_margin")
# The settings of the pictures that tools return (AgentOptions.tool_images).
IMAGE_SETTINGS: Final = ("tool_images", "tool_images_keep", "tool_images_retire")
# The settings of the model's thinking: what a turn that ends with only thinking
# becomes (AgentOptions.empty_reply_fallback), and on which steps it thinks
# (AgentOptions.thinking_steps).
FALLBACK_SETTINGS: Final = ("empty_reply_fallback", "thinking_steps")


def _configured_keys(
    cfg: Config, options: AgentOptions, names: tuple[str, ...]
) -> AgentOptions:
    """*options* with the config's ``[agent]`` *names*; ValueError for a bad value."""
    values = {name: getattr(cfg.agent, name) for name in names}
    if all(getattr(options, name) == value for name, value in values.items()):
        return options
    try:
        return dataclasses.replace(options, **values)
    except ValueError as exc:
        raise ValueError(f"[agent] {exc}") from None


def configured_image_options(cfg: Config, options: AgentOptions) -> AgentOptions:
    """*options* with the config's ``tool_images`` keys; ValueError for a bad value."""
    return _configured_keys(cfg, options, IMAGE_SETTINGS)


def configured_fallback_options(cfg: Config, options: AgentOptions) -> AgentOptions:
    """*options* with ``[agent] empty_reply_fallback`` and ``thinking_steps``; ValueError for a bad value."""
    return _configured_keys(cfg, options, FALLBACK_SETTINGS)


def configured_attempt_options(
    cfg: Config | None, options: AgentOptions, keep: Collection[str] = ()
) -> AgentOptions:
    """*options* with the config's reflection switch and attempt length.

    ``[agent] reflection = false`` makes ``attempt_policy`` ``"none"``: the run
    never reflects and goes straight to its total. ``true`` leaves the policy
    as the caller set it, so an explicit ``none`` stays. ``[agent]
    steps_per_attempt``, when set, is the attempt length. The names in *keep*
    (``attempt_policy``, ``steps_per_attempt``, ``extend_steps``) retain the
    caller's values, and ``max_steps`` there turns the extension off; a
    value the config holds that the options refuse raises ``ValueError``.
    """
    if cfg is None:
        return options
    changes: dict[str, Any] = {}
    if "attempt_policy" not in keep and cfg.agent.reflection is False:
        changes["attempt_policy"] = "none"
    if "steps_per_attempt" not in keep and cfg.agent.steps_per_attempt is not None:
        changes["steps_per_attempt"] = cfg.agent.steps_per_attempt
    if not options.extend_steps and not {"extend_steps", "max_steps"} & set(keep):
        # A person's session goes on by itself while it makes progress, instead
        # of stopping for them to press Continue (#547). A step budget the
        # person typed (--max-steps) is exact: it is not extended.
        changes["extend_steps"] = EXTEND_STEPS
    changes = {
        name: value
        for name, value in changes.items()
        if getattr(options, name) != value
    }
    if not changes:
        return options
    try:
        return dataclasses.replace(options, **changes)
    except ValueError as exc:
        raise ValueError(f"[agent] {exc}") from None


def configured_context_options(
    cfg: Config | None, options: AgentOptions, overrides: Mapping[str, int | None]
) -> AgentOptions:
    """Resolve context settings together so flags may override individual keys."""
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
        for name in CONTEXT_SETTINGS
    }
    if all(getattr(options, name) == value for name, value in fields.items()):
        return options
    return dataclasses.replace(options, **fields)


def configured_run_options(cfg: Config | None, options: AgentOptions) -> AgentOptions:
    """*options* for a run that is not a person's chat: a task's worker, the gateway's plain session.

    The config's context budget, cache settings, reflection switch and attempt
    length, as a chat gets them (:func:`configured_session_options`). A budget
    the config leaves unset is the model server's (``Agent`` asks it).
    """
    if cfg is None:
        return options
    try:
        options = configured_context_options(cfg, options, {})
    except ValueError as exc:
        raise ValueError(f"[agent] {exc}") from None
    return configured_attempt_options(cfg, configured_cache_options(cfg, options))


def configured_session_options(
    cfg: Config | None,
    options: AgentOptions | None = None,
    *,
    explicit: Collection[str] = (),
    budget_overrides: Mapping[str, int | None] | None = None,
    backend: str | None = None,
) -> AgentOptions:
    """Config defaults for a session; names in *explicit* retain the caller's values.

    A window that the config leaves unset is the default of *backend*, the key
    of the server the session talks to (``openai-compatible`` for a URL); it is
    ``cfg.chat_backend`` when None.
    """
    options = options or AgentOptions()
    overrides = {
        name: getattr(options, name) for name in CONTEXT_SETTINGS if name in explicit
    }
    overrides.update(budget_overrides or {})
    options = configured_context_options(cfg, options, overrides)
    if cfg is None:
        return options
    options = configured_attempt_options(cfg, options, explicit)
    if overrides:
        cfg = copy.deepcopy(cfg)
        for name in CONTEXT_SETTINGS:
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
            *FALLBACK_SETTINGS,
        )
        if name not in explicit
    }
    if "window_messages" in fields:
        fields["window_messages"] = cfg.agent.window_for(backend or cfg.chat_backend)
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


def review_body(llm: BaseLLMProvider, backend: str | None) -> dict[str, Any]:
    """The extra request body of a step review for *llm* on *backend*.

    Ollama's native API: ``think`` off and ``options`` with temperature 0 and
    :data:`REVIEW_MAX_TOKENS` (``num_predict``), over the provider's own
    options. An OpenAI-compatible server: temperature 0 and ``max_tokens``,
    and, where the server takes it, ``chat_template_kwargs.enable_thinking``
    off over the provider's own kwargs
    (:func:`~localmelo.support.providers.llm.thinking.thinking_off_fields`). A
    cloud backend gets nothing: its models answer fast, and some refuse these
    fields.
    """
    if not is_local(backend):
        return {}
    off = thinking_off_fields(llm, backend)
    if isinstance(llm, OllamaNativeChat):
        found = (getattr(llm, "extra_body", None) or {}).get("options")
        options: Mapping[str, Any] = found if isinstance(found, Mapping) else {}
        return {
            **off,
            "options": {
                **options,
                "temperature": 0,
                "num_predict": REVIEW_MAX_TOKENS,
            },
        }
    return {"temperature": 0, "max_tokens": REVIEW_MAX_TOKENS, **off}


def with_review_body(llm: BaseLLMProvider, backend: str | None) -> None:
    """Give *llm*'s step reviews :func:`review_body`, keeping its other kinds."""
    body = review_body(llm, backend)
    if body:
        llm.extra_body_by_kind = {**llm.extra_body_by_kind, "review": body}


def with_quiet_steps(
    llm: BaseLLMProvider, backend: str | None, options: AgentOptions
) -> None:
    """For ``thinking_steps = "first_and_failed"``: give *llm* the fields that turn thinking off on *backend* (#551).

    The agent sends them with the planning steps that need no thinking. A
    service whose provider already has a switch (the hosted ones) keeps it; a
    local server gets :func:`~localmelo.support.providers.llm.thinking.thinking_off_fields`.
    Under ``"all"`` nothing is touched, so the requests are as they were.
    """
    if options.thinking_steps == "all" or llm.thinking_off or llm.thinking_off_body:
        return
    if body := thinking_off_fields(llm, backend):
        llm.thinking_off_body = MappingProxyType(body)


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
    desktop: bool = True,
    web_fetch: bool = False,
    web_task: bool = False,
    web_search: bool = False,
    wrap_llm: LLMWrapper | None = None,
    wrap_embedding: EmbeddingWrapper | None = None,
    identify: bool = False,
    memory: ChatMemory | None = None,
    tools: Sequence[BaseToolProvider] = (),
    agent_class: type[Agent] | None = None,
    chat_id: str | None = None,
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
    the surfaces that ask a person (the terminal chat, a console chat and a
    console task a person's chat owns) pass it. *desktop* false keeps it out of
    other apps' windows whatever ``[browser] app_windows`` says (a console chat
    and its work, which may run with nobody at the screen). *web_fetch* gives it
    the ``web_fetch`` tool (see
    :mod:`localmelo.support.web`), and *web_task* says it is a task's, which
    reads no site with nobody to ask: the chats and the console's tasks pass
    ``[tools] web_fetch``; direct runs pass nothing, so they never have it.
    *web_search* gives it ``web_search`` (see :mod:`localmelo.support.web.search_tool`):
    the same surfaces pass ``[web] search``, and the tool may use the browser
    window when there is one.

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

    *tools* are more tool providers for this Agent (a console chat's task
    tools), and *agent_class* the class it is made of (a console chat's
    :class:`~localmelo.support.gateway.console.front.FrontAgent`); every other
    caller passes neither.

    *chat_id* is the chat this Agent is for: its browser keeps the files its
    pages hand over in the chat's work folder, which the Agent may read (see
    :mod:`localmelo.melo.contracts.work_folder`). Without one the folder is
    this run's own and goes when the run ends.
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

        browser = provider_for(
            browser_driver,
            keep_tabs=cfg is not None and cfg.browser.keep_tabs is True,
            app_windows=desktop and (cfg is None or cfg.browser.app_windows is True),
            opens_apps=desktop,
            live=cfg is None or cfg.browser.live is True,
            live_fps=cfg.browser.live_fps if cfg is not None else 15,
            chat_id=chat_id,
        )
        if browser is not None:
            providers.append(browser)
            agent_options = dataclasses.replace(
                agent_options,
                read_folders=(
                    *agent_options.read_folders,
                    work_folder.path_for(browser.work_id),
                ),
            )
    if web_fetch:
        from localmelo.support import web_local
        from localmelo.support.web import WebFetchProvider

        # The model servers of this run (the config's, and -u) and the console's
        # live port (this config's, which --port changes in memory only) are
        # ports that web_fetch never reads, even if [tools] web_fetch_allow_local
        # lists them.
        providers.append(
            WebFetchProvider(
                browser=browser_driver != "off",
                task=web_task,
                cfg=cfg,
                protect=(
                    base_url,
                    embedding_base_url,
                    *(web_local.urls_of(cfg) if cfg is not None else ()),
                ),
            )
        )
    if web_search:
        from localmelo.support.browser import BrowserToolProvider
        from localmelo.support.web.search_tool import WebSearchProvider

        # The browser window the chat has, if it has one, is a way to search.
        window = next(
            (p for p in providers if isinstance(p, BrowserToolProvider)), None
        )
        providers.append(WebSearchProvider(browser=window, cfg=cfg))
    llm: BaseLLMProvider | None = None
    embedding: BaseEmbeddingProvider | None = None
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
        # The providers must exist before the Agent's Chat is made, and are
        # closed below if a wrapper or the Agent refuses them.
        llm, embedding = providers_for(cfg, agent_options.reasoning)
    if identify and agent_options.identity is None:
        identity = None
        if base_url:
            kept = kept_across_chats(agent_options, has_embedding=embedding is not None)
            identity = url_identity(base_url, model, kept, notes)
        elif cfg is not None:
            kept = kept_across_chats(agent_options, has_embedding=embedding is not None)
            identity = config_identity(cfg, kept, notes)
        agent_options = dataclasses.replace(agent_options, identity=identity)
    try:
        assert llm is not None
        if retrieval_extra_body is not None:
            from localmelo.support.providers.llm._protocol import prepare_extra_body

            llm.extra_body_by_kind = {
                "retrieve": prepare_extra_body(
                    copy.deepcopy(dict(retrieval_extra_body))
                )
            }
        with_review_body(llm, None if cfg is None else cfg.chat_backend)
        # A server given by URL has no backend of its own to name (as in a direct run).
        with_quiet_steps(
            llm, None if base_url or cfg is None else cfg.chat_backend, agent_options
        )
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
