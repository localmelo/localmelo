"""The terminal UI's model backend: which server and model, and its Agents.

:class:`AgentFactory` knows where the session's model comes from (the
config, a URL given with ``--base-url``, a server found on this machine,
or the first run's answers), builds the session's Agents on it, lists the
server's models, and saves a new choice as the default when the session
may (``/model``, ``/backend``).
"""

from __future__ import annotations

import contextlib
import copy
import io
import os
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.support import config, onboard
from localmelo.support.backends import discovery
from localmelo.support.cli.views import BackendInfo
from localmelo.support.sessions.agents import build_session_agent


def _backend_label(cfg: config.Config, key: str, *, embedding: bool = False) -> str:
    """What the backend *key* of *cfg* is called, or *key* when it is not registered.

    A ``vllm`` backend is ``oMLX`` when the config or a listing says so.
    """
    from localmelo.support.backends.registry import get_backend

    try:
        return get_backend(key).label(cfg, embedding=embedding)
    except KeyError:
        return key


def _embedding_name(cfg: config.Config) -> str:
    """How the banner names the configured embedding: backend and model."""
    if not cfg.has_embedding:
        return ""
    key = cfg.embedding_backend
    section = getattr(cfg, key, None)
    model = getattr(section, "embedding_model", "")
    label = _backend_label(cfg, key, embedding=True)
    return f"{label} · {model}" if model else label


def config_backend(cfg: config.Config, model: str | None) -> BackendInfo:
    """The configured chat backend, with *model* when one is given."""
    key = cfg.chat_backend
    section = getattr(cfg, key, None)
    return BackendInfo(
        key=key,
        label=_backend_label(cfg, key),
        model=model or getattr(section, "chat_model", "") or "default",
        base_url=getattr(section, "chat_url", "") or getattr(section, "base_url", ""),
        embedding=_embedding_name(cfg),
        key_env=getattr(section, "api_key_env", ""),
    )


def url_backend(
    url: str, model: str | None, *, source: str, listed: int = 0, embedding: str = ""
) -> BackendInfo:
    """A server at *url* that is not the config's; *embedding* names its memory's."""
    return BackendInfo(
        key=discovery.OPENAI_COMPATIBLE,
        label=discovery.server_name(url),
        model=model or "default",
        base_url=url,
        embedding=embedding,
        source=source,
        listed=listed,
    )


def url_embedding(base_url: str | None, model: str | None) -> str:
    """How the banner names ``--embedding-base-url``'s server; ``""`` without one."""
    name = discovery.server_name(base_url or "")
    return f"{name} · {model or 'default'}" if base_url else ""


class AgentFactory:
    """Builds the session's Agents on its backend, and changes the backend."""

    def __init__(
        self,
        *,
        backend: BackendInfo,
        cfg: config.Config | None,
        options: AgentOptions,
        workspace: str,
        embedding_base_url: str | None = None,
        embedding_model: str | None = None,
        request_timeout_s: float | None = None,
        reasoning_field: str | None = None,
        retrieval_extra_body: Mapping[str, Any] | None = None,
        mcp_config: str | None = None,
        mcp_source: str = "",
        mcp_servers: list[Any] | None = None,
        saves: bool = True,
        browser_driver: str = "off",
    ) -> None:
        self.backend = backend
        self._cfg = cfg
        self._options = options
        self._workspace = workspace
        self._embedding_base_url = embedding_base_url
        self._embedding_model = embedding_model
        self._request_timeout_s = request_timeout_s
        self._reasoning_field = reasoning_field
        self._retrieval_extra_body = (
            copy.deepcopy(dict(retrieval_extra_body))
            if retrieval_extra_body is not None
            else None
        )
        # The MCP servers file, where it came from, and why MCP is off after
        # a server failed to start.
        self.mcp_config = mcp_config
        self.mcp_source = mcp_source
        self.mcp_error: str | None = None
        if mcp_servers is None and mcp_config:
            from localmelo.support.mcp import load_mcp_config

            mcp_servers = load_mcp_config(os.path.expanduser(mcp_config))
        self.mcp_servers: list[Any] = list(mcp_servers or [])
        # The browser the Agents may read pages in ([browser] driver); /browser changes it.
        self.browser_driver = browser_driver
        # Whether /model and /backend save their choice as the default: not
        # for a --base-url session whose save question was answered no.
        self.saves = saves
        # A configured session moved to a server /backend chose, not yet saved.
        self._new_server = False
        # Whether each server asked about answered like Ollama, by URL.
        self._ollama: dict[str, bool] = {}
        self._live: list[Agent] = []

    @property
    def memory(self) -> bool:
        return self._options.memory

    async def _is_ollama(self, url: str) -> bool:
        """Whether *url* is an Ollama server; asked once per URL, then remembered."""
        if url not in self._ollama:
            self._ollama[url] = await discovery.is_ollama(url)
        return self._ollama[url]

    async def build(self, model: str | None = None, *, url: str | None = None) -> Agent:
        """A started Agent on the backend, with *model* and on *url* when given.

        A configured session keeps its config: on a new *url*, the Agent
        runs on a copy of it as /backend saves it (see :meth:`use_url`).
        """
        model = model or self.backend.model
        try:
            agent = await self._build(
                model, url, self.mcp_config if self.mcp_error is None else None
            )
        except ValueError:
            # An invalid mode/provider pair is not an MCP startup failure.
            raise
        except Exception as exc:
            if self.mcp_config is None or self.mcp_error is not None:
                raise
            # A server that did not start: this session runs without MCP.
            self.mcp_error = str(exc) or type(exc).__name__
            agent = await self._build(model, url, None)
        self._live.append(agent)
        return agent

    async def _build(
        self, model: str, url: str | None, mcp_config: str | None
    ) -> Agent:
        if self._cfg is None:
            return await build_session_agent(
                cfg=None,
                base_url=url or self.backend.base_url,
                model=model,
                workspace=self._workspace,
                options=self._options,
                embedding_base_url=self._embedding_base_url,
                embedding_model=self._embedding_model,
                request_timeout_s=self._request_timeout_s,
                reasoning_field=self._reasoning_field,
                retrieval_extra_body=self._retrieval_extra_body,
                mcp_config=mcp_config,
                browser_driver=self.browser_driver,
            )
        cfg = self._cfg
        if url is not None:
            cfg = self._with_server(url, model, ollama=await self._is_ollama(url))
        return await build_session_agent(
            cfg=cfg,
            base_url=None,
            model=model,
            workspace=self._workspace,
            options=self._options,
            request_timeout_s=self._request_timeout_s,
            reasoning_field=self._reasoning_field,
            retrieval_extra_body=self._retrieval_extra_body,
            mcp_config=mcp_config,
            browser_driver=self.browser_driver,
        )

    def _with_server(self, url: str, model: str, *, ollama: bool) -> config.Config:
        """The session's config with its chat on *url*, as /backend saves it.

        *ollama* (the server answered like Ollama) makes the chat backend
        ``ollama``, which chats on Ollama's native API with its 300 s
        timeout; otherwise it is ``vllm``, the key for any OpenAI-compatible
        server. Embedding, the request timeout and the other table's values
        stay.
        """
        assert self._cfg is not None
        cfg = copy.deepcopy(self._cfg)
        if ollama or cfg.chat_backend != "vllm" or cfg.vllm.chat_url != url:
            cfg.chat_alias = ""  # the file's "omlx" named another server
        if ollama:
            cfg.chat_backend = "ollama"
            cfg.ollama.chat_url = url
            cfg.ollama.chat_model = model
        else:
            cfg.chat_backend = "vllm"
            cfg.vllm.chat_url = url
            cfg.vllm.chat_model = model
        return cfg

    async def close(self, agent: Agent) -> None:
        """Close *agent*, which this factory built."""
        with contextlib.suppress(ValueError):
            self._live.remove(agent)
        await agent.close()

    def live_agents(self) -> list[Agent]:
        """The Agents built and not yet closed (a switch may hold two)."""
        return list(self._live)

    def listable(self) -> bool:
        """Whether the backend's server can list its models (not a cloud vendor)."""
        return self.backend.local and bool(self.backend.base_url)

    async def list_models(self, url: str | None = None) -> list[str]:
        """The models the server lists; :class:`DiscoveryError` when it cannot.

        A listing of the session's own server can show it is oMLX: the
        backend is then named so from here on (see :meth:`relabel`).
        """
        key = discovery.OPENAI_COMPATIBLE if url else self.backend.key
        models = await discovery.list_models(key, url or self.backend.base_url)
        if url is None:
            self.relabel()
        return models

    def relabel(self) -> None:
        """Call the session's server oMLX if a model listing said it is one."""
        backend = self.backend
        if (
            backend.label != discovery.OMLX_NAME
            and backend.key in ("vllm", discovery.OPENAI_COMPATIBLE)
            and discovery.is_omlx(backend.base_url)
        ):
            self.backend = replace(backend, label=discovery.OMLX_NAME)

    def use_model(self, model: str) -> None:
        self.backend = replace(self.backend, model=model, listed=0)

    async def use_url(self, url: str, model: str) -> None:
        """Talk to the server at *url* from now on, with *model*."""
        if self._cfg is not None:
            self._cfg = self._with_server(url, model, ollama=await self._is_ollama(url))
            self._new_server = True
            self.backend = config_backend(self._cfg, model)
            return
        self.backend = url_backend(
            url,
            model,
            source="saved" if self.saves else "--base-url",
            embedding=self.backend.embedding,
        )

    async def save(self) -> str:
        """Save the backend as the default; ``""``, or why it was not saved.

        A configured backend keeps its key and gets the new model. A server
        /backend chose in a configured session is saved as the ``ollama``
        backend when it answered like Ollama, else as ``vllm`` (``omlx`` when
        a listing named oMLX), the rest of the config kept; any other server
        is saved as an OpenAI-compatible one with embedding off, as the first
        run saves it.
        """
        backend = self.backend
        if not self.saves:
            return "not saved (--url)"
        captured = io.StringIO()
        try:
            if self._new_server:
                ollama = await self._is_ollama(backend.base_url)
                with contextlib.redirect_stdout(captured):
                    saved = await onboard.save_chat_server(
                        backend.base_url,
                        backend.model,
                        keep_embedding=True,
                        ollama=ollama,
                    )
                if not saved:
                    return captured.getvalue().strip() or "not saved"
                self._new_server = False
                return ""
            if self._cfg is not None:
                # The session keeps the server it uses; a config file that
                # now names another one is not written over.
                cfg = config.load()
                section = getattr(cfg, cfg.chat_backend, None)
                if (
                    cfg.chat_backend != backend.key
                    or section is None
                    or config_backend(cfg, None).base_url != backend.base_url
                ):
                    return "not saved: the config file changed"
                section.chat_model = backend.model
                config.save(cfg)
                return ""
            ollama = await self._is_ollama(backend.base_url)
            with contextlib.redirect_stdout(captured):
                saved = await onboard.save_chat_server(
                    backend.base_url, backend.model, ollama=ollama
                )
        except (OSError, ValueError, config.ConfigError) as exc:
            # ValueError: a config file that is no longer valid TOML.
            return f"not saved: {exc}"
        if not saved:
            return captured.getvalue().strip() or "not saved"
        self.backend = replace(backend, source="saved", listed=0)
        return ""
